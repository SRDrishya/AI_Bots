from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import json
import logging
import re
import secrets
import time
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlencode, urljoin, urlsplit

import httpx
import keyring
from keyring.errors import KeyringError, PasswordDeleteError

from app.services.storage import Storage

log = logging.getLogger(__name__)

PROVIDERS = ("google", "microsoft")
PORTAL_DOMAINS = {
    "linkedin.com": "LinkedIn",
    "linkedinmail.com": "LinkedIn",
    "indeed.com": "Indeed",
    "indeedmail.com": "Indeed",
    "indeedemail.com": "Indeed",
    "naukri.com": "Naukri",
    "naukrimail.com": "Naukri",
}
# Portal senders that only send social/marketing mail (connection invites, newsletters, group posts).
NOISE_SENDERS = re.compile(
    r"^(?:invitations?|newsletters?(?:-noreply)?|editors?-noreply|groups?-noreply|security-noreply|"
    r"learning[-_]?\w*|marketing\w*|news\w*|birthdays?\w*)@",
    re.I,
)
JOB_TERMS = re.compile(
    r"\b(?:jobs?|career\w*|opportunit\w*|hiring|recruit\w*|vacan\w*|opening\w*|"
    r"application\w*|match\w*|position\w*|roles?)\b",
    re.I,
)
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
GOOGLE_AUTHORIZE = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
GOOGLE_API = "https://gmail.googleapis.com/gmail/v1"
MICROSOFT_AUTHORIZE = "https://login.microsoftonline.com/consumers/oauth2/v2.0/authorize"
MICROSOFT_TOKEN = "https://login.microsoftonline.com/consumers/oauth2/v2.0/token"
GRAPH_API = "https://graph.microsoft.com/v1.0"
FLOW_TTL_SECONDS = 600
FETCH_CONCURRENCY = 8
SCAN_DOC = "email_scan.json"
# Re-check a little before the last scan so mail delivered late (or with a skewed clock) isn't missed.
SCAN_OVERLAP = timedelta(hours=1)
KEEP_DAYS = 60          # saved email jobs older than this (by received date) are dropped
MAX_SAVED_JOBS = 500


class EmailIntegrationError(ValueError):
    """A safe, user-facing email integration error."""


class _EmailHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._active_links: list[dict[str, str]] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag in {"script", "style"}:
            self._skip_depth += 1
        if tag == "a" and attrs_dict.get("href"):
            self._active_links.append({"url": attrs_dict["href"] or "", "text": ""})

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "a" and self._active_links:
            link = self._active_links.pop()
            self.links.append((link["text"].strip(), link["url"]))

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self.text.append(data)
        for link in self._active_links:
            link["text"] += data


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _message_parts(payload: dict[str, Any]) -> tuple[str, str]:
    plain: list[str] = []
    markup: list[str] = []

    def walk(part: dict[str, Any]) -> None:
        mime = part.get("mimeType", "")
        data = (part.get("body") or {}).get("data", "")
        if data and mime in {"text/plain", "text/html"}:
            decoded = _b64url_decode(data).decode("utf-8", errors="replace")
            (plain if mime == "text/plain" else markup).append(decoded)
        for child in part.get("parts", []):
            walk(child)

    walk(payload)
    return "\n".join(plain), "\n".join(markup)


def _clean_content(plain: str, markup: str) -> tuple[str, list[tuple[str, str]]]:
    links: list[tuple[str, str]] = []
    if markup:
        parser = _EmailHtmlParser()
        parser.feed(markup)
        links.extend(parser.links)
        visible = " ".join(parser.text)
    else:
        visible = plain
    for url in URL_RE.findall(plain or visible):
        links.append(("", url.rstrip(".,;:!?")))
    visible = html.unescape(re.sub(r"\s+", " ", visible)).strip()
    return visible, links


def _portal(sender: str) -> str | None:
    address = parseaddr(sender)[1].lower()
    if NOISE_SENDERS.match(address):
        return None
    domain = address.rsplit("@", 1)[-1] if "@" in address else sender.lower().strip()
    for known_domain, label in PORTAL_DOMAINS.items():
        if domain == known_domain or domain.endswith("." + known_domain):
            return label
    return None


def _link_candidates(raw_links: list[tuple[str, str]], base_url: str = "") -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for label, raw_url in raw_links:
        url = urljoin(base_url, html.unescape(raw_url.strip()))
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            continue
        host = parts.hostname.lower()
        if (any(term in host for term in ("unsubscribe", "preferences", "tracking", "click."))
                or re.search(r"unsubscribe|preferences|opt.?out", parts.path, re.I)):
            continue
        label = re.sub(r"\s+", " ", label).strip()[:120]
        if not (JOB_TERMS.search(label) or re.search(r"\b(view|apply|see|open|details)\b", label, re.I)
                or re.search(r"/(?:jobs?|careers?|vacanc|openings?|positions?|apply)(?:/|$)", parts.path, re.I)
                or _portal(host)):
            continue
        if url in seen:
            continue
        seen.add(url)
        result.append({"label": label or f"Open {host}", "url": url})
        if len(result) == 10:
            break
    return result


def _iso_timestamp(value: str | int | None) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, int) or str(value).isdigit():
            return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).isoformat(timespec="seconds")
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")
    except (ValueError, OverflowError, OSError):
        return ""


def _is_opportunity(portal: str | None, subject: str, text: str) -> bool:
    return portal is not None and bool(JOB_TERMS.search(f"{subject} {text[:1000]}"))


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    if response.is_error:
        detail = ""
        try:
            body = response.json()
            detail = str(body.get("error_description") or body.get("error", {}).get("message") or "")
        except (ValueError, AttributeError):
            pass
        raise EmailIntegrationError(detail[:240] or f"Mail provider request failed ({response.status_code})")
    try:
        return response.json()
    except ValueError as exc:
        raise EmailIntegrationError("Mail provider returned an unreadable response") from exc


class EmailJobService:
    def __init__(
        self,
        storage: Storage,
        google_client_id: str = "",
        google_client_secret: str = "",
        microsoft_client_id: str = "",
        microsoft_client_secret: str = "",
        redirect_base: str = "http://127.0.0.1:8000",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.storage = storage
        self.credentials = {
            "google": (google_client_id.strip(), google_client_secret.strip()),
            "microsoft": (microsoft_client_id.strip(), microsoft_client_secret.strip()),
        }
        self.redirect_base = redirect_base.rstrip("/")
        self.transport = transport
        self.pending_flows: dict[str, dict[str, Any]] = {}
        self._scan_lock = asyncio.Lock()
        self.keyring_service = "job-search-agent-" + hashlib.sha256(
            str(storage.root.resolve()).encode("utf-8")
        ).hexdigest()[:24]

    def redirect_uri(self, provider: str) -> str:
        return f"{self.redirect_base}/api/emails/oauth/{provider}/callback"

    def _configured(self, provider: str) -> bool:
        client_id, client_secret = self.credentials[provider]
        return bool(client_id and client_secret)

    def accounts(self) -> dict[str, Any]:
        saved = self.storage.load_doc("email_accounts.json", {})
        accounts = []
        for provider in PROVIDERS:
            email = saved.get(provider, "")
            try:
                token_saved = bool(keyring.get_password(self.keyring_service, provider))
            except KeyringError as exc:
                log.error("OS credential store unavailable: %s", exc)
                raise EmailIntegrationError("The operating-system credential store is unavailable") from exc
            accounts.append({
                "provider": provider,
                "email": email,
                "configured": self._configured(provider),
                "connected": bool(email and token_saved),
            })
        return {"accounts": accounts}

    def begin_auth(self, provider: str) -> str:
        self._check_provider(provider)
        if not self._configured(provider):
            raise EmailIntegrationError(f"{provider.title()} OAuth credentials are not configured")
        self._expire_flows()
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        self.pending_flows[state] = {
            "provider": provider,
            "verifier": verifier,
            "created": time.monotonic(),
        }
        if provider == "google":
            params = {
                "client_id": self.credentials[provider][0],
                "redirect_uri": self.redirect_uri(provider),
                "response_type": "code",
                "scope": "openid email https://www.googleapis.com/auth/gmail.readonly",
                "access_type": "offline",
                "prompt": "consent",
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
            return f"{GOOGLE_AUTHORIZE}?{urlencode(params)}"
        params = {
            "client_id": self.credentials[provider][0],
            "redirect_uri": self.redirect_uri(provider),
            "response_type": "code",
            "response_mode": "query",
            "scope": (
                "openid email profile offline_access https://graph.microsoft.com/User.Read "
                "https://graph.microsoft.com/Mail.Read"
            ),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        return f"{MICROSOFT_AUTHORIZE}?{urlencode(params)}"

    async def finish_auth(self, provider: str, code: str, state: str) -> None:
        self._check_provider(provider)
        self._expire_flows()
        flow = self.pending_flows.pop(state, None)
        if not flow or flow["provider"] != provider:
            raise EmailIntegrationError("Email sign-in expired or failed its security check; please try again")
        client_id, client_secret = self.credentials[provider]
        token_url = GOOGLE_TOKEN if provider == "google" else MICROSOFT_TOKEN
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "code_verifier": flow["verifier"],
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri(provider),
        }
        payload = await self._post_json(token_url, data=data)
        refresh_token = payload.get("refresh_token")
        if not refresh_token:
            raise EmailIntegrationError("Provider did not return a refresh token; disconnect and reconnect with consent")
        access_token = payload.get("access_token")
        if not access_token:
            raise EmailIntegrationError("Provider did not return an access token")
        account = await self._get_account_email(provider, access_token)
        try:
            keyring.set_password(self.keyring_service, provider, refresh_token)
        except KeyringError as exc:
            log.error("Could not save OAuth token in OS credential store: %s", exc)
            raise EmailIntegrationError("Could not save the sign-in securely in the operating-system credential store") from exc
        saved = self.storage.load_doc("email_accounts.json", {})
        saved[provider] = account
        self.storage.save_doc("email_accounts.json", saved)

    async def disconnect(self, provider: str) -> None:
        self._check_provider(provider)
        try:
            keyring.delete_password(self.keyring_service, provider)
        except PasswordDeleteError:
            pass
        except KeyringError as exc:
            log.error("Could not remove OAuth token from OS credential store: %s", exc)
            raise EmailIntegrationError("Could not remove the sign-in from the operating-system credential store") from exc
        saved = self.storage.load_doc("email_accounts.json", {})
        saved.pop(provider, None)
        self.storage.save_doc("email_accounts.json", saved)
        scan = self._scan_doc()
        scan["cursors"].pop(provider, None)
        scan["jobs"] = [job for job in scan["jobs"] if job.get("provider") != provider]
        self.storage.save_doc(SCAN_DOC, scan)

    # ------------------------------------------------------------------ saved results
    def _scan_doc(self) -> dict[str, Any]:
        doc = self.storage.load_doc(SCAN_DOC, {})
        return {"cursors": dict(doc.get("cursors") or {}), "jobs": list(doc.get("jobs") or []),
                "last_scan_at": doc.get("last_scan_at", "")}

    def saved_opportunities(self) -> dict[str, Any]:
        """Jobs found by earlier scans, newest first — no mailbox access."""
        scan = self._scan_doc()
        connected = self.storage.load_doc("email_accounts.json", {})
        jobs = [dict(job, is_new=False) for job in scan["jobs"]
                if connected.get(job.get("provider")) == job.get("account")]
        jobs.sort(key=lambda item: item.get("received_at", ""), reverse=True)
        return {"jobs": jobs, "errors": [], "new_count": 0, "last_scan_at": scan["last_scan_at"],
                "scanned_since": {}}

    async def scan(self, *, full: bool = False, days: int = 7, limit: int = 100) -> dict[str, Any]:
        """Fetch only mail received since each account's previous scan, merge it into the saved list.

        `full=True` (or an account's first scan) looks back `days` days instead.
        """
        async with self._scan_lock:  # the daily schedule and the button must not interleave
            now = datetime.now(timezone.utc)
            scan = self._scan_doc()
            connected = self.storage.load_doc("email_accounts.json", {})
            since: dict[str, datetime] = {}
            for provider in PROVIDERS:
                if not connected.get(provider):
                    continue
                cursor = scan["cursors"].get(provider) or {}
                start = now - timedelta(days=days)
                if not full and cursor.get("account") == connected[provider] and cursor.get("scanned_at"):
                    try:
                        previous = datetime.fromisoformat(cursor["scanned_at"]) - SCAN_OVERLAP
                        start = max(previous, now - timedelta(days=365))
                    except ValueError:
                        pass
                since[provider] = start

            found = await self.find_opportunities(since=since, limit=limit)

            known = {job["id"]: job for job in scan["jobs"]}
            new_ids: set[str] = set()
            for job in found["jobs"]:
                if job["id"] not in known:
                    new_ids.add(job["id"])
                    known[job["id"]] = dict(job, first_seen_at=now.isoformat(timespec="seconds"))
            failed = {err["provider"] for err in found["errors"]}
            for provider in since:
                if provider not in failed:  # a failed account is retried from its old cursor next time
                    scan["cursors"][provider] = {"account": connected[provider],
                                                 "scanned_at": now.isoformat(timespec="seconds")}
            cutoff = (now - timedelta(days=KEEP_DAYS)).isoformat(timespec="seconds")
            jobs = sorted((job for job in known.values() if job.get("received_at", "") >= cutoff),
                          key=lambda item: item.get("received_at", ""), reverse=True)[:MAX_SAVED_JOBS]
            for job in jobs:
                job.pop("is_new", None)
            scan["jobs"] = jobs
            scan["last_scan_at"] = now.isoformat(timespec="seconds")
            self.storage.save_doc(SCAN_DOC, scan)

            listed = [dict(job, is_new=job["id"] in new_ids) for job in jobs
                      if connected.get(job.get("provider")) == job.get("account")]
            return {
                "jobs": listed,
                "errors": found["errors"],
                "new_count": len(new_ids),
                "last_scan_at": scan["last_scan_at"],
                "scanned_since": {p: d.isoformat(timespec="seconds") for p, d in since.items()},
            }

    async def find_opportunities(
        self, *, since: dict[str, datetime] | None = None, days: int = 30, limit: int = 100
    ) -> dict[str, Any]:
        """Search every connected inbox from `since[provider]` (default: the last `days` days).

        One failing account doesn't hide the others' results.
        """
        saved = self.storage.load_doc("email_accounts.json", {})
        connected = [p for p in PROVIDERS if saved.get(p)]
        default_since = datetime.now(timezone.utc) - timedelta(days=days)

        async def search(provider: str) -> list[dict[str, Any]]:
            start = (since or {}).get(provider, default_since)
            token = await self._refresh_access_token(provider)
            if provider == "google":
                return await self._google_messages(token, saved[provider], start, limit)
            return await self._microsoft_messages(token, saved[provider], start, limit)

        outcomes = await asyncio.gather(*(search(p) for p in connected), return_exceptions=True)
        jobs: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for provider, outcome in zip(connected, outcomes):
            if isinstance(outcome, EmailIntegrationError):
                errors.append({"provider": provider, "account": saved[provider], "message": str(outcome)})
            elif isinstance(outcome, BaseException):
                log.error("Email search failed for %s", provider, exc_info=outcome)
                errors.append({"provider": provider, "account": saved[provider],
                               "message": "Unexpected error while searching this inbox"})
            else:
                jobs.extend(outcome)
        jobs.sort(key=lambda item: item["received_at"], reverse=True)
        return {"jobs": jobs[:limit], "errors": errors}

    async def _refresh_access_token(self, provider: str) -> str:
        try:
            refresh_token = keyring.get_password(self.keyring_service, provider)
        except KeyringError as exc:
            log.error("OS credential store unavailable: %s", exc)
            raise EmailIntegrationError("The operating-system credential store is unavailable") from exc
        if not refresh_token:
            raise EmailIntegrationError(f"{provider.title()} is no longer connected; reconnect the account")
        client_id, client_secret = self.credentials[provider]
        if not self._configured(provider):
            raise EmailIntegrationError(f"{provider.title()} OAuth credentials are not configured")
        token_url = GOOGLE_TOKEN if provider == "google" else MICROSOFT_TOKEN
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
        if provider == "microsoft":
            data["scope"] = (
                "openid email profile offline_access https://graph.microsoft.com/User.Read "
                "https://graph.microsoft.com/Mail.Read"
            )
        payload = await self._post_json(token_url, data=data)
        access_token = payload.get("access_token")
        if not access_token:
            raise EmailIntegrationError("Could not refresh the sign-in; reconnect the account")
        rotated = payload.get("refresh_token")
        if rotated:
            try:
                keyring.set_password(self.keyring_service, provider, rotated)
            except KeyringError as exc:
                log.error("Could not update OAuth token in OS credential store: %s", exc)
                raise EmailIntegrationError("Could not update the sign-in securely in the operating-system credential store") from exc
        return access_token

    async def _google_messages(
        self, token: str, account: str, since: datetime, limit: int
    ) -> list[dict[str, Any]]:
        senders = " ".join(f"from:{domain}" for domain in PORTAL_DOMAINS)
        # after:<epoch seconds> is exact to the second, so a daily scan only lists that day's mail.
        query = f"after:{int(since.timestamp())} {{{senders}}}"
        async with self._http() as client:
            ids: list[str] = []
            page_token = ""
            # Gmail's query already restricts to portal senders; over-fetch since some aren't job mail.
            while len(ids) < limit * 2:
                params = {"q": query, "maxResults": "500"}
                if page_token:
                    params["pageToken"] = page_token
                listing = await self._get_json(
                    f"{GOOGLE_API}/users/me/messages", token, params=params, client=client
                )
                ids.extend(item["id"] for item in listing.get("messages", []))
                page_token = listing.get("nextPageToken", "")
                if not page_token:
                    break
            ids = ids[:limit * 2]
            gate = asyncio.Semaphore(FETCH_CONCURRENCY)

            async def fetch(message_id: str) -> dict[str, Any]:
                async with gate:
                    return await self._get_json(
                        f"{GOOGLE_API}/users/me/messages/{message_id}", token,
                        params={"format": "full"}, client=client,
                    )

            messages = await asyncio.gather(*(fetch(i) for i in ids), return_exceptions=True)
        failures = [m for m in messages if isinstance(m, BaseException)]
        if failures and len(failures) == len(messages):
            raise failures[0]
        results = []
        for message in messages:
            if isinstance(message, BaseException):
                log.warning("Skipped an unreadable Gmail message: %s", message)
                continue
            result = self._google_opportunity(message, account)
            if result:
                results.append(result)
        return results[:limit]

    def _google_opportunity(self, message: dict[str, Any], account: str) -> dict[str, Any] | None:
        payload = message.get("payload") or {}
        headers = {h.get("name", "").lower(): h.get("value", "") for h in payload.get("headers", [])}
        subject = headers.get("subject", "(No subject)")
        sender = headers.get("from", "")
        portal = _portal(sender)
        plain, markup = _message_parts(payload)
        body, raw_links = _clean_content(plain, markup)
        snippet = message.get("snippet", "") or body
        if not _is_opportunity(portal, subject, f"{snippet} {body}"):
            return None
        thread_id = message.get("threadId", "")
        # authuser opens the right Google account when several are signed in to the browser.
        message_url = (
            f"https://mail.google.com/mail/?{urlencode({'authuser': account})}#all/{thread_id}"
            if thread_id else ""
        )
        return {
            "id": f"google:{message.get('id', '')}",
            "provider": "google",
            "account": account,
            "portal": portal,
            "subject": subject,
            "sender": parseaddr(sender)[1] or sender,
            "received_at": _iso_timestamp(message.get("internalDate")),
            "preview": re.sub(r"\s+", " ", snippet).strip()[:320],
            "links": _link_candidates(raw_links),
            "message_url": message_url,
        }

    async def _microsoft_messages(
        self, token: str, account: str, since: datetime, limit: int
    ) -> list[dict[str, Any]]:
        since = since.astimezone(timezone.utc)
        select = "id,subject,from,receivedDateTime,bodyPreview,webLink"
        senders = " OR ".join(f"from:{domain}" for domain in PORTAL_DOMAINS)
        search_params = {
            "$search": f'"received>={since:%Y-%m-%d} AND ({senders})"',
            "$select": select,
            "$top": "50",
        }
        scan_params = {
            "$filter": f"receivedDateTime ge {since:%Y-%m-%dT%H:%M:%SZ}",
            "$select": select,
            "$top": "100",
            "$orderby": "receivedDateTime desc",
        }
        async with self._http() as client:
            # Server-side search reaches job mail anywhere in the date range, not just the newest messages.
            try:
                results = await self._microsoft_scan(
                    client, token, account, since, limit, search_params, max_pages=10
                )
            except EmailIntegrationError as exc:
                log.info("Outlook search unavailable, scanning by date instead: %s", exc)
                results = []
            if not results:
                results = await self._microsoft_scan(
                    client, token, account, since, limit, scan_params, max_pages=5
                )
        return results

    async def _microsoft_scan(
        self,
        client: httpx.AsyncClient,
        token: str,
        account: str,
        since: datetime,
        limit: int,
        params: dict[str, str] | None,
        *,
        max_pages: int,
    ) -> list[dict[str, Any]]:
        url = f"{GRAPH_API}/me/messages"
        since_iso = since.isoformat(timespec="seconds")
        results: list[dict[str, Any]] = []
        pages = 0
        while url and pages < max_pages and len(results) < limit:
            response = await self._get_json(url, token, params=params, client=client)
            params = None  # nextLink already carries the query
            for message in response.get("value", []):
                sender = ((message.get("from") or {}).get("emailAddress") or {}).get("address", "")
                received = _iso_timestamp(message.get("receivedDateTime"))
                if not received or received < since_iso:
                    continue
                if not _is_opportunity(
                    _portal(sender), message.get("subject", ""), message.get("bodyPreview", "")
                ):
                    continue
                body = await self._get_json(
                    f"{GRAPH_API}/me/messages/{message['id']}",
                    token,
                    params={"$select": "body"},
                    client=client,
                )
                message["body"] = body.get("body", {})
                result = self._microsoft_opportunity(message, account)
                if result:
                    results.append(result)
                    if len(results) >= limit:
                        break
            url = response.get("@odata.nextLink", "")
            pages += 1
        return results

    def _microsoft_opportunity(self, message: dict[str, Any], account: str) -> dict[str, Any] | None:
        sender = ((message.get("from") or {}).get("emailAddress") or {}).get("address", "")
        portal = _portal(sender)
        body_obj = message.get("body") or {}
        content = body_obj.get("content", "")
        if body_obj.get("contentType", "").lower() == "html":
            body, raw_links = _clean_content("", content)
        else:
            body = re.sub(r"\s+", " ", content).strip()
            raw_links = [("", url.rstrip(".,;:!?")) for url in URL_RE.findall(content)]
        subject = message.get("subject") or "(No subject)"
        preview = message.get("bodyPreview") or body
        if not _is_opportunity(portal, subject, f"{preview} {body}"):
            return None
        return {
            "id": f"microsoft:{message.get('id', '')}",
            "provider": "microsoft",
            "account": account,
            "portal": portal,
            "subject": subject,
            "sender": sender,
            "received_at": _iso_timestamp(message.get("receivedDateTime")),
            "preview": re.sub(r"\s+", " ", preview).strip()[:320],
            "links": _link_candidates(raw_links),
            "message_url": message.get("webLink", ""),
        }

    async def _get_account_email(self, provider: str, token: str) -> str:
        if provider == "google":
            payload = await self._get_json("https://openidconnect.googleapis.com/v1/userinfo", token)
            email = payload.get("email", "")
        else:
            payload = await self._get_json(
                f"{GRAPH_API}/me", token, params={"$select": "mail,userPrincipalName"}
            )
            email = payload.get("mail") or payload.get("userPrincipalName", "")
        if not email:
            raise EmailIntegrationError("Could not read the email address from the provider account")
        return email

    async def _get_json(
        self,
        url: str,
        token: str,
        params: dict[str, str] | list[tuple[str, str]] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {token}"}
        try:
            if client is not None:
                response = await client.get(url, params=params, headers=headers)
            else:
                async with self._http() as own:
                    response = await own.get(url, params=params, headers=headers)
        except httpx.RequestError as exc:
            raise EmailIntegrationError("Could not reach the email provider") from exc
        return _safe_json(response)

    def _http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport, timeout=25)

    async def _post_json(self, url: str, *, data: dict[str, str]) -> dict[str, Any]:
        async with self._http() as client:
            try:
                response = await client.post(url, data=data)
            except httpx.RequestError as exc:
                raise EmailIntegrationError("Could not reach the email provider") from exc
        return _safe_json(response)

    @staticmethod
    def _check_provider(provider: str) -> None:
        if provider not in PROVIDERS:
            raise EmailIntegrationError("Unknown email provider")

    def _expire_flows(self) -> None:
        cutoff = time.monotonic() - FLOW_TTL_SECONDS
        self.pending_flows = {
            state: flow for state, flow in self.pending_flows.items() if flow["created"] >= cutoff
        }
