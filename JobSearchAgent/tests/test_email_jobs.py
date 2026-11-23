from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from app.services.email_jobs import EmailIntegrationError, EmailJobService, _portal
from app.services.storage import Storage


RECENT = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _encoded(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def test_google_message_becomes_safe_job_opportunity(tmp_path):
    service = EmailJobService(Storage(tmp_path / "data"))
    message = {
        "id": "message-1",
        "threadId": "thread-1",
        "internalDate": "1730000000000",
        "snippet": "Data Engineer role in Chennai",
        "payload": {
            "headers": [
                {"name": "From", "value": "LinkedIn Jobs <alerts@linkedin.com>"},
                {"name": "Subject", "value": "LinkedIn job alert: Data Engineer"},
            ],
            "mimeType": "text/html",
            "body": {"data": _encoded(
                '<a href="https://www.linkedin.com/jobs/view/123">View job</a>'
                '<a href="https://www.linkedin.com/unsubscribe">Unsubscribe</a>'
            )},
        },
    }

    result = service._google_opportunity(message, "person@gmail.com")

    assert result is not None
    assert result["portal"] == "LinkedIn"
    assert result["subject"] == "LinkedIn job alert: Data Engineer"
    assert result["links"] == [{
        "label": "View job",
        "url": "https://www.linkedin.com/jobs/view/123",
    }]
    assert result["message_url"].endswith("#all/thread-1")
    assert "authuser=person%40gmail.com" in result["message_url"]


def test_unrelated_portal_message_is_not_a_job(tmp_path):
    service = EmailJobService(Storage(tmp_path / "data"))
    message = {
        "id": "message-2",
        "payload": {
            "headers": [
                {"name": "From", "value": "Indeed <updates@indeed.com>"},
                {"name": "Subject", "value": "Your weekly account summary"},
            ],
            "mimeType": "text/plain",
            "body": {"data": _encoded("Change your notification preferences.")},
        },
    }

    assert service._google_opportunity(message, "person@gmail.com") is None


def test_microsoft_message_lists_naukri_job_link(tmp_path):
    service = EmailJobService(Storage(tmp_path / "data"))
    message = {
        "id": "graph-message",
        "subject": "Python developer jobs matching your profile",
        "from": {"emailAddress": {"address": "alerts@naukri.com"}},
        "receivedDateTime": "2026-11-20T08:30:00Z",
        "bodyPreview": "New Python developer jobs",
        "body": {
            "contentType": "html",
            "content": '<a href="https://www.naukri.com/job-listings/123">Apply now</a>',
        },
        "webLink": "https://outlook.office.com/mail/id/graph-message",
    }

    result = service._microsoft_opportunity(message, "person@hotmail.com")

    assert result is not None
    assert result["portal"] == "Naukri"
    assert result["received_at"] == "2026-11-20T08:30:00+00:00"
    assert result["links"][0]["url"] == "https://www.naukri.com/job-listings/123"
    assert result["message_url"].startswith("https://outlook.office.com/")


@pytest.mark.asyncio
async def test_google_oauth_uses_state_and_stores_refresh_token_securely(tmp_path, monkeypatch):
    storage = Storage(tmp_path / "data")
    saved_credentials = {}

    def set_password(service, user, password):
        saved_credentials[(service, user)] = password

    monkeypatch.setattr("app.services.email_jobs.keyring.set_password", set_password)

    def handle(request):
        if request.url.host == "oauth2.googleapis.com":
            assert request.method == "POST"
            body = request.content.decode()
            assert "grant_type=authorization_code" in body
            return httpx.Response(200, json={"access_token": "short-lived", "refresh_token": "long-lived"})
        if request.url.host == "openidconnect.googleapis.com":
            assert request.headers["Authorization"] == "Bearer short-lived"
            return httpx.Response(200, json={"email": "person@gmail.com"})
        raise AssertionError(f"Unexpected request: {request.url}")

    service = EmailJobService(
        storage,
        google_client_id="client-id",
        google_client_secret="client-secret",
        transport=httpx.MockTransport(handle),
    )
    auth_url = service.begin_auth("google")
    query = parse_qs(urlsplit(auth_url).query)
    state = query["state"][0]
    assert query["scope"][0].endswith("gmail.readonly")
    assert query["code_challenge_method"] == ["S256"]

    await service.finish_auth("google", "auth-code", state)

    assert storage.load_doc("email_accounts.json", {}) == {"google": "person@gmail.com"}
    assert list(saved_credentials.values()) == ["long-lived"]
    with pytest.raises(EmailIntegrationError, match="security check"):
        await service.finish_auth("google", "auth-code", state)


@pytest.mark.asyncio
async def test_microsoft_search_fetches_bodies_only_for_matching_job_emails(tmp_path, monkeypatch):
    storage = Storage(tmp_path / "data")
    storage.save_doc("email_accounts.json", {"microsoft": "person@hotmail.com"})
    monkeypatch.setattr("app.services.email_jobs.keyring.get_password", lambda *args: "refresh-token")
    body_requests = []

    def handle(request):
        if request.url.host == "login.microsoftonline.com":
            return httpx.Response(200, json={"access_token": "access-token"})
        if request.url.path == "/v1.0/me/messages":
            assert "body" not in request.url.params["$select"].split(",")
            return httpx.Response(200, json={"value": [
                {
                    "id": "job-message",
                    "subject": "Indeed job alert: Data Analyst",
                    "from": {"emailAddress": {"address": "alerts@indeed.com"}},
                    "receivedDateTime": RECENT,
                    "bodyPreview": "New analyst jobs",
                    "webLink": "https://outlook.office.com/mail/id/job-message",
                },
                {
                    "id": "newsletter",
                    "subject": "Indeed weekly account summary",
                    "from": {"emailAddress": {"address": "updates@indeed.com"}},
                    "bodyPreview": "Change your preferences",
                },
            ]})
        if request.url.path == "/v1.0/me/messages/job-message":
            body_requests.append(request.url.path)
            assert request.url.params["$select"] == "body"
            return httpx.Response(200, json={
                "body": {
                    "contentType": "html",
                    "content": '<a href="https://www.indeed.com/viewjob?jk=1">View job</a>',
                },
            })
        raise AssertionError(f"Unexpected request: {request.url}")

    service = EmailJobService(
        storage,
        microsoft_client_id="client-id",
        microsoft_client_secret="client-secret",
        transport=httpx.MockTransport(handle),
    )

    result = await service.find_opportunities()

    assert [job["id"] for job in result["jobs"]] == ["microsoft:job-message"]
    assert result["errors"] == []
    assert body_requests == ["/v1.0/me/messages/job-message"]


def test_social_portal_senders_and_indeed_employer_mail():
    assert _portal("LinkedIn <invitations@linkedin.com>") is None
    assert _portal("newsletters-noreply@linkedin.com") is None
    assert _portal("LinkedIn Job Alerts <jobalerts-noreply@linkedin.com>") == "LinkedIn"
    assert _portal("Acme via Indeed <reply-abc@indeedemail.com>") == "Indeed"
    assert _portal("Naukri <info@naukri.com>") == "Naukri"


def _google_message(message_id: str, sender: str, subject: str) -> dict:
    return {
        "id": message_id,
        "threadId": f"thread-{message_id}",
        "internalDate": "1790000000000",
        "snippet": subject,
        "payload": {
            "headers": [{"name": "From", "value": sender}, {"name": "Subject", "value": subject}],
            "mimeType": "text/plain",
            "body": {"data": _encoded(f"{subject} https://www.naukri.com/job-listings/{message_id}")},
        },
    }


@pytest.mark.asyncio
async def test_one_failing_inbox_does_not_hide_the_other(tmp_path, monkeypatch):
    storage = Storage(tmp_path / "data")
    storage.save_doc("email_accounts.json", {"google": "me@gmail.com", "microsoft": "me@hotmail.com"})
    monkeypatch.setattr("app.services.email_jobs.keyring.get_password", lambda *args: "refresh-token")
    messages = {
        "m1": _google_message("m1", "Naukri <jobs@naukri.com>", "Python developer jobs for you"),
        "m2": _google_message("m2", "LinkedIn <invitations@linkedin.com>", "Ravi wants to connect"),
    }

    def handle(request):
        if request.url.host == "oauth2.googleapis.com":
            return httpx.Response(200, json={"access_token": "g-token"})
        if request.url.host == "login.microsoftonline.com":
            return httpx.Response(400, json={"error_description": "Token expired; reconnect"})
        if request.url.path == "/gmail/v1/users/me/messages":
            assert "from:naukri.com" in request.url.params["q"]
            return httpx.Response(200, json={"messages": [{"id": "m1"}, {"id": "m2"}]})
        if request.url.path.startswith("/gmail/v1/users/me/messages/"):
            return httpx.Response(200, json=messages[request.url.path.rsplit("/", 1)[-1]])
        raise AssertionError(f"Unexpected request: {request.url}")

    service = EmailJobService(
        storage,
        google_client_id="g", google_client_secret="g-secret",
        microsoft_client_id="m", microsoft_client_secret="m-secret",
        transport=httpx.MockTransport(handle),
    )
    result = await service.find_opportunities()

    assert [job["id"] for job in result["jobs"]] == ["google:m1"]
    assert result["jobs"][0]["links"][0]["url"] == "https://www.naukri.com/job-listings/m1"
    assert result["errors"] == [{
        "provider": "microsoft", "account": "me@hotmail.com", "message": "Token expired; reconnect",
    }]


@pytest.mark.asyncio
async def test_outlook_uses_sender_search_and_falls_back_to_date_scan(tmp_path, monkeypatch):
    storage = Storage(tmp_path / "data")
    storage.save_doc("email_accounts.json", {"microsoft": "me@hotmail.com"})
    monkeypatch.setattr("app.services.email_jobs.keyring.get_password", lambda *args: "refresh-token")
    list_requests = []

    def handle(request):
        if request.url.host == "login.microsoftonline.com":
            return httpx.Response(200, json={"access_token": "access-token"})
        if request.url.path == "/v1.0/me/messages":
            list_requests.append(dict(request.url.params))
            if "$search" in request.url.params:
                assert "from:linkedin.com" in request.url.params["$search"]
                return httpx.Response(400, json={"error": {"message": "Search not supported"}})
            return httpx.Response(200, json={"value": [{
                "id": "old-mail",
                "subject": "LinkedIn jobs for you",
                "from": {"emailAddress": {"address": "jobs-noreply@linkedin.com"}},
                "receivedDateTime": RECENT,
                "bodyPreview": "Data Engineer jobs",
            }]})
        if request.url.path == "/v1.0/me/messages/old-mail":
            return httpx.Response(200, json={"body": {"contentType": "text", "content": "See jobs"}})
        raise AssertionError(f"Unexpected request: {request.url}")

    service = EmailJobService(
        storage, microsoft_client_id="m", microsoft_client_secret="m-secret",
        transport=httpx.MockTransport(handle),
    )
    result = await service.find_opportunities()

    assert ["$search" in r for r in list_requests] == [True, False]
    assert [job["id"] for job in result["jobs"]] == ["microsoft:old-mail"]
    assert result["errors"] == []


@pytest.mark.asyncio
async def test_daily_scan_only_fetches_mail_since_previous_scan(tmp_path, monkeypatch):
    storage = Storage(tmp_path / "data")
    storage.save_doc("email_accounts.json", {"google": "me@gmail.com"})
    monkeypatch.setattr("app.services.email_jobs.keyring.get_password", lambda *args: "refresh-token")
    queries = []
    inbox = {"m1": _google_message("m1", "Naukri <jobs@naukri.com>", "Python developer jobs for you")}

    def handle(request):
        if request.url.host == "oauth2.googleapis.com":
            return httpx.Response(200, json={"access_token": "g-token"})
        if request.url.path == "/gmail/v1/users/me/messages":
            queries.append(request.url.params["q"])
            return httpx.Response(200, json={"messages": [{"id": i} for i in inbox]})
        if request.url.path.startswith("/gmail/v1/users/me/messages/"):
            return httpx.Response(200, json=inbox[request.url.path.rsplit("/", 1)[-1]])
        raise AssertionError(f"Unexpected request: {request.url}")

    service = EmailJobService(storage, google_client_id="g", google_client_secret="g-secret",
                              transport=httpx.MockTransport(handle))
    # Make saved jobs survive the 60-day pruning regardless of the fixture's fixed timestamp.
    monkeypatch.setattr("app.services.email_jobs.KEEP_DAYS", 100000)

    before = datetime.now(timezone.utc)
    first = await service.scan(days=7)
    first_after = int(queries[0].split()[0].removeprefix("after:"))
    assert abs(first_after - (before - timedelta(days=7)).timestamp()) < 60   # first scan: N days back
    assert first["new_count"] == 1 and first["jobs"][0]["is_new"]

    inbox["m2"] = _google_message("m2", "LinkedIn <jobs-noreply@linkedin.com>", "New jobs matching Data Engineer")
    second = await service.scan(days=7)
    second_after = int(queries[1].split()[0].removeprefix("after:"))
    # Next scan starts at the previous scan time (minus a 1h overlap), not 7 days back.
    assert abs(second_after - (before - timedelta(hours=1)).timestamp()) < 60
    assert second["new_count"] == 1
    assert {job["id"]: job["is_new"] for job in second["jobs"]} == {"google:m1": False, "google:m2": True}

    saved = service.saved_opportunities()
    assert [job["id"] for job in saved["jobs"]] and all(not job["is_new"] for job in saved["jobs"])
    assert len(saved["jobs"]) == 2

    await service.scan(full=True, days=30)                       # manual rescan ignores the cursor
    assert abs(int(queries[2].split()[0].removeprefix("after:"))
               - (before - timedelta(days=30)).timestamp()) < 60

    monkeypatch.setattr("app.services.email_jobs.keyring.delete_password", lambda *args: None)
    await service.disconnect("google")
    assert service._scan_doc()["jobs"] == [] and service._scan_doc()["cursors"] == {}


@pytest.mark.asyncio
async def test_failed_account_keeps_its_old_scan_time(tmp_path, monkeypatch):
    storage = Storage(tmp_path / "data")
    storage.save_doc("email_accounts.json", {"microsoft": "me@hotmail.com"})
    storage.save_doc("email_scan.json", {"cursors": {"microsoft": {
        "account": "me@hotmail.com", "scanned_at": "2026-01-01T00:00:00+00:00"}}, "jobs": []})
    monkeypatch.setattr("app.services.email_jobs.keyring.get_password", lambda *args: "refresh-token")

    def handle(request):
        return httpx.Response(400, json={"error_description": "Token expired; reconnect"})

    service = EmailJobService(storage, microsoft_client_id="m", microsoft_client_secret="m-secret",
                              transport=httpx.MockTransport(handle))
    result = await service.scan()

    assert result["errors"][0]["message"] == "Token expired; reconnect"
    assert service._scan_doc()["cursors"]["microsoft"]["scanned_at"] == "2026-01-01T00:00:00+00:00"
