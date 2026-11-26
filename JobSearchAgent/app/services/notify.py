"""Email digest of the new jobs found by a search run (SMTP, standard library only).

"New" means the app has not shown this job (same company + title) in any earlier run —
the same rule as the NEW badge — so a job that drops out of one run and reappears in
the next is not emailed twice.

Configured in .env (SMTP_HOST, SMTP_USER, SMTP_PASSWORD, NOTIFY_EMAIL_TO, ...).
Sending never fails a search: errors are logged and returned as a message.
"""
from __future__ import annotations

import asyncio
import html
import logging
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage

from app.config import IST, EnvSettings, get_env_settings
from app.models import ScoredJob, SearchResponse

log = logging.getLogger(__name__)

MAX_JOBS_IN_EMAIL = 50
SMTP_TIMEOUT = 30


def is_configured(env: EnvSettings) -> bool:
    return bool(env.smtp_host and env.notify_email_to)


def _where(job: ScoredJob) -> str:
    return job.location or ("Remote" if job.remote else "")


def build_message(response: SearchResponse, env: EnvSettings, *, trigger: str = "manual",
                  location: str = "", now: datetime | None = None) -> EmailMessage:
    new_jobs = sorted((j for j in response.jobs if j.is_new), key=lambda j: j.match_score, reverse=True)
    shown = new_jobs[:MAX_JOBS_IN_EMAIL]
    when = (now or datetime.now(IST)).astimezone(IST).strftime("%d %b %Y, %H:%M IST")
    kind = "Daily search" if trigger == "scheduled" else "Search"
    where = f" in {location}" if location else ""
    subject = (f"JobPilot: {len(new_jobs)} new job{'s' if len(new_jobs) != 1 else ''}{where}"
               if new_jobs else f"JobPilot: no new jobs{where}")
    intro = (f"{kind} of {when}: {len(new_jobs)} new of {response.total_returned} matches "
             f"(compared with earlier runs).")
    more = len(new_jobs) - len(shown)
    app_url = f"http://{env.host}:{env.port}/"

    # ---- plain text
    lines = [intro, ""]
    for j in shown:
        extra = " · ".join(filter(None, [_where(j), j.salary, j.source]))
        lines += [f"[{j.match_score}] {j.title} — {j.company}", f"    {extra}", f"    {j.url}", ""]
    if more:
        lines.append(f"…and {more} more in the app.")
    if not new_jobs:
        lines.append("Nothing new since the previous run.")
    lines.append(f"Open JobPilot: {app_url}")

    # ---- HTML
    e = html.escape
    rows = "".join(
        "<tr>"
        f"<td style='padding:6px 8px;font-weight:bold;color:{'#15803d' if j.match_score >= 70 else '#a16207' if j.match_score >= 50 else '#6b7280'}'>{j.match_score}</td>"
        f"<td style='padding:6px 8px'><a href='{e(j.url, quote=True)}'>{e(j.title)}</a>"
        f"<br><span style='color:#555'>{e(j.company)}</span></td>"
        f"<td style='padding:6px 8px;color:#555'>{e(_where(j))}{'<br>' + e(j.salary) if j.salary else ''}</td>"
        f"<td style='padding:6px 8px;color:#555'>{e(j.source)}</td>"
        "</tr>"
        for j in shown
    )
    table = (
        "<table style='border-collapse:collapse;font-size:14px' border='0'>"
        "<tr style='background:#f3f4f6;text-align:left'><th style='padding:6px 8px'>Match</th>"
        "<th style='padding:6px 8px'>Job</th><th style='padding:6px 8px'>Location</th>"
        f"<th style='padding:6px 8px'>Source</th></tr>{rows}</table>"
        if shown else "<p>Nothing new since the previous run.</p>"
    )
    body = (
        "<div style='font-family:Segoe UI,Arial,sans-serif'>"
        f"<p>{e(intro)}</p>{table}"
        + (f"<p>…and {more} more in the app.</p>" if more else "")
        + f"<p><a href='{e(app_url, quote=True)}'>Open JobPilot</a></p></div>"
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = env.smtp_from or env.smtp_user
    msg["To"] = ", ".join(env.notify_email_to)
    msg.set_content("\n".join(lines))
    msg.add_alternative(body, subtype="html")
    return msg


def _send(msg: EmailMessage, env: EnvSettings) -> None:
    context = ssl.create_default_context()
    if env.smtp_port == 465:
        server: smtplib.SMTP = smtplib.SMTP_SSL(env.smtp_host, env.smtp_port, timeout=SMTP_TIMEOUT, context=context)
    else:
        server = smtplib.SMTP(env.smtp_host, env.smtp_port, timeout=SMTP_TIMEOUT)
    with server:
        if env.smtp_port != 465:
            server.starttls(context=context)
        if env.smtp_user:
            server.login(env.smtp_user, env.smtp_password)
        server.send_message(msg)


async def notify_new_jobs(response: SearchResponse, *, trigger: str = "manual", location: str = "",
                          env: EnvSettings | None = None) -> str:
    """Email the new jobs of this run. Returns a short status message; never raises."""
    env = env or get_env_settings()
    if not is_configured(env):
        return ""
    if response.new_count == 0 and not env.notify_send_empty:
        return "No new jobs — email not sent"
    try:
        msg = build_message(response, env, trigger=trigger, location=location)
        await asyncio.to_thread(_send, msg, env)
    except Exception as exc:
        log.exception("Sending the new-jobs email failed")
        return f"Email failed: {str(exc)[:200] or type(exc).__name__}"
    log.info("New-jobs email sent to %s (%d new)", ", ".join(env.notify_email_to), response.new_count)
    return f"Emailed {response.new_count} new jobs"
