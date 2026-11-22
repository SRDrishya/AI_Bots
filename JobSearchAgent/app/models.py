from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class ResumeProfile(BaseModel):
    candidate_name: str = ""
    summary: str = ""
    target_roles: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    years_of_experience: float | None = None
    seniority: str = ""
    preferred_locations: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    raw_text_preview: str = ""
    analysis_engine: str = "local"


class ResumeFileInfo(BaseModel):
    """The original resume file kept in data/resume/."""
    filename: str
    size: int
    sha256: str
    uploaded_at: str


class ResumeUploadResult(BaseModel):
    profile: ResumeProfile
    file: ResumeFileInfo | None = None
    reused: bool = False  # True = same file as before; saved profile kept, no AI analysis


class ATSBoardConfig(BaseModel):
    provider: Literal["greenhouse", "lever", "ashby"]
    slug: str = Field(min_length=1, max_length=120)
    label: str = ""
    enabled: bool = True


class ApiKeysInput(BaseModel):
    gemini_api_key: str = ""
    gemini_model: str = ""
    adzuna_app_id: str = ""
    adzuna_app_key: str = ""
    rapidapi_key: str = ""
    serpapi_key: str = ""


class JobLink(BaseModel):
    source: str
    url: str


class JobListing(BaseModel):
    id: str
    title: str
    company: str
    location: str = "Remote"
    remote: bool = False
    url: str
    source: str
    posted_at: str = ""
    salary: str = ""
    employment_type: str = ""
    tags: list[str] = Field(default_factory=list)
    description: str = ""
    # Set by the location filter: "local" (in the target city) or "remote" (remote, open to India).
    location_match: Literal["", "local", "remote"] = ""
    # Same job found on other sources (filled in by de-duplication).
    alt_links: list[JobLink] = Field(default_factory=list)


TrackingStatus = Literal["saved", "applied", "interviewing", "offer", "rejected", "dismissed"]


class ScoredJob(JobListing):
    match_score: int = Field(ge=0, le=100)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    match_reasoning: str = ""
    scoring_engine: str = "local"
    # Filled in from storage when results are returned:
    key: str = ""                          # stable identity across sources (company + title)
    is_new: bool = False                   # not seen in any earlier search
    first_seen: str = ""                   # ISO timestamp of first sighting
    tracking_status: TrackingStatus | Literal[""] = ""
    appearances: int = 0                   # combined view only: number of saved searches it appeared in


class StatusChange(BaseModel):
    status: TrackingStatus
    at: str


class TrackedJob(BaseModel):
    key: str
    job: ScoredJob                         # snapshot when first tracked (kept even if the posting disappears)
    status: TrackingStatus
    notes: str = ""
    created_at: str
    updated_at: str
    history: list[StatusChange] = Field(default_factory=list)


class TrackJobRequest(BaseModel):
    job: ScoredJob
    status: TrackingStatus = "saved"
    notes: str = ""


class TrackUpdateRequest(BaseModel):
    status: TrackingStatus | None = None
    notes: str | None = None


class SearchSummary(BaseModel):
    id: str
    created_at: str
    trigger: Literal["manual", "scheduled"] = "manual"
    location: str
    queries: list[str]
    total_fetched: int
    total_returned: int
    new_count: int
    top_score: int
    scoring_engine: str


class SourceStatus(BaseModel):
    source: str
    status: Literal["ok", "skipped", "error"]
    count: int = 0
    message: str = ""


class SearchRequest(BaseModel):
    profile: ResumeProfile
    location: str = "Chennai"
    remote_only: bool = False
    max_results: int = Field(default=50, ge=5, le=200)
    # Key-based sources (adzuna, jsearch, serpapi) report "skipped" when no key is set.
    sources: list[str] = Field(
        default_factory=lambda: [
            "remotive",
            "arbeitnow",
            "himalayas",
            "jobicy",
            "ats_boards",
            "adzuna",
            "jsearch",
            "serpapi",
        ]
    )
    ats_boards: list[ATSBoardConfig] = Field(default_factory=list)
    custom_query: str = ""
    api_keys: ApiKeysInput = Field(default_factory=ApiKeysInput)


class SearchOptions(BaseModel):
    """What the web UI sends: the saved profile and .env keys are used server-side."""
    location: str = "Chennai"
    remote_only: bool = False
    max_results: int = Field(default=50, ge=5, le=200)
    custom_query: str = ""
    sources: list[str] | None = None  # None = all default sources


class SearchResponse(BaseModel):
    total_fetched: int
    total_returned: int
    scoring_engine: str
    sources: list[SourceStatus]
    # Why fetched jobs were dropped, e.g. {"remote_restricted": 41, "other_city": 49}.
    excluded: dict[str, int] = Field(default_factory=dict)
    duplicates_merged: int = 0
    jobs: list[ScoredJob]
    search_id: str = ""                    # set once the search is saved
    new_count: int = 0


class StoredSearch(BaseModel):
    summary: SearchSummary
    response: SearchResponse


class CombinedSearch(BaseModel):
    """Several saved searches merged into one de-duplicated list."""
    days: int
    start: str                             # ISO, start of the first calendar day (IST)
    end: str                               # ISO, when it was combined
    searches: list[SearchSummary]          # the searches included, newest first
    total_listings: int                    # jobs across all included searches, before de-dup
    duplicates_removed: int
    response: SearchResponse               # merged jobs, best score first


class CompanyLink(BaseModel):
    label: str
    url: str


class CompanyInsight(BaseModel):
    """AI summary of an employer. `grounded` = based on live Google Search (with sources);
    otherwise it's the model's own knowledge and may be outdated."""
    company: str
    role: str
    grounded: bool
    known: bool = True                     # False = the AI doesn't know this company well
    overview: str = ""
    rating: float | None = None            # out of 5
    rating_source: str = ""                # e.g. "AmbitionBox", "Glassdoor"
    review_count: str = ""
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    salary_range: str = ""                 # e.g. "INR 8-15 LPA"
    salary_note: str = ""
    questions_to_ask: list[str] = Field(default_factory=list)
    caveats: str = ""
    sources: list[CompanyLink] = Field(default_factory=list)
    generated_at: str = ""


class CompanyInsightsRequest(BaseModel):
    company: str = Field(min_length=1, max_length=200)
    role: str = Field(default="", max_length=200)
    location: str = "Chennai"
    regenerate: bool = False


class CompanyInsightsResponse(BaseModel):
    company: str
    links: list[CompanyLink]
    insight: CompanyInsight | None = None
    ai_error: str = ""                     # why there's no AI insight (links still work)


KitTone = Literal["professional", "warm", "concise"]


class ApplicationKit(BaseModel):
    """Gemini-written help for applying to one job."""
    job_key: str
    tone: KitTone
    cover_letter: str
    resume_tips: list[str] = Field(default_factory=list)     # edits to emphasise relevant experience
    keywords: list[str] = Field(default_factory=list)        # JD terms the candidate genuinely has (for ATS)
    talking_points: list[str] = Field(default_factory=list)  # interview prep, incl. how to address gaps
    created_at: str = ""


class ApplicationKitRequest(BaseModel):
    job: ScoredJob
    tone: KitTone = "professional"
    regenerate: bool = False
