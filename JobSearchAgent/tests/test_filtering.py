from __future__ import annotations

import pytest

from app.models import JobListing
from app.services.filtering import (
    LocationRule,
    classify,
    clean_title,
    deduplicate,
    normalize_company,
    refine_jobs,
)

_n = 0


def job(title="Data Engineer", company="Acme", location="Chennai, Tamil Nadu", remote=False,
        source="jsearch", url=None, **kw) -> JobListing:
    global _n
    _n += 1
    return JobListing(id=f"{source}:{_n}", title=title, company=company, location=location, remote=remote,
                      source=source, url=url or f"https://example.com/{_n}", **kw)


CHENNAI = LocationRule("Chennai")


# ------------------------------------------------------------------ titles
@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("EPAM!! Hiring Alert !! Sr Snowflake Data Engineer !! (Chennai)", "EPAM Sr Snowflake Data Engineer (Chennai)"),
        ("Urgent Hiring - Python Developer", "Python Developer"),
        ("Data Engineer || Immediate Joiners", "Data Engineer"),
        ("Senior Data Engineer", "Senior Data Engineer"),
        ("C# / .NET Developer", "C# / .NET Developer"),
    ],
)
def test_clean_title(raw, clean):
    assert clean_title(raw) == clean


# ------------------------------------------------------------------ location
@pytest.mark.parametrize(
    ("location", "remote", "title", "expected"),
    [
        ("Chennai, Tamil Nadu", False, "Data Engineer", ("local", "")),
        ("Madras", False, "Data Engineer", ("local", "")),
        ("Sholinganallur, OMR", False, "Data Engineer", ("local", "")),
        ("Seattle / Chennai", False, "Data Engineer", ("local", "")),
        ("India", False, "Data Engineer - Chennai", ("local", "")),       # city only in title
        ("Remote (India)", True, "Data Engineer", ("remote", "")),
        ("Remote (Anywhere)", True, "Data Engineer", ("remote", "")),
        ("Remote (Worldwide)", True, "Data Engineer", ("remote", "")),
        ("Remote (APAC)", True, "Data Engineer", ("remote", "")),
        ("Remote (India + 49 more countries)", True, "Data Engineer", ("remote", "")),
        ("Bangalore - Remote / Mumbai, India", True, "Data Engineer", ("remote", "")),
        ("Remote", True, "Data Engineer", ("remote", "")),
        ("Remote (USA)", True, "Data Engineer", ("", "remote_restricted")),
        ("Remote (Berlin)", True, "Data Engineer", ("", "remote_restricted")),
        ("Remote-Friendly, United States", True, "Data Engineer", ("", "remote_restricted")),
        ("Bengaluru, India", False, "Data Engineer", ("", "other_city")),
        ("Hyderabad", False, "Data Engineer", ("", "other_city")),
        ("San Francisco, CA", False, "Data Engineer", ("", "other_country")),
        ("Hybrid", False, "Data Engineer", ("", "unknown_location")),
        ("", False, "Data Engineer", ("", "unknown_location")),
    ],
)
def test_classify(location, remote, title, expected):
    assert classify(job(title=title, location=location, remote=remote), CHENNAI) == expected


@pytest.mark.parametrize(
    "description",
    [
        "This role is open to US-based candidates.",
        "Candidates must be located in the United States.",
        "You must reside within the EU.",
        "Must be authorized to work in the US without sponsorship.",
    ],
)
def test_plain_remote_with_restrictive_description(description):
    assert classify(job(location="Remote", remote=True, description=description), CHENNAI) == ("", "remote_restricted")


@pytest.mark.parametrize(
    "description",
    [
        "Join us to build data pipelines. Tell us about yourself.",
        "We have customers in the US and Europe.",
    ],
)
def test_plain_remote_without_restriction_is_kept(description):
    assert classify(job(location="Remote", remote=True, description=description), CHENNAI) == ("remote", "")


def test_remote_only_excludes_onsite_chennai():
    rule = LocationRule("Chennai", remote_only=True)
    assert classify(job(location="Chennai"), rule) == ("", "not_remote")
    assert classify(job(location="Remote (India)", remote=True), rule) == ("remote", "")


def test_other_city_rule():
    bangalore = LocationRule("Bengaluru")
    assert classify(job(location="Bangalore, Karnataka"), bangalore) == ("local", "")
    assert classify(job(location="Chennai"), bangalore) == ("", "other_city")


# ------------------------------------------------------------------ de-duplication
@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("5100 Kyndryl Solutions Private Limited", "kyndryl"),
        ("1401 Genpact India Pvt. Ltd", "genpact"),
        ("EPAM Systems", "epam"),
        ("Epam Systems", "epam"),
        ("Tata Consultancy Services", "tata"),
        ("Global Infovision Private", "global infovision"),
        ("The Company", "the company"),  # never strips to nothing
    ],
)
def test_normalize_company(raw, norm):
    assert normalize_company(raw) == norm


def test_dedup_merges_cross_source_and_keeps_alt_links():
    a = job("Senior Data Engineer – Databricks, dbt & GenAI", "5100 Kyndryl Solutions Private Limited",
            source="jsearch", url="https://workday/1", description="short")
    b = job("Senior Data Engineer - Databricks, dbt & GenAI", "Kyndryl", source="serpapi",
            url="https://linkedin/1", description="a much longer description " * 10, salary="INR 20L")
    c = job("EPAM!! Hiring Alert !! Sr Snowflake Data Engineer !! (Chennai)", "EPAM Systems")
    d = job("Senior Snowflake Data Engineer", "Epam Systems", location="Chennai")
    unique, merged = deduplicate([a, b, c, d], CHENNAI.city_re)
    assert merged == 2 and len(unique) == 2
    kyn = next(j for j in unique if "Databricks" in j.title)
    assert kyn.salary == "INR 20L" and "much longer" in kyn.description
    assert {link.url for link in kyn.alt_links} | {kyn.url} == {"https://workday/1", "https://linkedin/1"}


def test_dedup_prefers_company_ats_record():
    agg = job("Data Engineer", "Stripe", source="jsearch", url="https://linkedin/x", description="x" * 500)
    ats = job("Data Engineer", "Stripe", source="greenhouse", url="https://stripe.com/jobs/1", description="y" * 100)
    unique, _ = deduplicate([agg, ats])
    assert unique[0].source == "greenhouse" and unique[0].alt_links[0].source == "jsearch"


def test_dedup_same_url_different_text():
    unique, merged = deduplicate([job("Data Engineer", "A", url="https://same"), job("Data Eng.", "A Inc", url="https://same")])
    assert merged == 1 and len(unique) == 1


def test_dedup_keeps_distinct_roles():
    jobs = [
        job("Senior Data Engineer", "Acme"),
        job("Data Engineer", "Acme"),          # different seniority -> separate opening
        job("Lead Data Engineer", "Acme"),
        job("Data Scientist", "Acme"),
        job("Senior Data Engineer", "Other Co"),  # same title, other company
    ]
    unique, merged = deduplicate(jobs)
    assert merged == 0 and len(unique) == 5


# ------------------------------------------------------------------ pipeline
def test_refine_jobs_end_to_end():
    jobs = [
        job("EPAM!! Hiring Alert !! Data Engineer", "EPAM", location="Chennai"),
        job("Data Engineer", "Kyndryl", location="Remote (Anywhere)", remote=True, source="jsearch"),
        job("Data Engineer", "Kyndryl Solutions Pvt Ltd", location="Anywhere", remote=True, source="serpapi"),
        job("Data Engineer", "X", location="Remote (USA)", remote=True),
        job("Data Engineer", "Y", location="Bengaluru"),
        job("Data Engineer", "Z", location="London"),
    ]
    result = refine_jobs(jobs, "Chennai")
    assert [(j.company, j.location_match) for j in result.jobs] == [("EPAM", "local"), ("Kyndryl", "remote")]
    assert result.jobs[0].title == "EPAM Data Engineer"
    assert result.excluded == {"remote_restricted": 1, "other_city": 1, "other_country": 1}
    assert result.duplicates_merged == 1
    assert jobs[0].title.startswith("EPAM!!")  # inputs are not mutated
