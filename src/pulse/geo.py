"""Is the corresponding author US-based?  Returns 'US', 'nonUS' or None.

Evidence in order: corresponding email top-level domain, corresponding
address text, API corresponding-institution string, first affiliation country.
"""

from __future__ import annotations

import re

_STATES = (
    "Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|"
    "Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|"
    "Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|"
    "New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|"
    "South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|"
    "Wisconsin|Wyoming|District of Columbia"
)
_ABBR = (
    "AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|"
    "NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC"
)
US_TEXT = re.compile(
    rf"\b(?:USA|U\.S\.A\.?|United States|U\.S\.)\b|\b(?:{_STATES})\b|,\s?(?:{_ABBR})\b(?:\s?\d{{5}})?|"
    r"\b(?:Harvard|Yale|Stanford|Johns Hopkins|Mayo Clinic|Emory|Vanderbilt|Duke|Columbia University|"
    r"Northwestern University|Cornell|Princeton|MIT\b|Massachusetts General|Brigham|Kaiser Permanente|"
    r"Centers for Disease Control|\bCDC\b|\bNIH\b|National Institutes of Health|Fred Hutch|Broad Institute|"
    r"Mount Sinai|Icahn|Baylor|Scripps|Salk|Dana-Farber|Memorial Sloan|MD Anderson|Rockefeller University)"
)
# State names that collide with non-US places
_AMBIG = re.compile(r"\b(?:Georgia|New South Wales|Western Australia|Victoria|Queensland)\b")

_COUNTRIES = (
    "United Kingdom|UK|England|Scotland|Wales|Ireland|Canada|Australia|New Zealand|China|Hong Kong|"
    "Taiwan|Japan|Korea|Singapore|India|Pakistan|Bangladesh|Iran|Israel|Turkey|Saudi Arabia|Qatar|"
    "United Arab Emirates|Egypt|Nigeria|Kenya|Uganda|Tanzania|Ethiopia|South Africa|Ghana|Malawi|"
    "Zambia|Zimbabwe|Rwanda|Mozambique|Cameroon|Senegal|Brazil|Mexico|Argentina|Chile|Colombia|Peru|"
    "Germany|France|Italy|Spain|Portugal|Netherlands|Belgium|Switzerland|Austria|Sweden|Norway|"
    "Denmark|Finland|Iceland|Poland|Czech Republic|Czechia|Hungary|Greece|Romania|Russia|Ukraine|"
    "Thailand|Vietnam|Viet Nam|Malaysia|Indonesia|Philippines|Nepal|Sri Lanka|Lebanon|Jordan|"
    "Luxembourg|Estonia|Latvia|Lithuania|Slovenia|Croatia|Serbia|Cyprus|Malta"
)
OTHER_TEXT = re.compile(rf"\b(?:{_COUNTRIES})\b")
US_COUNTRY = re.compile(r"^(?:USA|U\.S\.A\.?|US|United States(?: of America)?)$", re.I)


def _from_email(email: str | None) -> str | None:
    if not isinstance(email, str) or "@" not in email:
        return None
    tld = email.rsplit(".", 1)[-1].lower().strip(">) ")
    if tld in {"edu", "gov", "mil", "us"}:
        return "US"
    if len(tld) == 2:
        return "nonUS"
    return None  # .com/.org/.net: uninformative


def _from_text(text: str | None) -> str | None:
    if not isinstance(text, str) or not text:
        return None
    other = OTHER_TEXT.search(text)
    us = US_TEXT.search(text)
    if us and not other and not (_AMBIG.search(text) and not re.search(r"USA|United States", text)):
        return "US"
    if other and not us:
        return "nonUS"
    return None


def corresponding_country(email=None, corresp=None, institution=None, first_aff_country=None) -> str | None:
    for v in (_from_email(email), _from_text(corresp), _from_text(institution)):
        if v:
            return v
    if isinstance(first_aff_country, str) and first_aff_country.strip():
        return "US" if US_COUNTRY.match(first_aff_country.strip()) else "nonUS"
    return None
