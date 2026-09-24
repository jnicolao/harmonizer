"""Turn a CROSS ruling into the product description a user would have typed.

Ruling text is a letter. After the salutation comes one boilerplate sentence
("In your letter dated ... you requested a ruling"), then the merchandise
description, then the legal analysis starting at phrases like "The applicable
subheading". We keep the middle part only, and mask any tariff codes so
retrieval cannot cheat off the answer.
"""

import re

SUBJECT_PREFIX = re.compile(
    r"^(?:re:\s*)?(?:the\s+)?(?:tariff\s+)?classification\s+(?:and\s+[\w\s]+?\s+)?of\s+",
    re.IGNORECASE,
)
SALUTATION = re.compile(r"Dear[^:]{0,60}:|Dear\s+\S+\s+\S+\s*")
REQUEST_SENTENCE = re.compile(r"^\s*In your .*?\.(?=\s+[A-Z“\"])\s*", re.IGNORECASE)
ANALYSIS_START = re.compile(
    r"\b(?:The applicable (?:subheading|tariff provision)"
    r"|You (?:suggest|propose|state that|indicate that the applicable)"
    r"|In your (?:letter|request|submission),? you (?:suggest|propose|request|state)"
    r"|We (?:agree|disagree)|Classification\s{2,}|Marking\s{2,}|Country of Origin"
    r"|The General Rules? of Interpretation|GRI\s?\d|Legal Note|Chapter Note|Section Note)",
    re.IGNORECASE,
)
DOTTED_CODE = re.compile(r"\b\d{4}\.\d{2}(?:\.\d{2,4})?\b")
HEADING_REF = re.compile(r"\b(?:sub)?headings?\s+\d{4}(?:\.\d{2,4})*", re.IGNORECASE)


def describe(subject: str, text: str, max_chars: int = 1200) -> str:
    subj = SUBJECT_PREFIX.sub("", subject or "").strip().rstrip(".")
    body = re.sub(r"\s+", " ", text or "")
    m = SALUTATION.search(body)
    if m:
        body = body[m.end() :]
    body = REQUEST_SENTENCE.sub("", body, count=1)
    m = ANALYSIS_START.search(body)
    if m and m.start() > 80:
        body = body[: m.start()]
    body = HEADING_REF.sub("[heading]", body)
    body = DOTTED_CODE.sub("[code]", body)
    return f"{subj}. {body}".strip()[:max_chars]
