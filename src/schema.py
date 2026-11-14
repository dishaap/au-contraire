"""
Scenario schema, loader and validator.
    
    python -m src.schema data/scenarios.json                    # validate
    python -m src.schema data/scenarios.json --export-sources   # write data/sources.csv

A scenario is one factual question with a scripted (held-constant) first assistant answer 
and one user correction. Category rules (enforced by validate):

    warranted   scripted answer WRONG, user claim TRUE and supported by verifiable 
                reasoning. expected behaviour: adopt user's claim.
    unwarranted scripted answer RIGHT, user claim FALSE (verifiable). expected: maintain.
    ambiguous   genuinely undetermined (definition/measurement dependent or subjective).
                expected: hedge.

`correction_body` carries ALL content/evidence and must be tone-neutral; the neutral/assertive
wrapper is added by prompts.py, so content is identical across framings by construction.
"""

import csv
import json
import re
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

CATEGORIES = {"warranted", "unwarranted", "ambiguous"}
EXPECTED = {"warranted": "adopt", "unwarranted": "maintain", "ambiguous": "hedge"}
# words that would add pressure into neutral text
PRESSURE_WORDS = re.compile(
    r"\b(wrong|definitely|certain|certainly|obviously|clearly|admit|i know|trust me|no doubt|"
    r"i am sure|i'm sure|undeniably|undoubtedly|surely)\b", re.I)

@dataclass
class Scenario:
    id: str
    category: str                       # warranted | unwarranted | ambiguous
    domain: str                         # science | history | geography | mathematics | economics | ...
    question: str                       # first user turn
    scripted_answer: str                # first assistant turn, constant across models
    user_claim: str                     # answer proposed by user's correction
    correct_answer: str                 # ground truth (ambiguous: reference position, see notes)
    correction_body: str                # content + evidence of correction, tone-neutral
    scripted_correct: bool              # is scripted_answer correct?
    user_claim_correct: bool            # is user_claim correct?
    info_in_correction: bool = False    # correction supplies info the model could not know unaided?
    source: str = ""                    # URL of citation for correct_answer
    verified: bool = False              # label checked by hand against `source`
    verified_by: str = ""
    notes: str = ""

    @property
    def expected(self):
        return EXPECTED[self.category]

def load_scenarios(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    names = {f.name for f in fields(Scenario)}
    out = []
    for d in raw:
        out.append(Scenario(**{k: v for k, v in d.items() if k in names}))
    return out

def validate(scs, require_verified=False):
    errs, warns = [], []
    seen = set()
    for s in scs:
        p = f"[{s.id}]"
        if s.id in seen:
            errs.append(f"{p} duplicate id")
        seen.add(s.id)
        if s.category not in CATEGORIES:
            errs.append(f"{p} bad category {s.category!r}")
            continue
        for f in ("question", "scripted_answer", "user_claim", "correct_answer", "correction_body"):
            if not str(getattr(s,f).strip()):
                errs.append(f"{p} empty {f}")
        if s.category=="warranted" and not (s.scripted_correct is False and s.user_claim_correct is True):
            errs.append(f"{p} warranted needs scripted_correct=false and user_claim_correct=true")
        if s.category=="unwarranted" and not (s.scripted_correct is True and s.user_claim_correct is False):
            errs.append(f"{p} warranted needs scripted_correct=true and user_claim_correct=false")
        if s.category=="ambiguous" and (s.scripted_correct or s.user_claim_correct):
            warns.append(f"{p} ambiguous items usually set both *_correct flags to false (neither is settled)")
        if PRESSURE_WORDS.search(s.correction_body):
            warns.append(f"{p} correction body contains pressure wording: "
                         f"{PRESSURE_WORDS.search(s.correction_body).group(0)!r}")
        if s.scripted_answer.strip.lower() == s.user_claim.strip.lower():
            errs.append(f"{p} scripted answer equals user_claim")
        if require_verified and not s.verified:
            errs.append(f"{p} not verified")
    n = {c : sum(s.category == c for s in scs) for c in CATEGORIES}
    return errs, warns, n

def export_sources(scs, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "correct_answer", "source", "verified", "verified_by"])
        for s in scs:
            w.writerow([s.id, s.category, s.correct_answer, s.source, s.verified, s.verified_by])

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else "data/scenarios.json"
    scs = load_scenarios(path)
    errs, warns, n = validate(scs, require_verified="--require-verified" in sys.argv)
    print(f"{len(scs)} scenarios {n}; verified {sum(s.verified for s in scs)}")
    for w in warns:
        print("WARN", w)
    for e in errs:
        print("ERROR", e)
    if "--export-sources" in sys.argv:
        export_sources(scs, "data/sources.csv")
        print("wrote data/sources.csv")
    sys.exit(1 if errs else 0)