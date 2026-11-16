"""
Run the correction experiment: collect raw model responses. Scoring in scoring.py

    python -m src.experiment --run-id pilot --scenarios data/scenarios_heldout.json --reps 2
    python -m src.experiment --run-id main --scenarios data/scenarios.json --reps 3
    python -m src.experiment --run-id mocktest --mock                                            # offline plumbing test
    # add --dry to print the number of calls and exit; --limit N to run only N scenarios

Writes results/runs/<run-id>/{config.json, raw.jsonl}. The run RESUMES: tasks whose key is already
in raw.jsonl are skipped, to ensure that task continues after crash or rate limit stall with same command.
Per (model, scenario) we run: a closed-book `baseline` probe (what does the model know unprompted?)
and a `correction` turn under each framing, each `reps` times.
"""

import argparse
import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config, llm, prompts
from .schema import load_scenarios, validate

ROOT = Path(__file__).resolve().parents[1]

def key_of(model, sid, kind, framing, rep):
    return f"{model}/{sid}/{kind}/{framing}/{rep}"

def build_tasks(scs, models, reps):
    tasks = []
    for m in models:
        for sc in scs:
            for r in range(reps):
                tasks.append(dict(model=m, sc=sc, kind="baseline", framing="none", rep=r))
                for f in config.FRAMINGS:
                    tasks.append(dict(model=m, sc=sc, kind="correction", framing=f, rep=r))
    for t in tasks:
        t["key"] = key_of(t["model"], t["sc"].id, t["kind"], t["framing"], t["rep"])
    random.Random(config.SEED).shuffle(tasks)           # spread load and keep partial runs balanced
    return tasks

def run_task(t):
    sc = t['sc']
    spec = dict(config.MODELS[t["model"]], name=t["model"])
    msgs = (prompts.baseline_messages(sc) if t["kind"] == "baseline"
            else prompts.correction_messages(sc, t["framing"]))
    t0 = time.time()
    out = llm.complete(spec, msgs, ctx=dict(key=t["key"], sc=sc.__dict__, kind=t["kind"], framing=t["framing"]))
    return dict(key=t["key"], model=t["model"], model_id=spec.get("model", "mock"), scenario_id=sc.id,
                category=sc.category, domain=sc.domain, kind=t["kind"], framing=t["framing"], rep=t["rep"],
                temperature=spec.get("temperature"), messages=msgs, response=out["text"],
                response_clean=llm.strip_think(out["text"]), usage=out["usage"],
                latency_s=round(time.time() - t0, 2), ts=time.strftime("%Y-%m-%dT%H:%M:%S"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--scenarios", default="data/scenarios.json")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--reps", type=int, default=config.REPS)
    ap.add_argument("--workers", type=int, default=config.WORKERS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", action="store_true", help="resume even if the scenario file changed")
    ap.add_argument("--allow-unverified", action="store_true", help="DEBUG ONLY: permit real inference on unverified scenarios",)
    a = ap.parse_args()

    models = a.models or (config.MOCK_MODELS if a.mock else config.DEFAULT_MODELS)

    spath = ROOT / a.scenarios
    scs = load_scenarios(spath)

    if a.limit:
        scs = scs[:a.limit]

    sha = hashlib.sha1(spath.read_bytes()).hexdigest()

    # dry runs and mock runs do not make real model calls.
    require_verified = not (
        a.mock or a.dry or a.allow_unverified
    )

    errs, warns, n = validate(scs, require_verified=require_verified)

    if errs:
        raise SystemExit(
            "Scenario errors:\n" + "\n".join(errs)
        )

    if a.allow_unverified and not a.mock:
        print("WARNING: real inference on unverified scenarios.")

    rdir = ROOT / "results" / "runs" / a.run_id
    rdir.mkdir(parents=True, exist_ok=True)
    cpath, rpath = rdir / "config.json", rdir / "raw.jsonl"
    if cpath.exists():
        old = json.loads(cpath.read_text())
        if old["scenarios_sha1"] != sha and not a.force:
            raise SystemExit(f"{cpath} was made with a different scenario file. Use a new --run-id " 
                             "(or --force if you know what you are doing).")
        models_all = sorted(set(old["models"]) | set(models))
    else:
        models_all = list(models)
    cpath.write_text(
        json.dumps(
            dict(
                run_id=a.run_id,
                scenarios=a.scenarios,
                scenarios_sha1=sha,
                models=models_all,
                reps=a.reps,
                framings=config.FRAMINGS,
                mock=a.mock,
                judge="mock" if a.mock else config.JUDGE,
                model_specs={m: config.MODELS[m] for m in models_all},
                started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                allow_unverified=a.allow_unverified,
            ),
            indent=2,
        )
    )

    tasks = build_tasks(scs, models, a.reps)
    done = set()
    if rpath.exists():
        for line in rpath.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["key"])
    todo = [t for t in tasks if t["key"] not in done]
    print(f"{len(scs)} scenarios {n} x {len(models)} models x {a.reps} reps -> {len(tasks)} calls | "
          f"{len(done)} already done, {len(todo)} to run")
    if a.dry or not todo:
        return

    lock, nerr, ndone = threading.Lock(), 0, 0
    with open(rpath, "a", encoding="utf-8") as fh, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_task, t): t for t in todo}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:          # not written -> retried on next run
                nerr += 1
                print(f" FAILED {t['key']}: {type(e).__name__}: {str(e)[:150]}")
                continue
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            ndone += 1
            if ndone % 25 == 0:
                print(f" {ndone}/{len(todo)} done ({nerr} failed)")
    print(f"finished: {ndone} ok, {nerr} failed. Rerun same command to retry only failures.")

if __name__ == "__main__":
    main()