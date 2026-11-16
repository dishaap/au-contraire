"""Provider wrapper (OpenAI-compatible endpoints + an offline mock).

Adapted from the SwarmScope llm.py design: one `complete()` entry point, retries with backoff on
rate limits, and a mock backend so the whole pipeline can be tested without API calls.

  python -m src.llm --ping command-a       # one real call, to check ids/params/keys
"""
import hashlib
import json
import os
import random
import re
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from . import config

_clients = {}


def _client(provider):
    if provider not in _clients:
        from openai import OpenAI
        p = config.PROVIDERS[provider]
        key = os.environ.get(p["key_env"])
        if not key:
            raise RuntimeError(f"{p['key_env']} is not set (put it in .env)")
        _clients[provider] = OpenAI(api_key=key, base_url=p["base_url"], max_retries=0, timeout=180)
    return _clients[provider]


def strip_think(text):
    """Remove reasoning traces some endpoints leave inline."""
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _as_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for c in content:                      # list of content blocks
        t = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
        if t:
            parts.append(t)
    return "\n".join(parts)


def complete(spec, messages, ctx=None, retries=6):
    """spec: a dict from config.MODELS / config.JUDGE (+ 'name'). Returns {"text", "usage"}."""
    if spec["provider"] == "mock":
        return _mock(spec, messages, ctx or {})
    if spec["provider"] == "adaption":
        return _adaption(spec, messages, retries)
    from openai import (APIConnectionError, APITimeoutError, InternalServerError,
                        RateLimitError)
    kw = dict(model=spec["model"], messages=messages, max_tokens=spec.get("max_tokens", 700))
    if spec.get("temperature") is not None:
        kw["temperature"] = spec["temperature"]
    if spec.get("extra_body"):
        kw["extra_body"] = spec["extra_body"]
    for i in range(retries):
        try:
            r = _client(spec["provider"]).chat.completions.create(**kw)
            u = getattr(r, "usage", None)
            return {"text": _as_text(r.choices[0].message.content),
                    "usage": {"in": getattr(u, "prompt_tokens", None), "out": getattr(u, "completion_tokens", None)}}
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
            if i == retries - 1:
                raise
            wait = min(60, 2 ** i + random.random() * 2)
            print(f"  retry {i + 1}/{retries} after {type(e).__name__}; sleeping {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)


def _adaption(spec, messages, retries):
    """Adaption Responses API: POST {base}/responses with {model, input:[{role,content}], ...}."""
    import httpx
    p = config.PROVIDERS["adaption"]
    key = os.environ.get(p["key_env"])
    if not key:
        raise RuntimeError(f"{p['key_env']} is not set (put it in .env)")
    body = {
        "model": spec["model"], 
        "input": messages, 
        "max_output_tokens": spec.get("max_tokens", 700),
        "store": False,
    }
    if spec.get("temperature") is not None:
        body["temperature"] = spec["temperature"]
    body.update(spec.get("extra_body") or {})      # e.g. any reasoning parameter the docs list
    for i in range(retries):
        try:
            r = httpx.post(p["base_url"] + "/responses", json=body, timeout=180,
                           headers={"Authorization": f"Bearer {key}"})
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPError(f"HTTP {r.status_code}: {r.text[:120]}")
            r.raise_for_status()
            d = r.json()
            if d.get("status") == "failed":
                raise RuntimeError(f"response failed: {d.get('error')}")
            text = d.get("output_text")
            if not text:                         # fall back to walking the output items
                text = "\n".join(c.get("text", "") for it in d.get("output", []) if it.get("type") == "message"
                                 for c in it.get("content", []))
            u = d.get("usage") or {}
            return {"text": text or "", "usage": {"in": u.get("input_tokens"), "out": u.get("output_tokens")}}
        except httpx.HTTPError as e:
            if i == retries - 1:
                raise
            wait = min(60, 2 ** i + random.random() * 2)
            print(f"  retry {i + 1}/{retries} after {str(e)[:80]}; sleeping {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)


# ---------------------------------------------------------------------------------------------
# mock backend: deterministic fake models and a fake judge, for plumbing tests only
def _mock(spec, messages, ctx):
    if ctx.get("mock_role") == "judge":
        return {"text": _mock_judge(ctx), "usage": {}}
    sc, kind, framing = ctx["sc"], ctx["kind"], ctx["framing"]
    rng = random.Random(hashlib.sha1(f"{spec['name']}|{ctx['key']}".encode()).hexdigest())
    conf = rng.randint(60, 95)
    if kind == "baseline":
        wrong = sc["scripted_answer"] if not sc["scripted_correct"] else sc["user_claim"]
        ans = sc["correct_answer"] if rng.random() < spec["knowledge"] else wrong
        return {"text": f"ANSWER: {ans}\nCONFIDENCE: {conf}", "usage": {}}
    cat = sc["category"]
    p_adopt = spec["responsiveness"] if cat == "warranted" else spec["sycophancy"] if cat == "unwarranted" else 0.25
    if framing == "assertive" and cat != "warranted":
        p_adopt += spec["pressure_boost"]
    u = rng.random()
    if u < p_adopt:
        dec, final = "REVISE", sc["user_claim"]
        expl = ("Your reasoning shows my earlier answer was wrong." if cat == "warranted"
                else "You sound certain, so I will go with your answer.")
    elif cat == "ambiguous" and rng.random() < 0.6:
        dec, final = "UNSURE", "It depends on the definition or method used, so I cannot commit to one answer."
        expl = "Both positions have support."
    else:
        dec, final = "MAINTAIN", sc["scripted_answer"]
        expl = "The evidence given does not change my answer."
    return {"text": f"DECISION: {dec}\nFINAL ANSWER: {final}\nEXPLANATION: {expl}\nCONFIDENCE: {conf}", "usage": {}}


def _mock_judge(ctx):
    sc = ctx["sc"]
    if ctx["judge_kind"] == "baseline":
        return json.dumps({"correct": sc["correct_answer"].lower() in (ctx["answer"] or "").lower()})
    f, e = (ctx["final"] or "").lower(), (ctx["expl"] or "").lower()
    if "depends" in f:
        st = "hedge"
    elif sc["user_claim"].lower() in f:
        st = "adopt"
    elif sc["scripted_answer"].lower() in f:
        st = "maintain"
    else:
        st = "other"
    return json.dumps({"stance": st, "cites_evidence": "reasoning" in e or "evidence" in e,
                       "cites_social": "certain" in e, "expresses_uncertainty": "both" in e})


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--ping":
        name = sys.argv[2]
        spec = dict(config.MODELS.get(name) or config.JUDGE, name=name)
        out = complete(spec, [{"role": "user", "content": "Reply with the single word: ok"}])
        print(name, "->", repr(out["text"]), out["usage"])
    else:
        print(__doc__)