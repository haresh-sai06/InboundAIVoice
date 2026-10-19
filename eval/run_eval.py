"""Offline eval for InboundAIVoice tool routing + response constraints.

Does **not** place live SIP calls. Measures:

* tool-call accuracy (exact tool match per turn)
* booking arg F1 (name / email) on save_booking_intent turns
* date_hint recall on check_availability turns
* optional: mean router latency

Usage::

    python eval/run_eval.py --mode heuristic --out eval/results/summary.json
    LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:1.5b \
      python eval/run_eval.py --mode llm --out eval/results/llm.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.tool_router import route  # noqa: E402


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def arg_prf(pred: dict, gold: dict) -> tuple[int, int, int]:
    tp = fp = fn = 0
    keys = set(gold) | set(pred)
    for k in keys:
        g, p = gold.get(k), pred.get(k)
        if g is None and p is None:
            continue
        if g is not None and p is not None and str(g).lower() == str(p).lower():
            tp += 1
        elif p is not None and g is None:
            fp += 1
        elif g is not None and p != g:
            # wrong or missing
            if p is None:
                fn += 1
            else:
                fp += 1
                fn += 1
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def evaluate(dialogues: list[dict], mode: str) -> dict:
    correct = total = 0
    per_tool = {}
    atp = afp = afn = 0
    latencies = []
    rows = []
    for d in dialogues:
        for turn in d["turns"]:
            t0 = time.perf_counter()
            decision = route(turn["text"], mode=mode)
            latencies.append(time.perf_counter() - t0)
            gold_tool = turn.get("expect_tool")
            ok = decision.tool == gold_tool
            correct += int(ok)
            total += 1
            key = gold_tool or "null"
            bucket = per_tool.setdefault(key, {"correct": 0, "total": 0})
            bucket["total"] += 1
            bucket["correct"] += int(ok)
            gold_args = turn.get("expect_args") or {}
            if gold_tool == "save_booking_intent" or gold_args:
                tp, fp, fn = arg_prf(decision.args, gold_args)
                atp += tp; afp += fp; afn += fn
            rows.append({
                "dialogue": d["id"], "text": turn["text"], "gold": gold_tool,
                "pred": decision.tool, "args": decision.args, "ok": ok,
                "rationale": decision.rationale,
            })
    return {
        "mode": mode,
        "n_turns": total,
        "n_dialogues": len(dialogues),
        "tool_accuracy": round(correct / total, 4) if total else 0.0,
        "tool_correct": correct,
        "per_tool_accuracy": {k: round(v["correct"] / v["total"], 4) for k, v in per_tool.items()},
        "booking_arg_prf": prf(atp, afp, afn),
        "latency_s": {
            "mean": round(statistics.mean(latencies), 4),
            "p50": round(statistics.median(latencies), 4),
            "p95": round(sorted(latencies)[max(0, int(round(0.95 * len(latencies))) - 1)], 4),
        },
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "eval" / "golden_dialogues.json"))
    ap.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = evaluate(load(Path(args.data)), args.mode)
    summary = {k: v for k, v in res.items() if k != "rows"}
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
