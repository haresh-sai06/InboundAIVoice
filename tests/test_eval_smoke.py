from pathlib import Path
from eval.run_eval import evaluate, load

DATA = Path(__file__).resolve().parents[1] / "eval" / "golden_dialogues.json"

def test_heuristic_accuracy_at_least_85_percent():
    res = evaluate(load(DATA), "heuristic")
    assert res["n_turns"] >= 20
    assert res["tool_accuracy"] >= 0.85
