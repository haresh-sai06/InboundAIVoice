from eval.agent_pure import estimate_cost, get_language_instruction

def test_language_presets():
    assert "Hindi" in get_language_instruction("hi")
    assert "language" in get_language_instruction("multilingual").lower()

def test_estimate_cost_positive():
    assert estimate_cost(120, 500) > 0
