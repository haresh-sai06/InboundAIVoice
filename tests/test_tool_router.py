from eval.tool_router import route_heuristic

def test_transfer():
    assert route_heuristic("Can I speak to the founder?").tool == "transfer_call"

def test_end_call():
    assert route_heuristic("Thanks, bye!").tool == "end_call"

def test_hours():
    assert route_heuristic("What are your business hours?").tool == "get_business_hours"

def test_availability_today():
    d = route_heuristic("Do you have any slots open today?")
    assert d.tool == "check_availability" and d.args.get("date_hint") == "today"

def test_booking_extracts_name_email():
    d = route_heuristic("3pm works. My name is Riya Sharma, email riya@smilecare.in")
    assert d.tool == "save_booking_intent"
    assert d.args["email"] == "riya@smilecare.in"
    assert "Riya" in d.args["name"]

def test_incomplete_email_does_not_book():
    d = route_heuristic("Book a call, my email is onlyhalf@")
    assert d.tool != "save_booking_intent"

def test_chitchat_no_tool():
    assert route_heuristic("I run a dental clinic in Pune.").tool is None
