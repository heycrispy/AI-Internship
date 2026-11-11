"""Simulated availability check against a small JSON calendar (calendar_fixtures.json).

Rules (booking_policies.md): 72 hours minimum notice, typical service window 7 AM-9 PM Pacific.
A 60-minute buffer on each side of an existing booking covers trailer travel, setup, and cleanup.
Results are labeled simulated; the owner confirms real scheduling.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")
FIXTURES = Path(__file__).with_name("calendar_fixtures.json")
MIN_NOTICE = timedelta(hours=72)
BUFFER = timedelta(minutes=60)
SERVICE_START, SERVICE_END = time(7, 0), time(21, 0)
SIMULATED_NOTE = ("Simulated demo calendar, not live availability. "
                  "The owner confirms final scheduling.")

_DATE_FORMATS = ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y")
_TIME_FORMATS = ("%H:%M", "%I:%M %p", "%I %p", "%I:%M%p", "%I%p")


def parse_date(text: str) -> date:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Could not read date '{text}'. Use YYYY-MM-DD.")


def parse_time(text: str) -> time:
    cleaned = text.strip().upper().replace(".", "")
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Could not read time '{text}'. Use HH:MM (24h) or e.g. 10:00 AM.")


def load_bookings(path: Path = FIXTURES) -> list[dict]:
    return json.loads(path.read_text())["bookings"]


def _at(d: date, t: time) -> datetime:
    return datetime.combine(d, t, tzinfo=PACIFIC)


def check(
    event_date: str,
    start_time: str,
    duration_hours: float,
    now: datetime | None = None,
    bookings: list[dict] | None = None,
) -> dict:
    """Return status available | unavailable | requires_review, with reasons."""
    now = now or datetime.now(PACIFIC)
    bookings = load_bookings() if bookings is None else bookings
    try:
        d, t = parse_date(event_date), parse_time(start_time)
    except ValueError as exc:
        return {"status": "invalid", "reasons": [str(exc)], "note": SIMULATED_NOTE}
    if not 0 < duration_hours <= 12:
        return {"status": "invalid", "reasons": ["Duration must be between 0 and 12 hours."],
                "note": SIMULATED_NOTE}

    start = _at(d, t)
    end = start + timedelta(hours=duration_hours)
    result = {
        "date": d.isoformat(),
        "requested_start": start.strftime("%I:%M %p").lstrip("0"),
        "requested_end": end.strftime("%I:%M %p").lstrip("0"),
        "buffer_minutes": int(BUFFER.total_seconds() // 60),
        "conflicts": [],
        "reasons": [],
        "note": SIMULATED_NOTE,
    }

    if start <= now:
        result.update(status="unavailable", reasons=["That date and time has already passed."])
        return result

    for b in bookings:
        b_start = _at(parse_date(b["date"]), parse_time(b["start"]))
        b_end = _at(parse_date(b["date"]), parse_time(b["end"]))
        if start < b_end + BUFFER and end > b_start - BUFFER:
            result["conflicts"].append(
                {"date": b["date"], "start": b["start"], "end": b["end"],
                 "event_type": b["event_type"], "status": b["status"]}
            )
    if result["conflicts"]:
        result["status"] = "unavailable"
        result["reasons"].append("Overlaps an existing booking (including the "
                                 f"{result['buffer_minutes']}-minute setup/travel buffer).")
        return result

    if start - now < MIN_NOTICE:
        result["reasons"].append("Less than 72 hours' notice: needs owner review.")
    if t < SERVICE_START or end > _at(d, SERVICE_END) or end.date() != d:
        result["reasons"].append("Outside the typical 7 AM-9 PM service window: needs owner review.")
    result["status"] = "requires_review" if result["reasons"] else "available"
    if result["status"] == "available":
        result["reasons"].append("No conflicts on the simulated calendar.")
    return result
