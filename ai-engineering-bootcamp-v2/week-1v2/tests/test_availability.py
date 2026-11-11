"""Unit tests for the simulated availability check. Run: python -m unittest discover tests"""

import unittest
from datetime import datetime

from golden_hour_agent.availability import PACIFIC, check

NOW = datetime(2026, 11, 11, 9, 0, tzinfo=PACIFIC)  # fixed "today" so tests don't drift
BOOKINGS = [
    {"date": "2026-11-21", "start": "10:00", "end": "13:00",
     "event_type": "Corporate event", "status": "confirmed"},
]


def run(date, start, hours, now=NOW):
    return check(date, start, hours, now=now, bookings=BOOKINGS)


class AvailabilityTests(unittest.TestCase):
    def test_exact_overlap_is_unavailable(self):
        r = run("2026-11-21", "10:00 AM", 3)
        self.assertEqual(r["status"], "unavailable")
        self.assertEqual(r["conflicts"][0]["event_type"], "Corporate event")

    def test_human_date_format(self):
        self.assertEqual(run("November 21, 2026", "10 AM", 3)["status"], "unavailable")

    def test_buffer_blocks_back_to_back(self):
        self.assertEqual(run("2026-11-21", "13:30", 2)["status"], "unavailable")

    def test_after_buffer_is_available(self):
        self.assertEqual(run("2026-11-21", "14:00", 2)["status"], "available")

    def test_free_day_is_available_and_labeled_simulated(self):
        r = run("2026-11-28", "09:00", 3)
        self.assertEqual(r["status"], "available")
        self.assertIn("Simulated", r["note"])

    def test_less_than_72_hours_needs_review(self):
        r = run("2026-11-13", "10:00", 2)
        self.assertEqual(r["status"], "requires_review")

    def test_outside_service_window_needs_review(self):
        self.assertEqual(run("2026-11-28", "19:00", 3)["status"], "requires_review")
        self.assertEqual(run("2026-11-28", "06:00", 2)["status"], "requires_review")

    def test_past_date_is_unavailable(self):
        self.assertEqual(run("2026-11-01", "10:00", 2)["status"], "unavailable")

    def test_invalid_input(self):
        self.assertEqual(run("next tuesday", "10:00", 2)["status"], "invalid")
        self.assertEqual(run("2026-11-28", "10:00", 0)["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
