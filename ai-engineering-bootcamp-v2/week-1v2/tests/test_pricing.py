"""Unit tests for the deterministic quote calculator. Run: python -m unittest discover tests"""

import unittest

from golden_hour_agent.pricing import quote


class QuoteTests(unittest.TestCase):
    def test_assignment_example_85_guests_3_hours_newport(self):
        q = quote(85, 3, "Newport Beach", "The Golden Hour")
        self.assertEqual(q["subtotal_usd"], 850)
        self.assertEqual(q["estimated_deposit_usd"], 255)
        self.assertEqual(q["remaining_balance_usd"], 595)
        self.assertTrue(any("taxes" in u.lower() for u in q["unresolved_charges"]))

    def test_no_package_recommends_cheapest_that_fits(self):
        q = quote(85, 3, "Newport Beach")  # Sunrise maxes out at 75 guests
        self.assertEqual(q["recommended"], "The Golden Hour")
        self.assertEqual(q["recommended_quote"]["subtotal_usd"], 850)
        self.assertNotIn("The Sunrise", [o["package"] for o in q["all_options"]])

    def test_extra_guests_and_partial_hours_round_up(self):
        q = quote(60, 2.5, "Irvine", "Sunrise")  # 10 extra guests, 0.5h -> 1 extra hour
        self.assertEqual(q["subtotal_usd"], 550 + 10 * 8 + 150)
        self.assertEqual(q["billed_hours"], 3)

    def test_no_discount_below_included_allowance(self):
        self.assertEqual(quote(20, 1, "Costa Mesa", "The Sunset")["subtotal_usd"], 1300)

    def test_travel_fee_bands(self):
        self.assertEqual(quote(40, 2, "Huntington Beach", "A")["subtotal_usd"], 600)
        self.assertEqual(quote(40, 2, "San Clemente, CA", "A")["subtotal_usd"], 650)

    def test_unknown_location_is_unresolved_not_guessed(self):
        q = quote(40, 2, "Long Beach", "A")
        self.assertEqual(q["subtotal_usd"], 550)
        self.assertTrue(any("Long Beach" in u for u in q["unresolved_charges"]))

    def test_capacity_exceeded_is_reported_not_silently_upgraded(self):
        q = quote(120, 3, "Irvine", "The Sunrise")
        self.assertEqual(q["status"], "exceeds_capacity")
        self.assertEqual(q["package"], "The Sunrise")
        self.assertIn("The Golden Hour", [a["package"] for a in q["alternatives"]])

    def test_above_200_guests_needs_custom_quote(self):
        self.assertEqual(quote(201, 4, "Irvine")["status"], "custom_quote_required")

    def test_sunset_extra_hour_rate(self):
        self.assertEqual(quote(150, 5, "Irvine", "C")["subtotal_usd"], 1300 + 175)

    def test_invalid_inputs(self):
        self.assertEqual(quote(50, 2, "Irvine", "Platinum")["status"], "error")
        with self.assertRaises(ValueError):
            quote(0, 2, "Irvine", "A")
        with self.assertRaises(ValueError):
            quote(10, 0, "Irvine", "A")


if __name__ == "__main__":
    unittest.main()
