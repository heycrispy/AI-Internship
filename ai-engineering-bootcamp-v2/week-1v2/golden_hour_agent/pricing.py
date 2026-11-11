"""Deterministic quote calculator for Golden Hour Coffee Co.

The knowledge base explains the packages to customers; THIS module is the source of truth for
money. The LLM never does pricing arithmetic: it calls calculate_quote, which calls quote().
Rules mirror sample_docs/golden_hour/catering_packages.md and booking_policies.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEPOSIT_RATE = 0.30
CUSTOM_QUOTE_ABOVE_GUESTS = 200
TAXES_AND_FEES = "Applicable taxes and fees (verified by the owner before confirmation)"


@dataclass(frozen=True)
class Package:
    key: str
    name: str
    base_price: int
    included_guests: int
    included_hours: int
    extra_guest_price: int
    extra_hour_price: int
    max_guests: int


PACKAGES: dict[str, Package] = {
    p.key: p
    for p in (
        Package("sunrise", "The Sunrise", 550, 50, 2, 8, 150, 75),
        Package("golden_hour", "The Golden Hour", 850, 100, 3, 7, 150, 150),
        Package("sunset", "The Sunset", 1300, 150, 4, 6, 175, 200),
    )
}

# Fictional service zones from booking_policies.md. Anything else -> manual owner review.
TRAVEL_FEES: dict[str, int] = {
    "newport beach": 0, "costa mesa": 0, "irvine": 0, "santa ana": 0,
    "huntington beach": 50, "laguna beach": 50, "anaheim": 50,
    "fullerton": 100, "san clemente": 100,
}

_PACKAGE_ALIASES = {
    "a": "sunrise", "package a": "sunrise", "sunrise": "sunrise", "the sunrise": "sunrise",
    "b": "golden_hour", "package b": "golden_hour", "golden hour": "golden_hour",
    "the golden hour": "golden_hour", "golden_hour": "golden_hour",
    "c": "sunset", "package c": "sunset", "sunset": "sunset", "the sunset": "sunset",
}


def find_package(name: str) -> Package | None:
    key = _PACKAGE_ALIASES.get(name.strip().lower().replace("package:", "").strip())
    return PACKAGES.get(key) if key else None


def travel_fee(location: str) -> int | None:
    """Fee for a city in a documented zone, or None when the owner must review it."""
    city = location.split(",")[0].strip().lower()
    return TRAVEL_FEES.get(city)


def _money(x: float) -> float | int:
    x = round(x, 2)
    return int(x) if x.is_integer() else x


def quote_package(pkg: Package, guests: int, hours: float, location: str) -> dict:
    """Itemized estimate for one package. Raises ValueError for invalid inputs."""
    if guests < 1:
        raise ValueError("Guest count must be at least 1.")
    if hours <= 0:
        raise ValueError("Service duration must be more than 0 hours.")
    if guests > pkg.max_guests:
        return {
            "status": "exceeds_capacity",
            "package": pkg.name,
            "message": f"{pkg.name} serves at most {pkg.max_guests} guests; {guests} requested.",
        }

    # Never discount below the included allowance; round partial extra hours UP.
    extra_guests = max(0, guests - pkg.included_guests)
    extra_hours = max(0, math.ceil(round(hours - pkg.included_hours, 6)))
    fee = travel_fee(location)

    items = [{"item": f"{pkg.name} package (up to {pkg.included_guests} guests, "
                      f"{pkg.included_hours} hours)", "amount_usd": pkg.base_price}]
    if extra_guests:
        items.append({"item": f"{extra_guests} additional guests x ${pkg.extra_guest_price}",
                      "amount_usd": extra_guests * pkg.extra_guest_price})
    if extra_hours:
        items.append({"item": f"{extra_hours} additional hours x ${pkg.extra_hour_price} "
                              "(partial hours rounded up)",
                      "amount_usd": extra_hours * pkg.extra_hour_price})
    unresolved = [TAXES_AND_FEES]
    if fee is None:
        unresolved.append(f"Travel fee for '{location}' (outside listed zones; owner review)")
    else:
        items.append({"item": f"Travel fee ({location.split(',')[0].strip()})", "amount_usd": fee})

    subtotal = sum(i["amount_usd"] for i in items)
    deposit = _money(subtotal * DEPOSIT_RATE)
    return {
        "status": "estimate",
        "package": pkg.name,
        "guests": guests,
        "service_hours": hours,
        "billed_hours": pkg.included_hours + extra_hours,
        "location": location,
        "line_items": items,
        "subtotal_usd": subtotal,
        "estimated_deposit_usd": deposit,
        "deposit_rate": DEPOSIT_RATE,
        "remaining_balance_usd": _money(subtotal - deposit),
        "unresolved_charges": unresolved,
        "note": "Estimate only. Not a confirmed booking; the owner verifies final pricing.",
    }


def quote(guests: int, hours: float, location: str, package: str | None = None) -> dict:
    """Quote one named package, or every package that fits and recommend the cheapest."""
    if guests > CUSTOM_QUOTE_ABOVE_GUESTS:
        return {"status": "custom_quote_required",
                "message": f"Events above {CUSTOM_QUOTE_ABOVE_GUESTS} guests need a custom "
                           "quote from the owner."}
    if package:
        pkg = find_package(package)
        if pkg is None:
            return {"status": "error",
                    "message": f"Unknown package '{package}'. Options: "
                               + ", ".join(p.name for p in PACKAGES.values())}
        result = quote_package(pkg, guests, hours, location)
        if result["status"] == "exceeds_capacity":
            # Don't silently upgrade: report it and show what does fit.
            result["alternatives"] = [q for q in _all_fitting(guests, hours, location)]
        return result

    options = _all_fitting(guests, hours, location)
    best = min(options, key=lambda q: q["subtotal_usd"])
    return {"status": "estimate", "recommended": best["package"],
            "recommended_quote": best, "all_options": options}


def _all_fitting(guests: int, hours: float, location: str) -> list[dict]:
    return [
        q for p in PACKAGES.values()
        if (q := quote_package(p, guests, hours, location))["status"] == "estimate"
    ]
