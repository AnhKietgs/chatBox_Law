from datetime import date


def is_effective(effective_from: date, effective_to: date | None, as_of: date) -> bool:
    return effective_from <= as_of and (effective_to is None or as_of <= effective_to)


def windows_overlap(start_a: date, end_a: date | None, start_b: date, end_b: date | None) -> bool:
    normalized_a = end_a or date.max
    normalized_b = end_b or date.max
    return start_a <= normalized_b and start_b <= normalized_a
