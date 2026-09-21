from datetime import date
from app.versioning import is_effective, windows_overlap


def test_effectivity_includes_boundaries():
    assert is_effective(date(2020, 1, 1), date(2020, 12, 31), date(2020, 12, 31))
    assert not is_effective(date(2020, 1, 1), date(2020, 12, 31), date(2021, 1, 1))


def test_open_ended_versions_overlap_later_version():
    assert windows_overlap(date(2020, 1, 1), None, date(2025, 1, 1), None)
