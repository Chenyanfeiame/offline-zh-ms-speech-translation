from __future__ import annotations

import pytest

from zhms_s2s.numbers import (
    canonical_number_tokens,
    normalize_malay_numbers,
    parse_malay_number,
)


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("tujuh", 7),
        ("tiga puluh lima", 35),
        ("empat puluh lima", 45),
        ("seratus dua puluh tiga", 123),
        ("dua ribu dua puluh lima", 2025),
    ],
)
def test_parse_malay_cardinals(spoken, expected):
    assert parse_malay_number(spoken) == expected


def test_adjacent_digits_are_not_misread_as_a_cardinal():
    assert parse_malay_number("tujuh tiga sembilan dua") is None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("empat puluh lima minit", "45 minit"),
        ("dua puluh lima November", "25 November"),
        ("pukul tiga petang", "pukul 3 petang"),
        ("A tujuh tiga sembilan dua", "A7392"),
        ("A tuju tiga sembilan dua", "A7392"),
        ("nombor tujuh tiga sembilan dua", "nombor 7392"),
    ],
)
def test_normalize_malay_numbers(source, expected):
    assert normalize_malay_numbers(source)[0] == expected


def test_uncued_adjacent_digits_are_left_unchanged():
    assert normalize_malay_numbers("tujuh tiga sembilan dua")[0] == (
        "tujuh tiga sembilan dua"
    )


def test_number_words_inside_lexicalized_hyphen_forms_are_not_rewritten():
    assert normalize_malay_numbers("pada kedua-dua frekuensi")[0] == (
        "pada kedua-dua frekuensi"
    )


def test_evaluation_number_slots_canonicalize_grouping_and_decimal_zeros():
    assert canonical_number_tokens("10,000 dan 5.00 GHz") == ["10000", "5"]
