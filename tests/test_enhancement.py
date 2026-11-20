from __future__ import annotations

from pathlib import Path

from zhms_s2s.enhancement import EntityNumberEnhancer, parse_user_terms


LEXICON = Path(__file__).resolve().parents[1] / "data/entity_lexicon.json"


def enhancer():
    return EntityNumberEnhancer(LEXICON)


def test_alias_entity_and_date_are_enhanced_without_losing_raw_text():
    raw = (
        "Sila tiba di KL Central pada pukul tiga petang, "
        "dua puluh lima November."
    )
    result = enhancer().enhance(raw, "ms")

    assert result.raw_text == raw
    assert result.enhanced_text == (
        "Sila tiba di KL Sentral pada pukul 3 petang, 25 November."
    )
    assert {item["canonical_name"] for item in result.entities} >= {
        "KL Sentral",
        "November",
    }
    assert any(item["correction_type"] == "entity_alias" for item in result.corrections)
    assert len(result.number_normalizations) == 2


def test_identifier_is_digitwise_but_cardinal_is_not():
    raw = (
        "Nombor pesanan saya ialah A tujuh tiga sembilan dua, "
        "lewat empat puluh lima minit."
    )
    result = enhancer().enhance(raw, "ms")

    assert "A7392" in result.enhanced_text
    assert "45 minit" in result.enhanced_text
    assert "A73592" not in result.enhanced_text


def test_high_confidence_pronunciation_variant_is_corrected():
    result = enhancer().enhance(
        "Penerbangan dari Beijing ke Pulau Pineng lewat.", "ms"
    )

    assert result.enhanced_text == (
        "Penerbangan dari Beijing ke Pulau Pinang lewat."
    )


def test_unrelated_words_are_not_forced_to_an_entity():
    raw = "Saya membaca buku baharu di rumah."
    result = enhancer().enhance(raw, "ms")

    assert result.enhanced_text == raw
    assert result.corrections == ()


def test_prompt_is_compact_and_scene_independent():
    prompt, terms = enhancer().build_prompt("ms", max_terms=12, max_characters=150)

    assert len(terms) <= 12
    assert len(prompt) <= 150
    assert {"jangan", "tidak", "bukan"}.issubset(terms)
    assert "KL Sentral" in terms


def test_user_terms_are_validated_without_requiring_a_scene():
    assert parse_user_terms("MH123, Restoran Rebung") == (
        "MH123",
        "Restoran Rebung",
    )
