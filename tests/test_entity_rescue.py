from pathlib import Path

from zhms_s2s.enhancement import EntityNumberEnhancer
from zhms_s2s.entity_rescue import SelectiveEntityRescuer


LEXICON = Path(__file__).resolve().parents[1] / "data/entity_lexicon.json"


class FakeChinesePhoneticEncoder:
    available = True

    _sounds = {
        "槟": "bin",
        "冰": "bing",
        "城": "cheng",
        "炒": "chao",
        "粿": "guo",
        "果": "guo",
        "条": "tiao",
        "叻": "le",
        "乐": "le",
        "沙": "sha",
    }

    def encode_many(self, values):
        return {
            value: "".join(
                self._sounds.get(character, f"u{ord(character):x}")
                for character in value
            )
            for value in dict.fromkeys(values)
        }


def rescuer(*, chinese_phonetics=False):
    return SelectiveEntityRescuer(
        EntityNumberEnhancer(LEXICON),
        phonetic_encoder=(
            FakeChinesePhoneticEncoder() if chinese_phonetics else None
        ),
    )


def test_retrieves_only_small_relevant_entity_set():
    candidates = rescuer().retrieve(
        "Number pasangan saya ialah A tujuh tiga sembilan dua.", "ms"
    )

    assert candidates[0].canonical_name == "nombor pesanan"
    assert len(candidates) <= 3
    assert "KL Sentral" not in {item.canonical_name for item in candidates}


def test_known_alias_is_canonicalized_without_requiring_second_pass():
    engine = rescuer()
    raw = "Sila tiba di KL Central petang ini."
    candidates = engine.retrieve(raw, "ms")
    result = engine.select(raw, "ms", candidates=candidates, contextual_text=None)

    assert result.selected_text == "Sila tiba di KL Sentral petang ini."
    assert result.accepted_contextual_pass is False
    assert result.raw_text == raw


def test_contextual_pass_requires_candidate_coverage_improvement():
    engine = rescuer()
    raw = "Number pasangan saya ialah A tujuh tiga sembilan dua."
    candidates = engine.retrieve(raw, "ms")
    contextual = "Nombor pesanan saya ialah A tujuh tiga sembilan dua."

    result = engine.select(
        raw, "ms", candidates=candidates, contextual_text=contextual
    )

    assert result.accepted_contextual_pass is True
    assert result.selected_text == contextual
    assert result.safety_checks["candidate_coverage_improved"] is True


def test_contextual_pass_cannot_drop_negation():
    engine = rescuer()
    raw = "Number pasangan saya, jangan buat bayaran dua kali."
    candidates = engine.retrieve(raw, "ms")
    contextual = "Nombor pesanan saya, buat bayaran dua kali."

    result = engine.select(
        raw, "ms", candidates=candidates, contextual_text=contextual
    )

    assert result.accepted_contextual_pass is False
    assert result.safety_checks["missing_critical_words"] == ["jangan"]


def test_contextual_pass_cannot_change_digits_or_inject_other_entities():
    engine = rescuer()
    raw = "Number pasangan A7392."
    candidates = engine.retrieve(raw, "ms")
    contextual = "Nombor pesanan A7393 di KL Sentral."

    result = engine.select(
        raw, "ms", candidates=candidates, contextual_text=contextual
    )

    assert result.accepted_contextual_pass is False
    assert result.safety_checks["digits_preserved"] is False
    assert result.safety_checks["unexpected_entities"] == ["KL Sentral"]


def test_request_specific_vocabulary_can_trigger_without_global_prompt():
    candidates = rescuer().retrieve(
        "Saya mahu ke Restoran Rebung.",
        "ms",
        user_terms=("Restoran Rebung",),
    )

    # It is already present, so no second pass is needed and no unrelated
    # global terms are appended.
    assert candidates == ()

    missing = rescuer().retrieve(
        "Saya mahu ke Restoran Rebong.",
        "ms",
        user_terms=("Restoran Rebung",),
    )
    assert [item.canonical_name for item in missing] == ["Restoran Rebung"]
    assert rescuer().build_prompt(missing) == "Restoran Rebung"


def test_multiword_entity_with_single_word_alias_keeps_multiword_fuzzy_policy():
    candidates = rescuer().retrieve(
        "Selamat datang ke demo suara luar darian.", "ms"
    )

    assert "luar talian" in {item.canonical_name for item in candidates}


def test_longer_alias_match_suppresses_overlapping_short_alias():
    candidates = rescuer().retrieve("Sila tiba di KL Central.", "ms")

    assert [item.canonical_name for item in candidates] == ["KL Sentral"]


def test_short_acronyms_do_not_match_ordinary_malay_words():
    engine = rescuer()

    assert engine.retrieve("Ia masih berada di sini.", "ms") == ()
    assert engine.retrieve("Bayar lima kali.", "ms") == ()
    assert engine.retrieve("Emas dan perak ialah logam.", "ms") == ()


def test_known_alias_does_not_waste_a_second_asr_pass():
    engine = rescuer()
    text = "Sila tiba di KL Central."
    candidates = engine.retrieve(text, "ms")

    assert engine.needs_contextual_pass(text, "ms", candidates) is False

    fuzzy_text = "Number pasangan saya ialah A7392."
    fuzzy_candidates = engine.retrieve(fuzzy_text, "ms")
    assert engine.needs_contextual_pass(
        fuzzy_text, "ms", fuzzy_candidates
    ) is True


def test_request_specific_terms_take_priority_over_dictionary_candidates():
    engine = rescuer()
    candidates = engine.retrieve(
        "Number pasangan ke KL Central.",
        "ms",
        user_terms=("Restoran Rebung", "MH123"),
    )

    assert [item.canonical_name for item in candidates[:2]] == [
        "MH123",
        "Restoran Rebung",
    ]


def test_request_term_keeps_matching_span_to_suppress_shorter_aliases():
    candidates = rescuer().retrieve(
        "请不要在 Kl Central 下车。",
        "zh",
        user_terms=("KL Sentral", "Pulau Pinang"),
    )

    assert [item.canonical_name for item in candidates] == [
        "KL Sentral",
        "Pulau Pinang",
    ]
    assert candidates[0].matched_text == "Kl Central"


def test_chinese_near_match_can_trigger_conservative_contextual_rescue():
    engine = rescuer()
    raw = "请抵达吉隆坡中央车占。"
    candidates = engine.retrieve(raw, "zh")

    assert [item.canonical_name for item in candidates] == ["吉隆坡中央车站"]
    assert engine.needs_contextual_pass(raw, "zh", candidates) is True

    result = engine.select(
        raw,
        "zh",
        candidates=candidates,
        contextual_text="请抵达吉隆坡中央车站。",
    )

    assert result.accepted_contextual_pass is True
    assert result.selected_text == "请抵达吉隆坡中央车站。"


def test_chinese_contextual_pass_cannot_drop_negation():
    engine = rescuer()
    raw = "不要去吉隆坡中央车占。"
    candidates = engine.retrieve(raw, "zh")
    result = engine.select(
        raw,
        "zh",
        candidates=candidates,
        contextual_text="去吉隆坡中央车站。",
    )

    assert result.accepted_contextual_pass is False
    assert "不要" in result.safety_checks["missing_critical_words"]


def test_chinese_phonetic_retrieval_is_general_not_error_alias_hardcoding():
    engine = rescuer(chinese_phonetics=True)

    location = engine.retrieve("我要去冰城。", "zh")
    food_one = engine.retrieve("我想吃炒果条。", "zh")
    food_two = engine.retrieve("这个乐沙很好吃。", "zh")

    assert [item.canonical_name for item in location] == ["槟城"]
    assert [item.canonical_name for item in food_one] == ["炒粿条"]
    assert [item.canonical_name for item in food_two] == ["叻沙"]
    assert all(
        item.reason == "offline-pinyin-near-match"
        for item in (*location, *food_one, *food_two)
    )


def test_chinese_phonetic_candidate_biases_second_pass_without_hard_replacement():
    engine = rescuer(chinese_phonetics=True)
    raw = "我想吃炒果条。"
    candidates = engine.retrieve(raw, "zh")

    assert engine.select(
        raw, "zh", candidates=candidates, contextual_text=None
    ).selected_text == raw

    result = engine.select(
        raw,
        "zh",
        candidates=candidates,
        contextual_text="我想吃炒粿条。",
    )

    assert result.accepted_contextual_pass is True
    assert result.selected_text == "我想吃炒粿条。"


def test_chinese_phonetic_retrieval_requires_semantic_context():
    engine = rescuer(chinese_phonetics=True)

    assert engine.retrieve("冰城是哈尔滨的别称。", "zh") == ()


def test_unresolved_exact_homophone_is_reranked_after_qwen_context_pass():
    engine = rescuer(chinese_phonetics=True)
    raw = "这个乐沙很好吃。"
    candidates = engine.retrieve(raw, "zh")

    result = engine.select(
        raw,
        "zh",
        candidates=candidates,
        contextual_text=raw,
    )

    assert result.accepted_contextual_pass is False
    assert result.selected_text == "这个叻沙很好吃。"
    assert result.decision_reason == "phonetic-rerank-after-context-rejection"
    assert result.safety_checks["phonetic_rerank_applied"] == ["叻沙"]
    assert result.corrections[0]["correction_type"] == "entity_phonetic_rerank"
