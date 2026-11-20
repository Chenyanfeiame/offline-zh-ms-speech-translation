from zhms_s2s.backends.mt import (
    _domain_translation_template,
    _protect_terminology,
    _restore_terminology,
    _split_translation_segments,
)


def test_splits_malay_multi_sentence_input():
    assert _split_translation_segments(
        "Nama saya Ali. Saya tinggal di Kuala Lumpur."
    ) == ["Nama saya Ali.", "Saya tinggal di Kuala Lumpur."]


def test_splits_question_and_answer():
    assert _split_translation_segments(
        "Di manakah stesen? Stesen itu di sebelah bank."
    ) == ["Di manakah stesen?", "Stesen itu di sebelah bank."]


def test_does_not_split_decimal_amount():
    assert _split_translation_segments(
        "Jumlahnya RM12.50. Bayar esok."
    ) == ["Jumlahnya RM12.50.", "Bayar esok."]


def test_protects_common_abbreviations_and_initials():
    assert _split_translation_segments(
        "Dr. Ahmad bekerja di Syarikat Maju Sdn. Bhd. Dia tinggal di Johor."
    ) == [
        "Dr. Ahmad bekerja di Syarikat Maju Sdn. Bhd.",
        "Dia tinggal di Johor.",
    ]


def test_protects_longest_place_names_bidirectionally():
    protected, replacements = _protect_terminology(
        "Saya pergi dari Johor Bahru ke Pulau Pinang.", "ms", "zh"
    )
    assert "Johor" not in protected
    assert "Pinang" not in protected
    assert _restore_terminology("XQ001X到XQ002X", replacements) == "新山到槟城"

    protected, replacements = _protect_terminology(
        "我从新山去槟城。", "zh", "ms"
    )
    assert _restore_terminology("XQ001X ke XQ002X", replacements) == (
        "Pulau Pinang ke Johor Bahru"
    )


def test_splits_chinese_sentences_without_spaces():
    assert _split_translation_segments("订单号是A73992。请不要重复付款。") == [
        "订单号是A73992。",
        "请不要重复付款。",
    ]
    assert _split_translation_segments("Nama saya A. Rahman. Saya doktor.") == [
        "Nama saya A. Rahman.",
        "Saya doktor.",
    ]


def test_critical_order_and_payment_templates_preserve_slots_bidirectionally():
    assert _domain_translation_template(
        "Nombor pesanan saya ialah A7-392.", "ms", "zh"
    ) == "我的订单号是A7-392。"
    assert _domain_translation_template(
        "Jangan buat bayaran dua kali.", "ms", "zh"
    ) == "请勿重复付款。"
    assert _domain_translation_template(
        "我的订单号是A7-392。", "zh", "ms"
    ) == "Nombor pesanan saya ialah A7-392."
    assert _domain_translation_template("请勿重复付款。", "zh", "ms") == (
        "Jangan buat bayaran dua kali."
    )
