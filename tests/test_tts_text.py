from zhms_s2s.backends.tts import build_tts_plan, normalize_tts_text


def test_malay_dates_numbers_times_and_ids_are_expanded_for_speech():
    assert normalize_tts_text(
        "Hari ini Khamis, 19 November. Pesanan A7392 pada 15:30, nilai 3.5.",
        "ms",
    ) == (
        "Hari ini Khamis, sembilan belas November. Pesanan A tujuh tiga sembilan dua "
        "pada lima belas tiga puluh, nilai tiga perpuluhan lima."
    )


def test_chinese_dates_numbers_times_and_ids_are_expanded_for_speech():
    assert normalize_tts_text(
        "今天是2026-11-19，订单A7392，时间15:30，温度3.5度。",
        "zh",
    ) == "今天是二零二六年十一月十九日，订单A 七三九二，时间十五点三十分，温度三点五度。"


def test_phone_like_numbers_are_read_digit_by_digit_in_both_languages():
    assert normalize_tts_text("01234567", "ms") == (
        "kosong satu dua tiga empat lima enam tujuh"
    )
    assert normalize_tts_text("01234567", "zh") == "零一二三四五六七"


def test_text_without_digits_is_unchanged():
    text = "Selamat pagi, Kuala Lumpur."
    assert normalize_tts_text(text, "ms") == text


def test_same_digits_are_spoken_by_semantic_context_without_changing_display():
    display = (
        "我的电话号码是013-4567890，订单号是1234，机票价格是1234令吉，"
        "航班MH1234预计延误45分钟。"
    )

    plan = build_tts_plan(display, "zh")

    assert plan.display_text == display
    assert plan.spoken_text == (
        "我的电话号码是零一三，四五六七八九零，订单号是一二三四，"
        "机票价格是一千二百三十四令吉，航班M H 一二三四预计延误四十五分钟。"
    )
    assert [span.semantic_type for span in plan.semantic_spans] == [
        "telephone",
        "identifier",
        "money",
        "flight_identifier",
        "duration",
    ]
    assert all(span.confidence == 1.0 for span in plan.semantic_spans)


def test_short_phone_and_order_id_are_digit_sequences_but_quantities_are_cardinal():
    assert normalize_tts_text("我的电话号码是1234。", "zh") == (
        "我的电话号码是一二三四。"
    )
    assert normalize_tts_text("订单号是1234。", "zh") == "订单号是一二三四。"
    assert normalize_tts_text("订单数量是1234个。", "zh") == (
        "订单数量是一千二百三十四个。"
    )
    assert normalize_tts_text("共有1234名乘客。", "zh") == (
        "共有一千二百三十四名乘客。"
    )


def test_malay_semantic_number_contexts_use_the_same_classifier():
    plan = build_tts_plan(
        "Nombor telefon 1234, nombor pesanan 5678, harga RM 1234, "
        "penerbangan MH1234 lewat 45 minit.",
        "ms",
    )

    assert [span.semantic_type for span in plan.semantic_spans] == [
        "telephone",
        "identifier",
        "money",
        "flight_identifier",
        "duration",
    ]
    assert "satu dua tiga empat" in plan.spoken_text
    assert "lima enam tujuh lapan" in plan.spoken_text
    assert "seribu dua ratus tiga puluh empat ringgit" in plan.spoken_text
    assert "empat puluh lima minit" in plan.spoken_text


def test_malay_currency_codes_are_expanded_for_speech_only():
    compact = build_tts_plan("Harga tiket ialah RM1234.", "ms")
    spaced = build_tts_plan("Harga tiket ialah MYR 1234.", "ms")

    assert compact.display_text == "Harga tiket ialah RM1234."
    assert compact.spoken_text == (
        "Harga tiket ialah seribu dua ratus tiga puluh empat ringgit."
    )
    assert spaced.display_text == "Harga tiket ialah MYR 1234."
    assert spaced.spoken_text == (
        "Harga tiket ialah seribu dua ratus tiga puluh empat ringgit."
    )
