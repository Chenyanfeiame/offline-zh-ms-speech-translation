from __future__ import annotations

from zhms_s2s.backends.hy_mt import HyMtLlamaTranslator


def make_backend(
    tmp_path,
    *,
    terminology_enabled=False,
    critical_slot_protection_enabled=False,
):
    model = tmp_path / "model.gguf"
    server = tmp_path / "llama-server"
    model.touch()
    server.touch()
    return HyMtLlamaTranslator(
        model,
        server,
        terminology_enabled=terminology_enabled,
        critical_slot_protection_enabled=critical_slot_protection_enabled,
    )


def test_malay_order_identifier_and_payment_negation_are_protected(tmp_path):
    backend = make_backend(tmp_path, critical_slot_protection_enabled=True)

    translated, pivot = backend.translate(
        "Nombor pesanan saya ialah A tujuh tiga sembilan dua. "
        "Jangan buat bayaran dua kali.",
        "ms",
        "zh",
    )

    assert translated == "我的订单号是A7392。请勿重复付款。"
    assert pivot is None


def test_compact_malay_order_identifier_and_payment_negation_are_protected(tmp_path):
    backend = make_backend(tmp_path, critical_slot_protection_enabled=True)

    translated, pivot = backend.translate(
        "Nombor pesanan saya ialah A73992. Jangan buat bayaran dua kali.",
        "ms",
        "zh",
    )

    assert translated == "我的订单号是A73992。请勿重复付款。"
    assert pivot is None


def test_normalized_number_keeps_payment_negation_protection(tmp_path):
    backend = make_backend(tmp_path, critical_slot_protection_enabled=True)

    translated, _ = backend.translate(
        "Nombor pesanan saya ialah A7392. Jangan buat bayaran 2 kali.",
        "ms",
        "zh",
    )

    assert translated == "我的订单号是A7392。请勿重复付款。"


def test_chinese_order_identifier_keeps_digit_sequence(tmp_path):
    backend = make_backend(tmp_path, critical_slot_protection_enabled=True)

    translated, _ = backend.translate(
        "我的订单编号是 A 七三九二，请不要重复付款。",
        "zh",
        "ms",
    )

    assert translated == (
        "Nombor pesanan saya ialah A7392. Jangan buat bayaran dua kali."
    )


def test_prompt_uses_official_terminology_intervention_shape(tmp_path):
    backend = make_backend(tmp_path, terminology_enabled=True)

    prompt = backend._prompt(
        "Penerbangan lewat empat puluh lima minit.", "ms", "zh"
    )

    assert "lewat 翻译成 延误" in prompt
    assert "pengecaman 翻译成 识别" in prompt
    assert "只需要输出翻译后的结果" in prompt


def test_backend_avoids_an_occupied_local_server_port(tmp_path, monkeypatch):
    class FakeSocket:
        def __init__(self, occupied, port=None):
            self.occupied = occupied
            self.port = port

        def bind(self, _address):
            if self.occupied:
                raise OSError("occupied")

        def getsockname(self):
            return ("127.0.0.1", self.port)

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    backend = make_backend(tmp_path)
    backend.port = 18081
    backend._base_url = "http://127.0.0.1:18081"
    sockets = iter((FakeSocket(True), FakeSocket(False, 23456)))
    monkeypatch.setattr(
        "zhms_s2s.backends.hy_mt.socket.socket", lambda *_args: next(sockets)
    )

    backend._move_to_free_port_if_needed()

    assert backend.port == 23456
    assert backend._base_url == "http://127.0.0.1:23456"


def test_prompt_accepts_a_small_glossary_without_expected_sentence(tmp_path):
    backend = make_backend(tmp_path)

    prompt = backend._prompt(
        "Saya mahu char kuey teow.",
        "ms",
        "zh",
        (("char kuey teow", "炒粿条"),),
    )

    assert "char kuey teow 翻译成 炒粿条" in prompt
    assert "我要一份" not in prompt


def test_default_prompt_contains_no_global_examples_or_prior_test_content(tmp_path):
    backend = make_backend(tmp_path)

    prompt = backend._prompt(
        "Jika sambungan Internet terputus, sistem masih berjalan.", "ms", "zh"
    )

    assert "KL Sentral" not in prompt
    assert "Jangan buat bayaran dua kali" not in prompt
    assert "请勿重复付款" not in prompt
    assert prompt.endswith(
        "Jika sambungan Internet terputus, sistem masih berjalan."
    )


def test_request_payload_is_fresh_single_message_and_disables_cache(tmp_path):
    backend = make_backend(tmp_path)

    first = backend._request_payload("first")
    second = backend._request_payload("second")
    first["messages"][0]["content"] = "mutated"

    assert second["messages"] == [{"role": "user", "content": "second"}]
    assert second["cache_prompt"] is False
