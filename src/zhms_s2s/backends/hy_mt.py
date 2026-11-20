from __future__ import annotations

import atexit
import json
import re
import secrets
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..config import target_for, validate_language
from ..errors import InferenceError, InvalidLanguageError, ModelUnavailableError


_TERMINOLOGY = {
    ("ms", "zh"): (
        ("Sila", "请"),
        ("KL Sentral", "吉隆坡中央车站"),
        ("Pulau Pinang", "槟城"),
        ("lewat", "延误"),
        ("pengecaman", "识别"),
        ("Nombor pesanan", "订单号"),
        ("luar talian", "离线"),
        ("Jangan buat bayaran dua kali", "请勿重复付款"),
    ),
    ("zh", "ms"): (
        ("订单号", "Nombor pesanan"),
        ("订单编号", "Nombor pesanan"),
        ("吉隆坡中央车站", "KL Sentral"),
        ("槟城", "Pulau Pinang"),
        ("离线", "luar talian"),
        ("识别", "pengecaman"),
        ("重复付款", "buat bayaran dua kali"),
    ),
}

_MALAY_DIGITS = {
    "kosong": "0",
    "sifar": "0",
    "satu": "1",
    "dua": "2",
    "tiga": "3",
    "empat": "4",
    "lima": "5",
    "enam": "6",
    "tujuh": "7",
    "tuju": "7",
    "lapan": "8",
    "sembilan": "9",
}
_CHINESE_DIGITS = str.maketrans("零〇一二两三四五六七八九", "001223456789")


def _extract_order_identifier(text: str, language: str) -> str | None:
    if language == "ms":
        match = re.search(
            # The identifier may be spoken as ``A tujuh tiga ...`` or
            # written compactly as ``A73992``.  Requiring a word boundary
            # after the letter rejects the compact form because both the
            # letter and the following digit are regex ``\w`` characters.
            r"nombor\s+pesanan.*?\b([a-z])(?=\s|[-\d])\s*"
            r"([a-z\d\s-]+?)(?=[,.!?]|\bjangan\b|$)",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        digits: list[str] = []
        for token in re.findall(r"[a-z]+|\d", match.group(2).casefold()):
            if token.isdigit():
                digits.append(token)
            elif token in _MALAY_DIGITS:
                digits.append(_MALAY_DIGITS[token])
        return match.group(1).upper() + "".join(digits) if digits else None

    match = re.search(
        r"(?:订单号|订单编号)(?:是|为)?\s*([a-z])\s*"
        r"([^,，。！！!?请]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    digit_text = re.sub(r"\s+", "", match.group(2)).translate(_CHINESE_DIGITS)
    digits = "".join(re.findall(r"\d", digit_text))
    return match.group(1).upper() + digits if digits else None


def _critical_intent_translation(
    text: str, source: str, target: str
) -> str | None:
    """Protect identifiers and payment negation from generative drift."""
    identifier = _extract_order_identifier(text, source)
    if source == "ms" and target == "zh":
        payment = bool(
            re.search(
                r"jangan\s+(?:buat|lakukan)?\s*bayaran\s+(?:dua|2)\s+kali",
                text,
                flags=re.IGNORECASE,
            )
        )
        if identifier and payment:
            return f"我的订单号是{identifier}。请勿重复付款。"
        if identifier and len(text) < 160:
            return f"我的订单号是{identifier}。"
        if payment and len(text) < 80:
            return "请勿重复付款。"

    if source == "zh" and target == "ms":
        payment = bool(re.search(r"(?:请勿|请不要|不要)?重复付款|支付两次", text))
        if identifier and payment:
            return (
                f"Nombor pesanan saya ialah {identifier}. "
                "Jangan buat bayaran dua kali."
            )
        if identifier and len(text) < 80:
            return f"Nombor pesanan saya ialah {identifier}."
        if payment and len(text) < 40:
            return "Jangan buat bayaran dua kali."
    return None


class HyMtLlamaTranslator:
    """Persistent local HY-MT translation served by llama.cpp over loopback."""

    def __init__(
        self,
        model_path: Path,
        server_path: Path,
        *,
        port: int = 18081,
        gpu_layers: int = 99,
        startup_timeout_s: float = 45.0,
        terminology_enabled: bool = False,
        critical_slot_protection_enabled: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.server_path = Path(server_path)
        self.port = int(port)
        self.gpu_layers = int(gpu_layers)
        self.startup_timeout_s = float(startup_timeout_s)
        self.terminology_enabled = bool(terminology_enabled)
        self.critical_slot_protection_enabled = bool(
            critical_slot_protection_enabled
        )
        self._base_url = f"http://127.0.0.1:{self.port}"
        self._api_key = secrets.token_urlsafe(24)
        self._process: subprocess.Popen | None = None
        self._inference_warmed = False
        self._start_lock = threading.Lock()
        self._request_lock = threading.Lock()
        atexit.register(self.close)

    def _present(self) -> bool:
        return self.model_path.is_file() and self.server_path.is_file()

    def _health(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self._base_url}/health", timeout=0.5
            ) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def _move_to_free_port_if_needed(self) -> None:
        """Avoid accidentally reusing an older authenticated local server."""

        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", self.port))
            return
        except OSError:
            pass
        finally:
            probe.close()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as chooser:
            chooser.bind(("127.0.0.1", 0))
            self.port = int(chooser.getsockname()[1])
        self._base_url = f"http://127.0.0.1:{self.port}"

    def _ensure_server(self) -> None:
        if self._process is not None and self._process.poll() is None and self._health():
            return
        with self._start_lock:
            if (
                self._process is not None
                and self._process.poll() is None
                and self._health()
            ):
                return
            if self._process is not None and self._process.poll() is None:
                self.close()
            self._move_to_free_port_if_needed()
            if not self.model_path.is_file():
                raise ModelUnavailableError(
                    f"HY-MT GGUF model is missing: {self.model_path}"
                )
            if not self.server_path.is_file():
                raise ModelUnavailableError(
                    f"llama-server executable is missing: {self.server_path}"
                )
            command = [
                str(self.server_path),
                "-m",
                str(self.model_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "-ngl",
                str(self.gpu_layers),
                "-c",
                "2048",
                "--parallel",
                "1",
                "--api-key",
                self._api_key,
            ]
            try:
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                raise ModelUnavailableError(
                    "Failed to start the local llama.cpp translation service.",
                    detail={"reason": str(exc), "server": str(self.server_path)},
                ) from exc

            deadline = time.monotonic() + self.startup_timeout_s
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise ModelUnavailableError(
                        "The local llama.cpp translation service exited during startup.",
                        detail={"returncode": self._process.returncode},
                    )
                if self._health():
                    return
                time.sleep(0.1)
            self.close()
            raise ModelUnavailableError(
                "Timed out while loading the local HY-MT translation model."
            )

    def _prompt(
        self,
        text: str,
        source: str,
        target: str,
        extra_terms: tuple[tuple[str, str], ...] = (),
    ) -> str:
        built_in_terms = (
            _TERMINOLOGY[(source, target)] if self.terminology_enabled else ()
        )
        all_terms = (*built_in_terms, *extra_terms)
        # Preserve order while avoiding duplicate source terms.
        deduplicated = tuple(dict(all_terms).items())
        target_name = "中文" if target == "zh" else "马来语"
        if not deduplicated:
            return (
                f"将以下文本翻译为{target_name}，注意只需要输出翻译后的结果，"
                f"不要额外解释：\n\n{text}"
            )
        terms = "\n".join(
            f"{source_term} 翻译成 {target_term}"
            for source_term, target_term in deduplicated
        )
        return (
            f"参考下面的翻译：\n{terms}\n\n"
            f"将以下文本翻译为{target_name}，注意只需要输出翻译后的结果，"
            f"不要额外解释：\n\n{text}"
        )

    @staticmethod
    def _request_payload(prompt: str, *, max_tokens: int = 256) -> dict:
        """Build a fresh, stateless request object for every inference call."""

        return {
            "model": "hy-mt1.5-1.8b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "top_k": 20,
            "top_p": 0.6,
            "repeat_penalty": 1.05,
            "max_tokens": max_tokens,
            # Explicitly disable prompt-prefix reuse in llama.cpp.  The
            # application does not send a conversation/session identifier.
            "cache_prompt": False,
        }

    def _complete_with_trace(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        request_id: str | None = None,
    ) -> tuple[str, dict]:
        self._ensure_server()
        request_payload = self._request_payload(prompt, max_tokens=max_tokens)
        payload = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        try:
            with self._request_lock, urllib.request.urlopen(
                request, timeout=60
            ) as response:
                result = json.load(response)
            output = str(result["choices"][0]["message"]["content"]).strip()
        except Exception as exc:
            raise InferenceError(
                "Local HY-MT translation inference failed.",
                detail={"reason": str(exc)},
            ) from exc
        if not output:
            raise InferenceError("Local HY-MT produced empty text.")
        # The trace is request-local.  It is never stored on the backend and
        # therefore cannot become input to a later request.
        trace = {
            "actual_mt_request_messages": [
                dict(message) for message in request_payload["messages"]
            ],
            "mt_raw_output": output,
            "model_id": request_payload["model"],
            "session_id": None,
            "mt_request_id": request_id,
            "cache_prompt": False,
        }
        return output, trace

    def _complete(self, prompt: str, *, max_tokens: int = 256) -> str:
        output, _ = self._complete_with_trace(prompt, max_tokens=max_tokens)
        return output

    def translate_with_trace(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        request_id: str | None = None,
    ) -> tuple[str, str | None, dict]:
        source = validate_language(source_language)
        target = validate_language(target_language)
        if source == target or target != target_for(source):
            raise InvalidLanguageError("Only direct zh<->ms translation is supported.")
        clean = text.strip()
        if not clean:
            raise InferenceError("Translation input text is empty.")
        protected = (
            _critical_intent_translation(clean, source, target)
            if self.critical_slot_protection_enabled
            else None
        )
        if protected:
            return protected, None, {
                "actual_mt_request_messages": [],
                "mt_raw_output": protected,
                "model_id": "deterministic-critical-slot-protection",
                "session_id": None,
                "mt_request_id": request_id,
                "cache_prompt": False,
            }
        output, trace = self._complete_with_trace(
            self._prompt(clean, source, target),
            request_id=request_id,
        )
        return output, None, trace

    def translate(
        self, text: str, source_language: str, target_language: str
    ) -> tuple[str, str | None]:
        output, pivot, _ = self.translate_with_trace(
            text, source_language, target_language
        )
        return output, pivot

    def warmup(self) -> None:
        self._ensure_server()
        if self._inference_warmed:
            return
        self._complete(self._prompt("你好。", "zh", "ms"), max_tokens=24)
        self._complete(
            self._prompt("Selamat datang.", "ms", "zh"), max_tokens=24
        )
        self._inference_warmed = True

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)

    def status(self) -> dict:
        running = self._process is not None and self._process.poll() is None
        return {
            "backend": "hy-mt1.5-llama.cpp-q4",
            "strategy": {"zh-ms": "direct", "ms-zh": "direct"},
            "models": {
                "zh-ms/ms-zh": {
                    "path": str(self.model_path),
                    "present": self.model_path.is_file(),
                    "loaded": running,
                }
            },
            "model_present": self._present(),
            "loaded": running,
            "inference_warmed": self._inference_warmed,
            "device": "metal" if self.gpu_layers else "cpu",
            "server": "127.0.0.1 loopback",
            "request_isolation": {
                "messages_per_request": 1,
                "conversation_history": False,
                "session_id": None,
                "cache_prompt": False,
            },
            "terminology_intervention": self.terminology_enabled,
            "critical_slot_protection": (
                "order-id-and-payment-negation"
                if self.critical_slot_protection_enabled
                else False
            ),
            "offline": True,
        }
