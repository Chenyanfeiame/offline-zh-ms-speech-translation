from __future__ import annotations

import re
from dataclasses import dataclass


_DIGIT_WORDS = {
    "kosong": "0",
    "sifar": "0",
    "satu": "1",
    "dua": "2",
    "tiga": "3",
    "empat": "4",
    "lima": "5",
    "enam": "6",
    "tujuh": "7",
    "lapan": "8",
    "sembilan": "9",
}
_SEQUENCE_DIGIT_WORDS = {**_DIGIT_WORDS, "tuju": "7"}
_UNITS = {word: int(value) for word, value in _DIGIT_WORDS.items()}
_SPECIAL = {"sepuluh": 10, "sebelas": 11}
_NUMBER_WORDS = tuple(
    sorted(
        {
            *_DIGIT_WORDS,
            *_SPECIAL,
            "belas",
            "puluh",
            "ratus",
            "seratus",
            "ribu",
            "seribu",
            "juta",
            "dan",
        },
        key=len,
        reverse=True,
    )
)
_SIMPLE_DIGIT = "(?:" + "|".join(map(re.escape, _SEQUENCE_DIGIT_WORDS)) + ")"
_NUMBER_WORD = "(?:" + "|".join(map(re.escape, _NUMBER_WORDS)) + ")"
_ALPHANUMERIC_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?P<prefix>[A-Z])"
    rf"(?P<body>(?:[\s-]+{_SIMPLE_DIGIT}){{2,}})(?![A-Za-z])"
)
_CODE_CUE_RE = re.compile(
    rf"\b(?P<cue>nombor|nomor|kod|telefon|tiket|siri|rujukan|akaun)"
    rf"(?P<separator>\s+)(?P<body>{_SIMPLE_DIGIT}(?:[\s-]+{_SIMPLE_DIGIT})+)\b",
    flags=re.IGNORECASE,
)
_CARDINAL_RE = re.compile(
    rf"(?<![A-Za-z-]){_NUMBER_WORD}(?:\s+{_NUMBER_WORD})*(?![A-Za-z-])",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class NumberNormalization:
    original: str
    normalized: str
    kind: str
    rule: str

    def as_dict(self) -> dict[str, str]:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "kind": self.kind,
            "rule": self.rule,
        }


def _parse_under_100(tokens: list[str]) -> int | None:
    if not tokens:
        return 0
    if len(tokens) == 1:
        if tokens[0] in _UNITS:
            return _UNITS[tokens[0]]
        return _SPECIAL.get(tokens[0])
    if len(tokens) == 2:
        first = _UNITS.get(tokens[0])
        if first is not None and first > 0 and tokens[1] == "belas":
            return 10 + first
        if first is not None and first > 0 and tokens[1] == "puluh":
            return first * 10
        return None
    if len(tokens) == 3:
        tens = _UNITS.get(tokens[0])
        unit = _UNITS.get(tokens[2])
        if (
            tens is not None
            and tens > 0
            and tokens[1] == "puluh"
            and unit is not None
        ):
            return tens * 10 + unit
    return None


def _parse_under_1000(tokens: list[str]) -> int | None:
    if not tokens:
        return 0
    if tokens[0] == "seratus":
        remainder = _parse_under_100(tokens[1:])
        return None if remainder is None else 100 + remainder
    if len(tokens) >= 2 and tokens[1] == "ratus":
        hundreds = _UNITS.get(tokens[0])
        if hundreds is None or hundreds == 0:
            return None
        remainder = _parse_under_100(tokens[2:])
        return None if remainder is None else hundreds * 100 + remainder
    return _parse_under_100(tokens)


def _parse_scaled(tokens: list[str], scale_word: str, scale: int) -> int | None:
    if scale_word not in tokens:
        return _parse_under_1000(tokens)
    if tokens.count(scale_word) != 1:
        return None
    index = tokens.index(scale_word)
    left = _parse_under_1000(tokens[:index])
    right = _parse_under_1000(tokens[index + 1 :])
    if left is None or right is None or left == 0:
        return None
    return left * scale + right


def parse_malay_number(words: str | list[str]) -> int | None:
    """Parse one grammatical Malay cardinal number.

    Adjacent digit words such as ``tujuh tiga`` are intentionally rejected;
    they only become a sequence when an identifier rule provides that context.
    """

    tokens = (
        re.findall(r"[A-Za-z]+", words.casefold())
        if isinstance(words, str)
        else [token.casefold() for token in words]
    )
    if not tokens:
        return None
    if "dan" in tokens:
        tokens = [token for token in tokens if token != "dan"]
        if not tokens:
            return None
    if tokens[0] == "seribu":
        remainder = _parse_under_1000(tokens[1:])
        return None if remainder is None else 1000 + remainder
    if "juta" in tokens:
        if tokens.count("juta") != 1:
            return None
        index = tokens.index("juta")
        left = _parse_scaled(tokens[:index], "ribu", 1000)
        right = _parse_scaled(tokens[index + 1 :], "ribu", 1000)
        if left is None or right is None or left == 0:
            return None
        return left * 1_000_000 + right
    return _parse_scaled(tokens, "ribu", 1000)


def normalize_malay_numbers(
    text: str,
) -> tuple[str, list[NumberNormalization]]:
    """Normalize Malay cardinals and explicit digit sequences conservatively."""

    changes: list[NumberNormalization] = []

    def replace_alpha_sequence(match: re.Match[str]) -> str:
        digits = "".join(
            _SEQUENCE_DIGIT_WORDS[word.casefold()]
            for word in re.findall(_SIMPLE_DIGIT, match.group("body"), re.I)
        )
        normalized = match.group("prefix") + digits
        changes.append(
            NumberNormalization(
                match.group(0), normalized, "identifier", "letter-plus-spoken-digits"
            )
        )
        return normalized

    normalized_text = _ALPHANUMERIC_RE.sub(replace_alpha_sequence, text)

    def replace_cued_sequence(match: re.Match[str]) -> str:
        digits = "".join(
            _SEQUENCE_DIGIT_WORDS[word.casefold()]
            for word in re.findall(_SIMPLE_DIGIT, match.group("body"), re.I)
        )
        replacement = match.group("cue") + match.group("separator") + digits
        changes.append(
            NumberNormalization(
                match.group("body"), digits, "digit_sequence", "identifier-cue"
            )
        )
        return replacement

    normalized_text = _CODE_CUE_RE.sub(replace_cued_sequence, normalized_text)

    def replace_cardinal(match: re.Match[str]) -> str:
        original = match.group(0)
        value = parse_malay_number(original)
        if value is None:
            return original
        replacement = str(value)
        changes.append(
            NumberNormalization(original, replacement, "cardinal", "malay-cardinal")
        )
        return replacement

    normalized_text = _CARDINAL_RE.sub(replace_cardinal, normalized_text)
    return normalized_text, changes


def canonical_number_tokens(text: str) -> list[str]:
    """Return normalized numeric/identifier slots for evaluation."""

    normalized, _ = normalize_malay_numbers(text)
    raw_tokens = re.findall(
        r"(?<![A-Za-z0-9])[A-Z]\d{2,}|\d+(?:[.,]\d+)?", normalized
    )
    canonical: list[str] = []
    for token in raw_tokens:
        if re.fullmatch(r"[A-Z]\d{2,}", token):
            canonical.append(token)
            continue
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", token):
            canonical.append(token.replace(",", ""))
            continue
        if "," in token:
            token = token.replace(",", ".")
        if "." in token:
            integer, fraction = token.split(".", 1)
            fraction = fraction.rstrip("0")
            token = integer if not fraction else f"{integer}.{fraction}"
        canonical.append(token)
    return canonical
