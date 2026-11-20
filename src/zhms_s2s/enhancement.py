from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from .numbers import normalize_malay_numbers


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_NORMALIZATION_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9'-]*|\d+(?:\.\d+)*|[\u3400-\u4dbf\u4e00-\u9fff]+"
)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_CONTEXT_CUES = {
    "location": {
        "di", "ke", "dari", "tiba", "stesen", "bandar", "lapangan", "jalan",
        "到", "去", "从", "在", "抵达", "位于", "车站", "机场", "城市", "飞往",
    },
    "food": {
        "makan", "minum", "pesan", "restoran", "hidangan", "menu",
        "吃", "喝", "点", "餐厅", "菜", "菜单",
    },
    "person_organization_brand": {
        "syarikat", "organisasi", "jenama", "produk", "encik", "puan",
        "公司", "组织", "品牌", "产品", "先生", "女士",
    },
    "transport_business": {
        "nombor", "kod", "penerbangan", "pesanan", "tiket", "tempahan", "laluan",
        "编号", "航班", "订单", "票", "预订", "线路",
    },
    "date_time_numeric": {
        "pada", "pukul", "tarikh", "jam", "minit", "ringgit", "kilometer",
        "日期", "时间", "点", "分钟", "令吉", "公里", "公斤",
    },
    "technical_term": {
        "sistem", "perisian", "rangkaian", "model", "internet", "komputer",
        "系统", "软件", "网络", "模型", "互联网", "电脑", "设备",
    },
}
_CRITICAL_PROMPT_TERMS = ("jangan", "tidak", "bukan")


def _normalized(value: str) -> str:
    return " ".join(_NORMALIZATION_RE.findall(value.casefold()))


@dataclass(frozen=True, slots=True)
class EntityEntry:
    canonical_name: str
    aliases: tuple[str, ...]
    language: str
    entity_type: str
    pronunciation_variants: tuple[str, ...]
    priority: int

    @property
    def variants(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (self.canonical_name, *self.aliases, *self.pronunciation_variants)
            )
        )


@dataclass(frozen=True, slots=True)
class EnhancementResult:
    raw_text: str
    enhanced_text: str
    entities: tuple[dict, ...]
    corrections: tuple[dict, ...]
    number_normalizations: tuple[dict, ...]


def load_entity_lexicon(path: Path) -> tuple[EntityEntry, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("entity_types")
    if not isinstance(groups, dict):
        raise ValueError("Entity lexicon must contain an 'entity_types' object.")
    entries: list[EntityEntry] = []
    for entity_type, values in groups.items():
        if not isinstance(values, list):
            raise ValueError(f"Entity group '{entity_type}' must be a list.")
        for value in values:
            declared_type = str(value.get("entity_type", entity_type))
            if declared_type != entity_type:
                raise ValueError(
                    f"Entity '{value.get('canonical_name')}' has inconsistent type."
                )
            canonical = str(value["canonical_name"]).strip()
            if not canonical:
                raise ValueError("Entity canonical_name cannot be empty.")
            entries.append(
                EntityEntry(
                    canonical_name=canonical,
                    aliases=tuple(str(item).strip() for item in value.get("aliases", [])),
                    language=str(value.get("language", "multi")),
                    entity_type=declared_type,
                    pronunciation_variants=tuple(
                        str(item).strip()
                        for item in value.get("pronunciation_variants", [])
                    ),
                    priority=max(1, min(5, int(value.get("priority", 3)))),
                )
            )
    return tuple(entries)


def parse_user_terms(value: str | None) -> tuple[str, ...]:
    if not value or not value.strip():
        return ()
    terms = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if len(terms) > 16:
        raise ValueError("At most 16 user context terms are allowed.")
    for term in terms:
        if len(term) > 64 or any(character in term for character in "\r\n\t"):
            raise ValueError("Each user context term must be a single line of at most 64 characters.")
    return terms


class EntityNumberEnhancer:
    """Offline, scene-independent entity matching and number normalization."""

    def __init__(self, lexicon_path: Path) -> None:
        self.lexicon_path = Path(lexicon_path)
        self.entries = load_entity_lexicon(self.lexicon_path)

    @staticmethod
    def _language_matches(entry: EntityEntry, language: str) -> bool:
        return entry.language in {"multi", language} or language in entry.language.split("+")

    def build_prompt(
        self,
        language: str,
        *,
        user_terms: tuple[str, ...] = (),
        max_terms: int = 24,
        max_characters: int = 280,
    ) -> tuple[str, tuple[str, ...]]:
        candidates = [
            entry.canonical_name
            for entry in sorted(
                self.entries,
                key=lambda entry: (-entry.priority, entry.entity_type, entry.canonical_name),
            )
            if entry.priority >= 5 and self._language_matches(entry, language)
        ]
        candidates = [*user_terms, *_CRITICAL_PROMPT_TERMS, *candidates]
        selected: list[str] = []
        character_count = 0
        for term in candidates:
            if term.casefold() in {value.casefold() for value in selected}:
                continue
            extra = len(term) + (1 if selected else 0)
            if len(selected) >= max_terms or character_count + extra > max_characters:
                break
            selected.append(term)
            character_count += extra
        return " ".join(selected), tuple(selected)

    @staticmethod
    def _variant_pattern(variant: str) -> re.Pattern[str]:
        if _CJK_RE.search(variant):
            compact = re.sub(r"\s+", "", variant)
            joined = r"\s*".join(re.escape(character) for character in compact)
            return re.compile(joined, re.I)
        words = _WORD_RE.findall(variant)
        if not words:
            return re.compile(re.escape(variant), re.I)
        joined = r"[\s-]+".join(re.escape(word) for word in words)
        return re.compile(rf"(?<![A-Za-z0-9]){joined}(?![A-Za-z0-9])", re.I)

    def _exact_entities(
        self, text: str, language: str
    ) -> tuple[str, list[dict], list[dict]]:
        entities: list[dict] = []
        corrections: list[dict] = []
        occupied: list[tuple[int, int]] = []
        replacements: list[tuple[int, int, str]] = []
        candidates = [
            (entry, variant)
            for entry in self.entries
            if self._language_matches(entry, language)
            for variant in entry.variants
        ]
        candidates.sort(key=lambda item: len(item[1]), reverse=True)
        for entry, variant in candidates:
            for match in self._variant_pattern(variant).finditer(text):
                span = match.span()
                if any(span[0] < end and span[1] > start for start, end in occupied):
                    continue
                occupied.append(span)
                original = match.group(0)
                canonical = entry.canonical_name
                replacement = (
                    canonical[:1].upper() + canonical[1:]
                    if original[:1].isupper() and canonical[:1].islower()
                    else canonical
                )
                changed = original != replacement
                entities.append(
                    {
                        "canonical_name": canonical,
                        "entity_type": entry.entity_type,
                        "matched_text": original,
                        "match_type": "exact" if _normalized(original) == _normalized(canonical) else "alias",
                        "confidence": 1.0,
                    }
                )
                if changed:
                    replacements.append((span[0], span[1], replacement))
                    corrections.append(
                        {
                            "original": original,
                            "replacement": replacement,
                            "correction_type": "entity_alias",
                            "confidence": 1.0,
                            "reason": f"lexicon:{entry.entity_type}",
                        }
                    )
        for start, end, replacement in sorted(replacements, reverse=True):
            text = text[:start] + replacement + text[end:]
        return text, entities, corrections

    def _fuzzy_entities(
        self, text: str, language: str
    ) -> tuple[str, list[dict], list[dict]]:
        word_matches = list(_WORD_RE.finditer(text))
        if not word_matches:
            return text, [], []
        sentence_words = {_normalized(match.group(0)) for match in word_matches}
        candidates: list[tuple[float, float, int, int, EntityEntry, str]] = []
        for entry in self.entries:
            if not self._language_matches(entry, language) or entry.priority < 4:
                continue
            canonical_words = _normalized(entry.canonical_name).split()
            if not canonical_words:
                continue
            has_context = bool(sentence_words & _CONTEXT_CUES.get(entry.entity_type, set()))
            if len(canonical_words) == 1 and not has_context:
                continue
            variants = tuple(_normalized(value) for value in entry.variants)
            for count in {max(1, len(value.split())) for value in variants}:
                for index in range(0, len(word_matches) - count + 1):
                    start = word_matches[index].start()
                    end = word_matches[index + count - 1].end()
                    original = text[start:end]
                    normalized_original = _normalized(original)
                    if normalized_original in variants or len(normalized_original) < 5:
                        continue
                    scores = sorted(
                        (SequenceMatcher(None, normalized_original, variant).ratio() for variant in variants),
                        reverse=True,
                    )
                    score = scores[0]
                    runner_up = scores[1] if len(scores) > 1 else 0.0
                    threshold = 0.90 if entry.priority == 5 else 0.94
                    if score >= threshold:
                        candidates.append((score, score - runner_up, start, end, entry, original))
        selected: list[tuple[float, int, int, EntityEntry, str]] = []
        occupied: list[tuple[int, int]] = []
        for score, margin, start, end, entry, original in sorted(candidates, reverse=True):
            if margin < 0.015 or any(start < b and end > a for a, b in occupied):
                continue
            occupied.append((start, end))
            selected.append((score, start, end, entry, original))
        entities: list[dict] = []
        corrections: list[dict] = []
        for score, start, end, entry, original in sorted(selected, key=lambda item: item[1], reverse=True):
            text = text[:start] + entry.canonical_name + text[end:]
            entity = {
                "canonical_name": entry.canonical_name,
                "entity_type": entry.entity_type,
                "matched_text": original,
                "match_type": "fuzzy",
                "confidence": round(score, 4),
            }
            entities.append(entity)
            corrections.append(
                {
                    "original": original,
                    "replacement": entry.canonical_name,
                    "correction_type": "entity_fuzzy",
                    "confidence": round(score, 4),
                    "reason": f"high-confidence-edit-similarity:{entry.entity_type}",
                }
            )
        return text, entities, corrections

    def enhance(
        self,
        text: str,
        language: str,
        *,
        enable_entities: bool = True,
        enable_numbers: bool = True,
    ) -> EnhancementResult:
        enhanced = text
        entities: list[dict] = []
        corrections: list[dict] = []
        number_changes: list[dict] = []
        if enable_entities:
            enhanced, exact, exact_corrections = self._exact_entities(enhanced, language)
            entities.extend(exact)
            corrections.extend(exact_corrections)
            enhanced, fuzzy, fuzzy_corrections = self._fuzzy_entities(
                enhanced, language
            )
            entities.extend(fuzzy)
            corrections.extend(fuzzy_corrections)
        if enable_numbers and language == "ms":
            enhanced, normalizations = normalize_malay_numbers(enhanced)
            number_changes = [item.as_dict() for item in normalizations]
            corrections.extend(
                {
                    "original": item.original,
                    "replacement": item.normalized,
                    "correction_type": "number_normalization",
                    "confidence": 1.0,
                    "reason": item.rule,
                }
                for item in normalizations
            )
        deduplicated_entities = tuple(
            dict.fromkeys(
                (
                    item["canonical_name"],
                    item["entity_type"],
                    item["matched_text"],
                    item["match_type"],
                    item["confidence"],
                )
                for item in entities
            )
        )
        entity_payload = tuple(
            {
                "canonical_name": canonical,
                "entity_type": entity_type,
                "matched_text": matched,
                "match_type": match_type,
                "confidence": confidence,
            }
            for canonical, entity_type, matched, match_type, confidence in deduplicated_entities
        )
        return EnhancementResult(
            raw_text=text,
            enhanced_text=enhanced,
            entities=entity_payload,
            corrections=tuple(corrections),
            number_normalizations=tuple(number_changes),
        )
