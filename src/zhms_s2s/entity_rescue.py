from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .chinese_phonetics import ChinesePhoneticEncoder
from .enhancement import EntityEntry, EntityNumberEnhancer, _CONTEXT_CUES, _normalized


_WORD_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9'-]*|[\u3400-\u4dbf\u4e00-\u9fff]"
)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_CRITICAL_WORDS = {"jangan", "tidak", "bukan"}
_CRITICAL_CHINESE_TERMS = {"不", "没", "没有", "别", "不要", "禁止", "未"}


@dataclass(frozen=True, slots=True)
class EntityRescueCandidate:
    canonical_name: str
    entity_type: str
    matched_text: str
    score: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "matched_text": self.matched_text,
            "score": self.score,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class EntityRescueResult:
    raw_text: str
    selected_text: str
    contextual_text: str | None
    prompt: str
    candidates: tuple[EntityRescueCandidate, ...]
    accepted_contextual_pass: bool
    decision_reason: str
    entities: tuple[dict, ...]
    corrections: tuple[dict, ...]
    safety_checks: dict


class SelectiveEntityRescuer:
    """Retrieve a tiny entity context and conservatively select a second pass.

    The retriever is scene-independent. It never sends the complete lexicon to
    ASR and never accepts a contextual transcript merely because it contains a
    prompted word.
    """

    def __init__(
        self,
        enhancer: EntityNumberEnhancer,
        *,
        max_candidates: int = 3,
        max_prompt_characters: int = 96,
        fuzzy_threshold: float = 0.72,
        phonetic_encoder: ChinesePhoneticEncoder | None = None,
        phonetic_margin: float = 0.04,
        phonetic_rerank_threshold: float = 0.98,
        max_token_change_ratio: float = 0.55,
        max_character_change_ratio: float = 0.25,
    ) -> None:
        self.enhancer = enhancer
        self.max_candidates = max(1, int(max_candidates))
        self.max_prompt_characters = max(16, int(max_prompt_characters))
        self.fuzzy_threshold = max(0.0, min(1.0, float(fuzzy_threshold)))
        self.phonetic_encoder = phonetic_encoder
        self.phonetic_margin = max(0.0, min(1.0, float(phonetic_margin)))
        self.phonetic_rerank_threshold = max(
            0.0, min(1.0, float(phonetic_rerank_threshold))
        )
        self.max_token_change_ratio = max(
            0.0, min(1.0, float(max_token_change_ratio))
        )
        self.max_character_change_ratio = max(
            0.0, min(1.0, float(max_character_change_ratio))
        )

    @staticmethod
    def _language_matches(entry: EntityEntry, language: str) -> bool:
        return EntityNumberEnhancer._language_matches(entry, language)

    @staticmethod
    def _contains_variant(text: str, entry: EntityEntry) -> tuple[str, str] | None:
        canonical = EntityNumberEnhancer._variant_pattern(entry.canonical_name)
        match = canonical.search(text)
        if match:
            return match.group(0), "canonical"
        for variant in (*entry.aliases, *entry.pronunciation_variants):
            match = EntityNumberEnhancer._variant_pattern(variant).search(text)
            if match:
                return match.group(0), "known-variant"
        return None

    @staticmethod
    def _word_windows(text: str, minimum: int, maximum: int):
        matches = list(_WORD_RE.finditer(text))
        for count in range(max(1, minimum), max(1, maximum) + 1):
            for index in range(0, len(matches) - count + 1):
                start = matches[index].start()
                end = matches[index + count - 1].end()
                yield text[start:end]

    def _fuzzy_candidate(
        self,
        text: str,
        entry: EntityEntry,
        *,
        normalized_text_words: set[str],
        window_cache: dict[int, tuple[str, ...]],
    ) -> EntityRescueCandidate | None:
        normalized_text = _normalized(text)
        context = bool(
            normalized_text_words & _CONTEXT_CUES.get(entry.entity_type, set())
        ) or any(
            _CJK_RE.search(cue) and cue in normalized_text
            for cue in _CONTEXT_CUES.get(entry.entity_type, set())
        )
        variants = tuple(
            value
            for value in (_normalized(item) for item in entry.variants)
            if value
        )
        if not variants:
            return None
        best: tuple[float, str] | None = None
        for variant in variants:
            word_count = len(_WORD_RE.findall(variant))
            if word_count == 0:
                continue
            windows = window_cache.get(word_count)
            if windows is None:
                windows = tuple(self._word_windows(text, word_count, word_count))
                window_cache[word_count] = windows
            single_word = word_count == 1
            compact = variant.replace(" ", "")
            cjk_variant = bool(_CJK_RE.search(variant))
            acronym_like = not cjk_variant and single_word and len(compact) <= 4
            threshold = max(
                self.fuzzy_threshold,
                1.0
                if cjk_variant and word_count <= 3
                else (
                    0.82
                    if cjk_variant
                    else (
                        0.92
                        if acronym_like
                        else (0.86 if single_word else (0.75 if not context else 0.72))
                    )
                ),
            )
            for window in windows:
                normalized_window = _normalized(window)
                if len(normalized_window) < 4:
                    continue
                score = SequenceMatcher(
                    None, normalized_window, variant
                ).ratio()
                if score < threshold or (
                    (single_word or cjk_variant) and not context
                ):
                    continue
                if best is None or score > best[0]:
                    best = (score, window)
        if best is None:
            return None
        return EntityRescueCandidate(
            canonical_name=entry.canonical_name,
            entity_type=entry.entity_type,
            matched_text=best[1],
            score=round(best[0], 4),
            reason="phonetic-orthographic-near-match",
        )

    @staticmethod
    def _chinese_context_types(text: str) -> set[str]:
        normalized_text = _normalized(text)
        return {
            entity_type
            for entity_type, cues in _CONTEXT_CUES.items()
            if any(
                _CJK_RE.search(cue) and cue in normalized_text
                for cue in cues
            )
        }

    @staticmethod
    def _cjk_windows(text: str, size: int) -> tuple[str, ...]:
        windows: list[str] = []
        for run in _CJK_RUN_RE.findall(text):
            windows.extend(
                run[index : index + size]
                for index in range(0, len(run) - size + 1)
            )
        return tuple(dict.fromkeys(windows))

    def _phonetic_candidates(
        self, text: str, language: str
    ) -> tuple[EntityRescueCandidate, ...]:
        """Retrieve Chinese entities by pronunciation, never by hard replacement."""

        if (
            language != "zh"
            or self.phonetic_encoder is None
            or not self.phonetic_encoder.available
        ):
            return ()
        context_types = self._chinese_context_types(text)
        if not context_types:
            return ()

        comparisons: list[tuple[EntityEntry, str, str, int]] = []
        values_to_encode: list[str] = []
        for entry in self.enhancer.entries:
            if (
                entry.priority < 4
                or entry.entity_type not in context_types
                or not self._language_matches(entry, language)
                or self._contains_variant(text, entry)
            ):
                continue
            for variant in entry.variants:
                cjk_variant = "".join(_CJK_RE.findall(variant))
                if len(cjk_variant) < 2:
                    continue
                for window in self._cjk_windows(text, len(cjk_variant)):
                    comparisons.append(
                        (entry, cjk_variant, window, len(cjk_variant))
                    )
                    values_to_encode.extend((cjk_variant, window))
        if not comparisons:
            return ()

        phonetic = self.phonetic_encoder.encode_many(values_to_encode)
        by_window: dict[str, dict[str, tuple[float, EntityEntry]]] = {}
        for entry, variant, window, length in comparisons:
            variant_key = phonetic.get(variant, "")
            window_key = phonetic.get(window, "")
            if not variant_key or not window_key or variant_key == window_key == "":
                continue
            score = SequenceMatcher(None, window_key, variant_key).ratio()
            threshold = 0.90 if length <= 3 else 0.86
            if score < threshold:
                continue
            canonical_key = entry.canonical_name.casefold()
            previous = by_window.setdefault(window, {}).get(canonical_key)
            if previous is None or score > previous[0]:
                by_window[window][canonical_key] = (score, entry)

        candidates: list[EntityRescueCandidate] = []
        for window, matches in by_window.items():
            ranked = sorted(
                matches.values(),
                key=lambda item: (-item[0], -item[1].priority, item[1].canonical_name),
            )
            if not ranked:
                continue
            best_score, best_entry = ranked[0]
            runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
            if best_score - runner_up < self.phonetic_margin:
                continue
            candidates.append(
                EntityRescueCandidate(
                    canonical_name=best_entry.canonical_name,
                    entity_type=best_entry.entity_type,
                    matched_text=window,
                    score=round(best_score, 4),
                    reason="offline-pinyin-near-match",
                )
            )
        return tuple(candidates)

    def retrieve(
        self,
        text: str,
        language: str,
        *,
        user_terms: tuple[str, ...] = (),
    ) -> tuple[EntityRescueCandidate, ...]:
        candidates: list[EntityRescueCandidate] = []
        normalized_text_words = set(_normalized(text).split())
        window_cache: dict[int, tuple[str, ...]] = {}
        for term in user_terms:
            if term.casefold() not in text.casefold():
                candidates.append(
                    EntityRescueCandidate(
                        canonical_name=term,
                        entity_type="user_term",
                        matched_text="",
                        score=1.0,
                        reason="request-specific-vocabulary",
                    )
                )
        for entry in self.enhancer.entries:
            if entry.priority < 4 or not self._language_matches(entry, language):
                continue
            exact = self._contains_variant(text, entry)
            if exact:
                matched, match_type = exact
                if match_type == "known-variant":
                    candidates.append(
                        EntityRescueCandidate(
                            canonical_name=entry.canonical_name,
                            entity_type=entry.entity_type,
                            matched_text=matched,
                            score=1.0,
                            reason="known-alias-or-pronunciation",
                        )
                    )
                continue
            fuzzy = self._fuzzy_candidate(
                text,
                entry,
                normalized_text_words=normalized_text_words,
                window_cache=window_cache,
            )
            if fuzzy:
                candidates.append(fuzzy)
        candidates.extend(self._phonetic_candidates(text, language))

        best_by_name: dict[str, EntityRescueCandidate] = {}
        for candidate in candidates:
            key = candidate.canonical_name.casefold()
            previous = best_by_name.get(key)
            if (
                previous is not None
                and previous.reason == "request-specific-vocabulary"
                and not previous.matched_text
                and candidate.matched_text
            ):
                best_by_name[key] = EntityRescueCandidate(
                    canonical_name=previous.canonical_name,
                    entity_type=previous.entity_type,
                    matched_text=candidate.matched_text,
                    score=previous.score,
                    reason=previous.reason,
                )
            elif previous is None or candidate.score > previous.score:
                best_by_name[key] = candidate
        ranked = sorted(
            best_by_name.values(),
            key=lambda item: (
                item.reason != "request-specific-vocabulary",
                -item.score,
                -len(_normalized(item.matched_text)),
                item.canonical_name.casefold(),
            ),
        )
        non_overlapping: list[EntityRescueCandidate] = []
        for candidate in ranked:
            matched = _normalized(candidate.matched_text)
            if matched and any(
                f" {matched} " in f" {_normalized(kept.matched_text)} "
                for kept in non_overlapping
            ):
                continue
            non_overlapping.append(candidate)
        return tuple(non_overlapping[: self.max_candidates])

    def build_prompt(
        self, candidates: tuple[EntityRescueCandidate, ...]
    ) -> str:
        selected: list[str] = []
        size = 0
        for candidate in candidates:
            term = candidate.canonical_name.strip()
            extra = len(term) + (1 if selected else 0)
            if size + extra > self.max_prompt_characters:
                break
            selected.append(term)
            size += extra
        return " ".join(selected)

    def warmup(self) -> None:
        if self.phonetic_encoder is None or not self.phonetic_encoder.available:
            return
        values = ["你好"]
        values.extend(
            variant
            for entry in self.enhancer.entries
            if entry.priority >= 4 and self._language_matches(entry, "zh")
            for variant in entry.variants
            if _CJK_RE.search(variant)
        )
        self.phonetic_encoder.encode_many(values)

    def needs_contextual_pass(
        self,
        text: str,
        language: str,
        candidates: tuple[EntityRescueCandidate, ...],
    ) -> bool:
        """Skip Qwen pass two when deterministic alias matching already suffices."""

        if not candidates:
            return False
        enhanced = self.enhancer.enhance(
            text, language, enable_entities=True, enable_numbers=False
        ).enhanced_text
        return len(self._candidate_coverage(enhanced, candidates)) < len(candidates)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return _WORD_RE.findall(text.casefold())

    @staticmethod
    def _critical_words(text: str) -> set[str]:
        critical = set(SelectiveEntityRescuer._tokens(text)) & _CRITICAL_WORDS
        critical.update(
            term for term in _CRITICAL_CHINESE_TERMS if term in text
        )
        return critical

    @staticmethod
    def _digits(text: str) -> tuple[str, ...]:
        return tuple(re.findall(r"\d+", text))

    @staticmethod
    def _candidate_coverage(
        text: str, candidates: tuple[EntityRescueCandidate, ...]
    ) -> set[str]:
        normalized = _normalized(text)
        return {
            candidate.canonical_name.casefold()
            for candidate in candidates
            if _normalized(candidate.canonical_name) in normalized
        }

    def _unexpected_entities(
        self,
        raw_text: str,
        contextual_text: str,
        candidates: tuple[EntityRescueCandidate, ...],
        language: str,
    ) -> list[str]:
        allowed = {item.canonical_name.casefold() for item in candidates}
        unexpected: list[str] = []
        for entry in self.enhancer.entries:
            if not self._language_matches(entry, language):
                continue
            if entry.canonical_name.casefold() in allowed:
                continue
            if self._contains_variant(raw_text, entry):
                continue
            if EntityNumberEnhancer._variant_pattern(entry.canonical_name).search(
                contextual_text
            ):
                unexpected.append(entry.canonical_name)
        return sorted(set(unexpected))

    def _apply_phonetic_rerank(
        self,
        text: str,
        language: str,
        candidates: tuple[EntityRescueCandidate, ...],
    ) -> tuple[str, tuple[dict, ...], tuple[dict, ...], tuple[str, ...]]:
        """Resolve only high-confidence phonetic candidates left by Qwen.

        This is a lexicon-wide phonetic reranker, not an error-alias table. It
        is deliberately restricted to candidates retrieved before decoding,
        revalidated against the delivered text and accepted only above a
        strict pronunciation threshold.
        """

        allowed = {
            candidate.canonical_name.casefold()
            for candidate in candidates
            if candidate.reason == "offline-pinyin-near-match"
        }
        if language != "zh" or not allowed:
            return text, (), (), ()
        reranked = [
            candidate
            for candidate in self._phonetic_candidates(text, language)
            if candidate.canonical_name.casefold() in allowed
            and candidate.score >= self.phonetic_rerank_threshold
        ]
        replacements: list[tuple[int, int, EntityRescueCandidate]] = []
        occupied: list[tuple[int, int]] = []
        for candidate in sorted(
            reranked,
            key=lambda item: (-item.score, -len(item.matched_text)),
        ):
            start = text.find(candidate.matched_text)
            if start < 0:
                continue
            end = start + len(candidate.matched_text)
            if any(start < previous_end and end > previous_start for previous_start, previous_end in occupied):
                continue
            occupied.append((start, end))
            replacements.append((start, end, candidate))

        entities: list[dict] = []
        corrections: list[dict] = []
        applied: list[str] = []
        for start, end, candidate in sorted(replacements, reverse=True):
            original = text[start:end]
            text = text[:start] + candidate.canonical_name + text[end:]
            entities.append(
                {
                    "canonical_name": candidate.canonical_name,
                    "entity_type": candidate.entity_type,
                    "matched_text": original,
                    "match_type": "phonetic_rerank",
                    "confidence": candidate.score,
                }
            )
            corrections.append(
                {
                    "original": original,
                    "replacement": candidate.canonical_name,
                    "correction_type": "entity_phonetic_rerank",
                    "confidence": candidate.score,
                    "reason": "offline-g2pw-semantic-context-rerank",
                }
            )
            applied.append(candidate.canonical_name)
        return text, tuple(entities), tuple(corrections), tuple(reversed(applied))

    def select(
        self,
        raw_text: str,
        language: str,
        *,
        candidates: tuple[EntityRescueCandidate, ...],
        contextual_text: str | None,
    ) -> EntityRescueResult:
        raw_enhancement = self.enhancer.enhance(
            raw_text, language, enable_entities=True, enable_numbers=False
        )
        selected_text = raw_enhancement.enhanced_text
        entities = raw_enhancement.entities
        corrections = raw_enhancement.corrections
        prompt = self.build_prompt(candidates)
        if not candidates or not contextual_text:
            return EntityRescueResult(
                raw_text=raw_text,
                selected_text=selected_text,
                contextual_text=contextual_text,
                prompt=prompt,
                candidates=candidates,
                accepted_contextual_pass=False,
                decision_reason=(
                    "no-suspected-entity"
                    if not candidates
                    else "contextual-pass-unavailable"
                ),
                entities=entities,
                corrections=corrections,
                safety_checks={"passed": True},
            )

        contextual_enhancement = self.enhancer.enhance(
            contextual_text, language, enable_entities=True, enable_numbers=False
        )
        first_tokens = self._tokens(raw_text)
        second_tokens = self._tokens(contextual_text)
        token_similarity = SequenceMatcher(None, first_tokens, second_tokens).ratio()
        character_similarity = SequenceMatcher(
            None, _normalized(raw_text), _normalized(contextual_text)
        ).ratio()
        length_ratio = len(second_tokens) / max(1, len(first_tokens))
        raw_coverage = self._candidate_coverage(selected_text, candidates)
        contextual_coverage = self._candidate_coverage(
            contextual_enhancement.enhanced_text, candidates
        )
        missing_critical = sorted(
            self._critical_words(raw_text) - self._critical_words(contextual_text)
        )
        digits_preserved = self._digits(raw_text) == self._digits(contextual_text)
        unexpected_entities = self._unexpected_entities(
            raw_text, contextual_enhancement.enhanced_text, candidates, language
        )
        checks = {
            "candidate_coverage_before": sorted(raw_coverage),
            "candidate_coverage_after": sorted(contextual_coverage),
            "candidate_coverage_improved": len(contextual_coverage) > len(raw_coverage),
            "missing_critical_words": missing_critical,
            "digits_preserved": digits_preserved,
            "unexpected_entities": unexpected_entities,
            "token_change_ratio": round(1.0 - token_similarity, 4),
            "character_change_ratio": round(1.0 - character_similarity, 4),
            "length_ratio": round(length_ratio, 4),
        }
        accepted = all(
            (
                checks["candidate_coverage_improved"],
                not missing_critical,
                digits_preserved,
                not unexpected_entities,
                0.65 <= length_ratio <= 1.35,
                (1.0 - token_similarity) <= self.max_token_change_ratio,
                (1.0 - character_similarity)
                <= self.max_character_change_ratio,
            )
        )
        checks["passed"] = accepted
        if accepted:
            rescued_names = contextual_coverage - raw_coverage
            rescue_corrections = tuple(
                {
                    "original": candidate.matched_text or "<unrecognized>",
                    "replacement": candidate.canonical_name,
                    "correction_type": "entity_contextual_rescue",
                    "confidence": candidate.score,
                    "reason": "candidate-retrieval-plus-safety-gates",
                }
                for candidate in candidates
                if candidate.canonical_name.casefold() in rescued_names
            )
            delivered_text, phonetic_entities, phonetic_corrections, applied = (
                self._apply_phonetic_rerank(
                    contextual_enhancement.enhanced_text,
                    language,
                    candidates,
                )
            )
            checks["phonetic_rerank_applied"] = list(applied)
            checks["candidate_coverage_delivered"] = sorted(
                self._candidate_coverage(delivered_text, candidates)
            )
            return EntityRescueResult(
                raw_text=raw_text,
                selected_text=delivered_text,
                contextual_text=contextual_text,
                prompt=prompt,
                candidates=candidates,
                accepted_contextual_pass=True,
                decision_reason="accepted-by-entity-and-semantic-safety-gates",
                entities=(*contextual_enhancement.entities, *phonetic_entities),
                corrections=(
                    *rescue_corrections,
                    *contextual_enhancement.corrections,
                    *phonetic_corrections,
                ),
                safety_checks=checks,
            )
        reranked_text, phonetic_entities, phonetic_corrections, applied = (
            self._apply_phonetic_rerank(selected_text, language, candidates)
        )
        checks["phonetic_rerank_applied"] = list(applied)
        checks["candidate_coverage_delivered"] = sorted(
            self._candidate_coverage(reranked_text, candidates)
        )
        return EntityRescueResult(
            raw_text=raw_text,
            selected_text=reranked_text,
            contextual_text=contextual_text,
            prompt=prompt,
            candidates=candidates,
            accepted_contextual_pass=False,
            decision_reason=(
                "phonetic-rerank-after-context-rejection"
                if applied
                else "rejected-by-semantic-safety-gates"
            ),
            entities=(*entities, *phonetic_entities),
            corrections=(*corrections, *phonetic_corrections),
            safety_checks=checks,
        )
