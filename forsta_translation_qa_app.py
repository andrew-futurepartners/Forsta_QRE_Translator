import os
import io
import json
import re
import time
import html
import asyncio
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import zipfile

from enum import Enum

from dotenv import load_dotenv

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    AsyncOpenAI = None
    OpenAI = None

# Load environment variables from a local .env file if present
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# ==========================
# LLM Client (Async)
# ==========================

_async_client: Optional["AsyncOpenAI"] = None


def get_async_client() -> "AsyncOpenAI":
    global _async_client
    if _async_client is not None:
        return _async_client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    _async_client = AsyncOpenAI(api_key=api_key)
    return _async_client


def reset_async_client() -> None:
    """Null the cached async client so the next call to get_async_client()
    creates a fresh instance bound to the current event loop.  Call once
    before creating the run loop and once after closing it."""
    global _async_client
    _async_client = None


# Per-run translation dedup cache.  Keyed on every output-affecting argument so
# identical English in identical context triggers exactly ONE model call; peers
# await the lock and reuse the cached result.  Reset at the start of each run
# via reset_translation_cache() so cross-run state never leaks.
_TRANSLATION_CACHE: Dict[tuple, dict] = {}
_TRANSLATION_CACHE_LOCKS: Dict[tuple, "asyncio.Lock"] = {}


def reset_translation_cache() -> None:
    """Clear the per-run translation dedup cache.  Must be called once at the
    start of each run (before the event loop is entered) so stale results from
    a previous run never propagate."""
    global _TRANSLATION_CACHE, _TRANSLATION_CACHE_LOCKS
    _TRANSLATION_CACHE = {}
    _TRANSLATION_CACHE_LOCKS = {}


# ==========================
# Configuration & Constants
# ==========================

DEFAULT_GLOBAL_CONTEXT = (
    "You are translating a market research questionnaire. "
    "The audience is respondents in the target locale. The text is survey content "
    "(questions, answer options, scale labels, messages)."
)

# Single source of truth for the model used everywhere (translation, style
# inference, consistency, back-translation, judge). Latest GA model mid-2026.
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.5")

# LLM judge (critique-and-revise) — opt-in quality loop.
# ENABLE_JUDGE_DEFAULT: UI checkbox default (False = off). Override via the checkbox.
# JUDGE_SCORE_THRESHOLD: judge scores below this (1-5 scale) trigger one retry.
#   Default 3 means scores 1-2 → retry; scores 3-5 → ship as-is.
ENABLE_JUDGE_DEFAULT: bool = False
JUDGE_SCORE_THRESHOLD: int = int(os.getenv("JUDGE_SCORE_THRESHOLD", "3"))

# Token cap passed to every create() call.  Prevents finish_reason='length'
# silent truncations.  GPT-5/o-series uses max_completion_tokens; older models
# use max_tokens.  Adjust the param name if the endpoint changes.
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "4096"))

# Best-effort determinism (Step 23). gpt-5-mini honors `seed` on a best-effort
# basis; setting it makes runs more reproducible and lets prompt changes be
# A/B-tested. Set TRANSLATION_SEED="" to disable (param passed as None == unset).
_seed_raw = os.getenv("TRANSLATION_SEED", "1234")
TRANSLATION_SEED: Optional[int] = int(_seed_raw) if _seed_raw.strip() else None

# Collected per run; surfaced in the UI so output provenance is reviewable.
_SYSTEM_FINGERPRINTS: set = set()


def reset_system_fingerprints() -> None:
    """Clear collected system_fingerprints. Call once at run start."""
    _SYSTEM_FINGERPRINTS.clear()


def _record_fingerprint(response) -> None:
    """Record a response.system_fingerprint if present (best-effort, never raises)."""
    try:
        fp = getattr(response, "system_fingerprint", None)
        if fp:
            _SYSTEM_FINGERPRINTS.add(fp)
    except Exception:
        pass


# Simple language mapping (expand as needed)
LANGUAGE_NAME_TO_CODE = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "portuguese": "pt",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "chinese": "zh",
    "dutch": "nl",
    "korean": "ko",
    "hindi": "hi",
}

# For UI labels
LANGUAGE_LABEL_TO_CODE = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Portuguese": "pt",
    "German": "de",
    "Italian": "it",
    "Japanese": "ja",
    "Chinese": "zh",
    "Dutch": "nl",
    "Korean": "ko",
    "Hindi": "hi",
}

# Sentinel shown in the language dropdown when filename detection fails.
# While selected, the Run button is blocked so the operator must choose explicitly.
SELECT_LANGUAGE_SENTINEL = "-- select language --"

# Locale mappings (focus on Spanish + English as requested)
SPANISH_LOCALE_NAME_TO_BCP47 = {
    "argentina": "es-AR",
    "mexico": "es-MX",
    "colombia": "es-CO",
    "chile": "es-CL",
    "peru": "es-PE",
    "spain": "es-ES",
    "es": "es",
}

ENGLISH_LOCALE_NAME_TO_BCP47 = {
    "uk": "en-GB",
    "united_kingdom": "en-GB",
    "gb": "en-GB",
    "britain": "en-GB",
    "usa": "en-US",
    "us": "en-US",
    "united_states": "en-US",
    "canada": "en-CA",
    "ca": "en-CA",
    "australia": "en-AU",
    "aus": "en-AU",
    "au": "en-AU",
    "en": "en",
}

# Shorthand tokens that imply English + a specific locale when they appear
# as the language segment in the filename (e.g., 260306_uk.xls).
_ENGLISH_LOCALE_SHORTHANDS = {
    "uk": "en-GB",
    "gb": "en-GB",
    "aus": "en-AU",
    "au": "en-AU",
    "ca": "en-CA",
}

# Common locale options per language (for UI dropdowns)
# (label shown in UI, BCP-47 code passed to the model)
LOCALE_OPTIONS = {
    "en": [
        ("Generic English (no specific country)", "en"),
        ("United States (en-US)", "en-US"),
        ("United Kingdom (en-GB)", "en-GB"),
        ("Canada (en-CA)", "en-CA"),
        ("Australia (en-AU)", "en-AU"),
    ],
    "es": [
        ("Generic Spanish (no specific country)", "es"),
        ("Mexico (es-MX)", "es-MX"),
        ("United States / US Hispanic (es-US)", "es-US"),
        ("Spain (es-ES)", "es-ES"),
        ("Argentina (es-AR)", "es-AR"),
        ("Colombia (es-CO)", "es-CO"),
        ("Chile (es-CL)", "es-CL"),
        ("Peru (es-PE)", "es-PE"),
    ],
    "fr": [
        ("Generic French (no specific country)", "fr"),
        ("France (fr-FR)", "fr-FR"),
        ("Canada (fr-CA)", "fr-CA"),
        ("Belgium (fr-BE)", "fr-BE"),
        ("Switzerland (fr-CH)", "fr-CH"),
    ],
    "pt": [
        ("Generic Portuguese (no specific country)", "pt"),
        ("Brazil (pt-BR)", "pt-BR"),
        ("Portugal (pt-PT)", "pt-PT"),
    ],
    "de": [
        ("Generic German (no specific country)", "de"),
        ("Germany (de-DE)", "de-DE"),
        ("Austria (de-AT)", "de-AT"),
        ("Switzerland (de-CH)", "de-CH"),
    ],
    "it": [
        ("Generic Italian (no specific country)", "it"),
        ("Italy (it-IT)", "it-IT"),
    ],
    "ja": [
        ("Japan (ja-JP)", "ja-JP"),
    ],
    "zh": [
        ("Generic Chinese (unspecified script)", "zh"),
        ("Chinese (Simplified, zh-CN)", "zh-CN"),
        ("Chinese (Traditional, zh-TW)", "zh-TW"),
        ("Chinese (Hong Kong, zh-HK)", "zh-HK"),
    ],
    "nl": [
        ("Generic Dutch (no specific country)", "nl"),
        ("Netherlands (nl-NL)", "nl-NL"),
        ("Belgium / Flemish (nl-BE)", "nl-BE"),
    ],
    "ko": [
        ("South Korea (ko-KR)", "ko-KR"),
    ],
    "hi": [
        ("Generic Hindi (no specific country)", "hi"),
        ("India (hi-IN)", "hi-IN"),
    ],
}

# ==========================
# Dialect Adaptation Dictionaries
# ==========================

# Per-locale vocabulary whitelist: only these US->local substitutions are
# permitted.  Terms NOT listed here must be left unchanged by the LLM.
# Phrase-level entries (multi-word keys) are checked before single-word entries.
DIALECT_VOCABULARY: Dict[str, Dict[str, str]] = {
    "en-GB": {
        # Phrase-level (checked first)
        "in line": "in a queue",
        # Single-word
        "vacation": "holiday",
        "vacations": "holidays",
        "airplane": "aeroplane",
        "airplanes": "aeroplanes",
        "transportation": "transport",
    },
    "en-AU": {
        "in line": "in a queue",
        "vacation": "holiday",
        "vacations": "holidays",
        "airplane": "aeroplane",
        "airplanes": "aeroplanes",
        "transportation": "transport",
    },
    "en-CA": {
        # CA uses "vacation" (NOT "holiday"), does NOT convert -ize/-ise.
        # Phrase-level mappings for adjective-form geographic terms:
        "state park": "provincial park",
        "state parks": "provincial parks",
        "state fair": "provincial fair",
        "state fairs": "provincial fairs",
        "state/county": "provincial/county",
        "national, state,": "national, provincial,",
    },
}

# Deterministic spelling corrections applied as a post-processing regex pass.
# Keyed by locale code; each value is a list of (US_word, local_word) tuples.
# Applied with word-boundary matching and case-preservation.
_DIALECT_SPELLING_UKAU: List[Tuple[str, str]] = [
    # -eling / -elling
    ("traveling", "travelling"),
    ("traveled", "travelled"),
    ("traveler", "traveller"),
    ("travelers", "travellers"),
    # -er / -re
    ("center", "centre"),
    ("centers", "centres"),
    # -or / -our
    ("neighborhood", "neighbourhood"),
    ("neighborhoods", "neighbourhoods"),
    ("color", "colour"),
    ("colors", "colours"),
    ("favor", "favour"),
    ("favors", "favours"),
    ("favorite", "favourite"),
    ("favorites", "favourites"),
    ("honor", "honour"),
    ("honors", "honours"),
    ("behavior", "behaviour"),
    ("behaviors", "behaviours"),
    ("humor", "humour"),
    ("labor", "labour"),
    # -ce / -se
    ("practiced", "practised"),
    ("practicing", "practising"),
    ("defense", "defence"),
    ("offense", "offence"),
    ("license", "licence"),
    # -ize / -ise  (UK/AU only — NOT CA)
    ("organize", "organise"),
    ("organizes", "organises"),
    ("organized", "organised"),
    ("organizing", "organising"),
    ("recognize", "recognise"),
    ("recognizes", "recognises"),
    ("recognized", "recognised"),
    ("recognizing", "recognising"),
    ("localize", "localise"),
    ("localizes", "localises"),
    ("localized", "localised"),
    ("localizing", "localising"),
    ("customize", "customise"),
    ("customized", "customised"),
    ("prioritize", "prioritise"),
    ("prioritized", "prioritised"),
    ("specialize", "specialise"),
    ("specialized", "specialised"),
    ("maximize", "maximise"),
    ("minimize", "minimise"),
    # -ization / -isation
    ("organization", "organisation"),
    ("organizations", "organisations"),
    ("localization", "localisation"),
    ("customization", "customisation"),
    ("specialization", "specialisation"),
    # misc
    ("gray", "grey"),
    ("analog", "analogue"),
    ("canceled", "cancelled"),
    ("canceling", "cancelling"),
]

# CA shares some UK/AU spelling but NOT -ize/-ise, NOT practiced, NOT color/favour
_DIALECT_SPELLING_CA: List[Tuple[str, str]] = [
    ("traveling", "travelling"),
    ("traveled", "travelled"),
    ("traveler", "traveller"),
    ("travelers", "travellers"),
    ("center", "centre"),
    ("centers", "centres"),
    ("neighborhood", "neighbourhood"),
    ("neighborhoods", "neighbourhoods"),
    ("canceled", "cancelled"),
    ("canceling", "cancelling"),
]

DIALECT_SPELLING: Dict[str, List[Tuple[str, str]]] = {
    "en-GB": _DIALECT_SPELLING_UKAU,
    "en-AU": _DIALECT_SPELLING_UKAU,
    "en-CA": _DIALECT_SPELLING_CA,
}

# Step 21: placeholder entries for recognized non-EN same-language dialect pairs.
# Vocabulary/spelling tables can be populated as needed; empty = conservative
# pass-through (no word substitutions, only LLM-driven adaptation).
for _dl in ["es-MX", "es-AR", "es-CO", "pt-BR", "pt-AO", "zh-TW", "zh-HK", "fr-CA", "fr-BE"]:
    DIALECT_VOCABULARY.setdefault(_dl, {})
    DIALECT_SPELLING.setdefault(_dl, [])
del _dl


def _case_preserving_replace(match: re.Match, replacement: str) -> str:
    """Replace with the same capitalization pattern as the matched text."""
    word = match.group(0)
    if word.isupper():
        return replacement.upper()
    if word[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def apply_dialect_spelling_corrections(context: "SurveyFileContext") -> int:
    """
    Deterministic post-processing pass: apply regex-based spelling corrections
    for the target dialect.  Catches any conversions the LLM missed.
    Skips content inside HTML tags and placeholder tokens.

    Returns the number of rows modified.
    """
    locale = context.locale_code
    corrections = DIALECT_SPELLING.get(locale, [])
    if not corrections:
        return 0

    # Pre-compile patterns (case-insensitive, word-boundary)
    compiled = [
        (re.compile(r'\b' + re.escape(us) + r'\b', re.IGNORECASE), local)
        for us, local in corrections
    ]

    fixed = 0
    for row in context.rows:
        if row.batch_translated:
            continue
        trl = row.new_translation or ""
        if not trl.strip():
            continue

        original = trl

        # Split on HTML tags / placeholders so we only touch visible text
        parts = re.split(r'(<[^>]+>|\{[^}]+\}|\[[^\]]+\])', trl)
        for i, part in enumerate(parts):
            if part.startswith(('<', '{', '[')):
                continue
            for pattern, replacement in compiled:
                part = pattern.sub(
                    lambda m, r=replacement: _case_preserving_replace(m, r),
                    part,
                )
            parts[i] = part

        new_trl = "".join(parts)
        if new_trl != original:
            row.new_translation = new_trl
            fixed += 1

    return fixed


# ── Step 18: v7 deterministic post-processors ────────────────────────────────

# ZH: "True"/"False" as standalone English words should become 是/否 in Chinese.
_ZH_BOOL_MAP: Dict[str, str] = {"true": "是", "false": "否"}


def _apply_zh_true_false(context: "SurveyFileContext") -> int:
    """Phase 4: replace English True/False with 是/否 in Chinese translations."""
    lc = context.language_code.lower()
    if not lc.startswith("zh"):
        return 0
    fixed = 0
    for row in context.rows:
        trl = (row.new_translation or "").strip()
        if trl.lower() in _ZH_BOOL_MAP:
            row.new_translation = _ZH_BOOL_MAP[trl.lower()]
            fixed += 1
    return fixed


# JA: standalone 4-digit years should carry the 年 suffix.
# Use (?<!\d) / (?!\d) instead of \b because Japanese characters are also \w
# in Unicode, so \b fails at digit/kana boundaries.
_JA_BARE_YEAR_RE = re.compile(r'(?<!\d)((?:19|20)\d{2})(?!\d)(?!\u5e74)')


def _apply_ja_year_suffix(context: "SurveyFileContext") -> int:
    """Phase 4: append 年 to bare 4-digit years in Japanese translations."""
    lc = context.language_code.lower()
    if not lc.startswith("ja"):
        return 0
    fixed = 0
    for row in context.rows:
        trl = row.new_translation or ""
        new_trl = _JA_BARE_YEAR_RE.sub(r'\g<1>年', trl)
        if new_trl != trl:
            row.new_translation = new_trl
            fixed += 1
    return fixed


# FR: fix common numeric-formatting issues (space before %, en-dash for ranges).
_FR_PCT_RE = re.compile(r'(\d)%')           # "5%" → "5 %"
_FR_RANGE_RE = re.compile(r'(\d)\s*-\s*(\d)')   # "10-20" → "10–20" (en-dash)


def _apply_fr_number_format(context: "SurveyFileContext") -> int:
    """Phase 4: apply French numeric-formatting conventions to translations."""
    lc = context.locale_code.lower()
    if not (lc.startswith("fr") or context.language_code.lower().startswith("fr")):
        return 0
    fixed = 0
    for row in context.rows:
        trl = row.new_translation or ""
        new_trl = _FR_PCT_RE.sub(r'\1\u00a0%', trl)          # non-breaking space
        new_trl = _FR_RANGE_RE.sub(r'\1\u2013\2', new_trl)   # en-dash
        if new_trl != trl:
            row.new_translation = new_trl
            fixed += 1
    return fixed


# ALL-CAPS emphasis: flag (advisory only) when English has an all-caps qualifier
# but the translation does not preserve any form of emphasis.
_CAPS_QUALIFIER_RE = re.compile(r'\b([A-Z]{2,})\b')
_PLACEHOLDER_RE_CAPS = re.compile(r'\{[^}]+\}|\[[^\]]+\]|\$\w+')


def _flag_emphasis_caps(context: "SurveyFileContext") -> int:
    """Phase 5 (flag-only): annotate qa_status when English has ALL-CAPS emphasis
    words (e.g. NOT, ONLY) that appear absent from the translation."""
    flagged = 0
    for row in context.rows:
        eng = row.english_text or ""
        trl = row.new_translation or ""
        if not eng or not trl:
            continue
        # Find ALL-CAPS words that are not placeholders
        eng_clean = _PLACEHOLDER_RE_CAPS.sub('', eng)
        caps_words = _CAPS_QUALIFIER_RE.findall(eng_clean)
        # Ignore single-letter caps and common abbreviations
        caps_words = [w for w in caps_words if len(w) > 1]
        if not caps_words:
            continue
        # If the translation contains no uppercase-emphasis at all, flag it.
        trl_clean = _PLACEHOLDER_RE_CAPS.sub('', trl)
        if not _CAPS_QUALIFIER_RE.search(trl_clean):
            note = f" | Emphasis check: English uses ALL-CAPS ({', '.join(caps_words)}) — verify emphasis is conveyed in translation."
            row.qa_status = (row.qa_status or "") + note
            flagged += 1
    return flagged


# Step 20: gender marker leak detector for scale-label rows.
_GENDER_MARKER_RE = re.compile(r'\(e\)|\(a\)|\(in\)|\*in\b', re.IGNORECASE)


def _flag_gender_marker_in_scale(context: "SurveyFileContext") -> int:
    """Phase 5: flag gender markers that leaked into SCALE_LABEL rows.
    SegmentType is resolved at call time (defined later in this module)."""
    flagged = 0
    for row in context.rows:
        if row.segment_type != SegmentType.SCALE_LABEL:
            continue
        trl = row.new_translation or ""
        if _GENDER_MARKER_RE.search(trl):
            row.qa_status = (row.qa_status or "") + " | Gender marker detected in scale label — verify."
            flagged += 1
    return flagged


def build_domain_prompt_fragment(global_context: str) -> str:
    """
    Build a domain-context string for injection into system prompts.
    Falls back to a generic description if global_context is empty.
    """
    if global_context and global_context.strip():
        return global_context.strip()
    return (
        "You are translating a market research questionnaire. "
        "The audience is survey respondents in the target locale."
    )


_GENDERED_LANGUAGE_CONFIG = {
    "fr": {
        "marker": "(e)",
        "positive_examples": "intéressé(e), satisfait(e), employé(e), disposé(e), retraité(e), étudiant(e)",
        "negative_examples": (
            "probable, excellent, élevé, civil, médical, suffisant, "
            "incapable, admirable, difficile, possible, raisonnable"
        ),
        "extra_rule": (
            "CRITICAL: Do NOT add (e) to past participles used with the auxiliary 'avoir' "
            "(e.g., 'avez mentionné', 'avez visité', 'avez indiqué'). "
            "Past participle agreement with avoir only occurs when the direct object precedes the verb, "
            "which is uncommon in survey questions. When in doubt, do NOT add (e) after avoir constructions."
        ),
    },
    "es": {
        "marker": "(a)",
        "positive_examples": "interesado(a), satisfecho(a), empleado(a), jubilado(a)",
        "negative_examples": (
            "probable, excelente, alto, civil, médico, "
            "incapaz, admirable, difícil, posible, razonable, importante"
        ),
        "extra_rule": (
            "Do NOT add (a) to adjectives ending in -ble, -nte, or -e that are already "
            "gender-invariable in Spanish (e.g., 'responsable', 'importante', 'independiente')."
        ),
    },
    "pt": {
        "marker": "(a)",
        "positive_examples": "interessado(a), satisfeito(a), empregado(a), aposentado(a)",
        "negative_examples": (
            "provável, excelente, elevado, civil, médico, "
            "incapaz, admirável, difícil, possível, razoável, importante"
        ),
        "extra_rule": (
            "Do NOT add (a) to adjectives ending in -vel, -nte, or -e that are already "
            "gender-invariable in Portuguese (e.g., 'responsável', 'importante', 'independente')."
        ),
    },
    "it": {
        "marker": "(a)",
        "positive_examples": "interessato(a), soddisfatto(a), impiegato(a), pensionato(a)",
        "negative_examples": (
            "probabile, eccellente, elevato, civile, medico, "
            "incapace, ammirabile, difficile, possibile, ragionevole, importante"
        ),
        "extra_rule": (
            "Do NOT add (a) to adjectives ending in -bile, -nte, or -e that are already "
            "gender-invariable in Italian (e.g., 'responsabile', 'importante', 'indipendente')."
        ),
    },
    "de": {
        "marker": "(in) or *in",
        "positive_examples": "Angestellte(r), Rentner(in), Student(in)",
        "negative_examples": "wahrscheinlich, ausgezeichnet, hoch, zivil",
    },
    "hi": {
        "marker": "agreement",
        "positive_examples": "satisfied (santusht), interested (ichchhuk), employed (karyarat)",
        "negative_examples": "probable, excellent, important",
        "extra_rule": (
            "Hindi uses grammatical gender agreement between adjectives/verbs and the noun they describe. "
            "Do NOT add parenthetical markers like (a) or (e). "
            "Use the correct masculine or feminine form as required by grammatical context. "
            "When the respondent's gender is unknown, default to the masculine form."
        ),
        "agreement_mode": True,
    },
}


# ── Step 19: LANGUAGE_CAPABILITIES registry ───────────────────────────────────
# Single source of truth for per-language feature flags.
# Keys are BCP-47 base codes (before the first "-").
# Fields:
#   first_person  - bool: reliable first-person detection/restyle supported
#   gender_marker - "marker" | "agreement" | "none"
#   has_articles  - bool: language uses definite articles on nouns (list-context gate)
#   cjk           - bool: CJK script (brand-name reading aids; skip capitalization adjust)
#   has_case      - bool: morphological case (informational, for future use)
#   dialect_codes - List[str]: recognized same-language dialect locales (for Step 21)
LANGUAGE_CAPABILITIES: Dict[str, Dict[str, object]] = {
    "en": {"first_person": True,  "gender_marker": "none",      "has_articles": True,  "cjk": False, "has_case": False, "dialect_codes": ["en-GB", "en-AU", "en-CA", "en-NZ", "en-IE", "en-ZA"]},
    "es": {"first_person": True,  "gender_marker": "marker",    "has_articles": True,  "cjk": False, "has_case": False, "dialect_codes": ["es-MX", "es-AR", "es-CO", "es-CL", "es-PE", "es-419", "es-US"]},
    "fr": {"first_person": True,  "gender_marker": "marker",    "has_articles": True,  "cjk": False, "has_case": False, "dialect_codes": ["fr-CA", "fr-BE", "fr-CH"]},
    "pt": {"first_person": True,  "gender_marker": "marker",    "has_articles": True,  "cjk": False, "has_case": False, "dialect_codes": ["pt-BR", "pt-AO", "pt-MZ"]},
    "de": {"first_person": True,  "gender_marker": "marker",    "has_articles": True,  "cjk": False, "has_case": True,  "dialect_codes": ["de-AT", "de-CH"]},
    "it": {"first_person": True,  "gender_marker": "marker",    "has_articles": True,  "cjk": False, "has_case": False, "dialect_codes": []},
    "nl": {"first_person": False, "gender_marker": "none",      "has_articles": True,  "cjk": False, "has_case": False, "dialect_codes": ["nl-BE"]},
    "ja": {"first_person": False, "gender_marker": "none",      "has_articles": False, "cjk": True,  "has_case": True,  "dialect_codes": []},
    "ko": {"first_person": False, "gender_marker": "none",      "has_articles": False, "cjk": True,  "has_case": True,  "dialect_codes": []},
    "zh": {"first_person": False, "gender_marker": "none",      "has_articles": False, "cjk": True,  "has_case": False, "dialect_codes": ["zh-TW", "zh-HK", "zh-SG"]},
    "hi": {"first_person": False, "gender_marker": "agreement", "has_articles": False, "cjk": False, "has_case": True,  "dialect_codes": []},
    "ar": {"first_person": False, "gender_marker": "agreement", "has_articles": True,  "cjk": False, "has_case": True,  "dialect_codes": []},
    "ru": {"first_person": False, "gender_marker": "agreement", "has_articles": False, "cjk": False, "has_case": True,  "dialect_codes": []},
    "pl": {"first_person": False, "gender_marker": "agreement", "has_articles": False, "cjk": False, "has_case": True,  "dialect_codes": []},
    "tr": {"first_person": False, "gender_marker": "none",      "has_articles": False, "cjk": False, "has_case": True,  "dialect_codes": []},
}

_LC_FALLBACK: Dict[str, object] = {
    "first_person": False, "gender_marker": "none", "has_articles": False,
    "cjk": False, "has_case": False, "dialect_codes": [],
}


def _get_lang_cap(language_code: str) -> Dict[str, object]:
    """Return capabilities for a base language code, or a safe fallback for unknowns."""
    lc = (language_code or "").lower().split("-")[0]
    return LANGUAGE_CAPABILITIES.get(lc, _LC_FALLBACK)


def build_brand_name_instruction(language_code: str) -> str:
    """
    For CJK languages, instruct the LLM to keep English brand names
    and add a local-language reading aid in parentheses.
    """
    if not _get_lang_cap(language_code).get("cjk"):
        return ""

    return (
        "BRAND NAMES: For well-known technology brand names and product names "
        "(e.g., ChatGPT, Copilot, Claude, Gemini), keep the original English name "
        "and add the local-language reading in parentheses. "
        "Example for Japanese: 'Copilot(コパイロット)', 'Claude(クロード)'. "
        "Do NOT fully transliterate brand names into the target script."
    )


def is_article_suppressed_list(
    answer_option_count: int,
    answer_option_avg_len: float,
    language_code: str,
) -> bool:
    """True when a block is a long, short-item list whose items the prompt asks to
    render bare/article-free with consistent casing.  Shared by the list-context
    prompt builder and adjust_capitalization_for_label so they never contradict."""
    if answer_option_count < 10:
        return False
    if answer_option_avg_len > 40:
        return False
    return bool(_get_lang_cap(language_code).get("has_articles"))


def build_list_context_instruction(
    answer_option_count: int,
    answer_option_avg_len: float,
    language_code: str,
) -> str:
    """
    For list-like blocks (many short options, e.g. country lists),
    instruct the LLM to omit definite articles for consistency.
    """
    if not is_article_suppressed_list(answer_option_count, answer_option_avg_len, language_code):
        return ""

    return (
        "LIST FORMAT: These items appear in a dropdown list or selection set. "
        "Do NOT add definite articles (le, la, les, o, a, os, as, el, los, il, der, die, das, de, het) "
        "before names. Use the bare name only for consistency within the list. "
        "Example: 'Bahamas' not 'Les Bahamas', 'Gambie' not 'La Gambie'."
    )


def build_gender_inclusive_instruction(
    language_code: str,
    enabled: bool,
    segment_type: Optional[str] = None,
) -> str:
    """
    Build the gender-inclusive prompt instruction appropriate for the target language.
    Only applies to languages with grammatical gender that affects adjective forms.
    segment_type: pass "SCALE_LABEL" to suppress the instruction for scale-label rows.
    """
    # Step 20: never emit gender-inclusive instruction for SCALE_LABEL segments.
    if segment_type == "SCALE_LABEL":
        return ""

    if not enabled:
        return (
            "Do NOT add parenthetical or slash-based gender variants "
            "(e.g., '(a)', '(e)', '/a') unless the English source explicitly includes them."
        )

    lc = (language_code or "").lower()

    config = None
    for prefix, cfg in _GENDERED_LANGUAGE_CONFIG.items():
        if lc.startswith(prefix):
            config = cfg
            break

    if not config:
        return (
            "GENDER-INCLUSIVE FORMS: Use locale-appropriate gender-inclusive language where "
            "standard practice in the target language. Do not add slash-based or parenthetical "
            "gender variants (e.g., '(e)', '/a') unless the English source explicitly includes them "
            "or there is a well-established convention for this language."
        )

    # Step 20: agreement-mode languages (e.g. Hindi) use grammatical agreement, not markers.
    if config.get("agreement_mode"):
        extra = config.get("extra_rule", "")
        return (
            f"GENDER-INCLUSIVE FORMS: This survey requires grammatically correct gender agreement. "
            f"Use the appropriate form of adjectives and verbs that refer to the respondent. "
            f"{extra} "
            f"Do NOT add slash-based or parenthetical markers (e.g., '(a)', '(e)', '/a')."
        )

    extra = config.get("extra_rule", "")
    return (
        f"GENDER-INCLUSIVE FORMS: This survey requires gender-inclusive language. "
        f"Use the {config['marker']} notation ONLY on adjectives and past participles "
        f"that DESCRIBE THE SURVEY RESPONDENT (the person answering). "
        f"Correct examples: {config['positive_examples']}. "
        f"Do NOT apply to adjectives that modify objects, concepts, costs, probabilities, "
        f"or quality ratings. Incorrect examples: {config['negative_examples']}. "
        f"{extra} "
        f"RULE: If the word describes a thing rather than the respondent, do NOT add "
        f"the inclusive marker. When in doubt, do NOT add it."
    )


# ==========================
# Structural Segment Types
# ==========================

class SegmentType(Enum):
    QUESTION = "question"
    INSTRUCTION = "instruction"
    ANSWER_OPTION = "answer_option"
    SCALE_LABEL = "scale_label"
    OTHER = "other"

# ==========================
# Data Classes
# ==========================

@dataclass
class SurveyRow:
    variable_name: str
    english_text: str
    existing_translation: str
    # True if Column C contained a *real* translation (i.e., not just a copy of English) at load time
    had_real_translation: bool = False
    # True if this run produced a brand new translation for a row that previously had only English/placeholder
    was_newly_translated: bool = False
    new_translation: Optional[str] = None
    suggested_translation: Optional[str] = None
    suggestion_reason: Optional[str] = None

    # Layer 1: structural classification
    segment_type: SegmentType = SegmentType.OTHER
    # Layer 2: block membership (question + options group)
    block_id: Optional[int] = None
    # True if this row was translated as part of a batch scale call
    batch_translated: bool = False
    # Read-only structural audit of the FINAL shipped value (set by audit_shipped_rows).
    # Never gates output — advisory column only.
    qa_status: Optional[str] = None

    # --- LLM judge state (set only when enable_judge=True; advisory, never gates) ---
    # judge_score: last score returned by judge_translation_async (T2 if retried, else T1)
    # judge_reason: specific reason sentence from the judge's last evaluation
    # judge_retried: True if a critique-and-revise retry (T2) was attempted
    # judge_outcome: "clean" | "retried_passed" | "retried_flagged" | None (judge did not run)
    judge_score: Optional[int] = None
    judge_reason: Optional[str] = None
    judge_retried: bool = False
    judge_outcome: Optional[str] = None

@dataclass
class QuestionBlock:
    block_id: int
    # All row indices belonging to this block (into SurveyFileContext.rows)
    row_indices: List[int]
    question_indices: List[int]
    instruction_indices: List[int]
    answer_option_indices: List[int]
    scale_label_indices: List[int]

@dataclass
class BlockStyle:
    block_id: int
    # For answer options
    options_grammatical_person: str = "unspecified"  # first_person|third_person|impersonal|unspecified
    options_phrase_form: str = "unspecified"         # clause|noun_phrase|short_phrase|unspecified
    options_tone: str = "formal_neutral"             # formal_neutral|casual_neutral|other
    # For scale labels
    scale_label_phrase_form: str = "short_phrase"    # short_phrase|noun_phrase|clause|unspecified
    # Optional notes from the style model (always English)
    notes: str = ""

@dataclass
class SurveyFileContext:
    filename: str
    language_code: str
    locale_code: str
    rows: List[SurveyRow]
    # normalized_english -> {"english": original_english, "translation": translation}
    translation_memory: Dict[str, Dict[str, str]]
    # Layer 2: list of question blocks (populated after loading)
    blocks: Optional[List[QuestionBlock]] = None
    # id -> block map (Step 27); built whenever blocks are (re)built so lookups
    # never depend on block_id == list position.
    blocks_by_id: Optional[Dict[int, "QuestionBlock"]] = None
    # Layer 3: style plan per question block
    block_styles: Optional[Dict[int, BlockStyle]] = None
    # True when source and target are both the same language but different locales
    # (e.g. en → en-GB, en → en-CA). Gates copy-check bypass and localization prompts.
    is_same_language_localization: bool = False
    # Step 22: resource-prefix allowlist for rows skipped in dialect/localization mode.
    # Defaults to the built-in US survey allowlist; can be extended per survey via the UI.
    skip_block_prefixes: Tuple[str, ...] = field(
        default_factory=lambda: _DIALECT_SKIP_BLOCK_PREFIXES
    )


# ==========================
# Utility Functions
# ==========================

# ==========================
# Structural Classification
# ==========================

def classify_segment_type(english_text: str, variable_name: str = "") -> SegmentType:
    """
    Classify a single English survey text into a coarse structural type:
    question / instruction / answer option / scale label / other.

    This is intentionally heuristic and language-agnostic. It does NOT try
    to recognize specific concepts like employment status; it only cares
    about structure and length.

    Uses HTML-stripped text for length checks and optionally the Forsta
    variable name pattern to better identify answer option rows.
    """
    s = (english_text or "").strip()
    if not s:
        return SegmentType.OTHER

    text_no_tags = re.sub(r"<[^>]+>", " ", s)
    lower = text_no_tags.lower().strip()

    # PRIORITY 1: Forsta variable-name pattern (structural signal overrides text heuristics).
    # qXXX,rN,cdata or qXXX,rN reliably indicates a response option row regardless of
    # what the English text looks like (e.g. "What to expect at..." is still an option).
    if variable_name and re.search(r',r\d+', variable_name):
        if re.search(
            r"\b(strongly|somewhat|agree|disagree|neither|satisfied|dissatisfied|"
            r"likely|unlikely|very|extremely|poor|excellent|good|bad|fair)\b",
            lower,
        ):
            return SegmentType.SCALE_LABEL
        return SegmentType.ANSWER_OPTION

    # PRIORITY 2: Text-based heuristics (only when no variable-name signal).

    # Likely question: visible text ends with '?' or starts with a question-like phrase
    if text_no_tags.rstrip().endswith("?") or re.match(
        r"^(how|what|which|when|where|who|do you|did you|have you|to what extent|please rate|on a scale)\b",
        lower,
    ):
        return SegmentType.QUESTION

    # Likely instruction text
    if re.search(
        r"(select one|select all that apply|check all that apply|please select|please choose|"
        r"mark all that apply|pick one|be specific|please specify|please describe|please explain|"
        r"provide details|give details|enter a number|enter your answer|write in your own words)",
        lower,
    ):
        return SegmentType.INSTRUCTION

    # Short, no sentence punctuation -> label-like thing: option or scale label
    # Use HTML-stripped text length for a more accurate cutoff
    stripped_len = len(text_no_tags.strip())
    if stripped_len <= 100 and not any(p in text_no_tags for p in ".?!;:"):
        if re.search(
            r"\b(strongly|somewhat|agree|disagree|neither|satisfied|dissatisfied|likely|unlikely|"
            r"very|extremely|poor|excellent|good|bad|fair)\b",
            lower,
        ):
            return SegmentType.SCALE_LABEL
        return SegmentType.ANSWER_OPTION

    # Everything else
    return SegmentType.OTHER


def classify_segments(context: SurveyFileContext) -> None:
    """
    Layer 1: assign a structural segment_type to each row in the file.
    """
    for row in context.rows:
        row.segment_type = classify_segment_type(row.english_text, row.variable_name)

def build_blocks(context: SurveyFileContext) -> List[QuestionBlock]:
    """
    Layer 2: group rows into question blocks.

    A block is typically:
      - one or more QUESTION rows,
      - followed by optional INSTRUCTION rows,
      - followed by ANSWER_OPTION and/or SCALE_LABEL rows,
      - possibly with some OTHER rows mixed in.

    We keep it simple and rely on document order:
    - A new QUESTION usually starts a new block.
    - Non-question rows before the first QUESTION are grouped into a "preamble" block.
    """
    blocks: List[QuestionBlock] = []
    current_block: Optional[QuestionBlock] = None

    def start_new_block(start_index: int, seg: SegmentType) -> QuestionBlock:
        block_id = len(blocks)
        block = QuestionBlock(
            block_id=block_id,
            row_indices=[start_index],
            question_indices=[start_index] if seg == SegmentType.QUESTION else [],
            instruction_indices=[start_index] if seg == SegmentType.INSTRUCTION else [],
            answer_option_indices=[start_index] if seg == SegmentType.ANSWER_OPTION else [],
            scale_label_indices=[start_index] if seg == SegmentType.SCALE_LABEL else [],
        )
        blocks.append(block)
        return block

    for idx, row in enumerate(context.rows):
        seg = row.segment_type or SegmentType.OTHER

        if seg == SegmentType.QUESTION:
            # Start a new block whenever we see a question
            current_block = start_new_block(idx, seg)
            row.block_id = current_block.block_id
            continue

        # Non-question row
        if current_block is None:
            # No question seen yet -> start a preamble block
            current_block = start_new_block(idx, seg)
        else:
            # Attach to current block
            current_block.row_indices.append(idx)
            if seg == SegmentType.INSTRUCTION:
                current_block.instruction_indices.append(idx)
            elif seg == SegmentType.ANSWER_OPTION:
                current_block.answer_option_indices.append(idx)
            elif seg == SegmentType.SCALE_LABEL:
                current_block.scale_label_indices.append(idx)
            # For OTHER, we just keep it in row_indices without a dedicated list

        row.block_id = current_block.block_id

    # Persist on context for future layers (style inference, block-level QA)
    context.blocks = blocks
    context.blocks_by_id = {b.block_id: b for b in blocks}
    return blocks


def get_block_by_id(context: "SurveyFileContext", block_id: Optional[int]) -> Optional[QuestionBlock]:
    """Look up a QuestionBlock by its identity (not list position). Lazily builds
    the id map if missing. Returns None when blocks are absent or id is unknown."""
    if context.blocks is None or block_id is None:
        return None
    if context.blocks_by_id is None:
        context.blocks_by_id = {b.block_id: b for b in context.blocks}
    return context.blocks_by_id.get(block_id)


def promote_scale_labels(context: SurveyFileContext) -> None:
    """
    Post-classification pass: when a block already contains >= 2 SCALE_LABEL
    items, reclassify any remaining ANSWER_OPTION items in the same block as
    SCALE_LABEL.  This fixes splits where some items in the same Likert-type
    scale lack trigger keywords (e.g. "Much more comfortable" has no keyword
    but "Somewhat more comfortable" does).

    Only short, label-like answer options are promoted — long or sentence-like
    options are left alone.
    """
    if not context.blocks:
        return

    for block in context.blocks:
        if len(block.scale_label_indices) < 2:
            continue
        if not block.answer_option_indices:
            continue

        indices_to_promote: List[int] = []
        for idx in block.answer_option_indices:
            if idx < 0 or idx >= len(context.rows):
                continue
            row = context.rows[idx]
            text = re.sub(r"<[^>]+>", " ", (row.english_text or "")).strip()
            if len(text) <= 100 and not any(p in text for p in ".?!;:"):
                indices_to_promote.append(idx)

        for idx in indices_to_promote:
            context.rows[idx].segment_type = SegmentType.SCALE_LABEL
            block.answer_option_indices.remove(idx)
            block.scale_label_indices.append(idx)

        block.scale_label_indices.sort()

    # Phase 2: structural-only promotion — no keyword required.
    # If a block has NO existing SCALE_LABELs but has ≥3 ANSWER_OPTION rows
    # that are all short (≤40 stripped chars) and have no terminal sentence
    # punctuation, treat the whole set as a scale.  This catches frequency
    # scales (Daily/Weekly/Monthly), numeric scales (1-5), and boolean pairs
    # that will benefit from the coherent scale-batch path.
    for block in context.blocks:
        if block.scale_label_indices:
            continue  # already handled above or by keyword classification
        if len(block.answer_option_indices) < 3:
            continue

        candidates: List[int] = []
        for idx in block.answer_option_indices:
            if idx < 0 or idx >= len(context.rows):
                continue
            row = context.rows[idx]
            text = re.sub(r"<[^>]+>", " ", (row.english_text or "")).strip()
            if len(text) <= 40 and not any(p in text for p in ".?!;:"):
                candidates.append(idx)

        # Only promote if ALL answer options in the block qualify —
        # a mixed block (some long, some short) is probably not a pure scale.
        if len(candidates) == len(block.answer_option_indices) and len(candidates) >= 3:
            for idx in candidates:
                context.rows[idx].segment_type = SegmentType.SCALE_LABEL
                block.scale_label_indices.append(idx)
            block.answer_option_indices.clear()
            block.scale_label_indices.sort()


# Ordered English intensity qualifiers (low -> high index = increasing intensity).
# Used only for an advisory monotonicity check on scale sets; English source is
# always available, so this stays language-agnostic on the routing side.
_INTENSITY_RANK_PATTERNS: List[Tuple[int, "re.Pattern"]] = [
    (0, re.compile(r"\bnot at all\b|\bnever\b|\bnone\b", re.I)),
    (1, re.compile(r"\bslightly\b|\brarely\b|\ba little\b", re.I)),
    (2, re.compile(r"\bsomewhat\b|\bsometimes\b|\bmoderately\b", re.I)),
    (3, re.compile(r"\bvery\b|\boften\b|\bquite\b", re.I)),
    (4, re.compile(r"\bextremely\b|\balways\b|\bcompletely\b", re.I)),
]


def english_intensity_rank(label: str) -> Optional[int]:
    """Return an intensity rank for an English scale label, or None if no known
    qualifier is present.  Deterministic, English-only."""
    s = strip_html_for_heuristics(label or "")
    if not s:
        return None
    for rank, pat in _INTENSITY_RANK_PATTERNS:
        if pat.search(s):
            return rank
    return None


# ── Step 24: concept-term extraction ─────────────────────────────────────────
# Patterns that identify the measurement concept in a survey question stem.
# These are English-source patterns only (the tool always has English input).
_CONCEPT_TERM_PATTERNS: List[re.Pattern] = [
    re.compile(r"how\s+(\w+)\s+(?:are|were|do|did|have|has)\s+you\b", re.IGNORECASE),
    re.compile(r"how\s+(?:much|many|often|likely|satisfied|comfortable)\s+(\w+)", re.IGNORECASE),
    re.compile(r"(?:rate|rating)\s+your\s+(?:level\s+of\s+)?(\w+)", re.IGNORECASE),
    re.compile(r"(?:your|your\s+level\s+of)\s+(\w+(?:ion|ness|ment|ity|ance))\b", re.IGNORECASE),
    re.compile(r"\b(satisfaction|comfort|likelihood|agreement|confidence|trust|importance|familiarity|frequency|ease)\b", re.IGNORECASE),
]


def _extract_concept_term(english_question: str) -> Optional[str]:
    """
    Heuristically extract the key measurement concept from an English survey
    question stem (e.g. 'comfortable' from 'How comfortable are you...').
    Returns the matched word lowercased, or None if no pattern fires.
    Language-agnostic — operates only on the English source.
    """
    q = (english_question or "").strip()
    if not q:
        return None
    for pat in _CONCEPT_TERM_PATTERNS:
        m = pat.search(q)
        if m:
            groups = [g for g in m.groups() if g]
            if groups:
                return groups[-1].lower()
    return None


# Resource prefixes for rows that must be skipped in dialect adaptation mode.
# These are hidden/system fields or US-only question blocks that are
# conditionally invisible to international respondents.
# Every row whose Resource starts with one of these prefixes is skipped
# (title, comment, ch## answer options, etc.)
_DIALECT_SKIP_BLOCK_PREFIXES = (
    "qZipCode",
    "qZipState",
    "qZipDivision",
    "qZipRegion",
    "qZipMarket",
    "qState",
    "vRESPDATA",
    "vHHINCOMELABELS",
)


def skip_dialect_excluded_rows(context: SurveyFileContext) -> int:
    """
    Pre-processing filter for dialect adaptation mode.  Marks rows that must
    NOT be adapted (hidden system fields, US-only question blocks) so they
    pass through with source text unchanged.

    Returns the number of rows skipped.
    """
    skipped = 0
    for row in context.rows:
        resource = (row.variable_name or "")

        # Check resource prefix against the skip list
        if any(resource.startswith(pfx) for pfx in context.skip_block_prefixes):
            row.new_translation = row.english_text
            row.batch_translated = True
            skipped += 1
            continue

        # Check for HIDDEN marker in visible text (strip HTML first)
        src = (row.english_text or "").strip()
        src_no_html = re.sub(r"<[^>]+>", " ", src).strip()
        if src_no_html.upper().startswith("HIDDEN"):
            row.new_translation = row.english_text
            row.batch_translated = True
            skipped += 1
            continue

    return skipped


def filename_without_extension(filename: str) -> str:
    return Path(filename).stem


def map_language_name_to_code(language_name: Optional[str]) -> str:
    """
    Map a language label (e.g., 'Spanish', 'spanish (only new)', 'Spanish - Mexico')
    to a code (e.g., 'es').

    If detection fails, return empty string so the UI can warn the user,
    instead of silently defaulting to Spanish.
    """
    if not language_name:
        return ""

    raw = language_name.strip().lower()

    # Strip any parenthetical metadata, e.g. "spanish (only new)" -> "spanish"
    raw_no_parens = re.sub(r"\(.*?\)", "", raw).strip()

    # Take the first token before whitespace or hyphen as the primary language token.
    # Examples:
    #   "spanish mexico"     -> "spanish"
    #   "spanish-mexico"     -> "spanish"
    #   "spanish only new"   -> "spanish"
    tokens = re.split(r"[\s\-]+", raw_no_parens)
    primary = tokens[0] if tokens else raw_no_parens

    # Direct match on primary token
    if primary in LANGUAGE_NAME_TO_CODE:
        return LANGUAGE_NAME_TO_CODE[primary]

    # Substring fallback removed: it causes false positives (e.g. a token containing
    # "it" or "de" matching Italian/German).  If the primary token doesn't match, we
    # return "" so the UI can prompt the operator rather than silently guessing.
    return ""


def map_language_and_locale_to_bcp47(language_code: str, locale_name: Optional[str]) -> str:
    if not language_code:
        return ""

    if not locale_name:
        # Use generic language-only code if no locale
        return language_code

    key = locale_name.strip().lower().replace(" ", "_")

    if language_code == "es":
        # Unknown locale token degrades to bare language code, never a wrong region.
        return SPANISH_LOCALE_NAME_TO_BCP47.get(key, language_code)
    if language_code == "en":
        return ENGLISH_LOCALE_NAME_TO_BCP47.get(key, language_code)

    # Extra: if filename already contains a BCP-47-like code (e.g. es-MX), just return it
    if re.match(rf"^{language_code}-[A-Za-z]+$", locale_name.strip()):
        return locale_name.strip()

    # Fallback for other languages: require exact normalized key match on the code,
    # not substring containment of the key inside the label (which was a false-positive risk).
    options = LOCALE_OPTIONS.get(language_code)
    if options:
        for label, code in options:
            if key == label.lower().replace(" ", "_") or key == code.lower():
                return code

    # Unknown locale: degrade to bare language code so the UI locale dropdown
    # defaults to the first option rather than fabricating a region.
    return language_code



def parse_language_and_locale_from_filename(filename: str) -> Tuple[str, str]:
    """
    Parse filenames like:
      12345_Spanish_Argentina.xlsx
      12345_English_UK.csv
      12345_English_CA.xls
      12345_uk.xls              (shorthand: implies English + en-GB)
      12345_aus.xls             (shorthand: implies English + en-AU)
      <survey_id>_<language>[_<localization>].ext

    Returns (language_code, locale_code), empty string when detection fails.
    """
    base_name = filename_without_extension(filename)
    parts = base_name.split("_")

    language_name = parts[1] if len(parts) >= 2 else None
    locale_name = parts[2] if len(parts) >= 3 else None

    language_code = map_language_name_to_code(language_name)

    # Shorthand fallback: if the language token itself is a known English locale
    # abbreviation (e.g., "uk", "aus", "ca"), treat it as English + that locale.
    if not language_code and language_name:
        shorthand_locale = _ENGLISH_LOCALE_SHORTHANDS.get(language_name.strip().lower())
        if shorthand_locale:
            return "en", shorthand_locale

    locale_code = map_language_and_locale_to_bcp47(language_code, locale_name) if language_code else ""

    return language_code, locale_code


def read_excel_or_csv(file) -> pd.DataFrame:
    filename = getattr(file, "name", "uploaded_file")
    suffix = Path(filename).suffix.lower()

    if suffix in [".xls", ".xlsx"]:
        df = pd.read_excel(file, keep_default_na=False, na_values=[""])
    elif suffix == ".csv":
        try:
            df = pd.read_csv(file, encoding="utf-8-sig", sep=None, engine="python",
                             keep_default_na=False, na_values=[""])
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="cp1252", sep=None, engine="python",
                             keep_default_na=False, na_values=[""])
    else:
        raise ValueError(f"Unsupported file type for {filename}. Use .xls, .xlsx, or .csv")

    return df


def normalize_english_text(text: str) -> str:
    """
    Simple normalization: lowercase, strip whitespace, remove most punctuation.
    Keeps word characters, digits, and whitespace.
    """
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_for_copy_check(text: str) -> str:
    """
    Normalize a string for the purpose of detecting 'no-op' translations:
    - strip HTML tags,
    - collapse whitespace,
    - lowercase.

    This is language-agnostic and only cares about whether the main human-readable
    content changed at all.
    """
    if not text:
        return ""
    # Remove HTML tags
    no_tags = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace and lowercase
    normalized = re.sub(r"\s+", " ", no_tags).strip().lower()
    return normalized


def is_effective_copy_of_english(english_text: str, candidate_translation: str) -> bool:
    """
    Return True if the candidate_translation is essentially just a copy
    of the English source (ignoring tags and whitespace). This is a strong
    signal that the model failed to translate.
    """
    src = normalize_for_copy_check(english_text)
    trg = normalize_for_copy_check(candidate_translation)
    if not src or not trg:
        return False
    return src == trg


def pick_final_translation(result: Dict[str, object], english_text: str = "") -> str:
    """
    Select the best translation from a model result dict applying the precedence:
      1. qa_checked_translation when non-empty and not an English copy.
      2. proposed_translation when non-empty.
      3. Empty string (caller must handle sentinel).

    If qa_checked_translation and proposed_translation diverge significantly
    (one is English, the other is not), prefer the non-English one.
    """
    qa = (result.get("qa_checked_translation") or "").strip()
    proposed = (result.get("proposed_translation") or "").strip()

    if qa and proposed:
        qa_is_copy = bool(english_text) and is_effective_copy_of_english(english_text, qa)
        proposed_is_copy = bool(english_text) and is_effective_copy_of_english(english_text, proposed)
        if qa_is_copy and not proposed_is_copy:
            return proposed  # QA step regressed to English; use the better first-pass
        return qa  # prefer QA revision (may be identical to proposed)
    return qa or proposed


TRANSLATABLE_SHORT_TERMS = {"none", "other", "yes", "no", "all", "any"}

# ---- Step 13: Semantic back-translation verification config ----
# Comparison is always English source vs back-translated English, so these are
# English-language sets and are fully language-agnostic on the target side.
_SEMANTIC_STOPWORDS = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or",
    "is", "are", "be", "you", "your", "i", "we", "they", "it",
    "this", "that", "with", "as", "at", "by", "do", "does",
})

# Qualifier/negation/polarity terms whose presence or absence on one side but not
# the other signals a semantically significant drift (e.g. "not" dropped).
_QUALIFIER_TERMS = frozenset({
    "not", "never", "no", "none", "always", "very", "somewhat", "slightly",
    "completely", "extremely", "rarely", "often", "sometimes", "fully",
    "partially", "strongly", "mostly",
})

# Content-word Jaccard overlap below this threshold flags potential drift.
# Configurable via environment variable so it can be tuned without a code change.
_SEMANTIC_OVERLAP_THRESHOLD = float(os.getenv("SEMANTIC_OVERLAP_THRESHOLD", "0.45"))

def should_run_copy_check(english_text: str, variable_name: str = "") -> bool:
    """
    Decide whether it makes sense to flag 'unchanged English' as a likely failure.

    We ONLY want to run the copy-check for content that clearly ought to be translated
    (questions, full phrases, longer labels), and we want to allow cases where it is
    normal for the English form to appear unchanged in the target language, such as:

      - pure numeric ranges or codes (e.g., '1970-1989', '2024'),
      - simple proper nouns / place names (e.g., 'Riverside', 'Disneyland Paris'),
      - very short single-word labels that are often the same across languages
        (e.g., 'No', 'OK').

    This function is deliberately language-agnostic; it only inspects the English form.
    """
    if not english_text:
        return False

    text = re.sub(r"<[^>]+>", " ", english_text)
    text = text.strip()
    if not text:
        return False

    if re.fullmatch(r"[\d\s\-\–/.,%+$€£¥]+", text):
        return False

    if text.lower() in TRANSLATABLE_SHORT_TERMS:
        return True

    tokens = text.split()

    # Multi-word proper noun heuristic: short title-cased phrases are likely
    # brand names or place names that stay the same across languages
    if 1 <= len(tokens) <= 5 and all(w[0].isupper() for w in tokens if w):
        return False

    # Additionally: short title-cased text in a Forsta response-option row
    # (variable name like qXXX,rN,cdata) is almost certainly a named entity
    if variable_name and re.search(r',r\d+', variable_name):
        if len(tokens) <= 5 and all(w[0].isupper() for w in tokens if w):
            return False

    if len(tokens) == 1:
        token = tokens[0]

        if re.fullmatch(r"[A-Z][A-Za-z0-9]*", token) or re.fullmatch(r"[A-Z0-9]{2,}", token):
            return False

        if len(token) <= 3:
            return False

    return True



# ---------------------------
# Invariant numeric/range guards
# ---------------------------
_PURE_NUMERIC_OR_RANGE_RE = re.compile(r"^[\d\s\-\–\—/.,%+$€£¥]+$")

def strip_html_for_heuristics(text: str) -> str:
    """Strip HTML tags and collapse whitespace for heuristics only."""
    if not text:
        return ""
    s = re.sub(r"<[^>]+>", " ", text)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_pure_numeric_or_range_code_like(text: str) -> bool:
    """True for strings like '1970-1989', '2024', '$100', '10-15%' (no letters)."""
    s = strip_html_for_heuristics(text)
    return bool(s) and bool(_PURE_NUMERIC_OR_RANGE_RE.fullmatch(s))


# Per-base-language numeric punctuation conventions for PURE numeric/range/code
# rows.  Value-preserving: only the thousands grouping and decimal separators are
# swapped; digits and their order never change.  English source convention is
# thousands=',' decimal='.'.  Languages absent here keep the source verbatim.
# Language-agnostic: routing reads this dict, never a hard-coded language branch.
_EN_THOUSANDS, _EN_DECIMAL = ",", "."
NUMERIC_LOCALE_FORMAT: Dict[str, Dict[str, str]] = {
    "de": {"thousands": ".", "decimal": ","},
    "es": {"thousands": ".", "decimal": ","},
    "fr": {"thousands": "\u00a0", "decimal": ","},   # NBSP grouping
    "it": {"thousands": ".", "decimal": ","},
    "pt": {"thousands": ".", "decimal": ","},
    "nl": {"thousands": ".", "decimal": ","},
    "ru": {"thousands": "\u00a0", "decimal": ","},
    "pl": {"thousands": "\u00a0", "decimal": ","},
    "tr": {"thousands": ".", "decimal": ","},
}

# A number token: optional currency/sign prefix, digit groups with EN separators,
# optional EN decimal part, optional percent.  Used only on pure-numeric rows.
_EN_NUMBER_RE = re.compile(
    r"(?P<pre>[$\u20ac\u00a3\u00a5+\-]?)"
    r"(?P<int>\d{1,3}(?:,\d{3})+|\d+)"
    r"(?P<dec>\.\d+)?"
    r"(?P<pct>%?)"
)


def format_pure_numeric_for_locale(text: str, language_code: str) -> str:
    """Reformat thousands/decimal separators of a PURE numeric/range/code string
    to the target base-language convention WITHOUT changing any digit or value.

    Only the integer grouping separator and the decimal point are translated.
    Returns *text* unchanged when the language has no override or no numbers."""
    if not text:
        return text
    fmt = NUMERIC_LOCALE_FORMAT.get((language_code or "").lower().split("-")[0])
    if not fmt:
        return text
    th, dec = fmt["thousands"], fmt["decimal"]
    if th == _EN_THOUSANDS and dec == _EN_DECIMAL:
        return text

    def _repl(m: "re.Match") -> str:
        int_part = m.group("int")
        digits = int_part.replace(_EN_THOUSANDS, "")
        # Re-group in 3s from the right only if the source was grouped.
        if _EN_THOUSANDS in int_part:
            grouped = ""
            for i, ch in enumerate(reversed(digits)):
                if i and i % 3 == 0:
                    grouped = th + grouped
                grouped = ch + grouped
            int_out = grouped
        else:
            int_out = digits
        dec_out = (dec + m.group("dec")[1:]) if m.group("dec") else ""
        return f"{m.group('pre')}{int_out}{dec_out}{m.group('pct')}"

    return _EN_NUMBER_RE.sub(_repl, text)


def is_label_like_english(text: str) -> bool:
    """
    Heuristic to detect short, stand-alone label-like English text
    (e.g., 'January', 'Very poor', 'Strongly Agree').

    We use this to optionally adjust capitalization of the translation for
    answer options, without touching full sentences.
    """
    s = strip_html_for_heuristics(text or "")
    if not s:
        return False
    if len(s) > 40:
        return False
    # If it has sentence punctuation, treat as sentence
    if any(p in s for p in ".?!;:"):
        return False

    # One or more words, each starting with uppercase OR the whole thing is ALL CAPS
    words = s.split()
    if not words:
        return False

    if all(w[0].isupper() for w in words if w):
        return True

    if s.isupper() and len(s) >= 2:
        return True

    return False


def adjust_capitalization_for_label(
    english_text: str,
    translation_text: str,
    language_code: str,
    answer_option_count: int = 0,
    answer_option_avg_len: float = 0.0,
) -> str:
    """
    For short, label-like English texts, adjust the translation so that the
    first alphabetic character is uppercase (for languages with case).

    This helps make response options like month names look like labels
    ('enero' -> 'Enero') without interfering with full sentences.

    Coordination guards:
    - Skip when the block is an article-suppressed list (the prompt already asks
      for bare, consistently-cased items; forcing uppercase would fight it).
    - Never alter a leading mixed-case brand token (iPhone, eBay, iOS) — its
      intentional lowercase initial is part of the brand.
    """
    if not translation_text:
        return translation_text

    if not is_label_like_english(english_text):
        return translation_text

    # Languages without case (CJK, Hindi) – do nothing.
    if language_code in {"ja", "zh", "ko", "hi"}:
        return translation_text

    # Coordination: do not re-case items in an article-suppressed list block.
    if is_article_suppressed_list(answer_option_count, answer_option_avg_len, language_code):
        return translation_text

    # Brand-token guard: a leading token with a lowercase initial followed by an
    # internal uppercase (iPhone, eBay, iOS) is intentionally cased — leave it.
    first_token = translation_text.split()[0] if translation_text.split() else ""
    if re.match(r"^[a-z]+[A-Z]", first_token):
        return translation_text

    chars = list(translation_text)
    for i, ch in enumerate(chars):
        if ch.isalpha():
            chars[i] = ch.upper()
            break
    return "".join(chars)


def options_look_like_short_labels(option_texts: List[str]) -> bool:
    """
    Heuristic: the option set looks like short labels rather than full sentences.
    Used to decide when peer-options guidance is helpful.
    """
    cleaned = [strip_html_for_heuristics(t) for t in (option_texts or [])]
    cleaned = [t for t in cleaned if t]
    if len(cleaned) < 2:
        return False

    punct_hits = sum(1 for t in cleaned if any(p in t for p in ".?!;:"))
    if punct_hits / len(cleaned) > 0.2:
        return False

    avg_words = sum(len(t.split()) for t in cleaned) / len(cleaned)
    if avg_words > 6:
        return False

    return True


def build_translation_memory(rows: List[SurveyRow]) -> Dict[str, Dict[str, str]]:
    """
    Build translation memory:
      normalized_english -> { "english": original_english, "translation": translation }

    Only includes rows that had a real existing translation (Column C != Column B) at load time.
    """
    memory: Dict[str, Dict[str, str]] = {}
    for r in rows:
        if not r.had_real_translation:
            continue

        eng = (r.english_text or "").strip()
        trl = (r.existing_translation or "").strip()
        if not eng or not trl:
            continue

        key = normalize_english_text(eng)
        if key and key not in memory:
            memory[key] = {"english": eng, "translation": trl}
    return memory


def _detect_same_language_localization(language_code: str, locale_code: str) -> bool:
    """
    Return True when the translation job is a same-language dialect adaptation
    (e.g. en -> en-GB, es -> es-MX, pt -> pt-BR, zh -> zh-TW).
    Uses LANGUAGE_CAPABILITIES.dialect_codes as the authoritative allowlist.
    """
    base_lang = (language_code or "").lower().split("-")[0]
    base_locale = (locale_code or "").lower().split("-")[0]
    if base_lang != base_locale:
        return False  # different base languages -> not a same-language localization
    cap = _get_lang_cap(base_lang)
    dialect_codes = cap.get("dialect_codes") or []
    normalized_locale = (locale_code or "").lower()
    return normalized_locale in [d.lower() for d in dialect_codes]


def load_forsta_export(
    file,
    language_code_override: Optional[str] = None,
    locale_code_override: Optional[str] = None,
) -> Tuple[SurveyFileContext, pd.DataFrame]:
    """
    Load Forsta export (3+ columns) and build SurveyFileContext + original DataFrame.

    IMPORTANT: Column C might be:
      - A real translation (target language)
      - A placeholder that just repeats the English (Column B)
      - Empty

    We treat "C == B" (after simple strip) as "no existing translation yet".
    """
    filename = getattr(file, "name", "uploaded_file")
    df = read_excel_or_csv(file)

    if df.shape[1] < 3:
        raise ValueError(
            f"File '{filename}' must have at least 3 columns "
            f"(variable_name, english_text, translation). Found {df.shape[1]}."
        )

    # Detect language/locale from filename
    detected_lang, detected_locale = parse_language_and_locale_from_filename(filename)
    language_code = language_code_override or detected_lang
    locale_code = locale_code_override or detected_locale

    rows: List[SurveyRow] = []
    for _, row in df.iterrows():
        var_name = row.iloc[0]
        eng_text = row.iloc[1]
        trl = row.iloc[2]

        var_name_str = "" if pd.isna(var_name) else str(var_name)
        eng_text_str = "" if pd.isna(eng_text) else str(eng_text)
        trl_str = "" if pd.isna(trl) else str(trl)

        # Step 26: detect "no real translation" after HTML-strip + whitespace-
        # collapse + lowercase, so a placeholder differing only by case/markup/
        # NBSP is not mistaken for a real prior translation.
        eng_norm = normalize_for_copy_check(eng_text_str)
        trl_norm = normalize_for_copy_check(trl_str)

        # Real existing translation only if Column C differs from Column B
        had_real_translation = bool(eng_norm and trl_norm and eng_norm != trl_norm)

        rows.append(
            SurveyRow(
                variable_name=var_name_str,
                english_text=eng_text_str,
                existing_translation=trl_str,
                had_real_translation=had_real_translation,
            )
        )

    context = SurveyFileContext(
        filename=filename,
        language_code=language_code or "",
        locale_code=locale_code or "",
        rows=rows,
        translation_memory={},  # filled next
    )
    context.translation_memory = build_translation_memory(context.rows)
    context.is_same_language_localization = _detect_same_language_localization(
        language_code, locale_code
    )
    return context, df


# ==========================
# LLM Integration
# ==========================

_llm_client: Optional["OpenAI"] = None

def get_llm_client() -> "OpenAI":
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    if OpenAI is None:
        raise RuntimeError(
            "The 'openai' package is not installed or incorrectly imported. "
            "Please install it with 'pip install openai'."
        )

    # Read from env (loaded via .env above)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please set it in your .env or environment.")

    _llm_client = OpenAI(api_key=api_key)
    return _llm_client


def _jaccard_word_similarity(text_a: str, text_b: str) -> float:
    """Simple word-level Jaccard similarity."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0


def sample_translation_memory_examples(
    translation_memory: Dict[str, Dict[str, str]],
    current_english: str = "",
    max_examples: int = 10,
) -> List[Tuple[str, str]]:
    """
    Return up to max_examples of (english, translation) pairs from translation_memory,
    ranked by relevance to current_english using Jaccard word similarity.
    """
    items = list(translation_memory.values())
    candidates: List[Tuple[float, str, str]] = []

    for val in items:
        if isinstance(val, dict):
            eng = val.get("english", "")
            trl = val.get("translation", "")
        else:
            eng = ""
            trl = str(val)
        if not eng or not trl:
            continue

        score = _jaccard_word_similarity(current_english, eng) if current_english else 0.0
        candidates.append((score, eng, trl))

    candidates.sort(key=lambda x: -x[0])

    examples: List[Tuple[str, str]] = []
    for _score, eng, trl in candidates[:max_examples]:
        examples.append((eng, trl))

    return examples


async def _call_translation_model_async_uncached(
        english_text: str,
        language_code: str,
        locale_code: str,
        global_context: str,
        translation_memory: Dict[str, Dict[str, str]],
        existing_translation: Optional[str] = None,
        segment_type: Optional[SegmentType] = None,
        block_style: Optional[BlockStyle] = None,
        peer_english_options: Optional[List[str]] = None,
        parent_context: str = "",
        gender_inclusive: bool = False,
        model_name: str = MODEL_NAME,
        answer_option_count: int = 0,
        answer_option_avg_len: float = 0.0,
        is_same_language_localization: bool = False,
) -> Dict[str, object]:
    client = get_async_client()
    existing_translation = existing_translation or ""

    # 1. Memory Construction
    memory_examples = sample_translation_memory_examples(translation_memory, current_english=english_text, max_examples=5)
    memory_str = "\n".join([f'- "{e}" -> "{t}"' for e, t in memory_examples]) if memory_examples else "None."

    # 2. Style & Context Construction
    segment_type_str = segment_type.value if isinstance(segment_type, SegmentType) else "other"

    # Block-level style plan (may be None)
    # Force null for same-language localization — style-driven restructuring
    # (first-person prefixes, phrase-form enforcement) must never apply.
    block_style_info = {}
    if not is_same_language_localization and isinstance(block_style, BlockStyle):
        block_style_info = {
            "options_style": {
                "grammatical_person": block_style.options_grammatical_person,
                "phrase_form": block_style.options_phrase_form,
                "tone": block_style.options_tone,
            },
            "scale_label_style": {
                "phrase_form": block_style.scale_label_phrase_form,
            },
        }
    block_style_json = json.dumps(block_style_info, ensure_ascii=False) if block_style_info else "null"

    # Context injection: only the string changes based on parent_context
    context_instruction = ""
    if parent_context:
        context_instruction = (
            f"Parent question for context only (do NOT change the meaning of the answer options): "
            f'"{parent_context}".\n'
            f"Ensure your translation fits grammatically as a response, but preserve the exact "
            f"semantic meaning of the English answer option text."
        )

    peer_options_instruction = ""
    if peer_english_options and segment_type_str == "answer_option":
        # Provide the full ordered peer set so the model can keep options parallel.
        peers_json = json.dumps(peer_english_options, ensure_ascii=False)
        peer_options_instruction = (
            f"Peer answer options in this same set (English, ordered): {peers_json}\n"
            "Keep this option grammatically and stylistically PARALLEL to its peers. "
            "Do not mix label types (e.g., person-noun labels vs adjective labels) within the same set. "
            "All peer options in this set MUST use the same grammatical structure. If most peers "
            "are infinitive phrases, this option must also be an infinitive phrase."
        )

    gender_inclusive_instruction = build_gender_inclusive_instruction(
        language_code, gender_inclusive,
        segment_type=segment_type.name if segment_type is not None else None,
    )

    brand_name_instruction = build_brand_name_instruction(language_code)

    # Grounding instruction: prevent semantic drift in answer option translations
    grounding_instruction = ""
    if segment_type in (SegmentType.ANSWER_OPTION, SegmentType.SCALE_LABEL):
        grounding_instruction = (
            "CRITICAL — MEANING PRESERVATION: Translate the semantic content of each answer "
            "option exactly. You MAY adapt grammatical form (verb conjugation, case endings, "
            "article agreement) as required by the target language. You MUST NOT change the "
            "underlying concept or qualifier. If the English says 'comfortable', the translation "
            "must convey comfort/ease — NOT interest, willingness, or any other concept. "
            "If the English says 'somewhat', 'not at all', or any intensity/negation qualifier, "
            "that qualifier must appear in the translation. "
            "For first-person restructuring: you may add only the grammatical subject ('I/je/ich') "
            "and the minimum required copula or auxiliary needed by the target language. "
            "Do NOT add volitional verbs ('want', 'intend', 'would like') that are not in the English."
        )

    # List-context instruction: omit articles for dropdown/list-like blocks
    list_context_instruction = build_list_context_instruction(
        answer_option_count, answer_option_avg_len, language_code
    )

    # --------- System & User Prompts (now outside the if-block) ---------
    domain_fragment = build_domain_prompt_fragment(global_context)

    # Build locale-specific vocabulary whitelist for the prompt
    _vocab_for_locale = DIALECT_VOCABULARY.get(locale_code, {})
    _vocab_prompt_lines = ""
    if _vocab_for_locale:
        _vocab_prompt_lines = "\n".join(
            f'   "{us}" -> "{local}"'
            for us, local in _vocab_for_locale.items()
        )

    if is_same_language_localization:
        _base_lang_name = language_code.upper() if language_code else "source"
        system_prompt = f"""
You are a professional copy editor specializing in regional dialect and locale adaptation
for market-research questionnaires.
Domain context: {domain_fragment}
You adapt survey text from one {_base_lang_name} variant into the {locale_code} locale
(e.g., US English to British English, es-ES to es-MX, pt-PT to pt-BR).

The ONLY changes you are permitted to make are:
1. Spelling conventions (e.g., color/colour, center/centre,
   traveled/travelled, canceled/cancelled, defense/defence, analog/analogue).
2. Vocabulary substitutions ONLY from the approved whitelist below. If a US term is
   NOT in this list, leave it unchanged.

APPROVED VOCABULARY WHITELIST for {locale_code}:
{_vocab_prompt_lines if _vocab_prompt_lines else "   (No vocabulary substitutions — spelling changes only.)"}

If the text is already correct for the target locale, return it UNCHANGED.
It is expected that the majority of rows will require zero changes.

You MUST NOT do any of the following:
- Do NOT change the grammatical structure, person, or phrasing of the source text.
  If the source says "Visited a UNESCO World Heritage Site", you must NOT add
  "I visited..." or any other prefix. Preserve the exact grammatical form.
- Do NOT insert, remove, or reorder words unless the change is a direct dialect
  vocabulary substitution from the whitelist above. Do NOT add words like "that",
  "which", or articles unless the whitelist entry includes them.
- Do NOT change adjectives or qualifiers (e.g., do NOT change "quiet" to "quieter").
- Do NOT rephrase, restructure, or "improve" text in any way beyond the two
  permitted change types above.
- Do NOT convert currency symbols, amounts, or formatting. Leave $, EUR, GBP and all
  monetary values exactly as they appear in the source. Do NOT add comma separators
  to currency amounts.
- Do NOT add content that is not in the source text (e.g., do NOT add metric/km
  equivalents, explanatory notes, or parenthetical additions).
- Do NOT change number formatting (e.g., do NOT change "5 - 10" to "5–10").
- Do NOT add, remove, or change any punctuation marks. Preserve the exact punctuation
  from the source, including periods, commas, quotation marks, hyphens, dashes, and
  ellipses. Do NOT "fix" or "correct" punctuation errors in the source text.
- Do NOT correct errors, typos, capitalization issues, or formatting inconsistencies
  in the source text. If the source has "District Of Columbia", output "District Of
  Columbia" exactly. Only change words that differ between the source variant and the
  target locale per the rules above.
- Do NOT convert "state" to "province" for UK or AU English. US geographic
  references (state, zip code) should remain as-is unless the whitelist above
  provides a specific conversion.
- When a US term is used as an adjective modifying a noun (e.g., "state park"),
  use the adjective form of the local equivalent ("provincial park"), not the noun
  form ("province park").

Structural safety:
- Do NOT change, remove, or re-order any HTML tags, placeholders, survey piping tokens
  or variable names. Only adapt the human-readable text between them.

Consistency:
- When the same phrase appears in multiple places, prefer consistent adaptation.
- Respect any adaptations shown in the translation memory.

Output format:
- You MUST always return a valid JSON object with the required keys and no extra text.
- The 'change_reason' field MUST always be written in ENGLISH.
"""
    else:
        system_prompt = f"""
You are a professional translator and QA specialist for market-research questionnaires.
Domain context: {domain_fragment}
You translate from English into a specified target language and locale.

RULE PRECEDENCE (highest to lowest — earlier rules override later ones):
  P1. Structural tokens — placeholders, HTML tags, and piping variables are NEVER changed.
  P2. Semantic meaning — every qualifier, distinction, and scale polarity in the English
      must be preserved. This rule cannot be overridden by style or form preferences.
  P3. Grammatical form — adapt to the target language's natural grammatical form only
      when required by the language; do NOT add new concepts while adapting form.
  P4. Secondary guidance — gender-inclusive forms, list-article rules, brand conventions,
      and block_style all apply only after P1-P3 are satisfied.

Your priorities, in order, are:

1. Semantic accuracy and measurement integrity
   - Preserve all meanings, distinctions and qualifiers from the English.
   - Do NOT add new concepts, remove options, or merge distinct categories in a way that
     would change the data collected.
   - Keep scale polarity and intensity intact (e.g., if the English scale goes from very
     positive to very negative, the target language scale must do the same).

2. Structural safety
   - Do NOT change, remove, or re-order any HTML tags, placeholders, survey piping tokens
     or variable names. Only translate the human-readable text between them.
   - Preserve numbers, numeric ranges, and currency symbols. You may adapt formatting
     (decimal/thousand separators, spacing) to the target locale, but the underlying
     values must stay the same.
   - When translating currency ranges within a question, use consistent number formatting
     appropriate for the target locale throughout the entire set of options. Do not mix
     English-format numbers with locale-format numbers within the same question block.

3. Tone and register
   - This is a survey for general consumers. Use standard, accessible language that any
     adult respondent can read without effort.
   - Avoid slang, jokes, or marketing hype. Also avoid legal/formal register (e.g., use
     the equivalent of 'live' rather than 'reside', 'airports' rather than 'airport hubs').
   - Do NOT add politeness markers (e.g., 'please', 'Por favor') to instruction text
     unless the English source explicitly includes them.
   - Use natural, idiomatic phrasing that a native speaker would use in everyday language.
     When a common English expression has a well-known idiomatic equivalent in the target
     language, prefer the idiomatic form (e.g., the equivalent of 'trips abroad' rather
     than a literal 'international voyages').
   - Target the register of a well-written consumer survey, not a legal document or
     institutional report.

4. Consistency and terminology
   - When the same English phrase appears in multiple places with the same meaning, you
     should prefer a consistent translation, unless local context clearly requires a
     different wording.
   - Respect any translations shown in the translation memory when they fit the local
     context; do not override them without a good reason.

5. Proper nouns, brands, and untranslatable items
   - Keep brand names, platform names and product names in their original form unless
     there is a widely used standard equivalent in the target language.
   - Do not translate internal variable names, placeholders, or piping tokens (for example:
     {{Q1}}, [PIPE:DESTINATION], [[VARNAME]], $VARNAME).
   - When the English source text contains a pattern of 'Full Name (ABBREVIATION)' for
     organizations, agencies, or programs (e.g., 'Transportation Security Administration (TSA)',
     'Customs and Border Protection (CBP)'), you MUST preserve the full expanded name in your
     translation. You may translate the full name into the target language or keep it in English,
     but you must NOT reduce it to just the abbreviation. The abbreviation in parentheses must
     also be preserved as-is.

6. Output format
   - You MUST always return a valid JSON object with the required keys and no extra text.
   - The 'change_reason' field MUST always be written in ENGLISH, even when you are
     translating into another language.
"""

    localization_mode_instruction = ""
    if is_same_language_localization:
        _vocab_block = ""
        if _vocab_for_locale:
            _lines = "\n".join(
                f"  {us} -> {local}"
                for us, local in _vocab_for_locale.items()
            )
            _vocab_block = (
                f"APPROVED VOCABULARY WHITELIST for {locale_code}:\n"
                f"{_lines}\n"
                "Only convert vocabulary items listed above. If a US term is NOT in the "
                "whitelist, leave it unchanged.\n\n"
            )
        localization_mode_instruction = (
            "DIALECT ADAPTATION MODE — STRICT RULES:\n"
            "The source and target are both English. You are adapting dialect-specific "
            "spelling and vocabulary ONLY. The majority of rows will require ZERO changes.\n\n"
            "PERMITTED changes (the ONLY changes you may make):\n"
            "- Spelling conventions: color/colour, center/centre, "
            "traveled/travelled, canceled/cancelled, defense/defence, analog/analogue\n"
            "- Vocabulary substitutions ONLY from the approved whitelist.\n\n"
            + _vocab_block
            + "PROHIBITED changes (you MUST NOT do any of the following):\n"
            "- Do NOT change grammatical structure, person, or phrasing. If the source says "
            "'Visited a UNESCO World Heritage Site', do NOT add 'I visited...' or any prefix.\n"
            "- Do NOT insert, remove, or reorder words (no adding 'that', 'which', articles) "
            "unless the whitelist entry includes them.\n"
            "- Do NOT change adjectives or qualifiers (e.g., do NOT change 'quiet' to 'quieter').\n"
            "- Do NOT convert currency symbols, amounts, or formatting. Leave $, EUR, etc. as-is.\n"
            "- Do NOT add content not in the source (no km equivalents, no explanatory notes).\n"
            "- Do NOT change number formatting (do NOT change '5 - 10' to '5–10').\n"
            "- Do NOT add, remove, or change punctuation. Keep all periods, commas, hyphens, "
            "dashes, and quotation marks exactly as they appear. Do NOT 'fix' punctuation.\n"
            "- Do NOT rephrase, restructure, or 'improve' the text in any way.\n"
            "- Do NOT correct errors, typos, or capitalization in the source. If the source "
            "has 'District Of Columbia', output 'District Of Columbia' exactly.\n"
            "- Do NOT convert 'state' to 'province' for UK or AU English.\n"
            "- When a US term is used as an adjective (e.g., 'state park'), use the adjective "
            "form of the local equivalent ('provincial park'), not the noun ('province park').\n\n"
            "If no dialect-specific spelling or vocabulary changes are needed, return the "
            "source text UNCHANGED.\n"
        )

    user_prompt = f"""
Target language code: {language_code}
Target locale code: {locale_code}

Global survey context:
{global_context}

{localization_mode_instruction}

Segment metadata for this element:
- segment_type: {segment_type_str}
- block_style (JSON, may be null): {block_style_json}

{context_instruction}

{peer_options_instruction}

{grounding_instruction}

{list_context_instruction}

{brand_name_instruction}

{gender_inclusive_instruction}

English source text:
\"\"\"{english_text}\"\"\"

Existing translation in the target language (may be empty or just a copy of the English):
\"\"\"{existing_translation}\"\"\"

Translation memory examples (English -> target translation):
{memory_str}

{"" if is_same_language_localization else """Interpretation of segment_type and block_style:
- If segment_type = "question":
    - Translate as a full, natural question in the target language, using the polite form
      that is standard for surveys in the target locale.
- If segment_type = "instruction":
    - Translate as a clear, polite imperative or directive, as a complete sentence.
      (For example: equivalents of "Select one option", "Select all that apply",
      "Enter a number", "Please describe".)
- If segment_type = "answer_option":
    - Treat this as a stand-alone answer choice shown under a question.
    - You MUST follow the block_style exactly for this answer option.
      - If grammatical_person is 'first_person' and phrase_form is 'clause': your translation
        MUST be a first-person statement (e.g., 'Quiero ver paisajes hermosos',
        'Je veux visiter des villes américaines', 'Voglio visitare i parchi nazionali').
      - If phrase_form is 'noun_phrase' or 'short_phrase': your translation MUST be a noun
        phrase or bare infinitive (e.g., 'Ver paisajes hermosos',
        'Visiter des villes américaines', 'Visitare i parchi nazionali').
      - Do NOT mix styles. If the plan says noun_phrase, do NOT produce a first-person clause.
      - INCORRECT example: block_style says noun_phrase but you write 'Quiero ir de compras'
        instead of 'Ir de compras'.
      - CORRECT example: block_style says noun_phrase, you write 'Ir de compras'.
    - It is acceptable to rewrite an existing translation to match this plan as long as you
      do NOT change the underlying meaning or distinctions.
    - Keep answer options concise and parallel in style within the block (all labels or
      all self-descriptions, not a random mix), as is natural in the target language.
    - COMPOUND OPTIONS: When an answer option contains an em-dash (\u2014), en-dash (\u2013),
      colon (:), or similar separator joining two distinct clauses, apply grammatical
      person to each clause independently based on the ENGLISH SOURCE TEXT:
      * The opt-out or categorical label part (e.g., "None of the above", "Other")
        should remain impersonal.
      * If the English source clause uses first-person pronouns (I, my, me, myself),
        that clause MUST be translated in first person REGARDLESS of block_style.
        This overrides block_style for that clause only.
      Example: English "None of the above \u2014 I do not use AI tools to plan travel"
      \u2192 Italian "Nessuna delle precedenti \u2014 Non utilizzo strumenti di IA per
      pianificare i viaggi" (first-person "utilizzo", NOT third-person "utilizza").
- If segment_type = "scale_label":
    - Treat this as a label on a rating scale (e.g., satisfaction, agreement).
    - Use short, symmetric phrases that clearly express the relative position on the
      scale (from most positive to most negative or vice versa), rather than long
      sentences or self-referential statements.
    - When an existing translation already forms a clear, well-ordered scale, avoid
      proposing changes that only reflect stylistic preferences. Only propose changes to scale
      labels when they fix problems with semantics, ordering, clarity, or obvious
      unnaturalness in the target language.
"""}
Instructions:
{"1. Only adapt dialect-specific spelling and vocabulary. If the source text is already correct for the target locale, return it unchanged. Returning the source text unchanged is the correct and expected behavior for most rows." if is_same_language_localization else "1. If the existing translation is effectively empty or simply repeats the English text, treat this as if there were no translation yet. In this case you MUST propose a high-quality translation whose main human-readable content is clearly in the target language, not in English. It is incorrect to simply copy the English sentence, except for proper names, brand names, and technical tokens."}
{"2. Do NOT change grammatical structure, person, or phrasing of the source text. Preserve the exact sentence structure, word order, and grammatical person. If the source uses a bare past tense ('Visited...'), you must keep it as a bare past tense. Do NOT add pronouns ('I visited...') or restructure in any way." if is_same_language_localization else """2. If the existing translation is non-empty and clearly already in the target language, treat it as the baseline
   and only propose changes if they improve:
   - semantic accuracy or preservation of qualifiers,
   - measurement safety (clearer distinctions or better ordered scale points),
   - structural safety (fixing issues with tags, placeholders, numbers),
   - localization (correct regional variant),
   - terminology consistency,
   - grammar, tense, and style,
   - handling of proper nouns and numeric formatting,
   - accents and punctuation.
   For scale_label elements in particular, you should NOT propose changes that only add
   stylistic variants (such as explicit gender marking or inclusive forms) when the
   existing label is already natural and forms part of a clear, symmetric scale."""}
3. Always perform a self-QA step on your proposed translation.{"" if is_same_language_localization else " If your proposed translation, after stripping HTML tags and condensing whitespace, is still essentially identical to the English text, you must reconsider and produce a real translation in the target language."}
4. You MUST NOT change or remove any HTML tags, placeholders, or piping tokens. Only {"adapt" if is_same_language_localization else "translate"} the text between them.
{"5. Do NOT change any punctuation, currency symbols, number formatting, or add any content not present in the source." if is_same_language_localization else """5. Style compliance check: Before returning your JSON, verify that your proposed_translation
   matches the block_style for this segment. If grammatical_person is 'first_person' but your
   translation does not use first-person phrasing, rewrite it. If phrase_form is 'noun_phrase'
   but your translation is a full clause, rewrite it. This check is mandatory."""}
6. Return ONLY a valid JSON object with the following keys:
   - "proposed_translation": your first-pass translation.
   - "qa_checked_translation": your self-QA revision of proposed_translation after
     checking P1-P4 above. If proposed_translation already satisfies all rules, copy
     it here exactly. If you find a rule violation and correct it, the corrected text
     goes here. This field is ALWAYS non-empty.
   - "needs_change": boolean
   - "change_reason": string (short explanation in English; empty if no change needed)

{"" if is_same_language_localization else """Very important when rewriting:
- You MUST keep all critical qualifiers from the English and any existing translation
  (for example: temporary vs permanent, unpaid vs paid, full-time vs part-time, looking
  for work vs not looking for work, disability, retired, etc.).
- You MUST preserve all tags, placeholders and piping tokens exactly.
"""}
Your response:
- Always perform a quick self-QA step before answering.
- Return ONLY a JSON object with these keys:
  {{
    "proposed_translation": "<string>",
    "qa_checked_translation": "<string>",
    "needs_change": <true or false>,
    "change_reason": "<short explanation in English; empty string if no change needed>"
  }}
"""

    # Call the model with retries
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=model_name,
                response_format={"type": "json_object"},
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                seed=TRANSLATION_SEED,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            _record_fingerprint(response)
            result = _safe_json(response)

            return {
                "proposed_translation": (result.get("proposed_translation") or "").strip(),
                "qa_checked_translation": (result.get("qa_checked_translation") or "").strip(),
                "needs_change": bool(result.get("needs_change", False)),
                "change_reason": (result.get("change_reason") or "").strip(),
                "error": False,
            }
        except Exception as e:
            if attempt == 2:
                return {
                    "error": True,
                    "change_reason": f"Error: {str(e)}",
                    "proposed_translation": existing_translation,
                    "qa_checked_translation": existing_translation,
                    "needs_change": False,
                }
            await asyncio.sleep(2 ** attempt)

    return {
        "error": True,
        "change_reason": "Unknown Error",
        "proposed_translation": existing_translation,
        "qa_checked_translation": existing_translation,
        "needs_change": False,
    }


async def call_translation_model_async(
        english_text: str,
        language_code: str,
        locale_code: str,
        global_context: str,
        translation_memory: Dict[str, Dict[str, str]],
        existing_translation: Optional[str] = None,
        segment_type: Optional[SegmentType] = None,
        block_style: Optional[BlockStyle] = None,
        peer_english_options: Optional[List[str]] = None,
        parent_context: str = "",
        gender_inclusive: bool = False,
        model_name: str = MODEL_NAME,
        answer_option_count: int = 0,
        answer_option_avg_len: float = 0.0,
        is_same_language_localization: bool = False,
) -> Dict[str, object]:
    """Per-run dedup-caching front for _call_translation_model_async_uncached.

    Identical (english_text, context-args) tuples share one model call within a
    run; the lock prevents concurrent requests for the same key from each making
    a separate call.  Cache is reset by reset_translation_cache() at run start.
    """

    def _bs_sig(bs: Optional[BlockStyle]):
        if not bs:
            return None
        return (bs.options_grammatical_person, bs.options_phrase_form,
                bs.options_tone, bs.scale_label_phrase_form)

    key: tuple = (
        english_text, language_code, locale_code,
        segment_type.value if segment_type else None,
        bool(gender_inclusive), bool(is_same_language_localization),
        _bs_sig(block_style), parent_context or "",
        tuple(peer_english_options) if peer_english_options else None,
        existing_translation or "",
        answer_option_count, round(answer_option_avg_len, 1),
        model_name,
        # global_context included so a CRITICAL-RETRY suffix (which appends to
        # global_context) always gets its own cache slot, never collides.
        global_context,
    )

    cached = _TRANSLATION_CACHE.get(key)
    if cached is not None:
        return dict(cached)

    # Create the lock inside the active event loop (per-run) so it never
    # accidentally binds to a prior run's loop.
    lock = _TRANSLATION_CACHE_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _TRANSLATION_CACHE_LOCKS[key] = lock

    async with lock:
        # Double-check: another coroutine may have populated the cache while
        # we waited for the lock.
        cached = _TRANSLATION_CACHE.get(key)
        if cached is not None:
            return dict(cached)

        result = await _call_translation_model_async_uncached(
            english_text, language_code, locale_code, global_context,
            translation_memory, existing_translation, segment_type, block_style,
            peer_english_options, parent_context, gender_inclusive, model_name,
            answer_option_count, answer_option_avg_len, is_same_language_localization,
        )

        # Cache only clean, non-English-copy output.  Errors and English-copies
        # are never cached so peers retry them independently rather than all
        # inheriting a known-bad result.
        candidate = result.get("qa_checked_translation") or result.get("proposed_translation") or ""
        if not result.get("error") and not is_effective_copy_of_english(english_text, candidate):
            _TRANSLATION_CACHE[key] = dict(result)

        return result


async def translate_scale_batch_async(
    context: SurveyFileContext,
    block: QuestionBlock,
    global_context: str,
    semaphore: asyncio.Semaphore,
    provide_suggestions: bool,
    gender_inclusive: bool = False,
    model_name: str = MODEL_NAME,
    translated_question_context: str = "",
    concept_term_english: Optional[str] = None,
) -> int:
    """
    Translate all scale labels in a block in a single LLM call so the model
    can produce a coherent, symmetric set of translations.

    Returns the number of rows that were successfully batch-translated.
    On failure, leaves batch_translated=False on all rows so they fall
    through to the row-by-row path.
    """
    if not block.scale_label_indices:
        return 0

    rows = context.rows
    # Always batch in ascending document order so write-back stays positionally
    # consistent even if scale_label_indices were appended out of order upstream.
    scale_indices = sorted(i for i in block.scale_label_indices if 0 <= i < len(rows))
    if len(scale_indices) < 2:
        return 0

    english_labels = []
    existing_translations = []
    all_have_real_translation = True
    any_have_real_translation = False

    for idx in scale_indices:
        row = rows[idx]
        eng = (row.english_text or "").strip()
        english_labels.append(eng)

        if row.had_real_translation:
            existing_translations.append((row.existing_translation or "").strip())
            any_have_real_translation = True
        else:
            existing_translations.append("")
            all_have_real_translation = False

    # If all labels already have real translations and suggestions are disabled, skip.
    if all_have_real_translation and not provide_suggestions:
        for idx in scale_indices:
            rows[idx].new_translation = rows[idx].existing_translation
            rows[idx].batch_translated = True
        return len(scale_indices)

    # Build question context
    q_texts = [
        rows[i].english_text for i in block.question_indices
        if 0 <= i < len(rows) and rows[i].english_text
    ]
    question_context = " ".join(q_texts)

    # Block style
    block_style = None
    if context.block_styles:
        block_style = context.block_styles.get(block.block_id)
    scale_phrase_form = getattr(block_style, "scale_label_phrase_form", "short_phrase")

    # Translation memory examples (use the combined English labels for relevance)
    combined_english = " ".join(english_labels)
    memory_examples = sample_translation_memory_examples(
        context.translation_memory, current_english=combined_english, max_examples=5
    )
    memory_str = "\n".join(
        [f'- "{e}" -> "{t}"' for e, t in memory_examples]
    ) if memory_examples else "None."

    gender_inclusive_instruction = build_gender_inclusive_instruction(
        context.language_code, gender_inclusive,
        segment_type="SCALE_LABEL",
    )

    brand_name_instruction = build_brand_name_instruction(context.language_code)

    labels_json = json.dumps(english_labels, ensure_ascii=False)
    existing_json = json.dumps(existing_translations, ensure_ascii=False)

    domain_fragment = build_domain_prompt_fragment(global_context)
    is_localization = context.is_same_language_localization

    _scale_vocab = DIALECT_VOCABULARY.get(context.locale_code, {})
    _scale_vocab_lines = ""
    if _scale_vocab:
        _scale_vocab_lines = "\n".join(
            f'   "{us}" -> "{local}"' for us, local in _scale_vocab.items()
        )

    if is_localization:
        _base_lang_name = context.language_code.upper() if context.language_code else "source"
        system_prompt = f"""You are a professional copy editor specializing in regional dialect and locale
adaptation for market-research questionnaires.
Domain context: {domain_fragment}
You adapt rating-scale label sets from one {_base_lang_name} variant into the {context.locale_code} locale
(e.g., US English to British English, es-ES to es-MX), changing ONLY dialect-specific
spelling and vocabulary.

The ONLY changes you are permitted to make are:
1. Spelling conventions (e.g., color/colour, center/centre, traveled/travelled).
2. Vocabulary substitutions ONLY from the approved whitelist:
{_scale_vocab_lines if _scale_vocab_lines else "   (No vocabulary substitutions — spelling changes only.)"}

If the labels are already correct for the target locale, return them UNCHANGED.

You MUST NOT:
- Change grammatical structure, person, or phrasing of any label.
- Convert currency symbols, amounts, or formatting.
- Add content not in the source (no km equivalents, no explanatory notes).
- Change number formatting (do NOT change "5 - 10" to "5-10").
- Add, remove, or change punctuation.
- Rephrase, restructure, or "improve" text beyond spelling/vocabulary.
- Correct errors, typos, or capitalization in the source text.

Maintain all labels as a coherent scale. Preserve scale polarity and intensity.
Preserve all HTML tags, placeholders, and piping tokens exactly.
Respect translation memory examples when they fit.
The 'notes' field MUST always be written in ENGLISH.
Return ONLY a valid JSON object with the required keys and no extra text."""
    else:
        system_prompt = f"""You are a professional translator for market-research questionnaires.
Domain context: {domain_fragment}
You translate rating-scale label sets from English into a specified target language and locale.

Your priorities:
1. Translate ALL labels as a single coherent scale — the set must be symmetric,
   monotonic, and use consistent vocabulary and grammatical structure throughout.
2. Use short, natural phrases appropriate for survey scale labels in the target locale.
   Avoid long sentences or self-referential statements. Use natural, idiomatic phrasing
   that a native speaker would use in everyday language — avoid overly literal or
   institutional translations. Target consumer-survey register, not legal or formal.
3. Preserve scale polarity and intensity (e.g., if the English goes from very positive
   to very negative, the target language set must do the same).
4. Translate the EXACT meaning of each English label. Do NOT reinterpret or adapt the
   concept based on the question stem. If the English label says 'comfortable', translate
   as 'comfortable/at ease', NOT as 'interested' or 'willing'.
5. Preserve all HTML tags, placeholders, and piping tokens exactly.
6. Respect translation memory examples when they fit.
7. The 'notes' field MUST always be written in ENGLISH.
8. Return ONLY a valid JSON object with the required keys and no extra text."""

    if all_have_real_translation:
        task_instruction = (
            "All labels already have existing translations. Review them as a SET for "
            "consistency, symmetry, and naturalness. If they already form a clear, "
            "well-ordered scale, return them unchanged. Only propose changes when there "
            "are problems with consistency across the set, semantic accuracy, ordering, "
            "clarity, or obvious unnaturalness."
        )
    elif any_have_real_translation:
        task_instruction = (
            "Some labels have existing translations and some do not (shown as empty strings). "
            "Fill in the missing translations so they match the style and vocabulary of "
            "the existing ones, forming a coherent set. You may also adjust existing "
            "translations if needed for set-wide consistency."
        )
    else:
        task_instruction = (
            "None of these labels have been translated yet. Produce a complete, coherent "
            "set of translations that form a natural rating scale in the target language."
        )

    translated_q_section = ""
    if translated_question_context:
        concept_constraint = ""
        if concept_term_english:
            concept_constraint = (
                f'\nThe key concept being measured is "{concept_term_english}". '
                f'Your translations MUST use the same root/word family that the '
                f'translated question above uses for this concept. '
                f'Do NOT substitute a synonym.'
            )
        translated_q_section = (
            f'\nTranslated question (target language):\n'
            f'"""{translated_question_context}"""\n\n'
            f'IMPORTANT: Your scale label translations MUST use vocabulary that is\n'
            f'semantically consistent with the translated question above. If the\n'
            f'question uses a specific word for the concept being measured (e.g.,\n'
            f'"comfortable", "satisfied", "likely"), your scale labels must echo\n'
            f'that same word or word family — do NOT use a different synonym.'
            + concept_constraint
        )

    localization_note = ""
    if is_localization:
        _scale_vocab_note = ""
        if _scale_vocab:
            _sv_lines = ", ".join(f"{us}->{local}" for us, local in _scale_vocab.items())
            _scale_vocab_note = (
                f"Approved vocabulary whitelist: {_sv_lines}. "
                "Only convert terms in this list.\n"
            )
        localization_note = (
            "\nDIALECT ADAPTATION MODE — STRICT RULES:\n"
            "Source and target are both English. Only adapt dialect-specific spelling "
            "and vocabulary from the approved whitelist. If labels are already correct "
            "for the target locale, return them UNCHANGED.\n"
            + _scale_vocab_note
            + "Do NOT change grammatical structure, punctuation, currency symbols, "
            "number formatting, or add any content not in the source.\n"
            "Do NOT correct errors, typos, or capitalization in the source text.\n"
        )

    user_prompt = f"""Target language code: {context.language_code}
Target locale code: {context.locale_code}

Global survey context:
{global_context}
{localization_note}
Question this scale belongs to (English):
\"\"\"{question_context}\"\"\"
{translated_q_section}

Scale label style: phrase_form = {scale_phrase_form}

{brand_name_instruction}

{gender_inclusive_instruction}

English scale labels (ordered):
{labels_json}

Existing translations (ordered, empty string means no translation yet):
{existing_json}

Translation memory examples (English -> target translation):
{memory_str}

Task:
{task_instruction}

Return ONLY a JSON object:
{{
  "translations": ["<label_1>", "<label_2>", ...],
  "needs_changes": [true/false, true/false, ...],
  "notes": "<short English explanation of approach>"
}}

CRITICAL: The "translations" array MUST have exactly {len(english_labels)} elements,
one for each English label, in the same order."""

    async with semaphore:
        client = get_async_client()
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    response_format={"type": "json_object"},
                    seed=TRANSLATION_SEED,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                _record_fingerprint(response)
                data = _safe_json(response)
                translations = data.get("translations", [])

                if not isinstance(translations, list) or len(translations) != len(scale_indices):
                    raise ValueError(
                        f"Expected {len(scale_indices)} translations, got {len(translations)}"
                    )

                needs_changes = data.get("needs_changes", [True] * len(translations))
                if not isinstance(needs_changes, list) or len(needs_changes) != len(translations):
                    needs_changes = [True] * len(translations)

                notes = data.get("notes", "")

                # Phase 1: stage every label — never commit partially.
                # A single empty or structurally-invalid label causes the whole
                # set to fall back to the row-by-row path (set_incomplete=True).
                staged: Dict[int, str] = {}
                set_incomplete = False
                for i, idx in enumerate(scale_indices):
                    row = rows[idx]
                    proposed = (translations[i] or "").strip()
                    if not proposed:
                        set_incomplete = True
                        continue
                    eng = (row.english_text or "").strip()
                    proposed = adjust_capitalization_for_label(eng, proposed, context.language_code)
                    is_ok, msg = validate_translation_structure(eng, proposed)
                    if not is_ok:
                        repaired = attempt_placeholder_repair(eng, proposed)
                        if repaired and repaired != proposed:
                            re_valid, _ = validate_translation_structure(eng, repaired)
                            if re_valid:
                                proposed, is_ok = repaired, True
                        if not is_ok:
                            set_incomplete = True
                            continue
                    staged[idx] = proposed

                # Phase 2: atomic commit or flag-and-fallback.
                if set_incomplete or len(staged) != len(scale_indices):
                    for idx in scale_indices:
                        rows[idx].suggestion_reason = (
                            ((rows[idx].suggestion_reason + " | ") if rows[idx].suggestion_reason else "")
                            + "Scale set incomplete — review whole scale."
                        )
                    return 0  # batch_translated stays False -> coherent row-by-row fallback

                translated_count = 0
                for i, idx in enumerate(scale_indices):
                    row = rows[idx]
                    proposed = staged[idx]
                    if not row.had_real_translation:
                        row.new_translation = proposed
                        row.was_newly_translated = True
                    else:
                        row.new_translation = row.existing_translation
                        if needs_changes[i] and proposed != row.existing_translation:
                            if provide_suggestions:
                                row.suggested_translation = proposed
                                detail = notes if notes else "adjusted for set consistency"
                                row.suggestion_reason = (
                                    (row.suggestion_reason or "") +
                                    f"Batch scale review: {detail}."
                                )
                            else:
                                row.new_translation = proposed
                    row.batch_translated = True
                    translated_count += 1

                # Monotonicity guard (advisory): map each scale position to its
                # English intensity rank.  Only evaluated when EVERY label carries
                # a recognized qualifier (a confident intensity scale); otherwise
                # skipped to avoid false positives on non-intensity sets.
                eng_ranks = [english_intensity_rank(rows[i].english_text) for i in scale_indices]
                if all(r is not None for r in eng_ranks) and len(eng_ranks) >= 2:
                    non_decr = all(eng_ranks[k] <= eng_ranks[k + 1] for k in range(len(eng_ranks) - 1))
                    non_incr = all(eng_ranks[k] >= eng_ranks[k + 1] for k in range(len(eng_ranks) - 1))
                    if not (non_decr or non_incr):
                        for idx in scale_indices:
                            rows[idx].qa_status = (
                                (rows[idx].qa_status or "")
                                + " | Scale intensity order is non-monotonic by English rank \u2014 verify scale point order."
                            )

                # Step 24: concept-term post-check (flag-only, non-destructive).
                # If the concept word is present in any English label, verify it
                # also appears (loosely) in the translated question context; if not,
                # annotate qa_status for the whole set.
                if concept_term_english and translated_question_context:
                    if not re.search(re.escape(concept_term_english), translated_question_context, re.IGNORECASE):
                        for idx in scale_indices:
                            context.rows[idx].qa_status = (
                                (context.rows[idx].qa_status or "")
                                + f" | Concept-term '{concept_term_english}' not detected in translated question — verify scale label vocabulary."
                            )

                return translated_count

            except Exception as e:
                if attempt == 2:
                    return 0
                await asyncio.sleep(2 ** attempt)

    return 0


def build_prompt_base_header(role_line: str, global_context: str) -> str:
    """Shared 2-line prompt header: role + domain context fragment.
    Keeps role/domain wording consistent across prompts without duplicating it."""
    return f"{role_line}\nDomain context: {build_domain_prompt_fragment(global_context)}\n"


async def infer_style_for_block_async(
    context: SurveyFileContext,
    block: QuestionBlock,
    global_context: str,
    semaphore: asyncio.Semaphore,
    model_name: str = MODEL_NAME,
) -> BlockStyle:
    """Async version of infer_style_for_block."""
    async with semaphore:
        client = get_async_client()
        rows = context.rows

        def get_texts(indices: List[int]) -> List[str]:
            texts: List[str] = []
            for i in indices:
                if 0 <= i < len(rows):
                    t = (rows[i].english_text or "").strip()
                    if t:
                        texts.append(t)
            return texts

        question_text = " ".join(get_texts(block.question_indices))
        instruction_texts = get_texts(block.instruction_indices)
        option_texts = get_texts(block.answer_option_indices)
        scale_label_texts = get_texts(block.scale_label_indices)

        block_data = {
            "block_id": block.block_id,
            "question_text": question_text,
            "instructions": instruction_texts,
            "options": option_texts,
            "scale_labels": scale_label_texts,
        }
        block_json = json.dumps(block_data, ensure_ascii=False)

        # Style inference is a CLASSIFICATION task only — it must not translate.
        # The prompt is stripped to a classifier so the model spends no tokens on
        # (and is not biased toward) producing translated text here.
        system_prompt = (
            build_prompt_base_header(
                "You are a survey-methodology classifier for market research questionnaires.",
                global_context,
            )
            + "You DO NOT translate. You only inspect one English question block and label its "
            "presentation style, so a later translation step can match it.\n\n"
            "Classify, using only these dimensions:\n"
            "- grammatical_person of the answer options: first_person | third_person | impersonal | unspecified.\n"
            "- phrase_form of the answer options: clause | noun_phrase | short_phrase | unspecified.\n"
            "- tone of the answer options: formal_neutral | casual_neutral | other.\n"
            "- phrase_form of the scale labels: short_phrase | noun_phrase | clause | unspecified.\n\n"
            "Rules:\n"
            "- Base every label on the ENGLISH source only; do not output any translated text.\n"
            "- Scale labels are almost always short, symmetric phrases, not full sentences.\n"
            "- Any explanation in the `notes` field MUST be in English.\n"
            "- Return ONLY valid JSON with the required keys and no extra commentary."
        )

        user_prompt = f"""
    Target language code: {context.language_code}
    Target locale code: {context.locale_code}

    Global context:
    {global_context}

    Here is one question block from the English source survey, expressed as JSON:
    {block_json}

    Guidance:
    - Look at the English question text and the list of options to infer what kind of thing is being asked.
    - If the options clearly describe the respondent themselves (their status, identity, situation, behavior, or attitudes),
      and it is natural in the target locale to answer with self-descriptions, you may choose "first_person" and "clause"
      for options. Typical English questions of this type include:
        - "Which best describes you?"
        - "What is your current employment status?"
        - "Which of the following statements best applies to you?"
    - If the options are better presented as short labels (e.g. brand names, countries, job titles, industries, or generic
      categories that are not self-statements), prefer "third_person" or "impersonal" with "noun_phrase" or "short_phrase".
    - Scale labels (e.g. "Very satisfied" to "Very dissatisfied") should almost always be short, symmetric phrases,
      not full self-referential sentences. For 5-point satisfaction or evaluation scales, short phrases equivalent to
      "Very good / Good / Neutral / Poor / Very poor" are preferred over long sentences.

    Return ONLY a JSON object of the form:
    {{
      "block_id": <int>,
      "options_style": {{
        "grammatical_person": "<first_person|third_person|impersonal|unspecified>",
        "phrase_form": "<clause|noun_phrase|short_phrase|unspecified>",
        "tone": "<formal_neutral|casual_neutral|other>"
      }},
      "scale_label_style": {{
        "phrase_form": "<short_phrase|noun_phrase|clause|unspecified>"
      }},
      "notes": "<short English explanation or empty string>"
    }}
    """

        max_retries = 3
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    response_format={"type": "json_object"},
                    seed=TRANSLATION_SEED,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                _record_fingerprint(response)
                data = _safe_json(response)

                options_style = data.get("options_style") or {}
                scale_style = data.get("scale_label_style") or {}

                return BlockStyle(
                    block_id=block.block_id,
                    options_grammatical_person=options_style.get("grammatical_person", "unspecified"),
                    options_phrase_form=options_style.get("phrase_form", "unspecified"),
                    options_tone=options_style.get("tone", "formal_neutral"),
                    scale_label_phrase_form=scale_style.get("phrase_form", "short_phrase"),
                    notes=data.get("notes", "") or "",
                )
            except Exception as e:
                last_exception = e
                status_code = getattr(e, "status_code", None)
                message = str(e).lower()
                is_rate_limit = (status_code == 429) or ("rate limit" in message)
                is_server_error = status_code is not None and 500 <= status_code < 600
                if (is_rate_limit or is_server_error) and attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                break

        return BlockStyle(
            block_id=block.block_id,
            options_grammatical_person="unspecified",
            options_phrase_form="unspecified",
            options_tone="formal_neutral",
            scale_label_phrase_form="short_phrase",
            notes=f"Style inference failed: {last_exception}",
        )


async def infer_block_styles_async(
    context: SurveyFileContext,
    global_context: str,
    semaphore: asyncio.Semaphore,
    model_name: str = MODEL_NAME,
) -> Dict[int, BlockStyle]:
    """Async version of infer_block_styles. Fires all block inferences concurrently."""
    block_styles: Dict[int, BlockStyle] = {}

    if not context.blocks:
        context.block_styles = block_styles
        return block_styles

    tasks = []
    blocks_to_infer = []

    for block in context.blocks:
        if block.answer_option_indices or block.scale_label_indices:
            tasks.append(
                infer_style_for_block_async(context, block, global_context, semaphore, model_name)
            )
            blocks_to_infer.append(block)

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for block, result in zip(blocks_to_infer, results):
            if isinstance(result, BlockStyle):
                block_styles[block.block_id] = result

    context.block_styles = block_styles
    return block_styles


def call_consistency_model(
    context: SurveyFileContext,
    phrase_groups: List[Dict[str, object]],
    global_context: str = "",
    model_name: str = MODEL_NAME,
) -> List[Dict[str, object]]:
    """
    LLM-powered survey-wide consistency checker.

    phrase_groups is a list of objects like:
      {
        "english_phrase": "...",
        "translations": [
          {"translation": "...", "indices": [0, 3, 7]},
          ...
        ]
      }

    Returns a list of issues:
      {
        "english_phrase": "...",
        "canonical_translation": "...",
        "indices_to_update": [0, 3],
        "notes": "optional explanation (ALWAYS in English)"
      }
    """
    if not phrase_groups:
        return []

    client = get_llm_client()

    groups_json = json.dumps(phrase_groups, ensure_ascii=False)

    domain_fragment = build_domain_prompt_fragment(global_context)
    system_prompt = (
        "You are a localization QA and terminology consistency specialist for market research questionnaires.\n"
        f"Domain context: {domain_fragment}\n"
        "You review translated survey text in the target language "
        "and recommend when the same English phrase should use a single canonical translation.\n\n"
        "Key requirements:\n"
        "- Respect the specified target language and locale.\n"
        "- Only unify translations when they clearly refer to the SAME concept in this survey context.\n"
        "- If the same English phrase may be used in different senses (polysemy), be cautious: either do "
        "  not unify, or explain the nuance in your notes.\n"
        "- Do NOT invent row indices; only use those provided in the input data.\n"
        "- Any explanations you provide in the `notes` field MUST always be written in English (even if the "
        "  target language is different).\n"
        "- Always return valid JSON with the required keys and no extra commentary."
    )

    user_prompt = f"""
Target language code: {context.language_code}
Target locale code: {context.locale_code}

You are given a JSON array called "groups". Each element has:
- "english_phrase": the original English text.
- "translations": an array of objects:
    - "translation": the current translation in the target language.
    - "indices": the row indices in the survey where this translation is used.

Your task:
1. For each english_phrase, look at its different translations.
2. Decide if they should share a single canonical translation in the target language.
3. If YES, choose the best canonical translation and list the row indices that should be updated.
4. If NO (because context likely differs), either skip that phrase or use an empty indices_to_update list and explain why in notes.

Return ONLY a JSON object like:
{{
  "issues": [
    {{
      "english_phrase": "...",
      "canonical_translation": "...",
      "indices_to_update": [list of integer row indices to update],
      "notes": "short explanation or rationale in English"
    }},
    ...
  ]
}}

Here is the JSON data for "groups":
{groups_json}
"""

    max_retries = 3
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                # GPT-5 models don’t support temperature; rely on default decoding.
                response_format={"type": "json_object"},
                seed=TRANSLATION_SEED,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            _record_fingerprint(response)
            data = _safe_json(response)
            issues = data.get("issues", [])
            if not isinstance(issues, list):
                raise ValueError("`issues` must be a list in consistency model response.")
            return issues
        except Exception as e:
            last_exception = e
            status_code = getattr(e, "status_code", None)
            message = str(e).lower()

            is_rate_limit = (status_code == 429) or ("rate limit" in message)
            is_server_error = status_code is not None and 500 <= status_code < 600

            if (is_rate_limit or is_server_error) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                break

    # Non-retryable failure or retries exhausted: degrade to no-op rather than
    # killing the run. The consistency pass is best-effort; a hard failure here
    # must never prevent the output file from being written and persisted.
    return []


# ==========================
# Safe JSON Parsing
# ==========================

class _RetryableModelError(Exception):
    """Raised by _safe_json when the model response is null, truncated, or
    malformed JSON.  Because it inherits from Exception, every existing
    ``except Exception`` retry loop in the codebase will catch it and retry
    without any additional changes."""


def _safe_json(response) -> dict:
    """Null-check, code-fence strip, and finish_reason guard around model JSON.

    Raises _RetryableModelError for every recoverable failure so the caller's
    existing retry loop re-tries transparently.  Never swallows errors silently.
    """
    try:
        choice = response.choices[0]
    except (AttributeError, IndexError, TypeError) as exc:
        raise _RetryableModelError(f"No choices in model response: {exc}") from exc

    finish_reason = getattr(choice, "finish_reason", None)
    content = getattr(getattr(choice, "message", None), "content", None)

    if content is None:
        raise _RetryableModelError(
            f"Null content from model (finish_reason={finish_reason!r})"
        )
    if finish_reason == "length":
        raise _RetryableModelError(
            "Response truncated by token limit (finish_reason='length'); will retry."
        )

    text = content.strip()
    if text.startswith("```"):
        # Strip optional language tag (```json) and closing fence
        text = re.sub(r"^```[A-Za-z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise _RetryableModelError(f"Malformed JSON from model: {exc}") from exc


# ==========================
# Structure Validation
# ==========================

def extract_numeric_tokens(text: str) -> List[str]:
    if not text:
        return []
    # Rough pattern for numbers with optional currency and percent
    pattern = r'[\$€£¥]?\d+(?:[.,]\d+)?%?'
    return re.findall(pattern, text)


# Shared placeholder-token regex: no-space inner content so prose phrases like
# "[see below]" are NOT treated as piping tokens.  Used by both the validator
# (extract_placeholder_tokens) and the repairer (attempt_placeholder_repair) so
# they are always in lockstep and cannot diverge on what counts as a token.
_PLACEHOLDER_TOKEN_RE = re.compile(
    r'\[\[[^\]\s]+\]\]'            # [[VAR]]   - double-bracket, no spaces
    r'|\{[^}\s]+\}'               # {token}   - curly brace, no spaces
    r'|\[[^\]\s]+\]'              # [token]   - single bracket, no spaces
    r'|\$[A-Za-z][A-Za-z0-9_]+'   # $VAR / $RESP
)


def extract_placeholder_tokens(text: str) -> List[str]:
    if not text:
        return []
    return sorted(set(_PLACEHOLDER_TOKEN_RE.findall(text)))


# HTML-tag fidelity: match opening and closing tags (e.g. <b>, </b>, <br>, <a href="...">).
# Whitespace inside tags is normalized so minor formatting differences don't count as distinct.
_TAG_RE = re.compile(r'</?[A-Za-z][^>]*>')


def extract_html_tags(text: str) -> List[str]:
    """Return a sorted list of HTML tags found in *text* (whitespace-normalized).

    Tags are not deduplicated so that Counter arithmetic gives correct multiset
    comparisons (e.g. two <b> tags in the source must have two in the translation).
    """
    if not text:
        return []
    return sorted(re.sub(r'\s+', ' ', t).strip() for t in _TAG_RE.findall(text))


def sanitize_reviewer_note(note: str) -> str:
    """Reviewer-facing rationale (change_reason/notes) is required to be English.
    When a model returns it in a non-Latin target script (CJK, Arabic, Devanagari,
    Cyrillic, etc.), prefix a visible flag so the English-reading reviewer is not
    silently handed untranslatable text.  Non-destructive: content is preserved.
    Language-agnostic (inspects Unicode script of the note itself, not a lang code).
    """
    if not note:
        return note
    alpha = [c for c in note if c.isalpha()]
    if not alpha:
        return note
    non_latin = sum(1 for c in alpha if ord(c) > 0x024F)  # beyond Latin Extended-B
    if non_latin / len(alpha) > 0.30 and "[Rationale not in English" not in note:
        return "[Rationale not in English \u2014 verify] " + note
    return note


def validate_translation_structure(english_text: str, translation_text: str) -> Tuple[bool, str]:
    """
    Validate that numeric/currency tokens and placeholder/piping tokens from the English
    text appear in the translation (allowing for locale-specific separators for numbers).

    Returns (is_valid, message).
    """
    english_text = english_text or ""
    translation_text = translation_text or ""

    if not english_text.strip() or not translation_text.strip():
        return True, ""

    # Numeric tokens: compare as a MULTISET of separator-normalized digit groups
    # so "100" no longer "passes" against "1000", and dropped duplicates are caught.
    def _digit_groups(text: str) -> List[str]:
        groups: List[str] = []
        for tok in extract_numeric_tokens(text):
            d = re.sub(r"\D", "", tok)
            if d:
                groups.append(d)
        return groups

    eng_groups = _digit_groups(english_text)
    trl_groups = _digit_groups(translation_text)
    # Counter subtraction: elements in eng_groups not (fully) covered by trl_groups
    missing_numeric = list((Counter(eng_groups) - Counter(trl_groups)).elements())

    # Range order: catch reversed ranges such as "18-34" → "34-18".
    # Only hyphen/en-dash/em-dash separators are considered; decimal points inside
    # numbers are stripped before comparison, so "1.5-2.5" → (15, 25) pair.
    # Known limitation: collapsing decimals means 1.5 and 15 look identical;
    # this is an acceptable trade-off for survey text where integer ranges dominate.
    _RANGE_RE = re.compile(r"(\d[\d.,]*)\s*[-\u2013\u2014]\s*(\d[\d.,]*)")

    def _range_pairs(text: str) -> List[tuple]:
        pairs: List[tuple] = []
        for a, b in _RANGE_RE.findall(text):
            ai = re.sub(r"\D", "", a)
            bi = re.sub(r"\D", "", b)
            if ai and bi:
                pairs.append((int(ai), int(bi)))
        return pairs

    trl_pairset = set(_range_pairs(translation_text))
    reversed_ranges = [
        f"{a}-{b}"
        for (a, b) in _range_pairs(english_text)
        if (a, b) not in trl_pairset and (b, a) in trl_pairset
    ]

    # Placeholder/tokens
    eng_placeholders = extract_placeholder_tokens(english_text)
    missing_placeholders = [tok for tok in eng_placeholders if tok not in translation_text]

    # HTML-tag fidelity (advertised in the app description; language-agnostic advisory flag)
    eng_tags = extract_html_tags(english_text)
    trl_tags = extract_html_tags(translation_text)
    missing_tags = list((Counter(eng_tags) - Counter(trl_tags)).elements())

    if missing_numeric or reversed_ranges or missing_placeholders or missing_tags:
        parts = []
        if missing_numeric:
            parts.append("numerics " + ", ".join(missing_numeric))
        if reversed_ranges:
            parts.append("reversed ranges " + ", ".join(reversed_ranges))
        if missing_placeholders:
            parts.append("placeholders " + ", ".join(missing_placeholders))
        if missing_tags:
            parts.append("HTML tags " + ", ".join(missing_tags))
        msg = "Missing or altered " + " and ".join(parts) + " compared to the English source."
        return False, msg

    return True, ""


def audit_shipped_rows(context: "SurveyFileContext") -> int:
    """Read-only structural audit of the FINAL shipped value of every row,
    including preserved and dialect/batch-skipped rows.

    Writes row.qa_status; NEVER mutates row text or new_translation.
    Language-agnostic (reuses validate_translation_structure which covers
    numerics, ranges, placeholders, and HTML tags).
    Returns the number of rows flagged.
    """
    flagged = 0
    for row in context.rows:
        eng = (row.english_text or "")
        final = row.new_translation if row.new_translation is not None else row.existing_translation
        final = str(final) if final is not None else ""
        if not eng.strip() or not final.strip():
            row.qa_status = ""
            continue
        ok, msg = validate_translation_structure(eng, final)
        row.qa_status = "" if ok else f"REVIEW: {msg}"
        if not ok:
            flagged += 1
    return flagged


def strip_question_punctuation_from_options(context: "SurveyFileContext") -> int:
    """
    Post-processing safety net: remove question-mark punctuation from rows
    classified as ANSWER_OPTION or SCALE_LABEL, since these should never
    be rendered as questions in the survey interface.

    Handles standard '?', fullwidth '？', Arabic '؟', and Spanish '¿...?' patterns.
    Also applies to suggested_translation (Deep L7).
    Returns the number of new_translation cells modified.
    """
    def _strip_q(trl: str) -> str:
        # Strip trailing question marks (standard, fullwidth, Arabic) with optional
        # preceding non-breaking space (French typography)
        trl = re.sub(r'[\u00A0\s]*[?？؟]\s*$', '', trl)
        # Strip leading inverted question mark (Spanish)
        trl = re.sub(r'^\s*¿\s*', '', trl)
        return trl

    fixed = 0
    for row in context.rows:
        if row.segment_type not in (SegmentType.ANSWER_OPTION, SegmentType.SCALE_LABEL):
            continue

        trl = row.new_translation or row.existing_translation or ""
        if trl.strip():
            new_trl = _strip_q(trl)
            if new_trl != trl:
                row.new_translation = new_trl
                row.suggestion_reason = (
                    (row.suggestion_reason or "")
                    + " | Auto-fix: removed question punctuation from answer option."
                )
                fixed += 1

        # Deep L7: same fix on the suggested_translation candidate.
        if row.suggested_translation and row.suggested_translation.strip():
            new_sugg = _strip_q(row.suggested_translation)
            if new_sugg != row.suggested_translation:
                row.suggested_translation = new_sugg

    return fixed


def preserve_source_punctuation(context: "SurveyFileContext") -> int:
    """
    Dialect-adaptation post-processing: ensure the translation's trailing
    punctuation matches the source exactly.  Catches cases where the LLM
    drops a period or adds one that wasn't there.

    Fixes (Deep L6):
    - Removes EXACTLY ONE terminal punctuation character (not a whole run).
    - Treats ellipsis (\u2026) as equivalent to a period so \u2026 \u2261 . match.
    - Skips rows whose translation ends in ) ] } \u00bb \" \u2019 \u201d (closing
      brackets/quotes) since stripping them would corrupt the meaning.

    Also applies the same correction to suggested_translation (Deep L7).
    Only runs for rows that were newly translated (not batch-skipped).
    Returns the number of new_translation cells corrected.
    """
    _PUNCT = set(".!,;?\u2026")
    _SKIP_ENDINGS = set(")]}\u00bb\"'\u201d\u2019")

    def _norm_end(ch: str) -> str:
        return "." if ch == "\u2026" else ch

    def _fix_terminal_punct(src: str, trl: str) -> str:
        if not src.strip() or not trl.strip():
            return trl
        src_plain = re.sub(r"<[^>]+>", "", src).rstrip()
        trl_plain = re.sub(r"<[^>]+>", "", trl).rstrip()
        if not src_plain or not trl_plain:
            return trl

        src_end = src_plain[-1]
        trl_end = trl_plain[-1]

        if _norm_end(src_end) == _norm_end(trl_end):
            return trl                          # \u2026 \u2261 . counts as a match
        if trl_end in _SKIP_ENDINGS:
            return trl                          # don't touch ) ] } \u00bb " ' endings

        # Source ends with terminal punctuation, translation doesn't -> append it.
        if src_end in _PUNCT and trl_end not in _PUNCT:
            return trl.rstrip() + src_end

        # Source has no terminal punctuation, translation does -> remove EXACTLY ONE.
        if src_end not in _PUNCT and trl_end in _PUNCT:
            stripped = trl.rstrip()
            return stripped[:-1]

        return trl

    fixed = 0
    for row in context.rows:
        if row.batch_translated:
            continue
        src = (row.english_text or "")

        new_trl = _fix_terminal_punct(src, row.new_translation or "")
        if new_trl != (row.new_translation or ""):
            row.new_translation = new_trl
            fixed += 1

        # Deep L7: apply the same correction to suggested_translation.
        if row.suggested_translation:
            new_sugg = _fix_terminal_punct(src, row.suggested_translation)
            if new_sugg != row.suggested_translation:
                row.suggested_translation = new_sugg

    return fixed


# Post-processor registry.  Each entry: (phase, name, fn, condition).
# phase: int 1-6 (6 = terminal-punctuation, always runs last).
# condition: a callable(context) -> bool, or None (always runs).
# fn: callable(context) -> int (returns rows-modified count).
# Defined here so all referenced functions (strip_question_punctuation_from_options,
# preserve_source_punctuation) are already in scope.
_POST_PROCESSORS: List[Tuple[int, str, object, object]] = [
    (1, "strip_question_punct",  strip_question_punctuation_from_options, None),
    (2, "dialect_spelling",      apply_dialect_spelling_corrections,      lambda c: c.is_same_language_localization),
    (4, "zh_true_false",         _apply_zh_true_false,                    None),
    (4, "ja_year_suffix",        _apply_ja_year_suffix,                   None),
    (4, "fr_number_format",      _apply_fr_number_format,                 None),
    (5, "emphasis_caps_flag",       _flag_emphasis_caps,           None),
    (5, "gender_marker_in_scale",   _flag_gender_marker_in_scale,  None),
    (6, "preserve_source_punct",    preserve_source_punctuation,   lambda c: c.is_same_language_localization),
]


def run_post_processors(
    context: "SurveyFileContext",
    phases: Optional[List[int]] = None,
    status_fn: Optional[object] = None,
) -> Dict[str, int]:
    """
    Run all registered post-processors whose condition is satisfied.
    When phases is given, only run processors in those phases.
    status_fn(msg) is called for non-zero results (optional, for UI feedback).
    Returns {name: rows_modified} for every processor that ran.
    """
    results: Dict[str, int] = {}
    for phase, name, fn, condition in _POST_PROCESSORS:
        if phases is not None and phase not in phases:
            continue
        if condition is not None and not condition(context):
            continue
        count = fn(context)
        results[name] = count
        if count and status_fn is not None:
            status_fn(f"Post-processing ({name}): {count} row(s) updated.")
    return results


def validate_abbreviation_preservation(english_text: str, translation_text: str) -> Tuple[bool, str]:
    """
    Check that 'Full Name (ABBREVIATION)' patterns from the English source
    are not reduced to just the abbreviation in the translation.

    Returns (is_valid, message).
    """
    if not english_text or not translation_text:
        return True, ""

    # Find patterns like "Full Name (ABBREV)" where ABBREV is 2-6 uppercase letters
    pattern = re.compile(r'([A-Z][A-Za-z\s]+?)\s*\(([A-Z]{2,6})\)')
    matches = pattern.findall(english_text)

    issues = []
    # A parenthetical containing 2+ letters anywhere in the translation is taken as
    # a (possibly localized) abbreviation form, e.g. EN "(WHO)" -> ES "(OMS)".
    any_paren_abbr = re.search(r'\([^)]*[A-Za-z]{2,}[^)]*\)', translation_text)
    for full_name, abbrev in matches:
        paren_pattern = re.compile(r'\([^)]*' + re.escape(abbrev) + r'[^)]*\)')
        has_exact_paren = bool(paren_pattern.search(translation_text))
        if has_exact_paren:
            continue  # parenthetical abbreviation preserved verbatim -> fine

        if abbrev in translation_text:
            # The bare abbreviation survived but the parenthetical wrapper did not:
            # the full descriptive name was likely dropped.
            issues.append(
                f"'{full_name} ({abbrev})' may have been shortened "
                f"to just '{abbrev}' — the full name should be preserved."
            )
        elif not any_paren_abbr:
            # Neither the abbreviation nor ANY parenthetical abbreviation survived:
            # the '(ABBREV)' parenthetical was dropped entirely. (Previously this
            # case hit `continue` and was never flagged — the inverted guard.)
            issues.append(
                f"'{full_name} ({abbrev})' — the parenthetical '({abbrev})' appears "
                f"to have been dropped from the translation."
            )

    if issues:
        return False, " | ".join(issues)
    return True, ""


def attempt_placeholder_repair(english_text: str, translation: str) -> str:
    """
    If the translation is missing EXACTLY ONE placeholder token from the English
    source AND there is a confident comma-gap anchor, re-insert it there.

    Uses the same _PLACEHOLDER_TOKEN_RE as extract_placeholder_tokens so validator
    and repairer are always in lockstep.  When more than one token is missing, or
    when no reliable anchor exists, returns the translation unmutated so the caller
    can flag the row without shipping a wrong positional guess.
    Language-agnostic.
    """
    translation = translation or ""
    eng_tokens = _PLACEHOLDER_TOKEN_RE.findall(english_text or "")
    missing = [t for t in eng_tokens if t not in translation]

    # Only auto-repair a single missing token with a confident neighbor anchor.
    if len(missing) != 1:
        return translation                   # leave unmutated -> caller flags

    placeholder = missing[0]
    # Anchor: a gap like "avec , quelle" where the LLM translated around the token
    gap_match = re.search(r'(\S)\s{0,2},\s+(?=\S)', translation)
    if gap_match:
        insert_pos = gap_match.start() + len(gap_match.group(1))
        return translation[:insert_pos] + f" {placeholder}" + translation[insert_pos:]

    # No reliable anchor -> leave unmutated rather than guessing a position
    return translation


# ==========================
# Block-level Style QA
# ==========================

def supports_first_person_detection(language_code: str) -> bool:
    """Return True iff first-person style can be reliably detected for this language.
    Reads from LANGUAGE_CAPABILITIES registry (Step 19)."""
    return bool(_get_lang_cap(language_code).get("first_person"))

# Fraction of a block's options that must already match the expected pattern before any
# per-row restyle is enforced (written to new_translation).  Below the threshold, restyle
# results go to suggested_translation only so human review is required.
_RESTYLE_CONSENSUS_THRESHOLD = float(os.getenv("RESTYLE_CONSENSUS_THRESHOLD", "0.70"))


def get_first_person_regexes(language_code: str) -> List[re.Pattern]:
    """
    Very lightweight heuristics for detecting first-person-like phrases
    in major languages. Used only for QA / style pattern detection.
    """
    lc = (language_code or "").lower()

    patterns: List[str] = []
    if lc.startswith("en"):
        # Unambiguous subject pronouns + common contractions.  Possessive "my" is
        # usually a reliable first-person signal in survey answer options.
        patterns = [
            r"\bi\b", r"\bi'm\b", r"\bi am\b", r"\bi've\b", r"\bi'd\b",
            r"\bmy\b", r"\bmyself\b",
        ]
    elif lc.startswith("es"):
        # Keep unambiguous subject pronoun + finite first-person verbs + clitic+verb.
        # Drop standalone \bme\b, \bmi\b, \bmis\b, \bmí\b — these are
        # object/possessive forms that appear in non-first-person noun phrases
        # (e.g. "mi país") and produce false positives.
        patterns = [
            r"\byo\b", r"\bsoy\b", r"\bestoy\b", r"\btrabajo\b", r"\btengo\b",
            r"\bme\s+\w+",   # clitic before verb: "me gusta", "me parece"
        ]
    elif lc.startswith("fr"):
        # Keep the unambiguous subject pronoun forms and compound patterns.
        # Drop standalone \bme\b, \bma\b, \bmon\b, \bmes\b, \bm['']\b — these
        # are object/possessive and fire on non-first-person phrases.
        patterns = [
            r"\bje\b", r"\bj['\u2019]",
            r"\bje suis\b", r"\bje me\b", r"\bje m['\u2019]",
        ]
    elif lc.startswith("pt"):
        # Keep unambiguous subject pronoun + finite first-person verbs + clitic+verb.
        # Drop standalone \bme\b and bare possessives (meu/minha/meus/minhas).
        patterns = [
            r"\beu\b", r"\bsou\b", r"\bestou\b", r"\btrabalho\b",
            r"\bme\s+\w+",   # clitic before verb: "me sinto", "me preocupo"
        ]
    elif lc.startswith("de"):
        # Subject pronoun "ich" is unambiguous; possessives are also fairly reliable
        # in German survey options (they typically indicate first-person framing).
        patterns = [
            r"\bich\b", r"\bich bin\b",
            r"\bmein\b", r"\bmeine\b", r"\bmeinem\b",
            r"\bmeinen\b", r"\bmeiner\b", r"\bmeines\b",
            r"\bmich\b", r"\bmir\b",
        ]
    elif lc.startswith("it"):
        # Keep unambiguous subject pronoun + finite verb + clitic+verb pattern.
        # Drop standalone \bmi\b, \bmio\b, \bmia\b, \bmiei\b, \bmie\b — these
        # are object/possessive forms that appear in non-first-person phrases.
        patterns = [
            r"\bio\b", r"\bsono\b",
            r"\bmi\s+\w+",   # clitic before verb: "mi piace", "mi sento"
        ]

    return [re.compile(pat, re.IGNORECASE) for pat in patterns]


def detect_option_style_pattern(
    translation_text: str,
    language_code: str,
) -> str:
    """
    Roughly classify an answer option translation as:
      - 'first_person_like'
      - 'short_label_like'
      - 'noun_phrase_like'
      - 'other'
    This is heuristic and only used for QA/warnings.
    """
    s = (translation_text or "").strip()
    if not s:
        return "unknown"

    lower = s.lower()

    for regex in get_first_person_regexes(language_code):
        if regex.search(lower):
            return "first_person_like"

    # Threshold aligned with classify_segment_type (100 chars on stripped text)
    if len(s) <= 100 and not any(p in s for p in ".?!;:"):
        return "short_label_like"

    # Longer text without question/exclamation marks -- typically multi-clause
    # answer options, long descriptions, or concern statements
    if len(s) <= 200 and not any(p in s for p in "?!"):
        return "noun_phrase_like"

    return "other"


def block_style_validation(context: SurveyFileContext) -> None:
    """
    Block-level style QA (Layer 5).

    For each question block, we:
      - Check answer options for mixed styles (e.g. some first-person-like, some short label-like).
      - Flag scale labels that look unusually long or self-referential.

    We only add/append to suggestion_reason; we NEVER change translations here.
    """
    if not context.blocks:
        return

    lang = context.language_code or ""

    for block in context.blocks:
        rows = context.rows

        # ----- Answer options: detect and compare patterns -----
        option_patterns: List[str] = []
        option_indices: List[int] = []

        for idx in block.answer_option_indices:
            if idx < 0 or idx >= len(rows):
                continue
            row = rows[idx]
            trl = (row.new_translation or row.existing_translation or "").strip()
            if not trl:
                continue

            pattern = detect_option_style_pattern(trl, lang)
            option_patterns.append(pattern)
            option_indices.append(idx)

        if option_patterns:
            unique_patterns = set(p for p in option_patterns if p != "unknown")
            if len(unique_patterns) > 1:
                # Find majority pattern (ignoring 'unknown')
                counts: Dict[str, int] = {}
                for p in option_patterns:
                    if p == "unknown":
                        continue
                    counts[p] = counts.get(p, 0) + 1

                if counts:
                    majority_pattern = max(counts, key=counts.get)
                    if majority_pattern in {"first_person_like", "short_label_like", "noun_phrase_like"}:
                        for idx, pat in zip(option_indices, option_patterns):
                            # noun_phrase_like and short_label_like are compatible
                            compatible = (
                                {pat, majority_pattern} <= {"short_label_like", "noun_phrase_like"}
                            )
                            if (
                                pat != majority_pattern
                                and pat != "unknown"
                                and pat != "other"
                                and not compatible
                            ):
                                row = rows[idx]
                                msg = (
                                    "Block-level style check: most answer options in this question "
                                    f"look '{majority_pattern.replace('_', ' ')}', but this option looks "
                                    f"'{pat.replace('_', ' ')}'. Consider aligning its style with the others."
                                )
                                if row.suggestion_reason:
                                    row.suggestion_reason = row.suggestion_reason + " | " + msg
                                else:
                                    row.suggestion_reason = msg

        # ----- Scale labels: warn on long or self-referential labels -----
        # Only run the first-person check for languages where detection is reliable;
        # for unsupported languages the regex list is empty anyway, but being explicit
        # prevents future false positives if patterns are ever added incompletely.
        first_person_regexes = (
            get_first_person_regexes(lang) if supports_first_person_detection(lang) else []
        )

        for idx in block.scale_label_indices:
            if idx < 0 or idx >= len(rows):
                continue
            row = rows[idx]
            trl = (row.new_translation or row.existing_translation or "").strip()
            if not trl:
                continue

            s = trl.strip()
            words = s.split()
            lower = s.lower()

            is_long = len(words) > 7
            is_first_personish = any(r.search(lower) for r in first_person_regexes)

            if is_long or is_first_personish:
                msg = (
                    "Block-level style check: this scale label looks unusually long or self-referential. "
                    "Scale labels in surveys are usually short, neutral phrases."
                )
                if row.suggestion_reason:
                    row.suggestion_reason = row.suggestion_reason + " | " + msg
                else:
                    row.suggestion_reason = msg

# ==========================
# Core Processing Pipeline
# ==========================


async def process_row_async(
        row: SurveyRow,
        context: SurveyFileContext,
        global_context: str,
        semaphore: asyncio.Semaphore,
        provide_suggestions: bool,
        gender_inclusive: bool = False,
        enable_judge: bool = False,
) -> SurveyRow:
    if row.batch_translated:
        return row

    # Acquire slot in the semaphore (e.g., max 20 active requests)
    async with semaphore:
        # Guard against pandas NaN (float NaN is truthy, so `or` won't catch it)
        if isinstance(row.english_text, float) and pd.isna(row.english_text):
            row.new_translation = ""   # never ship a NaN/float into the output cell
            return row
        eng_text = (row.english_text or "").strip()
        if not eng_text:
            row.new_translation = row.existing_translation
            return row
        # If this row already had a real translation and suggestions are disabled,
        # do not run QA and do not generate suggestions.
        if row.had_real_translation and not provide_suggestions:
            row.new_translation = row.existing_translation
            return row


        # Hard guard: if the source is *purely* numeric/range/code-like (e.g., '1970-1989'),
        # keep it as a pure range/code in the output. This prevents drift into prose like
        # 'Born between 1950 and 1969' and preserves visual parallelism across option sets.
        if row.segment_type in {SegmentType.ANSWER_OPTION, SegmentType.SCALE_LABEL} and is_pure_numeric_or_range_code_like(eng_text):
            localized_num = format_pure_numeric_for_locale(eng_text, context.language_code)
            if row.had_real_translation:
                # If an existing translation is already numeric/range-like, keep it. If it was paraphrased, suggest fixing it.
                if not is_pure_numeric_or_range_code_like(row.existing_translation):
                    row.new_translation = row.existing_translation
                    row.suggested_translation = localized_num
                    row.suggestion_reason = ((row.suggestion_reason + " | ") if row.suggestion_reason else "") + "Numeric/range option should remain a pure range/code (no prose rewrites)."
                else:
                    row.new_translation = row.existing_translation
                return row
            else:
                # Deterministic locale separator formatting; value never changes.
                row.new_translation = localized_num
                row.was_newly_translated = True
                return row

        # Logic to find Parent Context (Question Text)
        parent_context_str = ""
        peer_english_options = None

        # For short categorical label sets, provide peer options to encourage parallel translations.
        if row.segment_type == SegmentType.ANSWER_OPTION and context.blocks and row.block_id is not None:
            try:
                block = get_block_by_id(context, row.block_id)
                if block is not None:
                    opt_texts = [
                        strip_html_for_heuristics(context.rows[i].english_text)
                        for i in block.answer_option_indices
                        if i is not None and context.rows[i].english_text
                    ]
                    opt_texts = [t for t in opt_texts if t]
                    # Only include peers for small, label-like sets; skip large lists (cities, months, etc.).
                    if 2 <= len(opt_texts) <= 8 and options_look_like_short_labels(opt_texts):
                        peer_english_options = opt_texts
            except Exception:
                peer_english_options = None
        if row.segment_type in [SegmentType.ANSWER_OPTION, SegmentType.SCALE_LABEL]:
            block = get_block_by_id(context, row.block_id)
            if block is not None:
                # Get the question text(s) for this block
                q_texts = [context.rows[i].english_text for i in block.question_indices if 0 <= i < len(context.rows) and context.rows[i].english_text]
                parent_context_str = " ".join(q_texts)

        # Compute block-level answer option stats for list-context heuristic
        ao_count = 0
        ao_avg_len = 0.0
        if row.segment_type == SegmentType.ANSWER_OPTION and context.blocks and row.block_id is not None:
            try:
                blk = get_block_by_id(context, row.block_id)
                if blk is not None:
                    ao_texts = [
                        (context.rows[i].english_text or "").strip()
                        for i in blk.answer_option_indices
                        if i is not None and context.rows[i].english_text
                    ]
                    ao_count = len(ao_texts)
                    ao_avg_len = sum(len(t) for t in ao_texts) / ao_count if ao_count else 0.0
            except Exception:
                pass

        # Call the model
        result = await call_translation_model_async(
            english_text=eng_text,
            language_code=context.language_code,
            locale_code=context.locale_code,
            global_context=global_context,
            translation_memory=context.translation_memory,
            existing_translation=row.existing_translation if row.had_real_translation else None,
            segment_type=row.segment_type,
            block_style=(context.block_styles.get(row.block_id) if context.block_styles and row.block_id is not None else None),
            peer_english_options=peer_english_options,
            parent_context=parent_context_str,
            gender_inclusive=gender_inclusive,
            answer_option_count=ao_count,
            answer_option_avg_len=ao_avg_len,
            is_same_language_localization=context.is_same_language_localization,
        )

        # --- Process Result (Same logic as before, just adapted for async return) ---
        proposed = pick_final_translation(result, english_text=eng_text)

        is_ok, msg = validate_translation_structure(eng_text, proposed)
        if not is_ok:
            repaired = attempt_placeholder_repair(eng_text, proposed)
            if repaired and repaired != proposed:
                re_valid, _ = validate_translation_structure(eng_text, repaired)
                if re_valid:
                    proposed = repaired
                    is_ok = True

        if not is_ok:
            row.suggestion_reason = ((row.suggestion_reason + " | ") if row.suggestion_reason else "") + \
                                    f"Structure validation warning: {msg}"

        # Abbreviation preservation check
        abbrev_ok, abbrev_msg = validate_abbreviation_preservation(eng_text, proposed)
        if not abbrev_ok:
            row.suggestion_reason = ((row.suggestion_reason + " | ") if row.suggestion_reason else "") + \
                                    f"Abbreviation warning: {abbrev_msg}"

        # Copy check must run regardless of structure validation outcome.
        # Skip entirely for same-language localization (e.g. en → en-GB) where
        # identical output is expected and correct for most rows.
        if (not context.is_same_language_localization
                and not result.get("error")
                and not row.had_real_translation
                and should_run_copy_check(eng_text, variable_name=row.variable_name)):
            if is_effective_copy_of_english(eng_text, proposed):
                # Retry once with an explicit instruction to not return English
                retry_result = await call_translation_model_async(
                    english_text=eng_text,
                    language_code=context.language_code,
                    locale_code=context.locale_code,
                    global_context=(
                        global_context + "\n\nCRITICAL RETRY: Your previous attempt returned "
                        "the English text unchanged. You MUST translate this into the target "
                        "language. Do NOT return the English text."
                    ),
                    translation_memory=context.translation_memory,
                    existing_translation=None,
                    segment_type=row.segment_type,
                    block_style=(context.block_styles.get(row.block_id) if context.block_styles and row.block_id is not None else None),
                    parent_context=parent_context_str,
                    gender_inclusive=gender_inclusive,
                    peer_english_options=peer_english_options,
                    answer_option_count=ao_count,
                    answer_option_avg_len=ao_avg_len,
                    is_same_language_localization=context.is_same_language_localization,
                )
                retry_proposed = (
                    retry_result.get("qa_checked_translation")
                    or retry_result.get("proposed_translation")
                    or ""
                )
                if retry_proposed and not is_effective_copy_of_english(eng_text, retry_proposed):
                    proposed = retry_proposed
                else:
                    base = row.existing_translation or eng_text
                    row.new_translation = base
                    row.suggested_translation = base
                    row.suggestion_reason = (
                        (row.suggestion_reason or "")
                        + "Model output is effectively identical to the English source after retry; "
                          "translation likely failed. Please review/translate this row manually."
                    )
                    return row

        # If structure validation failed and copy check didn't fire, still SHIP a
        # translation (flagged) for a new row — never silently fall back to English.
        if not is_ok:
            if not row.had_real_translation:
                if proposed.strip():
                    row.new_translation = adjust_capitalization_for_label(eng_text, proposed, context.language_code, ao_count, ao_avg_len)
                    row.suggestion_reason = (
                        ((row.suggestion_reason + " | ") if row.suggestion_reason else "")
                        + "STRUCTURE FLAG: shipped translation despite failed structure validation; needs review."
                    )
                else:
                    row.new_translation = eng_text
                    row.suggestion_reason = (
                        ((row.suggestion_reason + " | ") if row.suggestion_reason else "")
                        + "STRUCTURE FLAG: model returned empty output; English retained; needs review."
                    )
                row.was_newly_translated = True
                row.suggested_translation = proposed
            else:
                # Existing translation: never overwrite it; surface the proposed fix.
                row.new_translation = row.existing_translation
                row.suggested_translation = proposed
            return row

        if result.get("error"):
            row.suggestion_reason = f"LLM Error: {result.get('change_reason')}"
            # For new rows, ship a visible sentinel rather than blank or plain English.
            row.new_translation = row.existing_translation or "[TRANSLATION FAILED — REVIEW]"
        elif not row.had_real_translation:
            # New Translation — guard empty output before writing anything.
            if not proposed.strip():
                row.new_translation = "[TRANSLATION FAILED — REVIEW]"
                row.was_newly_translated = True
                row.suggestion_reason = (
                    ((row.suggestion_reason + " | ") if row.suggestion_reason else "")
                    + "LLM Error: model returned an empty translation for a new row."
                )
                return row
            row.new_translation = adjust_capitalization_for_label(eng_text, proposed, context.language_code, ao_count, ao_avg_len)
            row.was_newly_translated = True
            existing = (row.existing_translation or "").strip()
            if existing and existing.lower() == eng_text.strip().lower():
                snippet = existing[:80]
                if context.is_same_language_localization:
                    row.suggestion_reason = (
                        (row.suggestion_reason or "")
                        + f"Localized from source English to {context.locale_code} conventions."
                    )
                else:
                    row.suggestion_reason = (
                        (row.suggestion_reason or "")
                        + f"Copy check: existing translation was identical to English source ('{snippet}'). Retranslated."
                    )
        else:
            # QA Existing
            row.new_translation = row.existing_translation
            if result.get("needs_change") and proposed != row.existing_translation:
                row.suggested_translation = adjust_capitalization_for_label(
                    eng_text, proposed, context.language_code, ao_count, ao_avg_len
                )
                row.suggestion_reason = result.get("change_reason")

        # --- LLM judge / critique-and-revise loop ---
        # Runs only when: the judge is enabled, this is a newly-translated row,
        # the translation is not a failure sentinel, and no API error occurred.
        if (
            enable_judge
            and row.was_newly_translated
            and (row.new_translation or "").strip()
            and not (row.new_translation or "").startswith("[TRANSLATION FAILED")
        ):
            bs = (
                context.block_styles.get(row.block_id)
                if context.block_styles and row.block_id is not None
                else None
            )
            style_note = _format_style_note(bs)
            t1 = row.new_translation

            # Judge T1
            j1 = await judge_translation_async(
                eng_text, t1, context.language_code, style_note
            )

            if j1.get("error") or j1.get("score") is None:
                # Non-fatal: keep T1, no flag
                pass
            elif j1["score"] >= JUDGE_SCORE_THRESHOLD:
                # T1 passes — record and ship
                row.judge_score = j1["score"]
                row.judge_reason = j1.get("reason")
                row.judge_outcome = "clean"
            else:
                # T1 below threshold: one critique-and-revise retry
                retry_ctx = (
                    global_context
                    + f"\n\nPrevious attempt had this issue: {j1.get('reason', '')}. "
                    "Please retranslate addressing this specific concern."
                )
                r2 = await call_translation_model_async(
                    english_text=eng_text,
                    language_code=context.language_code,
                    locale_code=context.locale_code,
                    global_context=retry_ctx,
                    translation_memory=context.translation_memory,
                    existing_translation=None,
                    segment_type=row.segment_type,
                    block_style=bs,
                    peer_english_options=peer_english_options,
                    parent_context=parent_context_str,
                    gender_inclusive=gender_inclusive,
                    answer_option_count=ao_count,
                    answer_option_avg_len=ao_avg_len,
                    is_same_language_localization=context.is_same_language_localization,
                )
                t2 = pick_final_translation(r2, english_text=eng_text)

                if t2.strip() and not r2.get("error"):
                    # Validate and apply T2
                    ok2, _ = validate_translation_structure(eng_text, t2)
                    if ok2:
                        row.new_translation = adjust_capitalization_for_label(
                            eng_text, t2, context.language_code, ao_count, ao_avg_len
                        )
                    # else: validation failed on T2 — still adopt it (already flagged by
                    # the existing structure-validation path upstream); don't revert to T1

                row.judge_retried = True

                # Judge T2 exactly once — no further retry regardless of score
                j2 = await judge_translation_async(
                    eng_text, row.new_translation, context.language_code, style_note
                )
                if not j2.get("error") and j2.get("score") is not None:
                    row.judge_score = j2["score"]
                    row.judge_reason = j2.get("reason")
                    if j2["score"] >= JUDGE_SCORE_THRESHOLD:
                        row.judge_outcome = "retried_passed"
                    else:
                        row.judge_outcome = "retried_flagged"
                        row.qa_status = (
                            (row.qa_status + " | ") if row.qa_status else ""
                        ) + f"JUDGE: low score {j2['score']} after retry — human review."
                # else: non-fatal judge failure on T2; ship T2 without a flag

        return row


async def restyle_mismatched_rows(
    context: SurveyFileContext,
    global_context: str,
    semaphore: asyncio.Semaphore,
    gender_inclusive: bool = False,
    provide_suggestions: bool = True,
) -> int:
    """
    Post-translation style re-check.

    For each question block, check that all answer options match the inferred
    block style. Re-translate any that don't match. Returns count of re-translated rows.

    When *provide_suggestions* is False, rows that already had a real (non-English-
    placeholder) translation are skipped — they were kept as-is during the main
    pass, so spending an LLM call to restyle them is unnecessary.
    """
    if not context.blocks or not context.block_styles:
        return 0

    lang = context.language_code or ""

    # -- Phase 1: collect every mismatched row that needs an LLM restyle call --
    # task_meta entries: (row, expected_pattern, enforce)
    # enforce=True  → write result to new_translation (overwrite the shipped cell)
    # enforce=False → write result to suggested_translation only (human review required)
    tasks = []
    task_meta: list[tuple] = []

    for block in context.blocks:
        if not context.block_styles:
            continue
        style = context.block_styles.get(block.block_id)
        if not style:
            continue
        if not block.answer_option_indices:
            continue

        person = style.options_grammatical_person or "unspecified"
        phrase = style.options_phrase_form or "unspecified"
        if person == "unspecified" and phrase == "unspecified":
            continue

        if style.options_grammatical_person == "first_person":
            expected_pattern = "first_person_like"
        elif style.options_phrase_form in ("noun_phrase", "short_phrase"):
            expected_pattern = "short_label_like"
        else:
            continue

        # Capability gate: if we cannot reliably detect first-person for this language,
        # skip the whole block rather than generating churn from unvalidatable LLM calls.
        if expected_pattern == "first_person_like" and not supports_first_person_detection(lang):
            continue

        parent_context = " ".join(
            context.rows[i].english_text
            for i in block.question_indices
            if context.rows[i].english_text
        )

        # Consensus gate: count how many of the block's options already match the
        # expected pattern.  Only enforce (overwrite new_translation) when ≥70% agree;
        # below the threshold, restyle results go to suggested_translation only.
        option_rows = [
            context.rows[i]
            for i in block.answer_option_indices
            if 0 <= i < len(context.rows)
        ]
        block_patterns = [
            detect_option_style_pattern(
                (r.new_translation or r.existing_translation or "").strip(), lang
            )
            for r in option_rows
        ]
        considered = [p for p in block_patterns if p != "unknown"]
        match_ratio = (
            sum(1 for p in considered if p == expected_pattern) / len(considered)
            if considered else 0.0
        )
        enforce_block = match_ratio >= _RESTYLE_CONSENSUS_THRESHOLD

        for idx in block.answer_option_indices:
            if idx < 0 or idx >= len(context.rows):
                continue
            row = context.rows[idx]

            if not provide_suggestions and row.had_real_translation:
                continue

            trl = (row.new_translation or row.existing_translation or "").strip()
            if not trl:
                continue

            actual_pattern = detect_option_style_pattern(trl, lang)

            if actual_pattern != expected_pattern and actual_pattern != "unknown":
                if actual_pattern == "noun_phrase_like" and expected_pattern == "short_label_like":
                    continue

                # Human-translated rows are NEVER auto-overwritten regardless of consensus.
                enforce = enforce_block and not row.had_real_translation

                async def _restyle_one(
                    _sem=semaphore, _row=row, _trl=trl, _style=style,
                    _parent_ctx=parent_context,
                ):
                    async with _sem:
                        return await call_translation_model_async(
                            english_text=_row.english_text,
                            language_code=context.language_code,
                            locale_code=context.locale_code,
                            global_context=global_context,
                            translation_memory=context.translation_memory,
                            existing_translation=_trl,
                            segment_type=_row.segment_type,
                            block_style=_style,
                            parent_context=_parent_ctx,
                            gender_inclusive=gender_inclusive,
                            is_same_language_localization=context.is_same_language_localization,
                        )

                tasks.append(_restyle_one())
                task_meta.append((row, expected_pattern, enforce))

    if not tasks:
        return 0

    # -- Phase 2: fire all LLM calls concurrently (semaphore throttles) --
    results = await asyncio.gather(*tasks, return_exceptions=True)

    retranslated = 0
    for (row, expected_pattern, enforce), result in zip(task_meta, results):
        if isinstance(result, Exception):
            continue

        new_trl = pick_final_translation(result, english_text=(row.english_text or ""))
        trl = (row.new_translation or row.existing_translation or "").strip()
        if new_trl and new_trl != trl:
            new_pattern = detect_option_style_pattern(new_trl, lang)
            pattern_ok = new_pattern == expected_pattern or new_pattern == "unknown"
            if pattern_ok and enforce:
                # Strong consensus + new row: overwrite the shipped cell.
                row.new_translation = new_trl
                retranslated += 1
            elif pattern_ok:
                # Good restyle but below consensus threshold, or human-translated row:
                # offer as a suggestion only — never overwrite.
                row.suggested_translation = new_trl
                row.suggestion_reason = (
                    (row.suggestion_reason or "")
                    + f" | Style suggestion: align this option to the block style ({expected_pattern})."
                )
            else:
                # LLM could not produce a matching pattern even after restyle.
                row.suggestion_reason = (
                    (row.suggestion_reason or "") +
                    " | Style re-check: could not align this option to the "
                    f"block style ({expected_pattern}). Manual review recommended."
                )

    return retranslated


# ==========================
# Step 13 — Semantic Back-Translation Verification
# ==========================

def _content_words(text: str) -> set:
    """
    Extract content words from English text for semantic overlap comparison.

    Strips placeholder tokens, HTML tags, lowercases, splits on non-word characters,
    and removes stopwords.  Returns a set of remaining word stems (no lemmatization;
    simple tokenization is sufficient for the overlap heuristic).
    """
    if not text:
        return set()
    # Remove placeholder tokens and HTML tags before tokenizing.
    cleaned = _PLACEHOLDER_TOKEN_RE.sub(" ", text)
    cleaned = _TAG_RE.sub(" ", cleaned)
    tokens = re.split(r"\W+", cleaned.lower())
    return {t for t in tokens if t and t not in _SEMANTIC_STOPWORDS}


async def call_backtranslation_model_async(
    target_text: str,
    target_language_code: str,
    semaphore: asyncio.Semaphore,
    model_name: str = MODEL_NAME,
) -> dict:
    """
    Back-translate *target_text* (in *target_language_code*) into literal English,
    with NO access to the original English source.

    Returns {"english": "<back-translation>"} on success, or
            {"english": "", "error": True} on terminal failure.

    Mirrors the retry/parse pattern of _call_translation_model_async_uncached.
    """
    lang_label = code_to_language_label(target_language_code) or target_language_code

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a precise translator.  Translate the {lang_label} text provided "
                "into literal English.  Do not interpret, rephrase, improve, or add anything.  "
                "Return ONLY valid JSON in the form: {\"english\": \"...\"}"
            ),
        },
        {
            "role": "user",
            "content": target_text,
        },
    ]

    last_error: Optional[str] = None
    for attempt in range(3):
        try:
            async with semaphore:
                client = get_async_client()
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                    temperature=0,
                    seed=TRANSLATION_SEED,
                )
            _record_fingerprint(response)
            parsed = _safe_json(response)
            if isinstance(parsed, dict) and "english" in parsed:
                return parsed
            last_error = f"Missing 'english' key in response: {parsed}"
        except _RetryableModelError as exc:
            last_error = str(exc)
            await asyncio.sleep(2 ** attempt)
        except Exception as exc:
            last_error = str(exc)
            await asyncio.sleep(2 ** attempt)

    return {"english": "", "error": True, "change_reason": last_error}


async def judge_translation_async(
    english_text: str,
    translation: str,
    language_code: str,
    style_note: str = "",
    model_name: str = MODEL_NAME,
) -> dict:
    """
    LLM judge for the critique-and-revise quality loop (Step 2 in the plan).

    Asks the model to score *translation* on a 1-5 naturalness/register scale
    and return a specific reason sentence.

    Returns {"score": int, "reason": str} on success, or
            {"score": None, "reason": "", "error": True} on terminal failure.

    Runs under the *caller's* semaphore slot (no acquire here) — same pattern as
    the copy-check retry in process_row_async.  Language-agnostic: only the
    language label and an optional free-text style_note enter the prompt.
    Non-fatal: any API error or malformed JSON returns {"error": True}.
    """
    lang = code_to_language_label(language_code) or language_code

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a survey-localization QA reviewer for {lang}. "
                f"Judge whether the translation reads as natural survey copy in {lang} "
                f"and whether its register is consistent with the style log provided. "
                f"Score 1 (very poor) to 5 (excellent). "
                f'Return ONLY valid JSON: {{"score": <integer 1-5>, "reason": "<one specific sentence explaining the score>"}}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"English source: {english_text}\n"
                f"Translation ({lang}): {translation}\n"
                f"Style log: {style_note or 'none'}"
            ),
        },
    ]

    last_error: Optional[str] = None
    for attempt in range(3):
        try:
            client = get_async_client()
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=0,
                seed=TRANSLATION_SEED,
            )
            _record_fingerprint(response)
            parsed = _safe_json(response)
            if isinstance(parsed, dict) and "score" in parsed:
                try:
                    score = int(parsed["score"])
                except (TypeError, ValueError):
                    last_error = f"Non-integer score in judge response: {parsed}"
                    continue
                if not (1 <= score <= 5):
                    last_error = f"Judge score out of range: {score}"
                    continue
                return {"score": score, "reason": str(parsed.get("reason") or "")}
            last_error = f"Missing 'score' key in judge response: {parsed}"
        except _RetryableModelError as exc:
            last_error = str(exc)
            await asyncio.sleep(2 ** attempt)
        except Exception as exc:
            last_error = str(exc)
            await asyncio.sleep(2 ** attempt)

    return {"score": None, "reason": "", "error": True, "change_reason": last_error}


def _format_style_note(block_style: Optional["BlockStyle"]) -> str:
    """Summarise a BlockStyle into a short English sentence for the judge prompt."""
    if not block_style:
        return ""
    parts = []
    if block_style.options_grammatical_person and block_style.options_grammatical_person != "unspecified":
        parts.append(f"person: {block_style.options_grammatical_person}")
    if block_style.options_phrase_form and block_style.options_phrase_form != "unspecified":
        parts.append(f"phrase form: {block_style.options_phrase_form}")
    if block_style.options_tone and block_style.options_tone != "formal_neutral":
        parts.append(f"tone: {block_style.options_tone}")
    if block_style.notes:
        parts.append(block_style.notes)
    return "; ".join(parts)


async def semantic_verification_pass(
    context: "SurveyFileContext",
    global_context: str,
    semaphore: asyncio.Semaphore,
    provide_suggestions: bool = True,
) -> int:
    """
    Suggest-only semantic drift check via back-translation (Step 13).

    For each QUESTION, SCALE_LABEL, or qualifier-bearing ANSWER_OPTION row,
    back-translates the shipped target text to English and compares content-word
    overlap + qualifier/negation presence against the original English source.
    Flags drift into suggestion_reason; NEVER modifies new_translation or
    suggested_translation.

    Returns the number of rows flagged.
    """
    if not provide_suggestions or context.is_same_language_localization:
        return 0

    lang = context.language_code or ""

    # --- Build in-scope row list ---
    in_scope: list[tuple] = []  # (row_idx, row)
    for idx, row in enumerate(context.rows):
        seg = row.segment_type
        shipped = (row.new_translation or row.existing_translation or "").strip()
        if not shipped or shipped.startswith("[TRANSLATION FAILED"):
            continue
        if seg == SegmentType.QUESTION or seg == SegmentType.SCALE_LABEL:
            in_scope.append((idx, row))
        elif seg == SegmentType.ANSWER_OPTION:
            # Include qualifier-bearing options (English source contains a qualifier term).
            if _content_words(row.english_text or "") & _QUALIFIER_TERMS:
                in_scope.append((idx, row))

    if not in_scope:
        return 0

    # Dedup: translate each unique (shipped_text, language) key only once.
    unique_keys: dict[tuple, list] = {}
    for idx, row in in_scope:
        shipped = (row.new_translation or row.existing_translation or "").strip()
        key = (shipped, lang)
        unique_keys.setdefault(key, []).append((idx, row))

    # Fire back-translation calls for each unique key.
    unique_list = list(unique_keys.items())
    tasks = [
        call_backtranslation_model_async(key[0], key[1], semaphore)
        for key, _ in unique_list
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    flagged = 0
    for (key, row_group), result in zip(unique_list, results):
        if isinstance(result, Exception) or result.get("error"):
            continue
        bt = (result.get("english") or "").strip()
        if not bt:
            continue

        for idx, row in row_group:
            eng_words = _content_words(row.english_text or "")
            bt_words = _content_words(bt)

            if not eng_words:
                continue

            overlap = _jaccard_word_similarity(
                " ".join(eng_words), " ".join(bt_words)
            )

            eng_q = eng_words & _QUALIFIER_TERMS
            bt_q = bt_words & _QUALIFIER_TERMS
            qualifier_mismatch = bool(eng_q ^ bt_q)

            if overlap < _SEMANTIC_OVERLAP_THRESHOLD or qualifier_mismatch:
                qual_note = ", qualifier/negation mismatch" if qualifier_mismatch else ""
                msg = (
                    f"Semantic check: back-translation '{bt}' diverges from source "
                    f"(overlap {overlap:.0%}{qual_note}). Manual review recommended."
                )
                row.suggestion_reason = (
                    (row.suggestion_reason + " | " + msg)
                    if row.suggestion_reason
                    else msg
                )
                flagged += 1

    return flagged


def consistency_pass(context: SurveyFileContext, apply_to_new_translations: bool = False, global_context: str = "") -> None:
    """
    Survey-wide consistency pass (LLM-powered) using Fuzzy Matching.

    Groups similar English phrases (e.g. "Select one." and "select one")
    to ensure they are translated consistently.

    Key hardening over the original:
    - Grouping key is (normalize_fuzzy(eng), segment_type, parent-question hash)
      so homographs with different senses in different blocks are never merged.
    - Applies the canonical ONLY to indices_to_update (not all group indices).
    - Deterministic canonical when nothing is locked (most-frequent; tie -> lowest index).
    - Re-validates the canonical against each target row's English before writing
      so a structure-breaking canonical is never applied.
    """

    def normalize_fuzzy(text: str) -> str:
        # NFKC + HTML-entity unescape so fullwidth/compatibility variants and
        # entities (&amp; &#39; <b>) collapse to one fuzzy key before stripping.
        text = html.unescape(text or "")
        text = unicodedata.normalize("NFKC", text)
        return text.lower().translate(str.maketrans('', '', string.punctuation)).replace(" ", "")

    def _parent_hash(row: SurveyRow) -> str:
        """Short hash of the parent question(s) for this row's block (context-aware grouping)."""
        blk = get_block_by_id(context, row.block_id)
        if blk is not None:
            q = " ".join(
                normalize_fuzzy(context.rows[i].english_text or "")
                for i in blk.question_indices
                if 0 <= i < len(context.rows)
            )
            return q[:64]
        return ""

    # Map: (fuzzy_norm_eng, segment_type, parent_hash) -> { original_english -> { translation -> [indices] } }
    fuzzy_map: Dict[tuple, Dict[str, Dict[str, List[int]]]] = {}

    for idx, row in enumerate(context.rows):
        eng = (row.english_text or "").strip()

        if "Structure validation warning" in (row.suggestion_reason or ""):
            continue

        trl = (row.new_translation or row.existing_translation or "").strip()
        if not eng or not trl:
            continue

        seg = row.segment_type.value if row.segment_type else ""
        fuzzy_key: tuple = (normalize_fuzzy(eng), seg, _parent_hash(row))

        if fuzzy_key not in fuzzy_map:
            fuzzy_map[fuzzy_key] = {}
        if eng not in fuzzy_map[fuzzy_key]:
            fuzzy_map[fuzzy_key][eng] = {}
        if trl not in fuzzy_map[fuzzy_key][eng]:
            fuzzy_map[fuzzy_key][eng][trl] = []
        fuzzy_map[fuzzy_key][eng][trl].append(idx)

    phrase_groups = []
    for fuzzy_key, eng_variants in fuzzy_map.items():
        all_translations: Dict[str, List[int]] = {}
        primary_english = list(eng_variants.keys())[0]
        for original_eng, trl_dict in eng_variants.items():
            for trl, indices in trl_dict.items():
                if trl not in all_translations:
                    all_translations[trl] = []
                all_translations[trl].extend(indices)
        if len(all_translations) > 1:
            phrase_groups.append({
                "english_phrase": primary_english,
                "translations": [{"translation": t, "indices": i} for t, i in all_translations.items()]
            })

    if not phrase_groups:
        return

    issues = call_consistency_model(context, phrase_groups, global_context=global_context)

    for issue in issues:
        canonical = (issue.get("canonical_translation") or "").strip()
        indices_to_update = [
            i for i in (issue.get("indices_to_update") or [])
            if isinstance(i, int) and 0 <= i < len(context.rows)
        ]
        notes = issue.get("notes") or ""

        if not canonical or not indices_to_update:
            continue

        # Match by index overlap (robust to homographs with different fuzzy_key tuples).
        group = None
        want = set(indices_to_update)
        for g in phrase_groups:
            gi = {
                i
                for t in g.get("translations", [])
                for i in (t.get("indices") or [])
                if isinstance(i, int)
            }
            if want & gi:
                group = g
                break
        if group is None:
            continue
        english_phrase = group.get("english_phrase", "")

        # Prefer a locked (pre-existing) translation; else deterministic pick.
        locked_counts: Dict[str, int] = {}
        for t in group.get("translations", []):
            trl = (t.get("translation") or "").strip()
            for idx in (t.get("indices") or []):
                if isinstance(idx, int) and 0 <= idx < len(context.rows) \
                        and not context.rows[idx].was_newly_translated and trl:
                    locked_counts[trl] = locked_counts.get(trl, 0) + 1

        if locked_counts:
            canonical_to_apply = max(locked_counts, key=locked_counts.get)
        else:
            freq: Dict[str, int] = {}
            first_idx: Dict[str, int] = {}
            for idx in sorted(indices_to_update):
                trl = (context.rows[idx].new_translation or context.rows[idx].existing_translation or "").strip()
                if not trl:
                    continue
                freq[trl] = freq.get(trl, 0) + 1
                first_idx.setdefault(trl, idx)
            if freq:
                canonical_to_apply = max(freq, key=lambda k: (freq[k], -first_idx[k]))
            else:
                canonical_to_apply = canonical

        # Apply ONLY to indices_to_update; re-validate against each row's English.
        for idx in indices_to_update:
            row = context.rows[idx]
            if "Structure validation warning" in (row.suggestion_reason or ""):
                continue
            ok, _ = validate_translation_structure(row.english_text or "", canonical_to_apply)
            if not ok:
                continue  # never overwrite with a structure-breaking canonical
            if apply_to_new_translations:
                if not row.was_newly_translated:
                    continue
                row.new_translation = canonical_to_apply
            else:
                if (row.suggested_translation or "").strip():
                    continue
                row.suggested_translation = canonical_to_apply
                row.suggestion_reason = (
                    f"LLM consistency suggestion: '{english_phrase}' appears with multiple "
                    f"translations. Suggested canonical: '{canonical_to_apply}'."
                    + (f" Note: {notes}" if notes else "")
                )
                if row.new_translation and not row.had_real_translation:
                    row.new_translation = canonical_to_apply

def build_block_style_log_df(context: SurveyFileContext) -> Optional[pd.DataFrame]:
    """
    Build a summary DataFrame of block-level style decisions.

    One row per QuestionBlock with:
      - block_id
      - question_text (English)
      - counts of options / scale labels
      - style decisions from BlockStyle (if available)
    """
    if not context.blocks:
        return None

    records: List[Dict[str, object]] = []

    for block in context.blocks:
        question_text_parts: List[str] = []
        for idx in block.question_indices:
            if 0 <= idx < len(context.rows):
                qt = (context.rows[idx].english_text or "").strip()
                if qt:
                    question_text_parts.append(qt)
        question_text = " ".join(question_text_parts)

        style = None
        if context.block_styles is not None:
            style = context.block_styles.get(block.block_id)

        records.append(
            {
                "block_id": block.block_id,
                "question_text_english": question_text,
                "num_rows_in_block": len(block.row_indices),
                "num_answer_options": len(block.answer_option_indices),
                "num_scale_labels": len(block.scale_label_indices),
                "options_grammatical_person": getattr(style, "options_grammatical_person", ""),
                "options_phrase_form": getattr(style, "options_phrase_form", ""),
                "options_tone": getattr(style, "options_tone", ""),
                "scale_label_phrase_form": getattr(style, "scale_label_phrase_form", ""),
                "style_notes": getattr(style, "notes", ""),
            }
        )

    if not records:
        return None

    return pd.DataFrame.from_records(records)


def write_output_file(
    context: SurveyFileContext,
    original_df: pd.DataFrame,
    include_suggestions: bool = True,
) -> Tuple[pd.DataFrame, str, bytes]:
    """
    Build the output DataFrame with 5 columns and serialize to Excel bytes.

    Columns:
      0: original column 0 (variable_name)
      1: original column 1 (english_text)
      2: translation (existing or new)
      3: suggested_translation
      4: suggestion_reason
    """
    df_out = original_df.copy()
    df_out = df_out.reset_index(drop=True)

    # Ensure at least 3 columns
    if df_out.shape[1] < 3:
        raise ValueError(
            f"Original DataFrame for '{context.filename}' has fewer than 3 columns."
        )

    translation_col_name = df_out.columns[2]

    if include_suggestions:
        # Add new columns if missing
        if "suggested_translation" not in df_out.columns:
            df_out["suggested_translation"] = ""
        if "suggestion_reason" not in df_out.columns:
            df_out["suggestion_reason"] = ""
    else:
        # Remove suggestion columns entirely when suggestions are disabled.
        df_out = df_out.drop(columns=["suggested_translation", "suggestion_reason"], errors="ignore")

    # qa_status is always emitted (read-only advisory, independent of suggestions mode).
    # Appended last so the first 3 input columns stay intact for Forsta re-import.
    if "qa_status" not in df_out.columns:
        df_out["qa_status"] = ""

    # judge_status: always emitted when the judge ran for at least one row;
    # harmless empty string when judge was disabled.
    if "judge_status" not in df_out.columns:
        df_out["judge_status"] = ""

    for i, row in enumerate(context.rows):
        # Column 2: translation (existing or new)
        final_translation = (
            row.new_translation
            if row.new_translation is not None
            else row.existing_translation
        )
        df_out.at[i, translation_col_name] = final_translation

        if include_suggestions:
            # Column 3 & 4: suggestions / warnings (if any)
            if row.suggested_translation:
                df_out.at[i, "suggested_translation"] = row.suggested_translation
            if row.suggestion_reason:
                df_out.at[i, "suggestion_reason"] = sanitize_reviewer_note(row.suggestion_reason)

        # Always populate qa_status (may be "" for clean rows)
        df_out.at[i, "qa_status"] = row.qa_status or ""

        # Populate judge_status when the judge produced an outcome.
        # "retried_flagged" rows are the human-review priority queue.
        if row.judge_outcome:
            score_note = (
                f" (score {row.judge_score}: {row.judge_reason})"
                if row.judge_score is not None
                else ""
            )
            df_out.at[i, "judge_status"] = row.judge_outcome + score_note

    # Determine if there are any suggestions or warnings (either column)
    if include_suggestions and "suggested_translation" in df_out.columns and "suggestion_reason" in df_out.columns:
        has_suggestions = (
            df_out["suggested_translation"].astype(str).str.strip().ne("").any()
            or df_out["suggestion_reason"].astype(str).str.strip().ne("").any()
        )
    else:
        has_suggestions = False

    base_name = filename_without_extension(context.filename)
    suffix = "_translated"
    if has_suggestions:
        suffix += "_WITH_SUGGESTIONS"
    output_filename = base_name + suffix + ".xlsx"

    # Strip control characters that can cause encoding errors in Excel writers
    _CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
    for col in df_out.columns:
        if df_out[col].dtype == object:
            df_out[col] = df_out[col].apply(
                lambda v: _CONTROL_CHAR_RE.sub("", v) if isinstance(v, str) else v
            )

    # Step 16: Guard Excel hard limits. Raise a clear error on row overflow;
    # flag (do not silently truncate) cells over the per-cell character limit.
    _XLSX_MAX_ROWS = 1_048_576
    _XLSX_MAX_CELL = 32_767
    if len(df_out) + 1 > _XLSX_MAX_ROWS:
        raise ValueError(
            f"'{context.filename}' has {len(df_out)} rows, exceeding the .xlsx "
            f"limit of {_XLSX_MAX_ROWS - 1}. Split the file before processing."
        )
    if "qa_status" in df_out.columns:
        qa_col = df_out.columns.get_loc("qa_status")
        for i in range(len(df_out)):
            for col in df_out.columns:
                v = df_out.iat[i, df_out.columns.get_loc(col)]
                if isinstance(v, str) and len(v) > _XLSX_MAX_CELL:
                    note = f" | Cell in '{col}' exceeds {_XLSX_MAX_CELL} chars (Excel will truncate on open) — review/split."
                    df_out.iat[i, qa_col] = (df_out.iat[i, qa_col] or "") + note

    # Serialize to XLSX via openpyxl (no 65,536-row cap, no 32,767-char silent truncation)
    import openpyxl

    def _write_df_to_openpyxl_sheet(workbook, df, sheet_name, first_sheet=False):
        """Write a DataFrame to an openpyxl sheet with a header row."""
        import numpy as np
        ws = workbook.active if first_sheet else workbook.create_sheet()
        ws.title = sheet_name[:31]  # Excel sheet-name limit
        ws.append([str(c) for c in df.columns])
        for row_idx in range(len(df)):
            out_row = []
            for col_idx in range(len(df.columns)):
                val = df.iat[row_idx, col_idx]
                if pd.isna(val):
                    out_row.append("")
                elif isinstance(val, str):
                    out_row.append(val)  # full value; over-long cells flagged in qa_status above
                elif isinstance(val, (np.integer,)):
                    out_row.append(int(val))
                elif isinstance(val, (np.floating,)):
                    out_row.append(float(val))
                elif isinstance(val, (np.bool_,)):
                    out_row.append(bool(val))
                else:
                    out_row.append(str(val))
            ws.append(out_row)

    wb = openpyxl.Workbook()
    _write_df_to_openpyxl_sheet(wb, df_out, "translations", first_sheet=True)

    # Optional style log sheet
    style_log_df = build_block_style_log_df(context)
    if style_log_df is not None and not style_log_df.empty:
        for col in style_log_df.columns:
            if style_log_df[col].dtype == object:
                style_log_df[col] = style_log_df[col].apply(
                    lambda v: _CONTROL_CHAR_RE.sub("", v) if isinstance(v, str) else v
                )
        _write_df_to_openpyxl_sheet(wb, style_log_df, "__style_log")

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    excel_bytes = buffer.getvalue()

    return df_out, output_filename, excel_bytes


# ==========================
# Streamlit App
# ==========================

def code_to_language_label(code: str) -> str:
    for label, c in LANGUAGE_LABEL_TO_CODE.items():
        if c == code:
            return label
    # Return empty string so callers can use SELECT_LANGUAGE_SENTINEL rather than
    # silently defaulting to Spanish.
    return ""


def main():
    st.set_page_config(
        page_title="Forsta Questionnaire Translation & QA Tool",
        layout="wide",
    )

    st.title("Forsta Questionnaire Translation & QA Tool")
    st.markdown(
        """
This app processes Forsta/Decipher translation exports (3-column Excel/CSV) and uses GPT
to:

- Generate high-quality localized translations.
- QA and optionally suggest improvements for existing translations (where Column C already differs from English).
- Optionally enforce survey-level consistency for recurring terms.
- Validate numeric ranges, placeholders, and HTML tags so survey structure is preserved.

**Expected input format:**
1. Column A: Variable name / ID  
2. Column B: English text  
3. Column C: Target translation (Previous translation OR English placeholder)
        """
    )

    # Hard-stop if API key is missing
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    if not api_key_present:
        st.error(
            "OPENAI_API_KEY is not set. Please configure it in your environment and reload the app "
            "before running the translation pipeline."
        )
        st.stop()

    uploaded_files = st.file_uploader(
        "Upload one or more Forsta translation export files",
        type=["xls", "xlsx", "csv"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.stop()

    st.subheader("Global Settings")

    global_context = st.text_area(
        "Global translation context",
        value=DEFAULT_GLOBAL_CONTEXT,
        help="This context is sent to the model with every row.",
        height=120,
    )

    enable_consistency = st.checkbox(
        "Enable survey-level consistency pass (LLM-powered)",
        value=True,
        help=(
            "After row-level QA, use a GPT model to analyze repeated English phrases and suggest "
            "canonical translations. Suggestions will remind you to check context in each case."
        ),
    )

    provide_suggestions = st.checkbox(
        "Provide suggestions",
        value=True,
        help=(
            "When enabled, the app will QA existing translations and populate "
            "'suggested_translation' / 'suggestion_reason'. When disabled, existing "
            "translations are not QA'd, but the survey-level consistency pass still "
            "runs to harmonize translations generated in this run."
        ),
    )

    enable_judge = st.checkbox(
        "Enable LLM judge (critique-and-revise quality loop)",
        value=ENABLE_JUDGE_DEFAULT,
        help=(
            "After each new translation, an LLM judge (1–5 scale) scores naturalness "
            f"and register. Scores below {JUDGE_SCORE_THRESHOLD} trigger one retranslation "
            "using the judge's specific feedback. Rows that still score low after retry "
            "are marked in the judge_status column for human review. Off by default."
        ),
    )

    st.subheader("Per-file Language & Locale Settings")

    file_configs = []

    for file in uploaded_files:
        filename = file.name
        detected_lang_code, detected_locale_code = parse_language_and_locale_from_filename(filename)

        detection_failed = not bool(detected_lang_code)
        # Do NOT silently coerce to Spanish.  Use the sentinel so the operator must
        # choose explicitly; the Run button is blocked until every file has a real selection.
        lang_label_default = code_to_language_label(detected_lang_code) if detected_lang_code else SELECT_LANGUAGE_SENTINEL

        # Auto-expand the settings panel when detection failed so the operator sees the warning.
        with st.expander(f"Settings for {filename}", expanded=detection_failed):
            if detection_failed:
                st.warning(
                    "Could not reliably detect target language from the filename. "
                    "Please select the target language below before running the pipeline."
                )

            # Language dropdown — sentinel is the first option so it displays when detection fails.
            lang_options = [SELECT_LANGUAGE_SENTINEL] + list(LANGUAGE_LABEL_TO_CODE.keys())
            if lang_label_default in lang_options:
                lang_default_index = lang_options.index(lang_label_default)
            else:
                lang_default_index = 0  # show sentinel
            selected_lang_label = st.selectbox(
                f"Target language for {filename}",
                options=lang_options,
                index=lang_default_index,
                key=f"lang_{filename}",
            )
            # Empty string when sentinel is still selected; blocks the Run button below.
            language_code = LANGUAGE_LABEL_TO_CODE.get(selected_lang_label, "")

            # Locale dropdown options are driven by the selected language
            if language_code:
                locale_options = LOCALE_OPTIONS.get(
                    language_code,
                    [(f"Generic ({language_code})", language_code)],
                )
            else:
                locale_options = [(SELECT_LANGUAGE_SENTINEL, "")]

            # Choose default locale: use detected locale if it matches one of the options,
            # otherwise default to the first option
            default_locale_code = (detected_locale_code or language_code or "").lower()
            default_locale_index = 0
            for i, (_, code) in enumerate(locale_options):
                if code.lower() == default_locale_code:
                    default_locale_index = i
                    break

            selected_locale_label = st.selectbox(
                f"Target locale for {filename}",
                options=[label for (label, _) in locale_options],
                index=default_locale_index,
                key=f"loc_label_{filename}",
            )
            # Map the selected label back to the BCP-47 code
            locale_code = next(
                (code for (label, code) in locale_options if label == selected_locale_label),
                language_code,
            )

            gender_inclusive = st.checkbox(
                f"Enable gender-inclusive forms for {filename}",
                value=False,
                help=(
                    "When enabled, the model will add gender-inclusive forms "
                    "(e.g., intéressé(e) in French) to adjective-based labels."
                ),
                key=f"gender_{filename}",
            )

            custom_skip_raw = st.text_input(
                "Custom skip prefixes (comma-separated)",
                value="",
                help=(
                    "Resource prefixes for rows that should be skipped without adaptation "
                    "(e.g. qZipCode, qState). Leave blank to use the built-in defaults."
                ),
                key=f"skip_pfx_{filename}",
            )

        file_configs.append(
            {
                "file": file,
                "language_code": language_code,
                "locale_code": locale_code,
                "gender_inclusive": gender_inclusive,
                "custom_skip_prefixes": custom_skip_raw,
            }
        )

    # Gate the Run button if any file still has no language selected.
    blocked = any(not cfg["language_code"] for cfg in file_configs)
    if blocked:
        st.error(
            "Select a target language for every file before running "
            "(filename detection was inconclusive for one or more files)."
        )

    # Initialise / read session state for processed results
    if "processed_results" not in st.session_state:
        st.session_state["processed_results"] = []

    run_pipeline = st.button("Run Translation Pipeline", disabled=blocked)

    # When the button is clicked, run the heavy pipeline and cache results in session_state
    if run_pipeline:
        # Create a new event loop for this run
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Null the cached async client so it binds to THIS loop, not a prior run's loop.
        reset_async_client()
        # Clear the per-run translation dedup cache (Step 11).
        reset_translation_cache()
        # Clear collected model fingerprints (Step 23).
        reset_system_fingerprints()

        st.session_state["processed_results"] = []

        for cfg in file_configs:
            try:
                file = cfg["file"]
                file.seek(0)
                context, original_df = load_forsta_export(file, cfg["language_code"], cfg["locale_code"])
                gender_inclusive = cfg.get("gender_inclusive", False)

                # Step 22: apply custom skip prefixes from UI if provided.
                _custom_skip = cfg.get("custom_skip_prefixes", "")
                if _custom_skip and _custom_skip.strip():
                    _extra = tuple(p.strip() for p in _custom_skip.split(",") if p.strip())
                    context.skip_block_prefixes = _DIALECT_SKIP_BLOCK_PREFIXES + _extra

                # Pre-processing layers (sync)
                classify_segments(context)
                build_blocks(context)
                promote_scale_labels(context)

                # Guard empty / header-only exports: nothing to translate and a
                # later `completed/total_rows` would divide by zero.
                if len(context.rows) == 0:
                    st.warning(f"Skipping {file.name}: no translatable rows found (empty or header-only file).")
                    continue

                if context.is_same_language_localization:
                    dialect_skipped = skip_dialect_excluded_rows(context)

                st.subheader(f"Processing: {file.name}")
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                # LIVE PREVIEW SETUP
                st.caption("Live Activity Log (Showing last 5 processed rows)")
                live_table_placeholder = st.empty()
                preview_data = []

                # SEMAPHORE: Control parallelism (e.g., 15 concurrent requests)
                semaphore = asyncio.Semaphore(15)

                async def run_file_processing():
                    # Async style inference (Layer 3)
                    # Skip for same-language localization — style-driven restructuring
                    # (first-person prefixes, phrase-form enforcement) must not run when
                    # the task is dialect adaptation, not full translation.
                    if not context.is_same_language_localization:
                        status_text.text("Inferring block styles...")
                        await infer_block_styles_async(context, global_context, semaphore)

                    # Batch-translate scale labels (one LLM call per block)
                    if context.blocks:
                        # Pre-translate question rows for blocks with scales so the
                        # scale batch call can align vocabulary with the translated stem.
                        scale_blocks = [
                            b for b in context.blocks
                            if b.scale_label_indices and len(b.scale_label_indices) >= 2
                        ]
                        if scale_blocks:
                            status_text.text("Pre-translating question stems for scale blocks...")
                            q_tasks = []
                            q_block_map: Dict[int, List[int]] = {}
                            for block in scale_blocks:
                                q_indices = [
                                    i for i in block.question_indices
                                    if 0 <= i < len(context.rows) and (context.rows[i].english_text or "").strip()
                                ]
                                if q_indices:
                                    q_block_map[block.block_id] = q_indices
                                    for qi in q_indices:
                                        qrow = context.rows[qi]
                                        if qrow.had_real_translation and not provide_suggestions:
                                            qrow.new_translation = qrow.existing_translation
                                            qrow.batch_translated = True
                                        else:
                                            q_tasks.append(
                                                call_translation_model_async(
                                                    english_text=qrow.english_text,
                                                    language_code=context.language_code,
                                                    locale_code=context.locale_code,
                                                    global_context=global_context,
                                                    translation_memory=context.translation_memory,
                                                    existing_translation=qrow.existing_translation if qrow.had_real_translation else None,
                                                    segment_type=qrow.segment_type,
                                                    block_style=(context.block_styles.get(qrow.block_id) if context.block_styles and qrow.block_id is not None else None),
                                                    gender_inclusive=gender_inclusive,
                                                    is_same_language_localization=context.is_same_language_localization,
                                                )
                                            )
                            if q_tasks:
                                q_results = await asyncio.gather(*q_tasks, return_exceptions=True)
                                result_iter = iter(q_results)
                                for block in scale_blocks:
                                    for qi in q_block_map.get(block.block_id, []):
                                        qrow = context.rows[qi]
                                        if qrow.batch_translated:
                                            continue
                                        result = next(result_iter)
                                        if isinstance(result, Exception):
                                            continue
                                        eng = (qrow.english_text or "").strip()
                                        proposed = pick_final_translation(result, english_text=eng)
                                        if not proposed:
                                            continue
                                        is_ok, _ = validate_translation_structure(eng, proposed)
                                        if not is_ok:
                                            repaired = attempt_placeholder_repair(eng, proposed)
                                            if repaired:
                                                re_ok, _ = validate_translation_structure(eng, repaired)
                                                if re_ok:
                                                    proposed = repaired
                                                    is_ok = True
                                        if not qrow.had_real_translation:
                                            qrow.new_translation = proposed
                                            qrow.was_newly_translated = True
                                        else:
                                            qrow.new_translation = qrow.existing_translation
                                            if result.get("needs_change") and proposed != qrow.existing_translation:
                                                if provide_suggestions:
                                                    qrow.suggested_translation = proposed
                                                    qrow.suggestion_reason = result.get("change_reason", "")
                                                else:
                                                    qrow.new_translation = proposed
                                        qrow.batch_translated = True

                        status_text.text("Translating scale labels...")
                        scale_tasks = []
                        for block in scale_blocks:
                            translated_q = " ".join(
                                (context.rows[i].new_translation or context.rows[i].existing_translation or "")
                                for i in block.question_indices
                                if 0 <= i < len(context.rows)
                            ).strip()
                            # Step 24: extract concept term from English question stem.
                            eng_q = " ".join(
                                (context.rows[i].english_text or "")
                                for i in block.question_indices
                                if 0 <= i < len(context.rows)
                            ).strip()
                            concept_term = _extract_concept_term(eng_q)
                            scale_tasks.append(
                                translate_scale_batch_async(
                                    context, block, global_context, semaphore,
                                    provide_suggestions, gender_inclusive=gender_inclusive,
                                    translated_question_context=translated_q,
                                    concept_term_english=concept_term,
                                )
                            )
                        if scale_tasks:
                            await asyncio.gather(*scale_tasks, return_exceptions=True)

                    tasks = []
                    total_rows = len(context.rows)

                    # Create tasks (batch-translated rows will return immediately)
                    for row in context.rows:
                        tasks.append(process_row_async(row, context, global_context, semaphore, provide_suggestions, gender_inclusive=gender_inclusive, enable_judge=enable_judge))

                    # Run tasks and update UI incrementally
                    completed_count = 0
                    for f in asyncio.as_completed(tasks):
                        try:
                            row = await f
                        except Exception as _row_err:
                            completed_count += 1
                            progress_bar.progress(completed_count / max(total_rows, 1))
                            status_text.text(f"Row error (skipped): {_row_err}")
                            continue
                        completed_count += 1

                        # Update Progress
                        progress = completed_count / total_rows
                        progress_bar.progress(progress)
                        status_text.text(f"Processed {completed_count}/{total_rows} rows...")

                        # Update Live Preview (Every 2 rows for smoother feedback)
                        if completed_count % 2 == 0:
                            eng_preview = (row.english_text or "")
                            new_preview = (row.new_translation or row.existing_translation or "")

                            preview_data.append({
                                "English Source": eng_preview,
                                "Translation": new_preview
                            })
                            # Only the last 5 are ever rendered; keep the list bounded.
                            if len(preview_data) > 5:
                                del preview_data[:-5]

                            # CHANGE 2: Overwrite the placeholder
                            live_table_placeholder.dataframe(
                                pd.DataFrame(preview_data[-5:]),
                                use_container_width=True,
                                hide_index=True
                            )

                    # Post-translation style re-check (skip for dialect adaptation)
                    if not context.is_same_language_localization:
                        status_text.text("Running style re-check...")
                        restyle_count = await restyle_mismatched_rows(
                            context, global_context, semaphore,
                            gender_inclusive=gender_inclusive,
                            provide_suggestions=provide_suggestions,
                        )
                        if restyle_count:
                            status_text.text(f"Style re-check: {restyle_count} rows re-translated for style alignment.")

                    # Suggest-only semantic verification via back-translation (Step 13).
                    # Runs after translations are stable; flags drift in suggestion_reason only.
                    if provide_suggestions and not context.is_same_language_localization:
                        status_text.text("Running semantic verification (back-translation)...")
                        drift = await semantic_verification_pass(
                            context, global_context, semaphore,
                            provide_suggestions=provide_suggestions,
                        )
                        if drift:
                            status_text.text(f"Semantic verification: {drift} row(s) flagged for review.")

                # Execute the async loop
                loop.run_until_complete(run_file_processing())

                # Phases 1, 2, 4, 5: structural fixes, v7 deterministic, emphasis flags.
                # Phase 6 (terminal punctuation) is deferred until after consistency.
                run_post_processors(context, phases=[1, 2, 4, 5],
                                    status_fn=status_text.text)

                # Post-processing
                status_text.text("Running Consistency Pass & Style Checks...")

                # Only do style warnings / suggestion columns when enabled
                # (skip entirely for dialect adaptation — no style enforcement)
                if provide_suggestions and not context.is_same_language_localization:
                    block_style_validation(context)

                # Consistency pass should run regardless; when suggestions are OFF,
                # apply the canonical form only to rows translated in this run.
                if enable_consistency:
                    try:
                        consistency_pass(context, apply_to_new_translations=(not provide_suggestions), global_context=global_context)
                    except Exception as _cp_err:
                        st.warning(f"Consistency pass skipped for {file.name} due to an error: {_cp_err}")

                    # Re-assert phase-1 (question-mark stripping) and phase-6
                    # (terminal punctuation) after consistency may have rewritten rows.
                    run_post_processors(context, phases=[1, 6])

                # Always-on structural audit of every shipped row (read-only, non-destructive).
                audit_shipped_rows(context)

                out_df, out_filename, excel_bytes = write_output_file(
                    context, original_df, include_suggestions=provide_suggestions
                )

                # Calculate Stats
                n_new = sum(1 for r in context.rows if r.was_newly_translated)
                n_sugg = (
                    sum(
                        1
                        for r in context.rows
                        if (r.suggested_translation or (r.suggestion_reason or "").strip())
                    )
                    if provide_suggestions
                    else 0
                )
                n_err = sum(
                    1
                    for r in context.rows
                    if (r.suggestion_reason or "").startswith("LLM Error")
                    or "translation likely failed" in (r.suggestion_reason or "")
                    or "STRUCTURE FLAG" in (r.suggestion_reason or "")
                    or "Structure validation warning" in (r.suggestion_reason or "")
                    or "[TRANSLATION FAILED" in (r.new_translation or "")
                )
                # Judge stats: only meaningful when enable_judge was True.
                n_judge_retried = sum(1 for r in context.rows if r.judge_retried)
                n_judge_flagged = sum(1 for r in context.rows if r.judge_outcome == "retried_flagged")

                result = {
                    "file_name": file.name,
                    "out_filename": out_filename,
                    "excel_bytes": excel_bytes,
                    "num_new_translations": n_new,
                    "num_suggestions": n_sugg,
                    "num_error_rows": n_err,
                    "num_judge_retried": n_judge_retried,
                    "num_judge_flagged": n_judge_flagged,
                    "is_same_language_localization": context.is_same_language_localization,
                }

                # Immediately persist to session state so the download survives
                # even if a later file errors out
                st.session_state["processed_results"].append(result)

                st.success(f"Completed {file.name}!")

                # Render download in a fragment so clicking it won't cancel the
                # processing loop for remaining files.
                _dl_idx = len(st.session_state["processed_results"]) - 1

                @st.fragment
                def _inline_download(idx=_dl_idx):
                    res = st.session_state["processed_results"][idx]
                    st.download_button(
                        label=f"Download: {res['out_filename']}",
                        data=res["excel_bytes"],
                        file_name=res["out_filename"],
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_inline_{res['file_name']}",
                    )

                _inline_download()

            except Exception as _file_err:
                fname = getattr(cfg.get('file'), 'name', str(cfg.get('file', '?')))
                st.error(f"Failed to process {fname}: {_file_err}")
                continue

        # All files processed; close the event loop and release the client so
        # a subsequent run in the same Streamlit session starts clean.
        loop.close()
        reset_async_client()

        if _SYSTEM_FINGERPRINTS:
            st.caption(
                "Model system_fingerprint(s) this run: "
                + ", ".join(sorted(_SYSTEM_FINGERPRINTS))
                + (f" | seed={TRANSLATION_SEED}" if TRANSLATION_SEED is not None else " | seed=unset")
            )

    # After possible run, render download buttons from cached results
    processed_results: List[Dict[str, object]] = st.session_state.get("processed_results", [])

    if not processed_results:
        st.info("Upload file(s), adjust settings, and click 'Run Translation Pipeline' to generate outputs.")
        st.stop()

    st.markdown("---")
    st.subheader("Download processed files")

    @st.fragment
    def _render_downloads():
        results = st.session_state.get("processed_results", [])
        for res in results:
            new_label = (
                "Rows localized" if res.get("is_same_language_localization")
                else "New translations (former English placeholders)"
            )
            judge_note = ""
            if res.get("num_judge_retried"):
                judge_note = (
                    f" | Judge retried: {res['num_judge_retried']}"
                    + (
                        f" (still flagged: {res['num_judge_flagged']} — see judge_status column)"
                        if res.get("num_judge_flagged")
                        else " (all passed)"
                    )
                )
            st.success(
                f"Finished processing `{res['file_name']}`. "
                f"{new_label}: {res['num_new_translations']} | "
                f"Rows with suggestions/warnings: {res['num_suggestions']} | "
                f"Rows needing review (errors/structure flags): {res['num_error_rows']}"
                + judge_note
            )

            st.download_button(
                label=f"Download processed file: {res['out_filename']}",
                data=res["excel_bytes"],
                file_name=res["out_filename"],
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_{res['file_name']}",
            )

        if len(results) > 1:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for res in results:
                    zf.writestr(res["out_filename"], res["excel_bytes"])
            zip_buffer.seek(0)

            st.download_button(
                label="Download ALL processed files as ZIP",
                data=zip_buffer.getvalue(),
                file_name="processed_translations.zip",
                mime="application/zip",
                key="download_all_zip",
            )

    _render_downloads()


if __name__ == "__main__":
    main()