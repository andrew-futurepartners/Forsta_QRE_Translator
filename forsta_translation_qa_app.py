import os
import io
import json
import re
import time
import asyncio
import string
from dataclasses import dataclass
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


# ==========================
# Configuration & Constants
# ==========================

DEFAULT_GLOBAL_CONTEXT = (
    "You are translating a travel & tourism market research questionnaire. "
    "The audience is respondents in the target locale. The text is survey content "
    "(questions, answer options, scale labels, messages), not marketing copy."
)

TRANSLATION_MODEL_NAME = os.getenv("TRANSLATION_MODEL", "gpt-5-mini")
CONSISTENCY_MODEL_NAME = os.getenv("CONSISTENCY_MODEL", "gpt-5-mini")

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
}

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
    "australia": "en-AU",
    "en": "en",
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
}

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

@dataclass
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
    # Layer 3: style plan per question block
    block_styles: Optional[Dict[int, BlockStyle]] = None


# ==========================
# Utility Functions
# ==========================

# ==========================
# Structural Classification
# ==========================

def classify_segment_type(english_text: str) -> SegmentType:
    """
    Classify a single English survey text into a coarse structural type:
    question / instruction / answer option / scale label / other.

    This is intentionally heuristic and language-agnostic. It does NOT try
    to recognize specific concepts like employment status; it only cares
    about structure and length.

    Improvements:
    - Strips HTML tags before classification so HTML-wrapped questions
      are still recognized as questions.
    - Broader set of common instruction phrases (e.g., "Be specific",
      "Please describe", "Provide details"), which improves behavior in
      all target languages.
    """
    s = (english_text or "").strip()
    if not s:
        return SegmentType.OTHER

    # Remove HTML tags for classification purposes (but do NOT modify the
    # original text elsewhere; this is only for deciding the segment type).
    text_no_tags = re.sub(r"<[^>]+>", " ", s)
    lower = text_no_tags.lower().strip()

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

    # Short, no sentence punctuation → label-like thing: option or scale label
    # (we use the original string here so HTML-only rows don't get mis-tagged)
    if len(s) <= 60 and not any(p in s for p in ".?!;:"):
        # Look for classic Likert / scale terms first (on the HTML-stripped text)
        if re.search(
            r"(strongly|somewhat|agree|disagree|neither|satisfied|dissatisfied|likely|unlikely|"
            r"very|extremely|poor|excellent|good|bad|fair)",
            lower,
        ):
            return SegmentType.SCALE_LABEL
        # Otherwise treat as generic answer option
        return SegmentType.ANSWER_OPTION

    # Everything else
    return SegmentType.OTHER


def classify_segments(context: SurveyFileContext) -> None:
    """
    Layer 1: assign a structural segment_type to each row in the file.
    """
    for row in context.rows:
        row.segment_type = classify_segment_type(row.english_text)

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
    return blocks

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

    # Fallback: substring search on raw string (in case language appears later)
    for lname, code in LANGUAGE_NAME_TO_CODE.items():
        if lname in raw_no_parens:
            return code

    # If we get here, we truly don't know.
    return ""


def map_language_and_locale_to_bcp47(language_code: str, locale_name: Optional[str]) -> str:
    if not language_code:
        return ""

    if not locale_name:
        # Use generic language-only code if no locale
        return language_code

    key = locale_name.strip().lower().replace(" ", "_")

    if language_code == "es":
        return SPANISH_LOCALE_NAME_TO_BCP47.get(key, "es")
    if language_code == "en":
        return ENGLISH_LOCALE_NAME_TO_BCP47.get(key, "en")

    # Extra: if filename already contains a BCP-47-like code (e.g. es-MX), just return it
    if re.match(rf"^{language_code}-[A-Za-z]+$", locale_name.strip()):
        return locale_name.strip()

    # Fallback for other languages: if LOCALE_OPTIONS has a matching label, use its code
    options = LOCALE_OPTIONS.get(language_code)
    if options:
        for label, code in options:
            if key in label.lower().replace(" ", "_"):
                return code

    # Generic fallback: language + uppercased locale token
    token = locale_name.strip().upper()
    return f"{language_code}-{token}"



def parse_language_and_locale_from_filename(filename: str) -> Tuple[str, str]:
    """
    Parse filenames like:
      12345_Spanish_Argentina.xlsx
      12345_English_UK.csv
      <survey_id>_<language>[_<localization>].ext

    Returns (language_code, locale_code), empty string when detection fails.
    """
    base_name = filename_without_extension(filename)
    parts = base_name.split("_")

    language_name = parts[1] if len(parts) >= 2 else None
    locale_name = parts[2] if len(parts) >= 3 else None

    language_code = map_language_name_to_code(language_name)
    locale_code = map_language_and_locale_to_bcp47(language_code, locale_name) if language_code else ""

    return language_code, locale_code


def read_excel_or_csv(file) -> pd.DataFrame:
    filename = getattr(file, "name", "uploaded_file")
    suffix = Path(filename).suffix.lower()

    if suffix in [".xls", ".xlsx"]:
        df = pd.read_excel(file)
    elif suffix == ".csv":
        df = pd.read_csv(file)
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

def should_run_copy_check(english_text: str) -> bool:
    """
    Decide whether it makes sense to flag 'unchanged English' as a likely failure.

    We ONLY want to run the copy-check for content that clearly ought to be translated
    (questions, full phrases, longer labels), and we want to allow cases where it is
    normal for the English form to appear unchanged in the target language, such as:

      - pure numeric ranges or codes (e.g., '1970-1989', '2024'),
      - simple proper nouns / place names (e.g., 'Riverside'),
      - very short single-word labels that are often the same across languages
        (e.g., 'No', 'OK').

    This function is deliberately language-agnostic; it only inspects the English form.
    """
    if not english_text:
        return False

    # Strip HTML tags for the purpose of this heuristic
    text = re.sub(r"<[^>]+>", " ", english_text)
    text = text.strip()
    if not text:
        return False

    # 1) Pure numeric / range / code-like content → usually safe to leave as-is
    #    (e.g., '1970-1989', '2024', '$100', '10-15%')
    if re.fullmatch(r"[\d\s\-\–/.,%+$€£¥]+", text):
        return False

    tokens = text.split()
    if len(tokens) == 1:
        token = tokens[0]

        # 2) Proper-noun / code-like single token:
        #    - Starts with uppercase followed by letters/digits (e.g., 'Riverside', 'Q1'),
        #    - OR is all caps / alphanumeric (e.g., 'USA', 'NYC').
        if re.fullmatch(r"[A-Z][A-Za-z0-9]*", token) or re.fullmatch(r"[A-Z0-9]{2,}", token):
            return False

        # 3) Very short single tokens (length <= 3) are often language-invariant labels
        #    like 'No', 'OK', etc. We avoid over-flagging these.
        if len(token) <= 3:
            return False

    # For longer/multi-word texts we DO want to check for no-op translations
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
def is_label_like_english(text: str) -> bool:
    """
    Heuristic to detect short, stand-alone label-like English text
    (e.g., 'January', 'Very poor', 'Strongly Agree').

    We use this to optionally adjust capitalization of the translation for
    answer options, without touching full sentences.
    """
    s = (text or "").strip()
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
) -> str:
    """
    For short, label-like English texts, adjust the translation so that the
    first alphabetic character is uppercase (for languages with case).

    This helps make response options like month names look like labels
    ('enero' -> 'Enero') without interfering with full sentences.
    """
    if not translation_text:
        return translation_text

    if not is_label_like_english(english_text):
        return translation_text

    # Languages without case (e.g. Japanese, Chinese) – do nothing
    if language_code in {"ja", "zh"}:
        return translation_text

    chars = list(translation_text)
    for i, ch in enumerate(chars):
        if ch.isalpha():
            chars[i] = ch.upper()
            break
    return "".join(chars)


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

        eng_norm = eng_text_str.strip()
        trl_norm = trl_str.strip()

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


def sample_translation_memory_examples(
    translation_memory: Dict[str, Dict[str, str]],
    max_examples: int = 10,
) -> List[Tuple[str, str]]:
    """
    Return up to max_examples of (english, translation) pairs from translation_memory.
    """
    items = list(translation_memory.values())
    examples: List[Tuple[str, str]] = []
    for val in items:
        if isinstance(val, dict):
            eng = val.get("english", "")
            trl = val.get("translation", "")
        else:
            # Backwards-compat, if ever a plain string sneaks in
            eng = ""
            trl = str(val)
        if eng and trl:
            examples.append((eng, trl))
        if len(examples) >= max_examples:
            break
    return examples


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
        model_name: str = TRANSLATION_MODEL_NAME,
) -> Dict[str, object]:
    client = get_async_client()
    existing_translation = existing_translation or ""

    # 1. Memory Construction
    memory_examples = sample_translation_memory_examples(translation_memory, max_examples=5)
    memory_str = "\n".join([f'- "{e}" -> "{t}"' for e, t in memory_examples]) if memory_examples else "None."

    # 2. Style & Context Construction
    segment_type_str = segment_type.value if isinstance(segment_type, SegmentType) else "other"

    # Block-level style plan (may be None)
    block_style_info = {}
    if isinstance(block_style, BlockStyle):
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
            f'CONTEXT ALERT: The text below is an answer option or label for this question: '
            f'"{parent_context}".\n'
            f"Ensure your translation fits grammatically and logically as a response to this question."
        )

    peer_options_instruction = ""
    if peer_english_options and segment_type_str == "answer_option":
        # Provide the full ordered peer set so the model can keep options parallel.
        peers_json = json.dumps(peer_english_options, ensure_ascii=False)
        peer_options_instruction = (
            f"Peer answer options in this same set (English, ordered): {peers_json}\n"
            "Keep this option grammatically and stylistically PARALLEL to its peers. "
            "Do not mix label types (e.g., person-noun labels vs adjective labels) within the same set."
        )

    # --------- System & User Prompts (now outside the if-block) ---------
    system_prompt = """
You are a professional translator and QA specialist for market-research questionnaires
in the travel & tourism domain. You translate from English into a specified target
language and locale.

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

3. Tone and register
   - Use a formal-neutral, polite tone appropriate for an official survey from a tourism
     board, municipality, or research agency.
   - Avoid slang, jokes, or marketing hype. Also avoid legalese or bureaucratic jargon
     unless the English clearly uses it.
   - Aim for clear, plain language a typical adult in the target locale would understand
     on first read.

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
     {Q1}, [PIPE:DESTINATION], [[VARNAME]], $VARNAME).

6. Output format
   - You MUST always return a valid JSON object with the required keys and no extra text.
   - The 'change_reason' field MUST always be written in ENGLISH, even when you are
     translating into another language.
"""

    user_prompt = f"""
Target language code: {language_code}
Target locale code: {locale_code}

Global survey context:
{global_context}

Segment metadata for this element:
- segment_type: {segment_type_str}
- block_style (JSON, may be null): {block_style_json}

{context_instruction}

{peer_options_instruction}

English source text:
\"\"\"{english_text}\"\"\"

Existing translation in the target language (may be empty or just a copy of the English):
\"\"\"{existing_translation}\"\"\"

Translation memory examples (English -> target translation):
{memory_str}

Interpretation of segment_type and block_style:
- If segment_type = "question":
    - Translate as a full, natural question in the target language, using the polite form
      that is standard for surveys in the target locale.
- If segment_type = "instruction":
    - Translate as a clear, polite imperative or directive, as a complete sentence.
      (For example: equivalents of "Select one option", "Select all that apply",
      "Enter a number", "Please describe".)
- If segment_type = "answer_option":
    - Treat this as a stand-alone answer choice shown under a question.
    - If block_style provides options_style (grammatical_person, phrase_form, tone),
      use it as the style plan for this block. It is acceptable to rewrite an existing
      translation to match this plan as long as you do NOT change the underlying meaning
      or distinctions.
    - Keep answer options concise and parallel in style within the block (all labels or
      all self-descriptions, not a random mix), as is natural in the target language.
- If segment_type = "scale_label":
    - Treat this as a label on a rating scale (e.g., satisfaction, agreement).
    - Use short, symmetric phrases that clearly express the relative position on the
      scale (from most positive to most negative or vice versa), rather than long
      sentences or self-referential statements.
    - Do NOT add parenthetical or slash-based variants (for example masculine/feminine
      endings like "(a)" or "/a", or multiple gender forms) unless the English source
      explicitly includes them. Keep each label as a single, simple form.
    - When an existing translation already forms a clear, well-ordered scale, avoid
      proposing changes that only reflect stylistic preferences (such as gender-inclusive
      morphology or alternative but equivalent wording). Only propose changes to scale
      labels when they fix problems with semantics, ordering, clarity, or obvious
      unnaturalness in the target language.


Instructions:
1. If the existing translation is effectively empty or simply repeats the English text, treat this as if there
   were no translation yet. In this case you MUST propose a high-quality translation whose main human-readable
   content is clearly in the target language, not in English. It is incorrect to simply copy the English sentence,
   except for proper names, brand names, and technical tokens.
2. If the existing translation is non-empty and clearly already in the target language, treat it as the baseline
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
   existing label is already natural and forms part of a clear, symmetric scale.
3. Always perform a self-QA step on your proposed translation. If your proposed translation, after stripping HTML
   tags and condensing whitespace, is still essentially identical to the English text, you must reconsider and
   produce a real translation in the target language.
4. You MUST NOT change or remove any HTML tags, placeholders, or piping tokens. Only translate the text between them.
5. Return ONLY a valid JSON object with the following keys:
   - "proposed_translation": string
   - "qa_checked_translation": string
   - "needs_change": boolean
   - "change_reason": string (short explanation in English; empty if no change needed)

Very important when rewriting:
- You MUST keep all critical qualifiers from the English and any existing translation
  (for example: temporary vs permanent, unpaid vs paid, full-time vs part-time, looking
  for work vs not looking for work, disability, retired, etc.).
- You MUST preserve all tags, placeholders and piping tokens exactly.

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
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            result = json.loads(response.choices[0].message.content)

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


def infer_style_for_block(
    context: SurveyFileContext,
    block: QuestionBlock,
    global_context: str,
    model_name: str = TRANSLATION_MODEL_NAME,
) -> BlockStyle:
    """
    Use the LLM to infer a style plan for a single question block.

    The model does NOT translate; it only decides how answer options and scale labels
    should be phrased in the target language/locale (e.g., first-person clauses vs
    short noun labels).
    """
    client = get_llm_client()

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

    system_prompt = (
        "You are a professional translator and QA specialist for travel and tourism "
        "market research questionnaires. You translate from English into the specified "
        "target language and locale. Your goal is to produce high-quality, locally natural "
        "survey text that preserves measurement properties and questionnaire structure.\n\n"
        "General requirements:\n"
        "- Use the standard variety of the target language that is appropriate for the given locale "
        "  (for example, the variety normally used in official surveys for that country or region).\n"
        "- Preserve the meaning of the source exactly: do not add or drop concepts, qualifiers, or exclusions.\n"
        "- Keep tense consistent with the English text and internally consistent between question stems and "
        "  answer options.\n"
        "- Use terminology that is clear to a general adult audience in the target market; prefer common, "
        "  everyday words over rare or overly technical synonyms unless the domain truly requires technical language.\n"
        "- When translating rating scales or Likert-type items, keep the same number of points, preserve the "
        "  direction (positive → negative or vice versa), and choose a set of labels that are monotonic and "
        "  stylistically consistent in the target language. Avoid mixing very formal/technical labels with casual ones "
        "  in the same scale.\n"
        "- For short routing or instruction texts (e.g. 'Select one', 'Select all that apply'), render them as "
        "  complete, natural-sounding instructions in the target language (the equivalent of 'Select one option' or "
        "  'Select all that apply') rather than fragmentary literal translations.\n"
        "- For short, stand-alone response options (for example month names or single-word labels), follow typical "
        "  survey label conventions for the target language: it is acceptable to capitalize them as labels (such as "
        "  starting with an uppercase letter) even if they would normally be lowercase in running text. Do not change "
        "  capitalization inside longer sentences.\n"
        "- For geographic names (cities, regions, valleys, etc.), use the form that is standard in the target "
        "  language for that locale when such a form exists; otherwise keep the original name. Do not translate "
        "  brand or platform names such as Forsta, Decipher, or DeepL.\n"
        "- Preserve all numbers, numeric ranges, and currency symbols exactly; adjust only the decimal/thousand "
        "  separators and spacing according to the target locale's conventions.\n"
        "- Preserve survey-specific markup, HTML tags, placeholders, and piping tokens exactly as they appear. "
        "  You MUST NOT change, drop, or re-order any tags or tokens; only translate the human-readable text "
        "  between them.\n"
        "- When the existing translation is empty or just a copy of the English, you MUST produce a translation "
        "  whose main human-readable content is in the target language, not in English. It is incorrect to simply "
        "  echo the English sentence (except for proper names, brand names, and technical tokens).\n"
        "- Any explanations you provide in the `change_reason` field must be written in English, regardless of "
        "  the target language.\n"
        "- Always return valid JSON with the required keys and no extra commentary."
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
            response = client.chat.completions.create(
                model=model_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
            data = json.loads(content)

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
                time.sleep(2 ** attempt)
                continue
            break

    # Fallback: default style if inference fails
    return BlockStyle(
        block_id=block.block_id,
        options_grammatical_person="unspecified",
        options_phrase_form="unspecified",
        options_tone="formal_neutral",
        scale_label_phrase_form="short_phrase",
        notes=f"Style inference failed: {last_exception}",
    )


def infer_block_styles(
    context: SurveyFileContext,
    global_context: str,
    model_name: str = TRANSLATION_MODEL_NAME,
) -> Dict[int, BlockStyle]:
    """
    Infer a style plan for each question block (Layer 3: style planning).
    Stores the result on context.block_styles and also returns it.
    """
    block_styles: Dict[int, BlockStyle] = {}

    if not context.blocks:
        context.block_styles = block_styles
        return block_styles

    for block in context.blocks:
        # Only infer style when there are options or scale labels; otherwise leave default.
        if not (block.answer_option_indices or block.scale_label_indices):
            continue

        style = infer_style_for_block(context, block, global_context, model_name=model_name)
        block_styles[block.block_id] = style

    context.block_styles = block_styles
    return block_styles


def call_consistency_model(
    context: SurveyFileContext,
    phrase_groups: List[Dict[str, object]],
    model_name: str = CONSISTENCY_MODEL_NAME,
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

    system_prompt = (
        "You are a localization QA and terminology consistency specialist for travel and tourism "
        "market research questionnaires. You review translated survey text in the target language "
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
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
            data = json.loads(content)
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
                raise RuntimeError(f"Consistency model call failed: {last_exception}")


# ==========================
# Structure Validation
# ==========================

def extract_numeric_tokens(text: str) -> List[str]:
    if not text:
        return []
    # Rough pattern for numbers with optional currency and percent
    pattern = r'[\$€£¥]?\d+(?:[.,]\d+)?%?'
    return re.findall(pattern, text)


def extract_placeholder_tokens(text: str) -> List[str]:
    if not text:
        return []
    tokens: List[str] = []
    patterns = [
        r'\{[^}]+\}',      # {placeholder}
        r'\[[^\]]+\]',     # [placeholder]
        r'<[^>]+>',        # <tag> or <% piping %>
        r'\$\w+',          # $VARNAME
        r'\[\[\w+\]\]',    # [[VARNAME]]
    ]
    for pat in patterns:
        tokens.extend(re.findall(pat, text))
    # Deduplicate
    return sorted(set(tokens))


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

    # Numeric tokens: compare digit sequences only
    eng_nums = extract_numeric_tokens(english_text)
    digits_in_trl = re.sub(r"\D", "", translation_text)

    missing_numeric: List[str] = []
    for tok in eng_nums:
        digits = re.sub(r"\D", "", tok)
        if digits and digits not in digits_in_trl:
            missing_numeric.append(tok)

    # Placeholder/tokens
    eng_placeholders = extract_placeholder_tokens(english_text)
    missing_placeholders = [tok for tok in eng_placeholders if tok not in translation_text]

    if missing_numeric or missing_placeholders:
        parts = []
        if missing_numeric:
            parts.append("numerics " + ", ".join(missing_numeric))
        if missing_placeholders:
            parts.append("placeholders " + ", ".join(missing_placeholders))
        msg = "Missing or altered " + " and ".join(parts) + " compared to the English source."
        return False, msg

    return True, ""

# ==========================
# Block-level Style QA
# ==========================

def get_first_person_regexes(language_code: str) -> List[re.Pattern]:
    """
    Very lightweight heuristics for detecting first-person-like phrases
    in major languages. Used only for QA / style pattern detection.
    """
    lc = (language_code or "").lower()

    patterns: List[str] = []
    if lc.startswith("en"):
        patterns = [
            r"\bi\b",
            r"\bi'm\b",
            r"\bi am\b",
            r"\bi’ve\b",
            r"\bi'd\b",
        ]
    elif lc.startswith("es"):
        patterns = [
            r"\byo\b",
            r"\bsoy\b",
            r"\bestoy\b",
            r"\btrabajo\b",
            r"\btengo\b",
        ]
    elif lc.startswith("fr"):
        patterns = [
            r"\bje\b",
            r"\bj['’]",     # j'...
            r"\bje suis\b",
        ]
    elif lc.startswith("pt"):
        patterns = [
            r"\beu\b",
            r"\bsou\b",
            r"\bestou\b",
            r"\btrabalho\b",
        ]
    elif lc.startswith("de"):
        patterns = [
            r"\bich\b",
            r"\bich bin\b",
        ]
    elif lc.startswith("it"):
        patterns = [
            r"\bio\b",
            r"\bsono\b",
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
      - 'other'
    This is heuristic and only used for QA/warnings.
    """
    s = (translation_text or "").strip()
    if not s:
        return "unknown"

    lower = s.lower()

    # First-person-like?
    for regex in get_first_person_regexes(language_code):
        if regex.search(lower):
            return "first_person_like"

    # Short label-like: short text, no sentence punctuation
    if len(s) <= 60 and not any(p in s for p in ".?!;:"):
        return "short_label_like"

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
                    # Only warn when the majority pattern is informative
                    if majority_pattern in {"first_person_like", "short_label_like"}:
                        for idx, pat in zip(option_indices, option_patterns):
                            if (
                                pat != majority_pattern
                                and pat != "unknown"
                                and pat != "other"
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
        first_person_regexes = get_first_person_regexes(lang)

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
        semaphore: asyncio.Semaphore,  # <--- Limit concurrency
) -> SurveyRow:
    # Acquire slot in the semaphore (e.g., max 20 active requests)
    async with semaphore:
        eng_text = (row.english_text or "").strip()
        if not eng_text:
            row.new_translation = row.existing_translation
            return row

        # Hard guard: if the source is *purely* numeric/range/code-like (e.g., '1970-1989'),
        # keep it as a pure range/code in the output. This prevents drift into prose like
        # 'Born between 1950 and 1969' and preserves visual parallelism across option sets.
        if row.segment_type in {SegmentType.ANSWER_OPTION, SegmentType.SCALE_LABEL} and is_pure_numeric_or_range_code_like(eng_text):
            if row.had_real_translation:
                # If an existing translation is already numeric/range-like, keep it. If it was paraphrased, suggest fixing it.
                if not is_pure_numeric_or_range_code_like(row.existing_translation):
                    row.new_translation = row.existing_translation
                    row.suggested_translation = eng_text
                    row.suggestion_reason = ((row.suggestion_reason + " | ") if row.suggestion_reason else "") + "Numeric/range option should remain a pure range/code (no prose rewrites)."
                else:
                    row.new_translation = row.existing_translation
                return row
            else:
                row.new_translation = eng_text
                row.was_newly_translated = True
                return row

        # Logic to find Parent Context (Question Text)
        parent_context_str = ""
        peer_english_options = None

        # For short categorical label sets, provide peer options to encourage parallel translations.
        if row.segment_type == SegmentType.ANSWER_OPTION and context.blocks and row.block_id is not None:
            try:
                block = context.blocks[row.block_id]
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
            if context.blocks and row.block_id is not None:
                block = context.blocks[row.block_id]
                # Get the question text(s) for this block
                q_texts = [context.rows[i].english_text for i in block.question_indices if context.rows[i].english_text]
                parent_context_str = " ".join(q_texts)

        # Call the model
        result = await call_translation_model_async(
            english_text=eng_text,
            language_code=context.language_code,
            locale_code=context.locale_code,
            global_context=global_context,
            translation_memory=context.translation_memory,
            existing_translation=row.existing_translation if row.had_real_translation else None,
            segment_type=row.segment_type,
            block_style=context.block_styles.get(row.block_id) if row.block_id is not None else None,
            peer_english_options=peer_english_options,
            parent_context=parent_context_str
        )

        # --- Process Result (Same logic as before, just adapted for async return) ---
        proposed = result.get("qa_checked_translation") or result.get("proposed_translation") or ""


        # Safety: if there was no prior real translation and the model's output is
        # effectively just the English again, treat this as a likely failure and flag it.
        # BUT only run this check for texts that clearly ought to be translated
        # (multi-word phrases, questions, longer labels). For simple numeric ranges,
        # proper nouns, or very short one-word labels, it's often correct for the
        # target-language form to match the English.
        if (not result.get("error")) and (not row.had_real_translation) and should_run_copy_check(eng_text):
            if is_effective_copy_of_english(eng_text, proposed):
                base = row.existing_translation or eng_text
                row.new_translation = base
                row.suggested_translation = base
                row.suggestion_reason = (
                    (row.suggestion_reason or "")
                    + "Model output is effectively identical to the English source; "
                      "translation likely failed. Please review/translate this row manually."
                )
                # Do NOT continue to treat this as a successful translation.
                return row


        if result.get("error"):
            row.suggestion_reason = f"LLM Error: {result.get('change_reason')}"
            row.new_translation = row.existing_translation or "[ERROR]"
        elif not row.had_real_translation:
            # New Translation
            row.new_translation = adjust_capitalization_for_label(eng_text, proposed, context.language_code)
            row.was_newly_translated = True
        else:
            # QA Existing
            row.new_translation = row.existing_translation
            if result.get("needs_change") and proposed != row.existing_translation:
                row.suggested_translation = adjust_capitalization_for_label(
                    eng_text, proposed, context.language_code
                )
                row.suggestion_reason = result.get("change_reason")

        return row


def consistency_pass(context: SurveyFileContext) -> None:
    """
    Survey-wide consistency pass (LLM-powered) using Fuzzy Matching.

    Groups similar English phrases (e.g. "Select one." and "select one")
    to ensure they are translated consistently.
    """

    # Helper for fuzzy normalization
    def normalize_fuzzy(text: str) -> str:
        # Lowercase and remove punctuation/whitespace
        return text.lower().translate(str.maketrans('', '', string.punctuation)).replace(" ", "")

    # Map: fuzzy_key -> { original_english -> { translation -> [indices] } }
    fuzzy_map: Dict[str, Dict[str, Dict[str, List[int]]]] = {}

    for idx, row in enumerate(context.rows):
        eng = (row.english_text or "").strip()

        # Skip rows already flagged for structural issues or empty translations
        if "Structure validation warning" in (row.suggestion_reason or ""):
            continue

        trl = (row.new_translation or row.existing_translation or "").strip()
        if not eng or not trl:
            continue

        fuzzy_key = normalize_fuzzy(eng)

        if fuzzy_key not in fuzzy_map: fuzzy_map[fuzzy_key] = {}
        if eng not in fuzzy_map[fuzzy_key]: fuzzy_map[fuzzy_key][eng] = {}
        if trl not in fuzzy_map[fuzzy_key][eng]: fuzzy_map[fuzzy_key][eng][trl] = []

        fuzzy_map[fuzzy_key][eng][trl].append(idx)

    # Convert the fuzzy map into phrase_groups for the LLM
    phrase_groups = []
    for fuzzy_key, eng_variants in fuzzy_map.items():
        # Aggregate all translations for this fuzzy meaning
        all_translations = {}  # translation -> indices

        # Pick the most common English variant as the "display" phrase, or just the first one
        primary_english = list(eng_variants.keys())[0]

        for original_eng, trl_dict in eng_variants.items():
            for trl, indices in trl_dict.items():
                if trl not in all_translations: all_translations[trl] = []
                all_translations[trl].extend(indices)

        # Only send to LLM if there is more than 1 distinct translation for this concept
        if len(all_translations) > 1:
            phrase_groups.append({
                "english_phrase": primary_english,
                "translations": [{"translation": t, "indices": i} for t, i in all_translations.items()]
            })

    if not phrase_groups:
        return  # nothing to do

    # Ask the LLM to decide canonical translations and which indices to update
    issues = call_consistency_model(context, phrase_groups)

    for issue in issues:
        english_phrase = issue.get("english_phrase", "")
        canonical = (issue.get("canonical_translation") or "").strip()
        indices_to_update = issue.get("indices_to_update") or []
        notes = issue.get("notes") or ""

        if not canonical or not indices_to_update:
            continue

        for idx in indices_to_update:
            # defensive: ensure idx is a valid row index
            if not isinstance(idx, int):
                continue
            if idx < 0 or idx >= len(context.rows):
                continue

            row = context.rows[idx]

            # Don't override existing suggestions from row-level QA or validation
            if (row.suggested_translation or "").strip():
                continue

            row.suggested_translation = canonical
            base_reason = (
                f"LLM consistency suggestion: The concept '{english_phrase}' "
                f"appears with multiple translations. "
                f"Suggested canonical: '{canonical}'."
            )
            if notes:
                row.suggestion_reason = base_reason + f" Note: {notes}"
            else:
                row.suggestion_reason = base_reason


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

    # Ensure at least 3 columns
    if df_out.shape[1] < 3:
        raise ValueError(
            f"Original DataFrame for '{context.filename}' has fewer than 3 columns."
        )

    translation_col_name = df_out.columns[2]

    # Add new columns if missing
    if "suggested_translation" not in df_out.columns:
        df_out["suggested_translation"] = ""
    if "suggestion_reason" not in df_out.columns:
        df_out["suggestion_reason"] = ""

    for i, row in enumerate(context.rows):
        # Column 2: translation (existing or new)
        final_translation = (
            row.new_translation
            if row.new_translation is not None
            else row.existing_translation
        )
        df_out.at[i, translation_col_name] = final_translation

        # Column 3 & 4: suggestions (if any)
        if row.suggested_translation is not None:
            df_out.at[i, "suggested_translation"] = row.suggested_translation
            df_out.at[i, "suggestion_reason"] = row.suggestion_reason or ""

    # Determine if there are any suggestions or warnings (either column)
    has_suggestions = (
        df_out["suggested_translation"].astype(str).str.strip().ne("").any()
        or df_out["suggestion_reason"].astype(str).str.strip().ne("").any()
    )

    base_name = filename_without_extension(context.filename)
    suffix = "_translated"
    if has_suggestions:
        suffix += "_WITH_SUGGESTIONS"
    output_filename = base_name + suffix + ".xlsx"

    # Serialize to Excel in memory, with an extra sheet for block-level style logs (Layer 4)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # Main translations sheet
        df_out.to_excel(writer, index=False, sheet_name="translations")

        # Optional style log sheet
        style_log_df = build_block_style_log_df(context)
        if style_log_df is not None and not style_log_df.empty:
            style_log_df.to_excel(writer, index=False, sheet_name="__style_log")

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
    # Default label for UI when unknown
    return "Spanish"


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

    st.subheader("Per-file Language & Locale Settings")

    file_configs = []

    for file in uploaded_files:
        filename = file.name
        detected_lang_code, detected_locale_code = parse_language_and_locale_from_filename(filename)

        detection_failed = not bool(detected_lang_code)
        # Fallback default for UI selection only (we still track that detection failed)
        if detection_failed:
            detected_lang_code = "es"
            detected_locale_code = "es"

        lang_label_default = code_to_language_label(detected_lang_code)

        with st.expander(f"Settings for {filename}", expanded=False):
            if detection_failed:
                st.warning(
                    "Could not reliably detect target language from the filename. "
                    "Please confirm the language and locale below before running the pipeline."
                )

            # Language dropdown
            selected_lang_label = st.selectbox(
                f"Target language for {filename}",
                options=list(LANGUAGE_LABEL_TO_CODE.keys()),
                index=list(LANGUAGE_LABEL_TO_CODE.keys()).index(lang_label_default)
                if lang_label_default in LANGUAGE_LABEL_TO_CODE
                else list(LANGUAGE_LABEL_TO_CODE.keys()).index("Spanish"),
                key=f"lang_{filename}",
            )
            language_code = LANGUAGE_LABEL_TO_CODE[selected_lang_label]

            # Locale dropdown options are driven by the selected language
            locale_options = LOCALE_OPTIONS.get(
                language_code,
                [(f"Generic ({language_code})", language_code)],
            )

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
                code for (label, code) in locale_options if label == selected_locale_label
            )

        file_configs.append(
            {
                "file": file,
                "language_code": language_code,
                "locale_code": locale_code,
            }
        )

    # Initialise / read session state for processed results
    if "processed_results" not in st.session_state:
        st.session_state["processed_results"] = []

    run_pipeline = st.button("Run Translation Pipeline")

    # When the button is clicked, run the heavy pipeline and cache results in session_state
    if run_pipeline:
        # Create a new event loop for this run
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        processed_results = []

        for cfg in file_configs:
            file = cfg["file"]
            file.seek(0)
            context, original_df = load_forsta_export(file, cfg["language_code"], cfg["locale_code"])

            # Pre-processing layers
            classify_segments(context)
            build_blocks(context)
            infer_block_styles(context, global_context)  # Still sync, but fast enough

            st.subheader(f"Processing: {file.name}")
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            # LIVE PREVIEW SETUP
            st.caption("Live Activity Log (Showing last 5 processed rows)")
            live_table_placeholder = st.empty()  # <--- CHANGE 1: Use empty() instead of container()
            preview_data = []

            # SEMAPHORE: Control parallelism (e.g., 15 concurrent requests)
            semaphore = asyncio.Semaphore(15)

            async def run_file_processing():
                tasks = []
                total_rows = len(context.rows)

                # Create tasks
                for row in context.rows:
                    tasks.append(process_row_async(row, context, global_context, semaphore))

                # Run tasks and update UI incrementally
                completed_count = 0
                for f in asyncio.as_completed(tasks):
                    row = await f
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

                        # CHANGE 2: Overwrite the placeholder
                        live_table_placeholder.dataframe(
                            pd.DataFrame(preview_data[-5:]),
                            use_container_width=True,
                            hide_index=True
                        )

            # Execute the async loop
            loop.run_until_complete(run_file_processing())

            # Post-processing
            status_text.text("Running Consistency Pass & Style Checks...")
            block_style_validation(context)
            if enable_consistency:
                consistency_pass(context)

            out_df, out_filename, excel_bytes = write_output_file(context, original_df)

            # Calculate Stats
            n_new = sum(1 for r in context.rows if r.was_newly_translated)
            n_sugg = sum(1 for r in context.rows if r.suggested_translation)

            processed_results.append({
                "file_name": file.name,
                "out_filename": out_filename,
                "excel_bytes": excel_bytes,
                "num_new_translations": n_new,
                "num_suggestions": n_sugg,
                "num_error_rows": 0
            })

            st.success(f"Completed {file.name}!")

        st.session_state["processed_results"] = processed_results

    # After possible run, render download buttons from cached results
    processed_results: List[Dict[str, object]] = st.session_state.get("processed_results", [])

    if not processed_results:
        st.info("Upload file(s), adjust settings, and click 'Run Translation Pipeline' to generate outputs.")
        st.stop()

    st.markdown("---")
    st.subheader("Download processed files")

    for res in processed_results:
        st.success(
            f"Finished processing `{res['file_name']}`. "
            f"New translations (former English placeholders): {res['num_new_translations']} | "
            f"Rows with suggestions/warnings: {res['num_suggestions']} | "
            f"Rows with LLM errors: {res['num_error_rows']}"
        )

        st.download_button(
            label=f"Download processed file: {res['out_filename']}",
            data=res["excel_bytes"],
            file_name=res["out_filename"],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_{res['file_name']}",
        )

    # Optional: one-click "download all" as a ZIP when there are multiple files
    if len(processed_results) > 1:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for res in processed_results:
                zf.writestr(res["out_filename"], res["excel_bytes"])
        zip_buffer.seek(0)

        st.download_button(
            label="Download ALL processed files as ZIP",
            data=zip_buffer.getvalue(),
            file_name="processed_translations.zip",
            mime="application/zip",
            key="download_all_zip",
        )


if __name__ == "__main__":
    main()