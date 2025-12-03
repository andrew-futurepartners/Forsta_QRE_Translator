import os
import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from dotenv import load_dotenv
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # We'll handle this gracefully later

# Load environment variables from a local .env file if present
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)


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


@dataclass
class SurveyFileContext:
    filename: str
    language_code: str
    locale_code: str
    rows: List[SurveyRow]
    # normalized_english -> {"english": original_english, "translation": translation}
    translation_memory: Dict[str, Dict[str, str]]


# ==========================
# Utility Functions
# ==========================

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

    # Fallback for other languages: language + uppercased locale token
    # e.g., "fr" + "CA" -> "fr-CA"
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
            "The 'openai' package is not installed. Install it with 'pip install openai'."
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


def call_translation_model(
    english_text: str,
    language_code: str,
    locale_code: str,
    global_context: str,
    translation_memory: Dict[str, Dict[str, str]],
    existing_translation: Optional[str] = None,
    model_name: str = TRANSLATION_MODEL_NAME,
) -> Dict[str, object]:
    """
    Calls the LLM to translate / QA a single survey row.

    Returns a dict with keys:
      - proposed_translation: str
      - qa_checked_translation: str
      - needs_change: bool
      - change_reason: str (ALWAYS in English)
      - error: bool
    """
    client = get_llm_client()

    memory_examples = sample_translation_memory_examples(translation_memory, max_examples=10)
    memory_str_lines = []
    for eng, trl in memory_examples:
        memory_str_lines.append(f'- "{eng}" -> "{trl}"')
    memory_str = "\n".join(memory_str_lines) if memory_str_lines else "None available."

    existing_translation = existing_translation or ""

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
        "- For geographic names (cities, regions, valleys, etc.), use the form that is standard in the target "
        "  language for that locale when such a form exists; otherwise keep the original name. Do not translate "
        "  brand or platform names such as Forsta, Decipher, or DeepL.\n"
        "- Preserve all numbers, numeric ranges, and currency symbols exactly; adjust only the decimal/thousand "
        "  separators and spacing according to the target locale's conventions.\n"
        "- Preserve survey-specific markup, HTML tags, placeholders, and piping tokens exactly as they appear. "
        "  You MUST NOT change, drop, or re-order any tags or tokens; only translate the human-readable text "
        "  between them.\n"
        "- Any explanations you provide in the `change_reason` field must be written in English, regardless of "
        "  the target language.\n"
        "- Always return valid JSON with the required keys and no extra commentary."
    )

    user_prompt = f"""
    You will translate and/or QA a single survey element.

    Global context (provided by the researcher):
    {global_context}

    Use this context to understand who the respondents are, what city/region the study focuses on,
    and what sort of organization is running the survey. Let that guide your choices of terminology,
    register (formality), and localization decisions in the target language and locale.

    Target language code: {language_code}
    Target locale code: {locale_code}

    English text (to translate from):
    \"\"\"{english_text}\"\"\"

    Existing translation in the target language (may be empty or just a copy of the English text):
    \"\"\"{existing_translation}\"\"\"

    Translation memory examples (English -> target translation):
    {memory_str}

    Instructions:
    1. If the existing translation is effectively empty or simply repeats the English text, treat this as if there "
       "were no translation yet, and propose a high-quality translation that follows all rules.
    2. If the existing translation is non-empty and clearly already in the target language, treat it as the baseline "
       "and only propose changes if they improve:
       - localization (correct regional variant),
       - terminology consistency,
       - grammar, tense, and style,
       - handling of proper nouns and numeric formatting,
       - accents and punctuation.
    3. Always perform a self-QA step on your proposed translation.
    4. You MUST NOT change or remove any HTML tags, placeholders, or piping tokens. Only translate the text between them.
    5. Return ONLY a valid JSON object with the following keys:
       - "proposed_translation": string
       - "qa_checked_translation": string
       - "needs_change": boolean
       - "change_reason": string (short explanation in English; empty if no change needed)
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
            result = json.loads(content)

            proposed = (result.get("proposed_translation") or "").strip()
            qa_checked = (result.get("qa_checked_translation") or proposed).strip()
            needs_change = bool(result.get("needs_change", False))
            change_reason = (result.get("change_reason") or "").strip()

            return {
                "proposed_translation": proposed,
                "qa_checked_translation": qa_checked,
                "needs_change": needs_change,
                "change_reason": change_reason,
                "error": False,
            }

        except Exception as e:
            last_exception = e
            status_code = getattr(e, "status_code", None)
            message = str(e).lower()

            is_rate_limit = (status_code == 429) or ("rate limit" in message)
            is_server_error = status_code is not None and 500 <= status_code < 600

            if is_rate_limit or is_server_error:
                # Exponential backoff
                sleep_seconds = 2 ** attempt
                time.sleep(sleep_seconds)
                continue
            else:
                break

    error_msg = f"LLM call failed: {last_exception}" if last_exception else "LLM call failed for unknown reasons."
    return {
        "proposed_translation": existing_translation or "",
        "qa_checked_translation": existing_translation or "",
        "needs_change": False,
        "change_reason": error_msg,
        "error": True,
    }


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

            if is_rate_limit or is_server_error and attempt < max_retries - 1:
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
    trl_nums = extract_numeric_tokens(translation_text)

    trl_digit_set = {re.sub(r"\D", "", t) for t in trl_nums if re.sub(r"\D", "", t)}
    missing_numeric: List[str] = []
    for tok in eng_nums:
        digits = re.sub(r"\D", "", tok)
        if digits and digits not in trl_digit_set:
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
# Core Processing Pipeline
# ==========================

def process_row_with_ai(
    row: SurveyRow,
    context: SurveyFileContext,
    global_context: str,
) -> None:
    """
    Process a single row through the translation + QA pipeline, with structure validation.

    Logic:
    - If Column B is empty: nothing to do.
    - If Column C was just a copy of Column B (no real translation), we treat it as "no existing translation"
      and *replace* Column C with a new translation.
    - If Column C was a real translation (different from B), we QA it and, if needed, add suggestions
      without overwriting Column C.
    """
    eng_text = (row.english_text or "").strip()
    if not eng_text:
        # Nothing to translate
        row.new_translation = row.existing_translation
        return

    normalized_key = normalize_english_text(eng_text)
    raw_existing = (row.existing_translation or "").strip()

    # Only treat as a real existing translation if we know it was different from English at load time
    if row.had_real_translation and raw_existing:
        existing_translation = raw_existing
    else:
        existing_translation = None

    result = call_translation_model(
        english_text=eng_text,
        language_code=context.language_code,
        locale_code=context.locale_code,
        global_context=global_context,
        translation_memory=context.translation_memory,
        existing_translation=existing_translation,
    )

    proposed = result.get("proposed_translation", "") or ""
    qa_checked = result.get("qa_checked_translation", "") or proposed
    needs_change = bool(result.get("needs_change", False))
    change_reason = result.get("change_reason", "") or ""
    error_flag = bool(result.get("error", False))

    # Handle LLM failure explicitly (must be visible in output)
    if error_flag:
        base = existing_translation or raw_existing
        row.new_translation = base
        if existing_translation:
            row.suggested_translation = base
        else:
            # Make the failure extremely obvious when there was no prior translation
            row.suggested_translation = "[NO TRANSLATION - LLM FAILED]"
        row.suggestion_reason = change_reason or "LLM call failed; translation not QA'd. Please review manually."
        return

    # Normal path: LLM call succeeded
    # First decide what will go into Column C and any suggestions, but do not update translation_memory yet.
    if existing_translation is None:
        # This row had no real translation yet (Column C was English or empty) -> brand new translation
        final_translation = qa_checked
        row.new_translation = final_translation
        row.was_newly_translated = True
    else:
        # There was an existing translation -> keep it in Column C, propose improvements as suggestions only
        row.new_translation = row.existing_translation
        final_translation = row.new_translation
        if needs_change and qa_checked.strip() != row.existing_translation.strip():
            row.suggested_translation = qa_checked
            # change_reason already requested in English
            row.suggestion_reason = change_reason or "Suggested improvement after QA."

    # Structure validation and translation memory update
    memory_update_candidate: Optional[str] = None

    # Validate final translation (Column C candidate)
    is_valid_final, validation_msg_final = validate_translation_structure(
        eng_text, final_translation
    )

    if not is_valid_final and final_translation.strip():
        # Flag the row explicitly without discarding the translation
        if not (row.suggested_translation or "").strip():
            row.suggested_translation = final_translation
            row.suggestion_reason = (
                f"Structure validation warning: {validation_msg_final} "
                "Please review numeric values and placeholders before using this translation."
            )
        else:
            extra = f" | Structure validation warning: {validation_msg_final}"
            row.suggestion_reason = (row.suggestion_reason or "") + extra
    else:
        # Only structurally valid final translations are candidates for memory
        if final_translation.strip():
            memory_update_candidate = final_translation

    # Validate suggested translation (if any), prefer valid suggestions for memory
    if (row.suggested_translation or "").strip():
        is_valid_sugg, validation_msg_sugg = validate_translation_structure(
            eng_text, row.suggested_translation or ""
        )
        if not is_valid_sugg:
            extra = (
                f" | Suggested translation may break numerics/placeholders: {validation_msg_sugg} "
                "Please verify carefully."
            )
            row.suggestion_reason = (row.suggestion_reason or "") + extra
        else:
            # If suggestion is structurally valid, prefer it as the canonical memory candidate
            memory_update_candidate = row.suggested_translation

    # Finally, update translation memory only with structurally valid candidate
    if memory_update_candidate and normalized_key:
        context.translation_memory[normalized_key] = {
            "english": eng_text,
            "translation": memory_update_candidate,
        }


def consistency_pass(context: SurveyFileContext) -> None:
    """
    Survey-wide consistency pass (LLM-powered).

    Steps:
    - Aggregate repeated English phrases and their different translations.
    - Send a compact summary to an LLM to decide where a canonical translation is helpful.
    - Add suggested_translation/suggestion_reason for rows the model flags.

    IMPORTANT:
    - This only adds suggested_translation/suggestion_reason when they are currently empty.
    - It does NOT overwrite Column C directly.
    - Rows already flagged with structure validation warnings are excluded from consideration.
    """
    # First, build a map: normalized English -> translation -> list of row indices
    term_map: Dict[str, Dict[str, List[int]]] = {}
    english_phrase_for_key: Dict[str, str] = {}

    for idx, row in enumerate(context.rows):
        eng = (row.english_text or "").strip()
        if not eng:
            continue

        # Skip rows already flagged for structural issues
        if "Structure validation warning" in (row.suggestion_reason or ""):
            continue

        key = normalize_english_text(eng)
        translation = (row.new_translation or row.existing_translation or "").strip()
        if not translation:
            continue

        term_map.setdefault(key, {}).setdefault(translation, []).append(idx)
        # Keep a representative English phrase for this normalized key
        english_phrase_for_key.setdefault(key, eng)

    # Build phrase_groups payload for the LLM: only keys with multiple translations
    phrase_groups: List[Dict[str, object]] = []
    for key, translations in term_map.items():
        if len(translations) <= 1:
            continue

        english_phrase = english_phrase_for_key.get(key, "")
        trans_list = []
        for t, indices in translations.items():
            trans_list.append({
                "translation": t,
                "indices": indices,
            })
        phrase_groups.append({
            "english_phrase": english_phrase,
            "translations": trans_list,
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
                f"LLM consistency suggestion: the English text '{english_phrase}' "
                f"appears with multiple translations across the survey. "
                f"Suggested canonical translation is '{canonical}'. "
                "Please check if this unified translation fits the context of this specific question/option."
            )
            if notes:
                row.suggestion_reason = base_reason + f" Notes from model: {notes}"
            else:
                row.suggestion_reason = base_reason


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

    # Serialize to Excel in memory
    buffer = io.BytesIO()
    df_out.to_excel(buffer, index=False)
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

- Generate high-quality localized translations where Column C currently just repeats the English text (or is empty).
- QA and optionally suggest improvements for existing translations (where Column C already differs from English).
- Optionally enforce survey-wide consistency for recurring terms.
- Validate numeric ranges, placeholders, and HTML tags so survey structure is preserved.

**Expected input format:**
1. Column A: variable name / ID  
2. Column B: English text  
3. Column C: target language translation (may be a real translation OR just repeat the English as a placeholder)
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

            selected_lang_label = st.selectbox(
                f"Target language for {filename}",
                options=list(LANGUAGE_LABEL_TO_CODE.keys()),
                index=list(LANGUAGE_LABEL_TO_CODE.keys()).index(lang_label_default)
                if lang_label_default in LANGUAGE_LABEL_TO_CODE
                else list(LANGUAGE_LABEL_TO_CODE.keys()).index("Spanish"),
                key=f"lang_{filename}",
            )
            language_code = LANGUAGE_LABEL_TO_CODE[selected_lang_label]

            locale_default_display = detected_locale_code or language_code
            locale_code = st.text_input(
                f"Target locale (BCP-47) for {filename}",
                value=locale_default_display,
                help="Examples: es-MX, es-AR, es-CO, es-ES, en-GB, en-US. "
                     "If unsure, you can leave it as-is.",
                key=f"loc_{filename}",
            )

        file_configs.append(
            {
                "file": file,
                "language_code": language_code,
                "locale_code": locale_code,
            }
        )

    run_pipeline = st.button("Run Translation Pipeline")

    if not run_pipeline:
        st.stop()

    # Process each file
    for cfg in file_configs:
        file = cfg["file"]
        language_code = cfg["language_code"]
        locale_code = cfg["locale_code"]

        st.markdown("---")
        st.subheader(f"Processing file: `{file.name}`")

        try:
            # Reset file pointer for each read
            file.seek(0)
            context, original_df = load_forsta_export(
                file,
                language_code_override=language_code,
                locale_code_override=locale_code,
            )
        except Exception as e:
            st.error(f"Failed to load {file.name}: {e}")
            continue

        n_rows = len(context.rows)
        st.write(
            f"Detected / selected language: `{context.language_code}` | "
            f"locale: `{context.locale_code}` | rows: {n_rows}"
        )

        progress_bar = st.progress(0.0)
        status_placeholder = st.empty()

        for idx, row in enumerate(context.rows):
            status_placeholder.text(f"Translating / QA row {idx + 1} of {n_rows}...")
            try:
                process_row_with_ai(row, context, global_context)
            except Exception as e:
                # Don't break the entire file on a single-row error; flag the row clearly.
                row.new_translation = row.existing_translation
                if not (row.suggested_translation or "").strip():
                    row.suggested_translation = row.existing_translation or "[NO TRANSLATION - ROW FAILED]"
                    row.suggestion_reason = f"Row-level processing failed: {e}"
                else:
                    row.suggestion_reason = (row.suggestion_reason or "") + f" | Row-level processing failed: {e}"

            if n_rows > 0:
                progress_bar.progress((idx + 1) / n_rows)

        status_placeholder.text("Row-level translation & QA complete.")

        if enable_consistency:
            try:
                consistency_pass(context)
                st.info("Survey-level consistency pass completed (check contexts before applying suggestions).")
            except Exception as e:
                st.warning(f"Consistency pass failed: {e}")

        # Build output
        try:
            out_df, out_filename, excel_bytes = write_output_file(context, original_df)
        except Exception as e:
            st.error(f"Failed to write output for {file.name}: {e}")
            continue

        # Summary statistics
        num_new_translations = sum(
            1 for r in context.rows if r.was_newly_translated
        )
        num_suggestions = sum(
            1
            for r in context.rows
            if ((r.suggested_translation and r.suggested_translation.strip())
                or (r.suggestion_reason and r.suggestion_reason.strip()))
        )
        num_error_rows = sum(
            1
            for r in context.rows
            if r.suggestion_reason and "LLM call failed" in r.suggestion_reason
        )

        st.success(
            f"Finished processing `{file.name}`. "
            f"New translations (former English placeholders): {num_new_translations} | "
            f"Rows with suggestions/warnings: {num_suggestions} | "
            f"Rows with LLM errors: {num_error_rows}"
        )

        st.download_button(
            label=f"Download processed file: {out_filename}",
            data=excel_bytes,
            file_name=out_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()