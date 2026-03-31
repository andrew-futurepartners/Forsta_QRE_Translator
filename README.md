# Forsta Questionnaire Translation & QA Tool

A Streamlit-powered application that translates and quality-assures Forsta/Decipher survey exports using OpenAI GPT models. It generates locale-aware translations across **any supported language and regional variant**, validates structural integrity (HTML tags, placeholders, numeric ranges), and enforces survey-wide terminology consistency — all through an interactive browser UI.

> **This is a general-purpose survey translator.** It is designed to handle any combination of target language and locale, for any survey domain. All heuristics, validation logic, style detection, and prompt construction operate at a language-agnostic level. See [Design Principles](#design-principles) below.

## Design Principles

This tool translates market-research questionnaires into many languages and regional localizations simultaneously. Every piece of logic — from segment classification to style enforcement to structural validation — **must remain language-agnostic**. Anyone modifying this codebase should follow these principles:

1. **No single-language assumptions.** Heuristics, thresholds, and guards must work correctly for all supported languages. A fix that improves French but breaks Japanese is not acceptable.
2. **Per-language configuration over hard-coded rules.** When behaviour genuinely varies by language (e.g. grammatical gender, first-person detection patterns), it is expressed as **data-driven lookup tables** keyed by language code — never as inline if/else branches scattered through business logic. See `_GENDERED_LANGUAGE_CONFIG` and `get_first_person_regexes()` for the established pattern.
3. **Domain-agnostic prompts.** The LLM system prompts reference a user-configurable global context string, not a hard-coded industry vertical. The tool works equally well for travel, healthcare, finance, or any other survey domain.
4. **Locale-aware, not locale-locked.** BCP-47 locale codes (e.g. `es-MX`, `fr-CA`, `pt-BR`) are passed through to the model for region-appropriate phrasing. Adding a new locale should only require adding an entry to `LOCALE_OPTIONS`.
5. **Adding a new language** should require only: a new entry in `LANGUAGE_NAME_TO_CODE` / `LANGUAGE_LABEL_TO_CODE`, a locale list in `LOCALE_OPTIONS`, and optional entries in the per-language config tables (gender config, first-person regexes). No structural changes to the pipeline.

## Features

- **GPT-powered translation & QA** — Translates untranslated rows and optionally reviews existing human translations, surfacing suggestions and warnings.
- **Multi-language & locale support** — Spanish (MX, AR, CO, CL, PE, ES), French (FR, CA), Portuguese (BR, PT), German (DE, AT, CH), Italian (IT, CH), Dutch (NL, BE), Japanese, Korean, Chinese (Simplified/Traditional/HK), and English (US, GB, CA, AU), with BCP-47 locale codes passed to the model for region-appropriate phrasing. Languages outside the built-in locale list fall back to a single generic option.
- **Configurable domain context** — Editable global context field lets you tailor every prompt to the survey's domain (e.g. healthcare, travel, finance) instead of relying on a hard-coded description.
- **Structural classification** — Automatically categorises each row as a question, instruction, answer option, or scale label using text heuristics and Forsta variable-name patterns, then groups them into question blocks for context-aware translation. A post-classification promotion step aligns short label-like answer options to scale labels when Likert-style blocks are detected.
- **Block-level style inference & enforcement** — Infers grammatical person, phrase form, and tone for each block via async LLM calls. Translations are held to hard style constraints, and a post-translation re-check pass automatically re-translates any answer options that still deviate from the inferred style. A `noun_phrase_like` category prevents false-positive restyle warnings on longer (60–100+ char) answer options.
- **Scale batch translation** — Scale-label blocks (2+ scale labels) are translated in a single JSON call per block after their parent question is pre-translated, ensuring coherent, parallel phrasing across the entire scale set.
- **Peer-option parallelism** — For small, label-like answer-option sets (2–8 items), the full ordered peer list is sent to the model so translations stay grammatically and stylistically parallel.
- **Gender-inclusive forms toggle** — Per-file checkbox (defaults to ON for French) that instructs the model to add parenthetical gender-inclusive forms on adjective-based labels. The instruction is **language-aware and scoped**: it only targets adjectives that describe the survey respondent (not objects, probabilities, or quality ratings), with per-language examples and markers for French, Spanish, Portuguese, Italian, and German. Languages without grammatical gender receive no instruction.
- **Survey-level consistency pass** — Uses fuzzy matching + an LLM call to identify repeated English phrases translated differently, then suggests (or auto-applies) a canonical translation. Freshly translated rows are auto-corrected in the Translation column; existing human translations are never overwritten ("locked"). Can be toggled on or off.
- **Structure validation & auto-repair** — Guards against missing/altered HTML tags, placeholders (`[piped text]`, `{TOKEN}`, `[[VARNAME]]`), and numeric ranges. When a placeholder is dropped during translation, the tool attempts automatic positional repair before falling back to a warning.
- **Numeric/range guards** — Pure numeric and range-code options (e.g. `1970–1989`, `$100`) are passed through unchanged, preventing unwanted prose rewrites.
- **Translation failure detection** — Flags rows where the model's output is effectively a copy of the English source, prompting manual review. Short proper nouns, named entities, and title-case Forsta option rows are exempted to avoid false positives.
- **Translation memory** — Reuses translations already present in the file for identical English strings, ranked by Jaccard word-similarity to the current row for maximum relevance (up to 5 examples injected into prompts).
- **Label capitalisation adjustment** — Short, label-like translations are automatically capitalised to match survey-label conventions in the target language (skipped for Japanese and Chinese).
- **Suggestions toggle** — Optionally suppress QA suggestions for existing translations; the consistency pass still harmonises newly generated translations.
- **Multi-file batch processing** — Upload multiple survey files at once; download individually or as a single ZIP archive.
- **Live progress UI** — Real-time progress bar, status text, and a live-updating preview table showing the last 5 processed rows (refreshed every 2 completions).
- **Async concurrency** — Row-level translation runs via `asyncio` with a semaphore (15 concurrent calls) and automatic retry with backoff (up to 3 attempts), maximising throughput while respecting API rate limits.

## Input Format

The tool expects Forsta/Decipher 3-column translation exports (`.xls`, `.xlsx`, or `.csv`):

| Column A | Column B | Column C |
|---|---|---|
| Variable name / ID | English source text | Target translation (or English placeholder) |

## Getting Started

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)

### Installation

```bash
pip install streamlit pandas openai python-dotenv openpyxl xlrd
```

### Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
TRANSLATION_MODEL=gpt-5-mini
CONSISTENCY_MODEL=gpt-5-mini
```

`TRANSLATION_MODEL` and `CONSISTENCY_MODEL` are optional and default to `gpt-5-mini`.

### Running the App

```bash
streamlit run forsta_translation_qa_app.py
```

The app will open in your browser. Upload one or more survey export files, configure language/locale settings per file, and click **Run Translation Pipeline**.

## Output

Each processed file is downloaded as an Excel workbook with:

- **translations** sheet — the original 3 columns plus (when suggestions are enabled) `suggested_translation` and `suggestion_reason` columns.
- **__style_log** sheet — block-level style analysis details for auditing (block ID, English question text, row counts, inferred style fields, and notes).

Output filenames follow the pattern `<original_name>_translated.xlsx`, with `_WITH_SUGGESTIONS` appended when suggestions or warnings are present. Illegal characters for Excel cells are automatically sanitised before writing.

## Project Structure

```
forsta_translation_qa_app.py   # Main application (Streamlit + processing pipeline)
.env                           # API keys (not committed)
Archive/                       # Previous iterations of the app
Feedback/                      # Reviewer feedback files (not committed)
```

## Processing Pipeline

1. **Load & parse** — Read the 3-column export; detect language and locale from the filename; build a translation memory from rows that already contain real translations.
2. **Structural classification (Layer 1)** — Classify each row by segment type (`question`, `instruction`, `answer_option`, `scale_label`, `other`) using text heuristics and Forsta variable-name patterns (e.g. `q…,rN`).
3. **Block grouping (Layer 2)** — Group rows into question blocks based on document order; assign `block_id` to each row.
4. **Scale-label promotion** — In blocks with 2+ scale labels, promote short label-like answer options to `scale_label` for Likert-scale alignment.
5. **Style inference (Layer 3)** — Async LLM calls infer grammatical person, phrase form, and tone for each block's answer options and scale labels (returns a `BlockStyle` per block).
6. **Scale batch translation** — Pre-translate question rows in scale-heavy blocks, then translate each block's scale labels as a single JSON batch call for coherent sets.
7. **Row-level translation/QA (Layer 4)** — Async GPT calls with concurrency control (semaphore of 15); translate new rows, QA existing ones, validate structure (with auto-repair for dropped placeholders), enforce style constraints, and detect translation failures. Rows already batch-translated in step 6 are skipped.
8. **Post-translation style re-check (Layer 4.5)** — Re-translate any answer options whose detected style pattern doesn't match the inferred block style; flag unresolvable mismatches for manual review.
9. **Block-level style validation (Layer 5)** — Flag remaining style mismatches within answer-option sets and scale-label groups (runs only when suggestions are enabled).
10. **Survey-level consistency pass** — Fuzzy-match repeated English phrases; auto-apply canonical translations to freshly translated rows, or surface as suggestions for existing human translations (when consistency pass is enabled).
11. **Output generation** — Write the final Excel file with translations, suggestions, and style logs.

## Core Data Structures

| Structure | Purpose |
|---|---|
| `SegmentType` (Enum) | Row classification: `question`, `instruction`, `answer_option`, `scale_label`, `other` |
| `SurveyRow` | Per-row state: variable name, English text, existing/new translations, flags (`had_real_translation`, `was_newly_translated`, `batch_translated`), suggestions, and segment metadata |
| `QuestionBlock` | Groups row indices by role (questions, instructions, options, scale labels) under a `block_id` |
| `BlockStyle` | Inferred style for a block: grammatical person, phrase form, tone, scale-label phrase form, and notes |
| `SurveyFileContext` | Top-level container: filename, language/locale codes, rows, translation memory, blocks, and block styles |

## Adding a New Language

1. Add entries to `LANGUAGE_NAME_TO_CODE` and `LANGUAGE_LABEL_TO_CODE`.
2. Add a locale list to `LOCALE_OPTIONS`.
3. *(Optional)* If the language has grammatical gender, add an entry to `_GENDERED_LANGUAGE_CONFIG`.
4. *(Optional)* If first-person style detection is needed, add patterns to `get_first_person_regexes()`.

No other pipeline changes should be necessary.
