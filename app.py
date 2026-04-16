import html
import hashlib
import json
import os
import re
from copy import deepcopy
from datetime import datetime
from uuid import uuid4

import gspread
import language_tool_python
import pandas as pd
import streamlit as st
from openai import OpenAI


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="CEFR Interactive Diagnostic",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------
# Grammar checker (cached — loads LanguageTool Java server once per session)
# -----------------------------
@st.cache_resource(show_spinner="Loading grammar checker…")
def _get_language_tool():
    return language_tool_python.LanguageTool("en-US")


# Rule categories to ignore — style/punctuation/redundancy are not core grammar errors for EFL learners
_GRAMMAR_IGNORE_CATEGORIES = {
    "STYLE",
    "REDUNDANCY",
    "PLAIN_ENGLISH",
    "MISC",
    "PUNCTUATION",   # Comma placement rules are stylistic, not grammar errors for EFL
    "TYPOGRAPHY",
}
# Specific noisy rules to ignore regardless of category
_GRAMMAR_IGNORE_RULE_IDS = {
    "SENTENCE_WHITESPACE",
    "COMMA_PARENTHESIS_WHITESPACE",
    "EN_QUOTES",
    "DOUBLE_PUNCTUATION",
    "WORD_CONTAINS_UNDERSCORE",
    "WHITESPACE_RULE",
}


# -----------------------------
# Constants
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
RESPONSES_DIR = os.path.join(OUTPUT_DIR, "responses")
EVALUATIONS_DIR = os.path.join(OUTPUT_DIR, "evaluations")
BY_STUDENT_RESPONSES_DIR = os.path.join(RESPONSES_DIR, "by_student")
BY_STUDENT_EVALUATIONS_DIR = os.path.join(EVALUATIONS_DIR, "by_student")
RESPONSES_FILE = os.path.join(RESPONSES_DIR, "questionnaire_responses.csv")
EVALUATIONS_FILE = os.path.join(EVALUATIONS_DIR, "state_evaluations.csv")
DEFAULT_RESPONSES_WORKSHEET = "questionnaire_responses"
DEFAULT_EVALUATIONS_WORKSHEET = "state_evaluations"
DEFAULT_LLM_MODEL = "gpt-5-mini"

MIN_SENTENCES = 1
RECOMMENDED_SENTENCES = 2
CEFR_LEVEL_OPTIONS = ["A1", "A2", "B1", "B2", "C1", "C2"]
TARGET_PASSAGE_SENTENCE_COUNT = 10
BACKGROUND_OPTIONS = [
    "Classroom",
    "Campus life",
    "Daily life",
    "Part-time job",
    "Travel",
    "Online communication",
    "Friendship",
    "Club activity",
    "Test preparation",
    "Self-study",
]
GENRE_OPTIONS = [
    "Narrative story",
    "Dialogue",
    "Message/chat",
    "Email",
    "Reflection journal",
    "Problem situation",
    "Short anecdote",
]

QUESTION_SECTION_BLUEPRINTS = [
    {
        "title": "Part 1. Understanding",
        "title_ko": "1부. 이해",
        "questions": [
            {
                "id": "q1",
                "label": "Q1",
                "layer": "Understanding",
                "type": "Character",
                "question_goal": "Ask the learner to explain the main situation, goal, or problem from the passage in their own words.",
            },
            {
                "id": "q2",
                "label": "Q2",
                "layer": "Understanding",
                "type": "Self",
                "depends_on": "q1",
                "question_goal": "Ask an adaptive self follow-up that uses the learner's Q1 interpretation and the passage, then moves into a similar situation in the learner's own life so the learner explains what would be similar, important, or difficult in connected detail.",
                "criteria_focus": "Generate an adaptive self follow-up based on the learner's Q1 answer and the passage. Move from the passage situation into a similar real-life situation for the learner. Ask the learner to connect their interpretation to their own experience, perspective, or likely response, while still anchoring the question in the key situation, reason, or detail from the passage. The question should invite a fuller connected response so fluency features such as explanation, connector use, linked ideas, organization, and emotional expression can be observed.",
            },
        ],
    },
    {
        "title": "Part 2. Emotion",
        "title_ko": "2부. 감정",
        "questions": [
            {
                "id": "q3",
                "label": "Q3",
                "layer": "Emotion",
                "type": "Character",
                "question_goal": "Ask how the main character feels in the situation and what in the passage shows that emotion.",
            },
            {
                "id": "q4",
                "label": "Q4",
                "layer": "Emotion",
                "type": "Self",
                "depends_on": "q3",
                "question_goal": "Ask how the learner would feel in a similar situation in their own life.",
                "criteria_focus": "Target anxiety versus enjoyment without naming technical labels. Encourage feelings such as nervous, worried, afraid, comfortable, relaxed, interested, happy, or enjoying the situation.",
            },
        ],
    },
    {
        "title": "Part 3. Cognition",
        "title_ko": "3부. 생각",
        "questions": [
            {
                "id": "q5",
                "label": "Q5",
                "layer": "Cognition",
                "type": "Character",
                "question_goal": "Ask what the main character might think, believe, or worry about in that moment.",
            },
            {
                "id": "q6",
                "label": "Q6",
                "layer": "Cognition",
                "type": "Self",
                "depends_on": "q5",
                "question_goal": "Ask what the learner would think about their own ability and how they would notice or manage difficulty in a similar situation.",
                "criteria_focus": "Target confidence and metacognitive reflection without naming technical labels. Encourage reflection on ability, confidence, noticing problems, and managing or improving performance in a natural way.",
            },
        ],
    },
    {
        "title": "Part 4. Behavior",
        "title_ko": "4부. 행동",
        "questions": [
            {
                "id": "q7",
                "label": "Q7",
                "layer": "Behavior",
                "type": "Character",
                "question_goal": "Ask what the main character could do next and which action seems most likely or most helpful.",
            },
            {
                "id": "q8",
                "label": "Q8",
                "layer": "Behavior",
                "type": "Self",
                "depends_on": "q7",
                "question_goal": "Ask what the learner would do in a similar situation in their own life.",
                "criteria_focus": "Target communication, coping, and engagement without naming technical labels. Encourage reflection on speaking, staying quiet, asking for help, preparing, practicing, avoiding, or participating.",
            },
        ],
    },
    {
        "title": "Part 5. Strategy",
        "title_ko": "5부. 전략",
        "questions": [
            {
                "id": "q9",
                "label": "Q9",
                "layer": "Strategy",
                "type": "Character",
                "question_goal": "Ask what the best strategy is for the character and why it would work in that situation.",
            },
            {
                "id": "q10",
                "label": "Q10",
                "layer": "Strategy",
                "type": "Self",
                "depends_on": "q9",
                "question_goal": "Ask what strategy would help the learner most in a similar situation in their own life.",
                "criteria_focus": "Target strategy type and quality without naming technical labels. Encourage strategy examples such as practice, planning, calming down, asking others, guessing, or using simple words.",
            },
        ],
    },
]

QUESTION_GENERATION_CRITERIA = """
Layer and evaluation targets:

Fluency
- Low: very short or isolated responses with little connection between ideas.
- Mid: basic connected response using at least one connector such as and, because, but, so, when, if, or then.
- High: longer and more organized response with connected ideas, some clause or sentence complexity, clear flow, and an explanation of what the learner would do, think, or try.
- Observable features to support fluency judgment:
  Sentence length: enough development beyond a fragment or one short line.
  Connector use: and, because, but, so, when, if, or then.
  Structure complexity: clauses, combined ideas, or more than one linked statement.
  Organization: order or logic such as first, next, then, later, finally, so, or in the end.
  Strategy expression: explaining a solution, plan, reason, coping action, or next step.

Emotion
- FLA (Foreign Language Anxiety)
  High: strong anxiety expressions such as nervous, afraid, worry, avoid speaking.
  Mid: anxiety plus control expressions such as nervous but try, worried but prepare.
  Low: almost no anxiety such as comfortable, relaxed.
- FLE (Foreign Language Enjoyment)
  High: clear enjoyment or interest such as enjoy, fun, interesting, happy.
  Mid: mixed positive and neutral such as okay, sometimes enjoyable.
  Low: low interest or negative response such as boring, do not like.

Cognition
- Self-efficacy
  High: confidence about performance such as I can speak well, I am confident.
  Mid: limited confidence such as I can, but..., simple English only.
  Low: low belief in performance such as I cannot speak, I will make mistakes.
- Metacognition
  High: self-awareness plus regulation such as I know my problem and try to fix it.
  Mid: partial awareness such as I know I am weak, but no clear action.
  Low: little or no reflection.

Behavior
- WTC (Willingness to Communicate)
  High: active speaking such as I try to speak, I talk often.
  Mid: situation-dependent such as sometimes speak, depends.
  Low: avoidance such as stay quiet, do not speak.
- Coping
  Active: problem-solving behaviors such as practice, try, ask, prepare.
  Mixed: trying plus avoidance.
  Avoidant: mainly avoidance such as avoid, give up, stay silent.
- Engagement
  High: active participation such as participate, ask questions, interact.
  Mid: limited participation such as listen but rarely speak.
  Low: non-participation such as passive, no interaction.

Strategy
- Strategy Type
  Cognitive: practice, repetition, note-taking.
  Metacognitive: planning, monitoring, self-check.
  Affective: calming, positive thinking, anxiety control.
  Social: asking, interacting, peer learning.
  Compensation: guessing, using gestures, using simple words.
  Mixed: using two or more strategy types together.
- Strategy Quality
  Effective: strategy matches the goal and shows regulation.
  Limited: simple or repetitive strategy such as only memorize.
  Avoidant: mainly avoidance.
"""

PERSONALIZED_SELF_QUESTION_SPECS = {
    "q2": {
        "state_targets": "Understanding self-application / fluency",
        "criteria_focus": (
            "Generate an adaptive self follow-up question based on the learner's Q1 answer and the passage. "
            "Move from the passage situation into a similar real-life situation for the learner. "
            "Ask the learner to connect their interpretation to their own experience, perspective, or likely response while still anchoring the question in the key situation, reason, or detail from the passage. "
            "The question should encourage a fuller connected explanation so fluency features such as sentence development, connector use, linked ideas, organization, and emotional expression can be observed."
        ),
        "note": "This follow-up is linked to your answer in Q1 and turns the passage situation into an adaptive self question.",
    },
    "q4": {
        "state_targets": "FLA / FLE",
        "criteria_focus": (
            "Generate a self question for a similar real-life situation that connects the learner's character-emotion answer "
            "to their own emotional state. It should help reveal anxiety versus enjoyment without naming technical labels."
        ),
        "note": "This follow-up is linked to your answer in Q3 and checks your emotion in a similar situation.",
    },
    "q6": {
        "state_targets": "Self-efficacy / Metacognition",
        "criteria_focus": (
            "Generate a self question for a similar real-life situation that connects the learner's character-thinking answer "
            "to their own confidence, self-awareness, noticing problems, and regulation."
        ),
        "note": "This follow-up is linked to your answer in Q5 and checks your thinking in a similar situation.",
    },
    "q8": {
        "state_targets": "WTC / Coping / Engagement",
        "criteria_focus": (
            "Generate a self question for a similar real-life situation that connects the learner's character-behavior answer "
            "to their own willingness to communicate, coping under difficulty, and participation."
        ),
        "note": "This follow-up is linked to your answer in Q7 and checks your behavior in a similar situation.",
    },
    "q10": {
        "state_targets": "Strategy Type / Strategy Quality",
        "criteria_focus": (
            "Generate a self question for a similar real-life situation that connects the learner's character-strategy answer "
            "to their own learning strategies. It should help reveal strategy type and quality without naming the labels."
        ),
        "note": "This follow-up is linked to your answer in Q9 and checks your strategy use in a similar situation.",
    },
}

# -----------------------------
# Output and storage helpers
# -----------------------------
def ensure_output_directories():
    os.makedirs(RESPONSES_DIR, exist_ok=True)
    os.makedirs(EVALUATIONS_DIR, exist_ok=True)
    os.makedirs(BY_STUDENT_RESPONSES_DIR, exist_ok=True)
    os.makedirs(BY_STUDENT_EVALUATIONS_DIR, exist_ok=True)


def extract_spreadsheet_id(value: str) -> str:
    if not value:
        return ""

    value = value.strip()
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", value)
    if match:
        return match.group(1)

    return value


def normalize_private_key(private_key: str) -> str:
    if not private_key:
        return ""

    normalized = private_key.strip()
    if normalized.startswith('"') and normalized.endswith('"'):
        normalized = normalized[1:-1]

    normalized = normalized.replace("\\n", "\n")
    normalized = normalized.replace("\r\n", "\n")

    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    if not lines:
        return ""

    header = lines[0] if lines[0].startswith("-----BEGIN") else "-----BEGIN PRIVATE KEY-----"
    footer = lines[-1] if lines[-1].startswith("-----END") else "-----END PRIVATE KEY-----"
    body_lines = lines[1:-1] if lines[0].startswith("-----BEGIN") and lines[-1].startswith("-----END") else lines

    # Join body and normalize URL-safe base64 → standard base64
    body = "".join(body_lines)
    body = body.replace("-", "+").replace("_", "/")

    # Fix missing base64 padding
    padding_needed = (4 - len(body) % 4) % 4
    body += "=" * padding_needed

    # Re-chunk into standard 64-character lines
    chunked = "\n".join(body[i:i + 64] for i in range(0, len(body), 64))
    return f"{header}\n{chunked}\n{footer}\n"


def normalize_service_account_info(service_account_info: dict) -> dict:
    normalized = dict(service_account_info)
    if "private_key" in normalized:
        normalized["private_key"] = normalize_private_key(str(normalized["private_key"]))
    return normalized


def normalize_worksheet_name(value: str, default_name: str) -> str:
    normalized = str(value or "").strip()
    return normalized or default_name


def resolve_google_worksheet_names(sheet_settings: dict) -> tuple[str, str]:
    responses_worksheet = normalize_worksheet_name(
        sheet_settings.get("responses_worksheet", ""),
        DEFAULT_RESPONSES_WORKSHEET,
    )
    evaluations_worksheet = normalize_worksheet_name(
        sheet_settings.get("evaluations_worksheet", ""),
        DEFAULT_EVALUATIONS_WORKSHEET,
    )

    if responses_worksheet == evaluations_worksheet:
        fallback_evaluations_name = DEFAULT_EVALUATIONS_WORKSHEET
        if fallback_evaluations_name == responses_worksheet:
            fallback_evaluations_name = f"{DEFAULT_EVALUATIONS_WORKSHEET}_tab"
        evaluations_worksheet = fallback_evaluations_name

    return responses_worksheet, evaluations_worksheet


def get_google_sheets_config() -> dict:
    google_service_account = None
    sheet_settings = None

    if "google_service_account" in st.secrets and "google_sheets" in st.secrets:
        google_service_account = dict(st.secrets["google_service_account"])
        sheet_settings = dict(st.secrets["google_sheets"])
    elif "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
        connection_settings = dict(st.secrets["connections"]["gsheets"])
        google_service_account = {
            key: connection_settings[key]
            for key in [
                "type",
                "project_id",
                "private_key_id",
                "private_key",
                "client_email",
                "client_id",
                "auth_uri",
                "token_uri",
                "auth_provider_x509_cert_url",
                "client_x509_cert_url",
                "universe_domain",
            ]
            if key in connection_settings
        }
        sheet_settings = connection_settings
    else:
        return {"enabled": False}

    spreadsheet_id = extract_spreadsheet_id(sheet_settings.get("spreadsheet_id", ""))
    if not spreadsheet_id:
        spreadsheet_id = extract_spreadsheet_id(sheet_settings.get("spreadsheet_url", ""))

    if not spreadsheet_id or not google_service_account:
        return {"enabled": False}

    responses_worksheet, evaluations_worksheet = resolve_google_worksheet_names(sheet_settings)

    return {
        "enabled": True,
        "service_account": normalize_service_account_info(google_service_account),
        "spreadsheet_id": spreadsheet_id,
        "responses_worksheet": responses_worksheet,
        "evaluations_worksheet": evaluations_worksheet,
    }


def get_llm_config() -> dict:
    api_key = ""
    model = DEFAULT_LLM_MODEL

    if "openai" in st.secrets:
        openai_settings = dict(st.secrets["openai"])
        api_key = str(openai_settings.get("api_key", "")).strip()
        model = str(
            openai_settings.get("question_model")
            or openai_settings.get("model")
            or model
        ).strip()

    if not api_key and "OPENAI_API_KEY" in st.secrets:
        api_key = str(st.secrets["OPENAI_API_KEY"]).strip()

    env_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    env_model = os.getenv("OPENAI_QUESTION_MODEL", "").strip()

    if env_api_key and not api_key:
        api_key = env_api_key
    if env_model:
        model = env_model

    return {
        "enabled": bool(api_key),
        "api_key": api_key,
        "model": model or DEFAULT_LLM_MODEL,
    }


@st.cache_resource
def get_gspread_client(service_account_info: dict):
    return gspread.service_account_from_dict(service_account_info)


@st.cache_resource
def get_openai_client(api_key: str):
    return OpenAI(api_key=api_key)


def get_google_spreadsheet(config: dict):
    client = get_gspread_client(config["service_account"])
    return client.open_by_key(config["spreadsheet_id"])


def ensure_worksheet(spreadsheet, worksheet_name: str, headers: list):
    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=worksheet_name,
            rows=max(1000, len(headers) + 10),
            cols=max(26, len(headers) + 10),
        )

    existing_headers = worksheet.row_values(1)
    merged_headers = list(existing_headers)

    for header in headers:
        if header not in merged_headers:
            merged_headers.append(header)

    if len(merged_headers) > worksheet.col_count:
        worksheet.add_cols(len(merged_headers) - worksheet.col_count)

    if not existing_headers:
        worksheet.update("A1", [merged_headers])
    elif merged_headers != existing_headers:
        worksheet.update("A1", [merged_headers])

    return worksheet, merged_headers


def append_rows_to_google_sheet(spreadsheet, worksheet_name: str, rows: list):
    if not rows:
        return

    headers = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)

    worksheet, merged_headers = ensure_worksheet(spreadsheet, worksheet_name, headers)
    values = [[row.get(header, "") for header in merged_headers] for row in rows]
    worksheet.append_rows(values, value_input_option="USER_ENTERED")


def append_rows(file_path: str, rows: list):
    ensure_output_directories()
    df_new = pd.DataFrame(rows)

    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(file_path, index=False, encoding="utf-8-sig")


def save_rows(rows: list, local_file_path: str, google_worksheet_name: str):
    google_config = get_google_sheets_config()

    if google_config["enabled"]:
        spreadsheet = get_google_spreadsheet(google_config)
        append_rows_to_google_sheet(spreadsheet, google_worksheet_name, rows)
        return {
            "backend": "google_sheets",
            "spreadsheet_id": google_config["spreadsheet_id"],
            "spreadsheet_url": f"https://docs.google.com/spreadsheets/d/{google_config['spreadsheet_id']}",
            "worksheet_name": google_worksheet_name,
        }

    append_rows(local_file_path, rows)
    return {
        "backend": "local_csv",
        "file_path": local_file_path,
    }


# -----------------------------
# Questionnaire generation helpers
# -----------------------------
def get_question_slots() -> list:
    return [
        question
        for section in QUESTION_SECTION_BLUEPRINTS
        for question in section["questions"]
    ]


def get_all_questions(questionnaire: dict) -> list:
    return [
        question
        for section in questionnaire["sections"]
        for question in section["questions"]
    ]


def get_question_lookup(questionnaire: dict) -> dict:
    return {question["id"]: question for question in get_all_questions(questionnaire)}


def build_question_slot_instructions() -> str:
    lines = []
    for question in get_question_slots():
        line = (
            f"- {question['id']} | layer={question['layer']} | type={question['type']} | "
            f"goal={question['question_goal']}"
        )
        if question.get("criteria_focus"):
            line += f" | criteria={question['criteria_focus']}"
        lines.append(line)
    return "\n".join(lines)


def extract_json_object(text: str) -> dict:
    if not text:
        raise ValueError("The model returned an empty response.")

    code_block_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1)

    start_index = text.find("{")
    end_index = text.rfind("}")

    if start_index < 0 or end_index < 0 or end_index <= start_index:
        raise ValueError("No JSON object was found in the model response.")

    return json.loads(text[start_index : end_index + 1])


def format_exception_message(error: Exception) -> str:
    message = str(error).strip()
    if message:
        return f"{type(error).__name__}: {message}"
    return type(error).__name__


def format_passage_markdown(title: str, sentences: list) -> str:
    return "\n".join([f"**{title}**", "", *sentences])


def build_participant_signature(student_name: str, student_number: str) -> str:
    return f"{student_name.strip()}::{student_number.strip()}"


def normalize_prompt_map(payload: dict) -> dict:
    questions_payload = payload.get("questions", [])

    if isinstance(questions_payload, dict):
        items = []
        for question_id, value in questions_payload.items():
            if isinstance(value, dict):
                items.append({"id": question_id, **value})
    elif isinstance(questions_payload, list):
        items = questions_payload
    else:
        items = []

    normalized = {}
    for item in items:
        if not isinstance(item, dict):
            continue

        question_id = str(item.get("id", "")).strip()
        prompt = str(item.get("prompt", "")).strip()
        prompt_ko = str(item.get("prompt_ko", "")).strip()

        if question_id and prompt and prompt_ko:
            normalized[question_id] = {
                "prompt": prompt,
                "prompt_ko": prompt_ko,
            }

    expected_question_ids = {question["id"] for question in get_question_slots()}
    if set(normalized.keys()) != expected_question_ids:
        missing = sorted(expected_question_ids - set(normalized.keys()))
        raise ValueError(f"Missing generated questions: {', '.join(missing)}")

    return normalized


def normalize_sentence_payload(payload: dict, key: str) -> list:
    value = payload.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list.")

    sentences = [str(item).strip() for item in value if str(item).strip()]
    if len(sentences) != TARGET_PASSAGE_SENTENCE_COUNT:
        raise ValueError(f"{key} must contain exactly {TARGET_PASSAGE_SENTENCE_COUNT} sentences.")

    return sentences


def apply_prompt_map_to_sections(prompt_map: dict) -> list:
    sections = deepcopy(QUESTION_SECTION_BLUEPRINTS)

    for section in sections:
        for question in section["questions"]:
            generated = prompt_map[question["id"]]
            question["base_prompt"] = generated["prompt"]
            question["base_prompt_ko"] = generated["prompt_ko"]
            question["prompt"] = generated["prompt"]
            question["prompt_ko"] = generated["prompt_ko"]

    return sections


def normalize_experiment_payload(payload: dict, cefr_level: str) -> dict:
    story_title = str(payload.get("story_title", "")).strip()
    story_title_ko = str(payload.get("story_title_ko", "")).strip()
    character_focus = str(
        payload.get("central_characters", "") or payload.get("character_focus", "")
    ).strip()
    relationship_focus = str(
        payload.get("situation_focus", "") or payload.get("relationship_focus", "")
    ).strip()
    relationship_focus_ko = str(
        payload.get("situation_focus_ko", "") or payload.get("relationship_focus_ko", "")
    ).strip()

    if not all([story_title, story_title_ko, character_focus, relationship_focus, relationship_focus_ko]):
        raise ValueError("The model returned an incomplete experiment header.")

    passage_sentences = normalize_sentence_payload(payload, "passage_sentences")
    passage_ko_sentences = normalize_sentence_payload(payload, "passage_ko_sentences")
    prompt_map = normalize_prompt_map(payload)

    questionnaire = {
        "id": "cefr_dynamic",
        "page_title": "CEFR Interactive Diagnostic",
        "page_title_ko": "CEFR 상호작용 진단",
        "story_title": story_title,
        "story_title_ko": story_title_ko,
        "central_characters": character_focus,
        "character_focus": character_focus,
        "situation_focus": relationship_focus,
        "relationship_focus": relationship_focus,
        "relationship_focus_ko": relationship_focus_ko,
        "intro": (
            "Read the CEFR-level passage on the left and answer every question on the right in English. "
            "Short answers are allowed, and longer answers are welcome."
        ),
        "intro_ko": (
            "왼쪽 CEFR 수준 지문을 읽고 오른쪽 질문에 모두 영어로 답하세요. "
            "짧은 답변도 가능하며, 더 길게 써도 됩니다."
        ),
        "cefr_level": cefr_level,
        "passage_sentences": passage_sentences,
        "passage_ko_sentences": passage_ko_sentences,
        "selected_background": "",
        "selected_genre": "",
        "text": format_passage_markdown(story_title, passage_sentences),
        "text_ko": format_passage_markdown(story_title_ko, passage_ko_sentences),
        "passage_plain": " ".join(passage_sentences),
        "passage_ko_plain": " ".join(passage_ko_sentences),
        "sections": apply_prompt_map_to_sections(prompt_map),
    }

    return questionnaire


def generate_experiment_with_openai(
    cefr_level: str,
    student_name: str,
    student_number: str,
    variation_seed: str,
    selected_background: str,
    selected_genre: str,
    api_key: str,
    model: str,
) -> dict:
    client = get_openai_client(api_key)
    question_slot_instructions = build_question_slot_instructions()

    system_prompt = """
You design one bilingual CEFR-based reading-and-response experiment for an English-education diagnostic app.
Return valid JSON only.

Requirements:
- Create one passage with exactly 10 English sentences.
- Match the CEFR level requested by the user.
- Use one or two clear central characters in one meaningful situation with emotional, social, or decision-making stakes.
- The passage can be about English learning or any broader real-life situation, as long as it supports emotionally and cognitively meaningful responses and gives enough context for evaluating fluency, emotion, cognition, behavior, and strategy through the follow-up questions.
- The selected personal background and selected genre are mandatory constraints, not optional hints.
- Structure the passage with a clear beginning, development, turning point, and closing outcome or reflection.
- Create a fresh scenario for this participant and avoid repeating the same stock situation across different students.
- Make the situation rich enough to support emotion, cognition, behavior, and strategy evaluation.
- Create exactly 10 student-facing questions with ids q1 to q10.
- Make the questions interactive, specific, learner-friendly, and natural enough to sound like real student prompts instead of a psychology checklist.
- Make the questions open enough to reveal fluency features such as sentence length, connector use, structure complexity, organization, and strategy expression.
- Include depth 2 by making the second question in each section meaningfully deeper than the first.
- Keep self questions clearly connected to the learner's own feelings, thoughts, actions, or strategies in a similar situation.
- Do not use technical labels such as FLA, FLE, self-efficacy, metacognition, WTC, coping, engagement, or Oxford strategy type in the student-facing questions.
- Make the Korean translations natural and faithful.
"""

    user_prompt = f"""
Target CEFR level: {cefr_level}
Participant name: {student_name}
Participant number: {student_number}
Variation seed: {variation_seed}
Selected personal background: {selected_background}
Selected genre: {selected_genre}

Create a noticeably different passage across participants when possible.

Design guidance:
- Respect the selected background and genre throughout the passage.
- The passage can be about English learning or any broader real-life situation, as long as it contains clear feelings, thoughts, decisions, and coping or strategy opportunities.
- Let the 10 sentences move through introduction, development, tension or decision, and ending.
- Make the questions useful for evaluation, but phrase them naturally for students.
- Give students room to show fluency through connected explanation, not only through one-word or one-phrase answers.

Evaluation criteria:
{QUESTION_GENERATION_CRITERIA}

Question slots:
{question_slot_instructions}

Return JSON only in this format:
{{
  "story_title": "...",
  "story_title_ko": "...",
  "central_characters": "...",
  "situation_focus": "...",
  "situation_focus_ko": "...",
  "passage_sentences": ["...", "...", "...", "...", "...", "...", "...", "...", "...", "..."],
  "passage_ko_sentences": ["...", "...", "...", "...", "...", "...", "...", "...", "...", "..."],
  "questions": [
    {{"id": "q1", "prompt": "....", "prompt_ko": "...."}},
    {{"id": "q2", "prompt": "....", "prompt_ko": "...."}},
    {{"id": "q3", "prompt": "....", "prompt_ko": "...."}},
    {{"id": "q4", "prompt": "....", "prompt_ko": "...."}},
    {{"id": "q5", "prompt": "....", "prompt_ko": "...."}},
    {{"id": "q6", "prompt": "....", "prompt_ko": "...."}},
    {{"id": "q7", "prompt": "....", "prompt_ko": "...."}},
    {{"id": "q8", "prompt": "....", "prompt_ko": "...."}},
    {{"id": "q9", "prompt": "....", "prompt_ko": "...."}},
    {{"id": "q10", "prompt": "....", "prompt_ko": "...."}}
  ]
}}
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
    )

    response_text = completion.choices[0].message.content or ""
    payload = extract_json_object(response_text)
    questionnaire = normalize_experiment_payload(payload, cefr_level)
    questionnaire["selected_background"] = selected_background
    questionnaire["selected_genre"] = selected_genre
    questionnaire["question_source"] = "openai"
    questionnaire["question_model"] = model
    questionnaire["question_generation_note"] = (
        "Passage and interactive questions were generated with OpenAI "
        f"for background '{selected_background}' and genre '{selected_genre}'."
    )
    return questionnaire


@st.cache_data(show_spinner=False, ttl=86400)
def build_generated_experiment(
    cefr_level: str,
    student_name: str,
    student_number: str,
    selected_background: str,
    selected_genre: str,
    api_key: str,
    model: str,
    nonce: str,
) -> dict:
    if not api_key:
        raise ValueError(
            "OpenAI API key is not configured. Set [openai].api_key or OPENAI_API_KEY in Streamlit secrets."
        )

    return generate_experiment_with_openai(
        cefr_level=cefr_level,
        student_name=student_name,
        student_number=student_number,
        variation_seed=nonce[:12],
        selected_background=selected_background,
        selected_genre=selected_genre,
        api_key=api_key,
        model=model,
    )


# -----------------------------
# Answer validation and follow-up generation
# -----------------------------
# -----------------------------
# Answer validation and follow-up generation
# -----------------------------
def get_personalized_self_spec(question_id: str) -> dict:
    return PERSONALIZED_SELF_QUESTION_SPECS.get(question_id, {})


def normalize_single_prompt_payload(payload: dict) -> dict:
    prompt = str(payload.get("prompt", "")).strip()
    prompt_ko = str(payload.get("prompt_ko", "")).strip()

    if not prompt or not prompt_ko:
        raise ValueError("The model did not return both prompt and prompt_ko.")

    return {
        "prompt": prompt,
        "prompt_ko": prompt_ko,
    }


def normalize_answer_feedback_payload(payload: dict) -> dict:
    status = str(payload.get("status", "")).strip().lower()
    message = str(payload.get("message", "")).strip()
    message_ko = str(payload.get("message_ko", "")).strip()

    if status not in {"valid", "rewrite"}:
        raise ValueError("Feedback status must be valid or rewrite.")
    if not message or not message_ko:
        raise ValueError("Feedback must include message and message_ko.")

    return {
        "status": status,
        "message": message,
        "message_ko": message_ko,
    }


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.strip().split())


def sentence_count(text: str) -> int:
    if not text:
        return 0
    sentences = re.split(r"[.!?]+", text.strip())
    sentences = [sentence for sentence in sentences if sentence.strip()]
    return len(sentences)


def contains_any(text: str, keywords: list) -> bool:
    return any(keyword in text for keyword in keywords)


def count_matches(text: str, keywords: list) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def generate_answer_feedback_with_openai(
    questionnaire: dict,
    question: dict,
    answer: str,
    api_key: str,
    model: str,
) -> dict:
    client = get_openai_client(api_key)

    system_prompt = """
You evaluate one learner answer for an English-education diagnostic app.
Return valid JSON only.

Rules:
- Mark status as "rewrite" only when the answer is empty, says I don't know or no idea, is only symbols or random text, or is so strange that it cannot be used.
- Mark status as "valid" for short, weak, grammatically incorrect, or partially incorrect English if it still looks like an attempt to answer.
- If status is "rewrite", the message must explicitly say the answer is incomplete and must be rewritten.
- Keep feedback short, warm, and direct.
"""

    user_prompt = f"""
Passage title: {questionnaire['story_title']}
Passage:
{questionnaire['passage_plain']}

Question layer: {question['layer']}
Question type: {question['type']}
Question:
{question['prompt']}

Learner answer:
{answer}

Return JSON only:
{{
  "status": "valid" or "rewrite",
  "message": "...",
  "message_ko": "..."
}}
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
    )

    response_text = completion.choices[0].message.content or ""
    payload = extract_json_object(response_text)
    return normalize_answer_feedback_payload(payload)


def period_sentence_count(text: str) -> int:
    if not text:
        return 0
    parts = [part.strip() for part in text.split(".") if part.strip()]
    return len(parts)


def contains_korean(text: str) -> bool:
    return bool(re.search(r"[\uAC00-\uD7AF\u3130-\u318F]", text))


def is_sentence_copied_from_passage(answer: str, passage_sentences: list) -> bool:
    normalized_answer = re.sub(r"\s+", " ", answer.strip().lower())
    for sentence in passage_sentences:
        normalized_sentence = re.sub(r"\s+", " ", sentence.strip().lower())
        # Only check sentences that are long enough to be meaningful (5+ words)
        if len(normalized_sentence.split()) >= 5 and normalized_sentence in normalized_answer:
            return True
    return False


def build_local_answer_feedback(question: dict, answer: str, passage_sentences: list = None) -> dict:
    stripped = answer.strip()
    normalized_lower = re.sub(r"\s+", " ", stripped.lower())

    if not stripped:
        return {
            "status": "rewrite",
            "message": "Your answer is incomplete. Please write something for this question.",
            "message_ko": "답변이 비어 있습니다. 이 질문에 대해 내용을 적어 주세요.",
        }

    if re.fullmatch(r"[\W_]+", stripped, flags=re.UNICODE):
        return {
            "status": "rewrite",
            "message": "Your answer is incomplete. Please write words, not only symbols or punctuation.",
            "message_ko": "답변이 불완전합니다. 기호만 쓰지 말고 단어로 적어 주세요.",
        }

    if contains_korean(stripped):
        return {
            "status": "rewrite",
            "message": "Please write your answer in English only. Korean answers cannot be accepted.",
            "message_ko": "답변은 영어로만 작성해 주세요. 한국어 답변은 인정되지 않습니다.",
        }

    if re.fullmatch(r"(?:i\s+don't\s+know|i\s+do\s+not\s+know|dont\s+know|idk|no\s+idea)[.!? ]*", normalized_lower):
        return {
            "status": "rewrite",
            "message": "Your answer is incomplete because it says I don't know. Please try to write anything you can.",
            "message_ko": "답변이 I don't know로 되어 있어 불완전합니다. 할 수 있는 만큼이라도 적어 주세요.",
        }

    if passage_sentences and is_sentence_copied_from_passage(stripped, passage_sentences):
        return {
            "status": "rewrite",
            "message": "Your answer copies a sentence from the passage. Please write the answer in your own words.",
            "message_ko": "지문의 문장을 그대로 복사했습니다. 자신의 말로 다시 작성해 주세요.",
        }

    if question.get("id") == "q1" and period_sentence_count(stripped) < 2:
        return {
            "status": "rewrite",
            "message": "Q1 is not complete yet. Please write at least 2 sentences and separate them with periods.",
            "message_ko": "Q1이 아직 완료되지 않았습니다. 마침표(.)를 사용해 최소 2문장 이상으로 작성해 주세요.",
        }

    if question.get("id") == "q1":
        return {
            "status": "valid",
            "message": "Q1 is complete. This answer can be used to create Question 2.",
            "message_ko": "Q1이 완료되었습니다. 이 답변을 바탕으로 2번 질문을 만들 수 있습니다.",
        }

    return {
        "status": "valid",
        "message": "Recorded. This answer can be used to create the next question.",
        "message_ko": "기록되었습니다. 이 답변을 바탕으로 다음 질문을 만들 수 있습니다.",
    }


@st.cache_data(show_spinner=False, ttl=86400)
def get_answer_feedback(
    questionnaire_json: str,
    question_json: str,
    answer: str,
    api_key: str,
    model: str,
) -> dict:
    question = json.loads(question_json)
    questionnaire = json.loads(questionnaire_json)
    passage_sentences = questionnaire.get("passage_sentences", [])
    return build_local_answer_feedback(question, answer, passage_sentences)


def build_pending_self_prompt(
    question: dict,
    source_question: dict,
    source_feedback: dict,
) -> dict:
    reason_en = source_feedback.get(
        "message",
        "The previous answer needs to be rewritten before the next question can be created.",
    )
    reason_ko = source_feedback.get(
        "message_ko",
        "다음 질문을 만들기 전에 이전 답변을 다시 작성해야 합니다.",
    )

    return {
        "prompt": (
            f"Finish {source_question['label']} first. The interactive follow-up question for this part will appear here after your answer is usable."
        ),
        "prompt_ko": (
            f"먼저 {source_question['label']}에 답해 주세요. 이 파트의 상호작용형 후속 질문은 답변을 사용할 수 있게 되면 여기에 나타납니다."
        ),
        "prompt_source": "pending",
        "prompt_note": (
            f"Rewrite {source_question['label']} first. {reason_en}\n\n"
            f"{source_question['label']}을 먼저 다시 작성해 주세요. {reason_ko}"
        ),
        "ready_for_answer": False,
    }


def generate_personalized_self_prompt_with_openai(
    questionnaire: dict,
    question: dict,
    source_question: dict,
    source_answer: str,
    api_key: str,
    model: str,
) -> dict:
    client = get_openai_client(api_key)
    spec = get_personalized_self_spec(question["id"])

    q2_understanding_guidance = ""
    if question["id"] == "q2":
        q2_understanding_guidance = """
Additional instruction for Q2:
- Use both the passage and the learner's Q1 answer, not only one of them.
- Keep Q2 anchored in the Understanding layer, but move it into the learner's own life as a self-applied follow-up.
- Ask the learner to connect the main situation, problem, or key detail from the passage to a similar real-life situation for themselves.
- Ask the learner to expand, clarify, support, or refine the idea they already gave in Q1.
- If the learner's Q1 answer is short, partial, or grammatically weak but still meaningful, treat it as usable and build the next question from the main idea the learner actually gave.
- Keep the question open enough to invite a longer connected response, not a yes/no answer or a single quoted detail.
- Help reveal fluency features such as sentence development, connector use, linked ideas, organization, and emotional expression.
""".strip()

    system_prompt = """
You generate one interactive follow-up question for an English-education diagnostic app.
Return valid JSON only.
The question must be the second question in a part and must clearly connect to the learner's answer to the first question.
If the follow-up question type is Self, it must move into the learner's own experience in a similar real-life situation.
If the follow-up question type is Depth 2, it must stay focused on the passage and deepen the learner's reasoning.
If the follow-up is Q2 in the Understanding section, it must clearly use both the passage and the learner's Q1 answer, then turn that into a self-applied follow-up about a similar real-life situation for the learner. It should invite a fuller connected explanation so fluency features such as sentence development, connector use, linked ideas, organization, and emotional expression can be observed. Even when the learner's Q1 answer is short, partial, or grammatically weak, use the meaningful part of that answer instead of treating it as unusable.
Use simple, learner-friendly English.
Do not mention technical labels such as FLA, FLE, self-efficacy, metacognition, WTC, coping, engagement, or strategy type.
Make the Korean translation natural and faithful.
"""

    user_prompt = f"""
Passage title: {questionnaire['story_title']}
Passage:
{questionnaire['passage_plain']}

Current layer: {question['layer']}
Follow-up question type: {question['type']}
State targets: {spec.get('state_targets', '')}
Instruction:
{spec.get('criteria_focus', '')}

Base follow-up direction:
{question.get('base_prompt', question.get('prompt', ''))}

First question in this part:
{source_question['prompt']}

Learner's answer to the first question:
{source_answer}

{q2_understanding_guidance}

Create one interactive follow-up question that:
1. Feels clearly linked to the learner's answer above.
2. Matches the layer and question type.
3. Uses the passage and the learner's answer together when deciding what to ask next.
4. Sounds natural for a student questionnaire.
5. Uses 1 or 2 short sentences.

Return JSON only:
{{
  "prompt": "....",
  "prompt_ko": "...."
}}
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
    )

    response_text = completion.choices[0].message.content or ""
    payload = extract_json_object(response_text)
    normalized = normalize_single_prompt_payload(payload)
    normalized["prompt_source"] = "openai_personalized"
    normalized["prompt_note"] = spec["note"]
    normalized["ready_for_answer"] = True
    return normalized


@st.cache_data(show_spinner=False, ttl=86400)
def build_personalized_self_prompt(
    questionnaire_json: str,
    question_json: str,
    source_question_json: str,
    source_answer: str,
    source_feedback_json: str,
    api_key: str,
    model: str,
) -> dict:
    questionnaire = json.loads(questionnaire_json)
    question = json.loads(question_json)
    source_question = json.loads(source_question_json)
    source_feedback = json.loads(source_feedback_json)

    if not source_answer.strip() or source_feedback.get("status") != "valid":
        return build_pending_self_prompt(question, source_question, source_feedback)

    if not api_key:
        raise ValueError("OpenAI API key is not configured. Personalized follow-up generation requires OpenAI.")

    return generate_personalized_self_prompt_with_openai(
        questionnaire=questionnaire,
        question=question,
        source_question=source_question,
        source_answer=source_answer,
        api_key=api_key,
        model=model,
    )


def materialize_questionnaire_for_answers(
    questionnaire: dict,
    answers: dict,
    llm_config: dict,
) -> dict:
    materialized = deepcopy(questionnaire)
    question_lookup = get_question_lookup(materialized)
    questionnaire_json = json.dumps(questionnaire, ensure_ascii=False, sort_keys=True)

    for question in get_all_questions(materialized):
        question["prompt"] = question.get("base_prompt", question.get("prompt", ""))
        question["prompt_ko"] = question.get("base_prompt_ko", question.get("prompt_ko", ""))
        question["prompt_source"] = materialized.get("question_source", "openai")
        question["prompt_note"] = ""
        question["ready_for_answer"] = True

        if not question.get("depends_on"):
            continue

        source_question = question_lookup[question["depends_on"]]
        source_answer = answers.get(source_question["id"], "").strip()
        try:
            source_feedback = get_answer_feedback(
                questionnaire_json=questionnaire_json,
                question_json=json.dumps(source_question, ensure_ascii=False, sort_keys=True),
                answer=source_answer,
                api_key=llm_config["api_key"],
                model=llm_config["model"],
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to evaluate {source_question['id']} before generating {question['id']}. {format_exception_message(error)}"
            ) from error

        try:
            personalized = build_personalized_self_prompt(
                questionnaire_json=questionnaire_json,
                question_json=json.dumps(question, ensure_ascii=False, sort_keys=True),
                source_question_json=json.dumps(source_question, ensure_ascii=False, sort_keys=True),
                source_answer=source_answer,
                source_feedback_json=json.dumps(source_feedback, ensure_ascii=False, sort_keys=True),
                api_key=llm_config["api_key"],
                model=llm_config["model"],
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to generate personalized follow-up for {question['id']}. {format_exception_message(error)}"
            ) from error
        question.update(personalized)

    return materialized


@st.cache_data(show_spinner=False, ttl=86400)
def materialize_questionnaire_for_answers_cached(
    questionnaire_json: str,
    answers_json: str,
    api_key: str,
    model: str,
) -> dict:
    questionnaire = json.loads(questionnaire_json)
    answers = json.loads(answers_json)
    return materialize_questionnaire_for_answers(
        questionnaire=questionnaire,
        answers=answers,
        llm_config={
            "api_key": api_key,
            "model": model,
        },
    )


@st.cache_data(show_spinner=False, ttl=86400)
def build_answer_feedback_map(
    questionnaire_json: str,
    answers_json: str,
    api_key: str,
    model: str,
) -> dict:
    questionnaire = json.loads(questionnaire_json)
    answers = json.loads(answers_json)
    feedback_map = {}

    for question in get_all_questions(questionnaire):
        if not question.get("ready_for_answer", True):
            continue

        answer = answers.get(question["id"], "").strip()
        if not answer:
            continue

        try:
            feedback_map[question["id"]] = get_answer_feedback(
                questionnaire_json=questionnaire_json,
                question_json=json.dumps(question, ensure_ascii=False, sort_keys=True),
                answer=answer,
                api_key=api_key,
                model=model,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to evaluate {question['id']}. {format_exception_message(error)}"
            ) from error

    return feedback_map


# -----------------------------
# Evaluation helpers
# -----------------------------
def detect_grammar_errors(text: str, api_key: str = "", model: str = "") -> str:
    """
    Use OpenAI to count grammar errors and return High / Mid / Low.

    Thresholds (absolute error count):
      High : 0 errors
      Mid  : 1–3 errors
      Low  : 4+ errors

    Falls back to LanguageTool if OpenAI is unavailable.
    """
    if not text or not text.strip():
        return "High"

    # --- OpenAI path ---
    if api_key:
        try:
            client = get_openai_client(api_key)
            response = client.chat.completions.create(
                model=model or DEFAULT_LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a grammar checker for EFL learners. "
                            "Count the number of grammar errors in the given text. "
                            "Focus on: subject-verb agreement, verb tense, article usage, "
                            "preposition errors, word form errors, and capitalization of 'I'. "
                            "Return only valid JSON with one key: {\"error_count\": <integer>}. "
                            "Do not include any explanation."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=20,
            )
            raw = response.choices[0].message.content.strip()
            error_count = json.loads(raw).get("error_count", 0)
            if error_count == 0:
                return "High"
            if error_count <= 3:
                return "Mid"
            return "Low"
        except Exception:
            pass  # fall through to LanguageTool

    # --- LanguageTool fallback ---
    try:
        tool = _get_language_tool()
        matches = tool.check(text)
        filtered = [
            m for m in matches
            if m.category not in _GRAMMAR_IGNORE_CATEGORIES
            and m.rule_id not in _GRAMMAR_IGNORE_RULE_IDS
        ]
        error_count = len(filtered)
        if error_count == 0:
            return "High"
        wc = max(len(text.split()), 1)
        density = error_count / wc * 100
        if density <= 30:
            return "Mid"
        return "Low"
    except Exception:
        return "High"


def detect_connectors(text: str) -> list:
    if not text:
        return []

    lower_text = text.lower()
    candidates = ["because", "but", "and", "so", "when", "if", "or", "then"]

    return [
        connector
        for connector in candidates
        if re.search(rf"\b{re.escape(connector)}\b", lower_text)
    ]


def detect_structure_markers(text: str) -> list:
    if not text:
        return []

    lower_text = text.lower()
    candidates = [
        "because",
        "when",
        "if",
        "while",
        "although",
        "though",
        "before",
        "after",
        "so that",
        "which",
        "who",
        "that",
    ]

    return [
        marker
        for marker in candidates
        if re.search(rf"\b{re.escape(marker)}\b", lower_text)
    ]


def detect_organization_markers(text: str) -> list:
    if not text:
        return []

    lower_text = text.lower()
    candidates = [
        "first",
        "next",
        "then",
        "later",
        "after that",
        "finally",
        "in the end",
        "at first",
        "for example",
        "for instance",
        "also",
    ]

    return [
        marker
        for marker in candidates
        if re.search(rf"\b{re.escape(marker)}\b", lower_text)
    ]


def detect_strategy_expressions(text: str) -> list:
    if not text:
        return []

    lower_text = text.lower()
    candidates = [
        "try",
        "plan",
        "practice",
        "prepare",
        "ask",
        "check",
        "use",
        "change",
        "solve",
        "fix",
        "explain",
        "decide",
        "manage",
        "review",
        "repeat",
        "calm down",
    ]

    return [
        marker
        for marker in candidates
        if re.search(rf"\b{re.escape(marker)}\b", lower_text)
    ]


def score_to_label(score: float) -> str:
    if score >= 1.4:
        return "High"
    if score >= 0.6:
        return "Mid"
    return "Low"


def label_to_score(label: str) -> int:
    return {"Low": 0, "Mid": 1, "High": 2}.get(label, 0)


def analyze_fluency_features(text: str, api_key: str = "", model: str = "") -> dict:
    wc = word_count(text)
    sc = sentence_count(text)
    connectors = detect_connectors(text)
    structure_markers = detect_structure_markers(text)
    organization_markers = detect_organization_markers(text)
    strategy_expressions = detect_strategy_expressions(text)
    grammar_label = detect_grammar_errors(text, api_key=api_key, model=model)

    avg_sentence_length = wc / sc if sc else 0.0

    if wc >= 18 or (sc >= 2 and avg_sentence_length >= 9):
        sentence_length_label = "High"
    elif wc >= 8 or avg_sentence_length >= 5:
        sentence_length_label = "Mid"
    else:
        sentence_length_label = "Low"

    if len(connectors) >= 2:
        connector_label = "High"
    elif len(connectors) >= 1:
        connector_label = "Mid"
    else:
        connector_label = "Low"

    if len(structure_markers) >= 2 or (sc >= 2 and len(connectors) >= 2):
        structure_label = "High"
    elif len(structure_markers) >= 1 or sc >= 2:
        structure_label = "Mid"
    else:
        structure_label = "Low"

    if len(organization_markers) >= 2:
        organization_label = "High"
    elif len(organization_markers) >= 1:
        organization_label = "Mid"
    else:
        organization_label = "Low"

    if len(strategy_expressions) >= 2:
        strategy_label = "High"
    elif len(strategy_expressions) >= 1:
        strategy_label = "Mid"
    else:
        strategy_label = "Low"

    non_low_feature_count = sum(
        label != "Low"
        for label in [
            sentence_length_label,
            connector_label,
            structure_label,
            organization_label,
            strategy_label,
            grammar_label,
        ]
    )
    high_feature_count = sum(
        label == "High"
        for label in [
            sentence_length_label,
            connector_label,
            structure_label,
            organization_label,
            strategy_label,
            grammar_label,
        ]
    )

    if (
        sentence_length_label != "Low"
        and connector_label != "Low"
        and non_low_feature_count >= 4
        and high_feature_count >= 2
    ):
        overall_label = "High"
    elif sentence_length_label != "Low" and non_low_feature_count >= 2:
        overall_label = "Mid"
    else:
        overall_label = "Low"

    return {
        "word_count": wc,
        "sentence_count": sc,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "connectors": connectors,
        "structure_markers": structure_markers,
        "organization_markers": organization_markers,
        "strategy_expressions": strategy_expressions,
        "sentence_length_label": sentence_length_label,
        "connector_label": connector_label,
        "structure_label": structure_label,
        "organization_label": organization_label,
        "strategy_label": strategy_label,
        "grammar_label": grammar_label,
        "overall_label": overall_label,
    }


def fluency_label(text: str) -> str:
    return analyze_fluency_features(text)["overall_label"]


def evaluate_fluency(answers: dict, api_key: str = "", model: str = "") -> dict:
    analyses = [
        analyze_fluency_features(answer, api_key=api_key, model=model)
        for answer in answers.values()
        if answer.strip()
    ]

    if not analyses:
        return {
            "Fluency": "Low",
            "Fluency_sentence_length": "Low",
            "Fluency_connector_use": "Low",
            "Fluency_structure_complexity": "Low",
            "Fluency_organization": "Low",
            "Fluency_strategy_expression": "Low",
            "Fluency_grammar": "High",
            "Fluency_state": (
                "Fluency: Low / Sentence length: Low / Connector use: Low / "
                "Structure complexity: Low / Organization: Low / Strategy expression: Low / Grammar: High"
            ),
            "Fluency_connector_examples": "",
            "Fluency_feature_note": "No usable responses were available for fluency analysis.",
        }

    def average_feature(feature_key: str) -> float:
        return sum(label_to_score(item[feature_key]) for item in analyses) / len(analyses)

    sentence_length_label = score_to_label(average_feature("sentence_length_label"))
    connector_label = score_to_label(average_feature("connector_label"))
    structure_label = score_to_label(average_feature("structure_label"))
    organization_label = score_to_label(average_feature("organization_label"))
    strategy_label = score_to_label(average_feature("strategy_label"))
    grammar_label = score_to_label(average_feature("grammar_label"))

    averaged_non_low = sum(
        label != "Low"
        for label in [
            sentence_length_label,
            connector_label,
            structure_label,
            organization_label,
            strategy_label,
            grammar_label,
        ]
    )
    averaged_high = sum(
        label == "High"
        for label in [
            sentence_length_label,
            connector_label,
            structure_label,
            organization_label,
            strategy_label,
            grammar_label,
        ]
    )

    if (
        sentence_length_label != "Low"
        and connector_label != "Low"
        and averaged_non_low >= 4
        and averaged_high >= 2
    ):
        overall_label = "High"
    elif sentence_length_label != "Low" and averaged_non_low >= 2:
        overall_label = "Mid"
    else:
        overall_label = "Low"

    connector_examples = sorted(
        {
            connector
            for item in analyses
            for connector in item["connectors"]
        }
    )

    return {
        "Fluency": overall_label,
        "Fluency_sentence_length": sentence_length_label,
        "Fluency_connector_use": connector_label,
        "Fluency_structure_complexity": structure_label,
        "Fluency_organization": organization_label,
        "Fluency_strategy_expression": strategy_label,
        "Fluency_grammar": grammar_label,
        "Fluency_state": (
            f"Fluency: {overall_label} / Sentence length: {sentence_length_label} / "
            f"Connector use: {connector_label} / Structure complexity: {structure_label} / "
            f"Organization: {organization_label} / Strategy expression: {strategy_label} / "
            f"Grammar: {grammar_label}"
        ),
        "Fluency_connector_examples": ", ".join(connector_examples),
        "Fluency_feature_note": (
            "Based on sentence length, connector use, structure complexity, "
            "organization, strategy expression, and grammar across the learner's responses."
        ),
    }


def classify_fla_fle(text: str) -> tuple:
    lower_text = text.lower()

    anxiety_keywords = [
        "nervous",
        "afraid",
        "scared",
        "worry",
        "worried",
        "anxious",
        "shy",
        "stress",
        "stressed",
        "panic",
        "embarrassed",
        "hurt",
        "sad",
        "avoid",
        "quiet",
        "mistake",
        "judge",
        "ashamed",
    ]
    anxiety_control_keywords = [
        "but i try",
        "but i still try",
        "but sometimes i try",
        "try",
        "prepare",
        "practice",
        "check",
    ]
    calm_keywords = ["comfortable", "relaxed", "calm", "fine"]

    enjoyment_keywords = [
        "enjoy",
        "fun",
        "interesting",
        "happy",
        "excited",
        "good",
        "comfortable",
        "relaxed",
        "warm",
        "like",
        "love",
        "confident",
    ]
    mixed_enjoyment_keywords = [
        "okay",
        "sometimes enjoyable",
        "sometimes fun",
        "a little fun",
        "not bad",
    ]
    low_enjoyment_keywords = [
        "boring",
        "do not like",
        "don't like",
        "not interesting",
        "hate",
        "bad",
    ]

    anxiety_score = count_matches(lower_text, anxiety_keywords)
    has_avoidance = contains_any(
        lower_text,
        ["avoid", "stay quiet", "silent", "do not speak", "don't speak"],
    )
    has_anxiety_control = anxiety_score > 0 and contains_any(lower_text, anxiety_control_keywords)

    if anxiety_score >= 2 or (anxiety_score >= 1 and has_avoidance):
        fla = "High"
    elif anxiety_score >= 1 or has_anxiety_control:
        fla = "Mid"
    elif contains_any(lower_text, calm_keywords):
        fla = "Low"
    else:
        fla = "Low"

    enjoyment_score = count_matches(lower_text, enjoyment_keywords)

    if enjoyment_score >= 2:
        fle = "High"
    elif enjoyment_score >= 1 or contains_any(lower_text, mixed_enjoyment_keywords):
        fle = "Mid"
    elif contains_any(lower_text, low_enjoyment_keywords) or anxiety_score >= 1:
        fle = "Low"
    else:
        fle = "Low"

    return fla, fle


def classify_self_efficacy_metacognition(text: str) -> tuple:
    lower_text = text.lower()

    high_selfeff = [
        "i can speak well",
        "i am confident",
        "i can do it",
        "i can answer",
        "i can explain",
        "i can speak in english",
    ]
    mid_selfeff = [
        "i can, but",
        "i can but",
        "simple english",
        "a little confident",
        "sometimes i can",
        "i can try",
    ]
    low_selfeff = [
        "i cannot",
        "i can't",
        "i am not good",
        "i will make mistakes",
        "i am bad",
        "too difficult",
        "hard for me",
        "people may judge me",
        "i am not good at english",
    ]

    meta_high = [
        "i know my problem",
        "i try to fix",
        "i try to improve",
        "i check",
        "i plan",
        "i prepare",
        "i monitor",
        "i practice more",
        "i review",
    ]
    meta_mid = [
        "i know i am weak",
        "i know i need",
        "i think i am weak",
        "sometimes",
        "maybe",
        "i know",
    ]

    if contains_any(lower_text, low_selfeff):
        self_efficacy = "Low"
    elif contains_any(lower_text, high_selfeff):
        self_efficacy = "High"
    elif contains_any(lower_text, mid_selfeff):
        self_efficacy = "Mid"
    else:
        self_efficacy = "Mid"

    if contains_any(lower_text, meta_high):
        metacognition = "High"
    elif contains_any(lower_text, meta_mid):
        metacognition = "Mid"
    else:
        metacognition = "Low"

    return self_efficacy, metacognition


def classify_behavior(text: str) -> tuple:
    lower_text = text.lower()

    high_wtc_keywords = [
        "i try to speak",
        "i speak",
        "i talk",
        "i ask",
        "i answer",
        "i join",
    ]
    mid_wtc_keywords = [
        "sometimes i speak",
        "sometimes i try",
        "depends",
        "if the teacher asks",
        "rarely speak",
    ]
    low_wtc_keywords = [
        "stay quiet",
        "silent",
        "do not speak",
        "don't speak",
        "avoid speaking",
        "i ignore",
    ]

    active_keywords = [
        "practice",
        "prepare",
        "ask",
        "try",
        "study",
        "review",
        "check",
        "breathe",
        "calm",
    ]
    avoidant_keywords = [
        "avoid",
        "give up",
        "stay quiet",
        "do nothing",
        "stop",
        "ignore",
    ]

    high_engagement_keywords = [
        "participate",
        "ask questions",
        "interact",
        "join",
        "answer",
        "talk with others",
    ]
    mid_engagement_keywords = [
        "listen",
        "sometimes participate",
        "rarely speak",
        "listen but",
    ]
    low_engagement_keywords = [
        "passive",
        "no interaction",
        "do not join",
        "stay quiet",
        "just listen",
    ]

    if contains_any(lower_text, low_wtc_keywords):
        wtc = "Low"
    elif contains_any(lower_text, high_wtc_keywords):
        wtc = "High"
    elif contains_any(lower_text, mid_wtc_keywords) or "sometimes" in lower_text:
        wtc = "Mid"
    else:
        wtc = "Mid"

    active = contains_any(lower_text, active_keywords)
    avoidant = contains_any(lower_text, avoidant_keywords)

    if active and avoidant:
        coping = "Mixed"
    elif active:
        coping = "Active"
    else:
        coping = "Avoidant"

    if contains_any(lower_text, low_engagement_keywords):
        engagement = "Low"
    elif contains_any(lower_text, high_engagement_keywords):
        engagement = "High"
    elif contains_any(lower_text, mid_engagement_keywords):
        engagement = "Mid"
    else:
        engagement = "Mid"

    return wtc, coping, engagement


def classify_strategy(text: str) -> tuple:
    lower_text = text.lower()

    strategy_types = []

    cognitive_keywords = ["practice", "repeat", "write", "note", "read again", "memorize"]
    metacognitive_keywords = ["plan", "check", "monitor", "prepare", "set a goal", "review"]
    affective_keywords = ["calm", "relax", "positive", "encourage", "breathe", "feel better"]
    social_keywords = ["ask friends", "ask teacher", "talk with friends", "study group", "peer"]
    compensation_keywords = ["guess", "gesture", "simple words", "easy words", "use other words"]
    avoidant_keywords = ["avoid", "give up", "stay quiet", "do nothing", "ignore"]
    limited_keywords = ["only memorize", "just memorize"]

    if contains_any(lower_text, cognitive_keywords):
        strategy_types.append("Cognitive")
    if contains_any(lower_text, metacognitive_keywords):
        strategy_types.append("Metacognitive")
    if contains_any(lower_text, affective_keywords):
        strategy_types.append("Affective")
    if contains_any(lower_text, social_keywords):
        strategy_types.append("Social")
    if contains_any(lower_text, compensation_keywords):
        strategy_types.append("Compensation")

    if not strategy_types:
        strategy_type = "Unclear"
    elif len(strategy_types) == 1:
        strategy_type = strategy_types[0]
    else:
        strategy_type = "Mixed: " + ", ".join(strategy_types)

    if contains_any(lower_text, avoidant_keywords):
        strategy_quality = "Avoidant"
    elif strategy_type == "Unclear":
        strategy_quality = "Limited"
    elif contains_any(lower_text, limited_keywords):
        strategy_quality = "Limited"
    elif len(strategy_types) >= 2 or "helps" in lower_text:
        strategy_quality = "Effective"
    else:
        strategy_quality = "Limited"

    return strategy_type, strategy_quality


def evaluate_state(answers: dict, api_key: str = "", model: str = "") -> dict:
    fluency = evaluate_fluency(answers, api_key=api_key, model=model)
    fla, fle = classify_fla_fle(answers.get("q4", ""))
    self_efficacy, metacognition = classify_self_efficacy_metacognition(answers.get("q6", ""))
    wtc, coping, engagement = classify_behavior(answers.get("q8", ""))
    strategy_type, strategy_quality = classify_strategy(answers.get("q10", ""))

    return {
        "evaluation_method": "criteria_rules_v3_dynamic_cefr",
        "Fluency": fluency["Fluency"],
        "Fluency_sentence_length": fluency["Fluency_sentence_length"],
        "Fluency_connector_use": fluency["Fluency_connector_use"],
        "Fluency_structure_complexity": fluency["Fluency_structure_complexity"],
        "Fluency_organization": fluency["Fluency_organization"],
        "Fluency_strategy_expression": fluency["Fluency_strategy_expression"],
        "Fluency_grammar": fluency["Fluency_grammar"],
        "Fluency_state": fluency["Fluency_state"],
        "Fluency_connector_examples": fluency["Fluency_connector_examples"],
        "Fluency_feature_note": fluency["Fluency_feature_note"],
        "FLA": fla,
        "FLE": fle,
        "Emotion_state": f"FLA: {fla} / FLE: {fle}",
        "Self_efficacy": self_efficacy,
        "Metacognition": metacognition,
        "Cognition_state": f"Self-efficacy: {self_efficacy} / Metacognition: {metacognition}",
        "WTC": wtc,
        "Coping": coping,
        "Engagement": engagement,
        "Behavior_state": f"WTC: {wtc} / Coping: {coping} / Engagement: {engagement}",
        "Strategy_type": strategy_type,
        "Strategy_quality": strategy_quality,
        "Strategy_state": f"Strategy Type: {strategy_type} / Strategy Quality: {strategy_quality}",
    }


# -----------------------------
# Submission row builders
# -----------------------------
def sanitize_filename_component(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "-", name.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "participant"


def create_submission_id(student_name: str, student_number: str) -> str:
    source_value = student_number.strip() or student_name.strip()
    safe_value = re.sub(r"[^A-Za-z0-9_-]+", "-", source_value)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_value}_{timestamp}_{uuid4().hex[:6]}"


def build_response_row(
    submission_id: str,
    student_name: str,
    student_number: str,
    cefr_level: str,
    questionnaire: dict,
    answers: dict,
) -> dict:
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "submission_id": submission_id,
        "student_name": student_name,
        "student_number": student_number,
        "cefr_level": cefr_level,
        "selected_background": questionnaire.get("selected_background", ""),
        "selected_genre": questionnaire.get("selected_genre", ""),
        "questionnaire_id": questionnaire["id"],
        "questionnaire_title": questionnaire["page_title"],
        "questionnaire_title_ko": questionnaire["page_title_ko"],
        "central_characters": questionnaire.get("central_characters", questionnaire.get("character_focus", "")),
        "situation_focus": questionnaire.get("situation_focus", questionnaire.get("relationship_focus", "")),
        "situation_focus_ko": questionnaire.get("relationship_focus_ko", ""),
        "story_title": questionnaire["story_title"],
        "story_title_ko": questionnaire.get("story_title_ko", ""),
        "passage_text": questionnaire.get("passage_plain", ""),
        "passage_text_ko": questionnaire.get("passage_ko_plain", ""),
        "question_source": questionnaire.get("question_source", "openai"),
        "question_model": questionnaire.get("question_model", ""),
        "question_generation_note": questionnaire.get("question_generation_note", ""),
        "total_questions": len(get_all_questions(questionnaire)),
        "minimum_sentences_required": MIN_SENTENCES,
        "recommended_sentences": RECOMMENDED_SENTENCES,
        "sentence_count_rule": "A sentence is counted when it ends with ., !, or ?, and an unpunctuated response is treated as one sentence.",
    }

    for question in get_all_questions(questionnaire):
        answer = answers.get(question["id"], "").strip()
        row[f"{question['id']}_layer"] = question["layer"]
        row[f"{question['id']}_type"] = question["type"]
        row[f"{question['id']}_prompt"] = question["prompt"]
        row[f"{question['id']}_prompt_ko"] = question["prompt_ko"]
        row[f"{question['id']}_prompt_source"] = question.get("prompt_source", questionnaire.get("question_source", "openai"))
        row[f"{question['id']}_prompt_note"] = question.get("prompt_note", "")
        row[f"{question['id']}_response"] = answer
        row[f"{question['id']}_word_count"] = word_count(answer)
        row[f"{question['id']}_sentence_count"] = sentence_count(answer)
        row[f"{question['id']}_fluency"] = fluency_label(answer)
        row[f"{question['id']}_connectors"] = ", ".join(detect_connectors(answer))

    return row


def build_evaluation_row(
    submission_id: str,
    student_name: str,
    student_number: str,
    cefr_level: str,
    questionnaire: dict,
    answers: dict,
    api_key: str = "",
    model: str = "",
) -> dict:
    evaluation = evaluate_state(answers, api_key=api_key, model=model)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "submission_id": submission_id,
        "student_name": student_name,
        "student_number": student_number,
        "cefr_level": cefr_level,
        "selected_background": questionnaire.get("selected_background", ""),
        "selected_genre": questionnaire.get("selected_genre", ""),
        "questionnaire_id": questionnaire["id"],
        "questionnaire_title": questionnaire["page_title"],
        "questionnaire_title_ko": questionnaire["page_title_ko"],
        "central_characters": questionnaire.get("central_characters", questionnaire.get("character_focus", "")),
        "situation_focus": questionnaire.get("situation_focus", questionnaire.get("relationship_focus", "")),
        "situation_focus_ko": questionnaire.get("relationship_focus_ko", ""),
        "story_title": questionnaire["story_title"],
        "question_source": questionnaire.get("question_source", "openai"),
        "question_model": questionnaire.get("question_model", ""),
        "evaluation_method": evaluation["evaluation_method"],
        "fluency_layer": "Cross-question",
        "Fluency": evaluation["Fluency"],
        "Fluency_sentence_length": evaluation["Fluency_sentence_length"],
        "Fluency_connector_use": evaluation["Fluency_connector_use"],
        "Fluency_structure_complexity": evaluation["Fluency_structure_complexity"],
        "Fluency_organization": evaluation["Fluency_organization"],
        "Fluency_strategy_expression": evaluation["Fluency_strategy_expression"],
        "Fluency_state": evaluation["Fluency_state"],
        "Fluency_connector_examples": evaluation["Fluency_connector_examples"],
        "Fluency_feature_note": evaluation["Fluency_feature_note"],
        "emotion_layer": "Emotion",
        "FLA": evaluation["FLA"],
        "FLE": evaluation["FLE"],
        "Emotion_state": evaluation["Emotion_state"],
        "cognition_layer": "Cognition",
        "Self_efficacy": evaluation["Self_efficacy"],
        "Metacognition": evaluation["Metacognition"],
        "Cognition_state": evaluation["Cognition_state"],
        "behavior_layer": "Behavior",
        "WTC": evaluation["WTC"],
        "Coping": evaluation["Coping"],
        "Engagement": evaluation["Engagement"],
        "Behavior_state": evaluation["Behavior_state"],
        "strategy_layer": "Strategy",
        "Strategy_type": evaluation["Strategy_type"],
        "Strategy_quality": evaluation["Strategy_quality"],
        "Strategy_state": evaluation["Strategy_state"],
        "emotion_self_response": answers.get("q4", "").strip(),
        "cognition_self_response": answers.get("q6", "").strip(),
        "behavior_self_response": answers.get("q8", "").strip(),
        "strategy_self_response": answers.get("q10", "").strip(),
    }


def validate_participant_inputs(
    student_name: str,
    student_number: str,
    cefr_level: str,
    selected_background: str,
    selected_genre: str,
) -> str:
    if not student_name.strip():
        return "Please enter Name / 이름을 입력해주세요."
    if not student_number.strip():
        return "Please enter Student Number / 학번을 입력해주세요."
    if cefr_level not in CEFR_LEVEL_OPTIONS:
        return "Please choose a CEFR level / CEFR 레벨을 선택해주세요."
    if selected_background not in BACKGROUND_OPTIONS:
        return "Please choose a background / 배경을 선택해주세요."
    if selected_genre not in GENRE_OPTIONS:
        return "Please choose a genre / 장르를 선택해주세요."
    return ""


def validate_submission_answers(
    questionnaire: dict,
    answers: dict,
    llm_config: dict,
    feedback_map: dict | None = None,
) -> str:
    questionnaire_json = json.dumps(questionnaire, ensure_ascii=False, sort_keys=True)

    for question in get_all_questions(questionnaire):
        answer = answers.get(question["id"], "").strip()

        if not question.get("ready_for_answer", True):
            return (
                f"{question['label']} is still locked. Please fix the previous answer first. / "
                f"{question['label']}이 아직 잠겨 있습니다. 이전 답변을 먼저 수정해 주세요."
            )

        if not answer:
            return f"Please answer all questions / 모든 질문에 답해주세요."

        feedback = None
        if feedback_map is not None:
            feedback = feedback_map.get(question["id"])

        if feedback is None:
            feedback = get_answer_feedback(
                questionnaire_json=questionnaire_json,
                question_json=json.dumps(question, ensure_ascii=False, sort_keys=True),
                answer=answer,
                api_key=llm_config["api_key"],
                model=llm_config["model"],
            )
        if feedback["status"] != "valid":
            return (
                f"{question['label']} needs revision: {feedback['message']} / "
                f"{question['label']} 수정 필요: {feedback['message_ko']}"
            )

    return ""


# -----------------------------
# Session and UI helpers
# -----------------------------
def widget_key(question_id: str) -> str:
    return f"question_{question_id}"


def initialize_session_state():
    defaults = {
        "student_name_input": "",
        "student_number_input": "",
        "cefr_level_input": "A2",
        "selected_background_input": "Daily life",
        "selected_genre_input": "Narrative story",
        "student_name_value": "",
        "student_number_value": "",
        "cefr_level_value": "",
        "selected_background_value": "",
        "selected_genre_value": "",
        "active_questionnaire": {},
        "saved_answers": {},
        "current_section_index": 0,
        "generation_nonce": "",
        "materialized_questionnaire": {},
        "materialized_signature": "",
        "feedback_map": {},
        "feedback_signature": "",
        "submission_complete": False,
        "last_submission_id": "",
        "last_response_file": "",
        "last_evaluation_file": "",
        "last_storage_backend": "",
        "generation_error": "",
        "last_evaluation_result": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_question_widgets():
    for question in get_question_slots():
        st.session_state.pop(widget_key(question["id"]), None)


def get_saved_answers() -> dict:
    return dict(st.session_state.get("saved_answers", {}))


def set_saved_answer(question_id: str, answer: str):
    saved_answers = get_saved_answers()
    saved_answers[question_id] = answer
    st.session_state.saved_answers = saved_answers


def get_effective_answer(question_id: str, saved_answers: dict | None = None) -> str:
    if saved_answers is None:
        saved_answers = get_saved_answers()

    widget_value = str(st.session_state.get(widget_key(question_id), "")).strip()
    saved_value = str(saved_answers.get(question_id, "")).strip()

    if widget_value:
        return widget_value
    if saved_value:
        return saved_value
    return ""


def build_answers_snapshot(questionnaire: dict) -> dict:
    saved_answers = get_saved_answers()
    snapshot = {}

    for question in get_all_questions(questionnaire):
        snapshot[question["id"]] = get_effective_answer(question["id"], saved_answers)

    return snapshot


def serialize_answers_snapshot(questionnaire: dict, answers: dict) -> str:
    ordered_answers = {
        question["id"]: answers.get(question["id"], "").strip()
        for question in get_all_questions(questionnaire)
    }
    return json.dumps(ordered_answers, ensure_ascii=False, sort_keys=True)


def reset_session_state():
    clear_question_widgets()
    keys_to_clear = [
        "student_name_input",
        "student_number_input",
        "cefr_level_input",
        "selected_background_input",
        "selected_genre_input",
        "student_name_value",
        "student_number_value",
        "cefr_level_value",
        "selected_background_value",
        "selected_genre_value",
        "active_questionnaire",
        "saved_answers",
        "current_section_index",
        "generation_nonce",
        "materialized_questionnaire",
        "materialized_signature",
        "feedback_map",
        "feedback_signature",
        "submission_complete",
        "last_submission_id",
        "last_response_file",
        "last_evaluation_file",
        "last_storage_backend",
        "generation_error",
        "last_evaluation_result",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def inject_custom_styles():
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 238, 214, 0.88), transparent 34%),
                radial-gradient(circle at top right, rgba(221, 236, 255, 0.7), transparent 28%),
                linear-gradient(180deg, #f7efe2 0%, #fcfaf6 46%, #eef5fb 100%) !important;
            color: #12263a !important;
            color-scheme: light !important;
        }

        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0) !important;
        }

        .stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
            color: #12263a !important;
        }

        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input {
            background: #fffdf9 !important;
            color: #12263a !important;
            border-radius: 18px !important;
            border: 1.4px solid #d8cbb7 !important;
            box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.03);
        }

        .stTextArea textarea {
            min-height: 170px !important;
            font-size: 1rem !important;
            line-height: 1.72 !important;
        }

        .stTextInput input:focus,
        .stTextArea textarea:focus {
            border-color: #bb6b22 !important;
            box-shadow: 0 0 0 0.18rem rgba(187, 107, 34, 0.12) !important;
        }

        .stSelectbox > div[data-baseweb="select"] > div {
            background: #fffdf9 !important;
            color: #12263a !important;
            border-radius: 18px !important;
            border: 1.4px solid #d8cbb7 !important;
        }

        .stButton button {
            border-radius: 999px !important;
            border: 1px solid #b35f1f !important;
            background: linear-gradient(180deg, #cb7a30 0%, #b56022 100%) !important;
            color: #fffaf4 !important;
            font-weight: 800 !important;
            letter-spacing: 0.01em;
            min-height: 3rem;
            box-shadow: 0 12px 24px rgba(177, 95, 34, 0.18);
        }

        .stButton button:hover {
            border-color: #a95318 !important;
            filter: brightness(1.02);
        }

        div[data-testid="stSpinner"] {
            position: fixed !important;
            inset: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background: rgba(249, 242, 231, 0.78) !important;
            z-index: 9999 !important;
        }

        div[data-testid="stSpinner"] > div {
            background: #fffaf2 !important;
            border: 1px solid #e6d1b3 !important;
            border-radius: 24px !important;
            padding: 1rem 1.25rem !important;
            box-shadow: 0 24px 60px rgba(18, 38, 58, 0.16) !important;
        }

        .hero-title {
            font-size: 2.1rem;
            font-weight: 900;
            letter-spacing: -0.03em;
            margin-bottom: 0.35rem;
            color: #102437;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            line-height: 1.65;
            color: #55687c;
            margin-bottom: 1.2rem;
            max-width: 76rem;
        }

        .mode-notice {
            border-radius: 22px;
            padding: 0.95rem 1.05rem;
            background: rgba(255, 250, 241, 0.92);
            border: 1px solid rgba(179, 95, 31, 0.16);
            color: #5c4d39 !important;
            margin-bottom: 1rem;
            box-shadow: 0 12px 28px rgba(18, 38, 58, 0.06);
        }

        .participant-shell {
            border-radius: 28px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            background: rgba(255, 251, 245, 0.95);
            border: 1px solid #e8dccb;
            box-shadow: 0 22px 48px rgba(18, 38, 58, 0.08);
            margin-bottom: 1.1rem;
        }

        .participant-title {
            font-size: 1.1rem;
            font-weight: 800;
            color: #14283c !important;
            margin-bottom: 0.3rem;
        }

        .participant-subtitle {
            color: #65788b !important;
            margin-bottom: 0.95rem;
            font-size: 0.95rem;
        }

        .panel-title {
            font-size: 1.15rem;
            font-weight: 900;
            color: #13263b !important;
            margin: 0 0 0.2rem 0;
        }

        .panel-subtitle {
            color: #607386 !important;
            margin: 0 0 0.9rem 0;
            line-height: 1.6;
        }

        .passage-panel {
            position: sticky;
            top: 1rem;
            border-radius: 28px;
            padding: 1.2rem 1.2rem 1.1rem 1.2rem;
            background: linear-gradient(180deg, rgba(255, 251, 245, 0.96) 0%, rgba(246, 248, 252, 0.96) 100%);
            border: 1px solid #dfd2c0;
            box-shadow: 0 22px 48px rgba(18, 38, 58, 0.08);
        }

        .passage-kicker {
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #a55b1f !important;
            margin-bottom: 0.55rem;
        }

        .story-title {
            font-size: 1.6rem;
            font-weight: 900;
            line-height: 1.2;
            color: #102438 !important;
            margin-bottom: 0.25rem;
        }

        .story-title-ko {
            font-size: 0.98rem;
            color: #728496 !important;
            margin-bottom: 0.9rem;
        }

        .meta-row {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
            margin-bottom: 0.95rem;
        }

        .meta-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.32rem 0.7rem;
            border-radius: 999px;
            background: #e9f1fb;
            color: #224968 !important;
            border: 1px solid #cfe0f2;
            font-size: 0.78rem;
            font-weight: 800;
        }

        .passage-block {
            border-radius: 22px;
            padding: 1rem 1rem 0.9rem 1rem;
            background: #fffdf9;
            border: 1px solid #ece0cf;
            margin-bottom: 0.85rem;
        }

        .passage-block h4 {
            margin: 0 0 0.65rem 0;
            color: #123049 !important;
            font-size: 0.98rem;
            font-weight: 900;
            letter-spacing: 0.01em;
        }

        .passage-sentence {
            margin: 0 0 0.5rem 0;
            line-height: 1.72;
            color: #1f3448 !important;
        }

        .question-shell {
            border-radius: 28px;
            padding: 1.2rem 1.15rem 1.05rem 1.15rem;
            background: rgba(255, 251, 245, 0.9);
            border: 1px solid #e8dccb;
            box-shadow: 0 22px 48px rgba(18, 38, 58, 0.08);
        }

        .section-title {
            margin-top: 1rem;
            margin-bottom: 0.2rem;
            font-size: 1.2rem;
            font-weight: 900;
            color: #13263b !important;
            letter-spacing: -0.02em;
        }

        .section-subtitle {
            margin-bottom: 0.8rem;
            color: #5f7286 !important;
            font-size: 0.94rem;
        }

        .question-card {
            border-radius: 22px;
            padding: 1rem 1.05rem 0.95rem 1.05rem;
            margin: 0.78rem 0 0.45rem 0;
            border: 1px solid #dcd2c3;
            box-shadow: 0 14px 30px rgba(21, 43, 68, 0.07);
        }

        .question-card-character {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border-left: 7px solid #2f6fed;
        }

        .question-card-self {
            background: linear-gradient(180deg, #fffef8 0%, #fff7e8 100%);
            border-left: 7px solid #d97706;
        }

        .question-card-depth {
            background: linear-gradient(180deg, #fffdfa 0%, #fbf4ec 100%);
            border-left: 7px solid #8f5a2a;
        }

        .question-card-pending {
            background: linear-gradient(180deg, #f8fafc 0%, #eef4f8 100%);
            border-left-color: #7b8da2;
        }

        .question-top {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
            align-items: center;
            margin-bottom: 0.7rem;
        }

        .question-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.28rem 0.62rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .question-chip-label {
            background: #12263f;
            color: #ffffff !important;
        }

        .question-chip-layer {
            background: #dce9ff;
            color: #1b4fb8 !important;
        }

        .question-chip-type {
            background: #fff0d6;
            color: #9a5a00 !important;
        }

        .question-complete-badge {
            margin-left: auto;
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            background: rgba(220, 252, 231, 0.9);
            color: #1f5b33 !important;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }

        .question-incomplete-badge {
            margin-left: auto;
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            background: rgba(241, 245, 249, 0.88);
            color: #64748b !important;
            border: 1px solid rgba(148, 163, 184, 0.3);
        }

        .question-prompt {
            font-size: 1.14rem;
            line-height: 1.58;
            font-weight: 800;
            color: #13263b !important;
            margin-bottom: 0.65rem;
        }

        .question-ko {
            font-size: 0.97rem;
            line-height: 1.62;
            color: #4d6177 !important;
            margin-bottom: 0.55rem;
        }

        .question-note {
            font-size: 0.9rem;
            line-height: 1.58;
            color: #6a4b0a !important;
            background: rgba(255, 244, 214, 0.7);
            border-radius: 14px;
            padding: 0.6rem 0.75rem;
            border: 1px solid rgba(217, 119, 6, 0.18);
            white-space: pre-line;
        }

        .question-note-muted {
            color: #56697d !important;
            background: rgba(235, 242, 248, 0.8);
            border: 1px solid rgba(123, 141, 162, 0.18);
        }

        .status-box {
            border-radius: 16px;
            padding: 0.78rem 0.9rem;
            margin: 0.25rem 0 1.15rem 0;
            font-size: 0.92rem;
            line-height: 1.58;
        }

        .status-box-valid {
            background: rgba(220, 252, 231, 0.75);
            color: #1f5b33 !important;
            border: 1px solid rgba(34, 197, 94, 0.18);
        }

        .status-box-rewrite {
            background: rgba(255, 241, 242, 0.88);
            color: #7f1d1d !important;
            border: 1px solid rgba(244, 63, 94, 0.18);
        }

        .status-box-neutral {
            background: rgba(241, 245, 249, 0.88);
            color: #516476 !important;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }

        .check-box {
            border-radius: 14px;
            padding: 0.65rem 0.9rem;
            margin: 0.25rem 0 1rem 0;
            display: flex;
            align-items: flex-start;
            gap: 0.65rem;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .check-box-complete {
            background: rgba(220, 252, 231, 0.8);
            border: 1.5px solid rgba(34, 197, 94, 0.35);
            color: #1f5b33 !important;
        }

        .check-box-pending {
            background: rgba(241, 245, 249, 0.88);
            border: 1.5px solid rgba(148, 163, 184, 0.3);
            color: #516476 !important;
        }

        .check-box-rewrite {
            background: rgba(255, 241, 242, 0.88);
            border: 1.5px solid rgba(244, 63, 94, 0.25);
            color: #7f1d1d !important;
        }

        .check-box-icon {
            font-size: 1.25rem;
            line-height: 1;
            flex-shrink: 0;
            margin-top: 0.05rem;
        }

        .check-box-header {
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.15rem;
        }

        .check-box-detail {
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .question-label-prefix {
            color: #a55b1f;
            font-weight: 900;
            margin-right: 0.25em;
        }

        .checklist-grid {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
            margin: 0.6rem 0 1rem 0;
        }

        .checklist-row {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.5rem 0.75rem;
            border-radius: 12px;
            background: rgba(241, 245, 249, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.18);
        }

        .checklist-section-title {
            font-size: 0.78rem;
            font-weight: 700;
            color: #374151 !important;
            min-width: 140px;
            flex-shrink: 0;
        }

        .checklist-questions {
            display: flex;
            gap: 0.5rem;
        }

        .checklist-item {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.22rem 0.6rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            background: #f1f5f9;
            color: #94a3b8 !important;
            border: 1px solid rgba(148, 163, 184, 0.25);
        }

        .checklist-item-answered {
            background: rgba(219, 234, 254, 0.8);
            color: #1d4ed8 !important;
            border: 1px solid rgba(59, 130, 246, 0.25);
        }

        .checklist-item-valid {
            background: rgba(220, 252, 231, 0.9);
            color: #166534 !important;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }

        .placeholder-shell {
            border-radius: 28px;
            padding: 1.4rem 1.2rem;
            background: rgba(255, 251, 245, 0.92);
            border: 1px solid #eadfcf;
            box-shadow: 0 22px 48px rgba(18, 38, 58, 0.08);
            text-align: center;
        }

        .result-page-header {
            font-size: 1.7rem;
            font-weight: 900;
            color: #102437 !important;
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        }

        .result-page-sub {
            font-size: 1rem;
            color: #5a7085 !important;
            margin-bottom: 1.5rem;
        }

        .result-section-title {
            font-size: 1.1rem;
            font-weight: 900;
            color: #13263b !important;
            margin: 1.4rem 0 0.6rem 0;
            letter-spacing: -0.01em;
        }

        .result-card {
            border-radius: 22px;
            padding: 1.1rem 1.2rem 1rem 1.2rem;
            margin-bottom: 0.75rem;
            border: 1px solid #e0d5c5;
            background: rgba(255, 251, 245, 0.95);
            box-shadow: 0 8px 20px rgba(18, 38, 58, 0.06);
        }

        .result-card-title {
            font-size: 0.75rem;
            font-weight: 900;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #8a7060 !important;
            margin-bottom: 0.55rem;
        }

        .result-overall-row {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.8rem;
        }

        .result-level-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 1rem;
            border-radius: 999px;
            font-size: 1rem;
            font-weight: 900;
            letter-spacing: 0.02em;
        }

        .result-level-high {
            background: rgba(220, 252, 231, 0.9);
            color: #166534 !important;
            border: 1.5px solid rgba(34, 197, 94, 0.35);
        }

        .result-level-mid {
            background: rgba(254, 243, 199, 0.9);
            color: #92400e !important;
            border: 1.5px solid rgba(245, 158, 11, 0.35);
        }

        .result-level-low {
            background: rgba(254, 226, 226, 0.9);
            color: #991b1b !important;
            border: 1.5px solid rgba(239, 68, 68, 0.3);
        }

        .result-level-label {
            font-size: 0.88rem;
            color: #445566 !important;
        }

        .result-sub-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.1rem;
        }

        .result-sub-item {
            display: inline-flex;
            flex-direction: column;
            align-items: flex-start;
            padding: 0.4rem 0.75rem;
            border-radius: 12px;
            background: rgba(241, 245, 249, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.2);
            min-width: 110px;
        }

        .result-sub-name {
            font-size: 0.7rem;
            font-weight: 700;
            color: #64748b !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.2rem;
        }

        .result-sub-value {
            font-size: 0.85rem;
            font-weight: 800;
        }

        .result-sub-high { color: #166534 !important; }
        .result-sub-mid  { color: #92400e !important; }
        .result-sub-low  { color: #991b1b !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header():
    st.markdown(
        """
        <div class="hero-title">CEFR Interactive Diagnostic</div>
        <div class="hero-subtitle">
            Generate one CEFR-level passage that follows the participant's selected background and genre, keep the passage visible on the left,
            and let participants move part by part through adaptive follow-up questions.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="mode-notice">
            Light-view readability is enforced in this app because some system or dark-mode environments showed white text problems.
            If your browser still looks dark, refresh once in light mode for the cleanest participant view.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_participant_header():
    st.markdown(
        """
        <div class="participant-shell">
            <div class="participant-title">Participant Setup</div>
            <div class="participant-subtitle">
                이름, 학번, CEFR 레벨, 배경, 장르를 먼저 선택한 뒤 지문과 질문을 생성하세요. 생성 이후 참가자 정보나 선택값을 바꾸면 다시 생성해야 새 실험 세트가 반영됩니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_placeholder():
    st.markdown(
        """
        <div class="placeholder-shell">
            <div class="panel-title">Ready to Build the Experiment</div>
            <div class="panel-subtitle">
                Enter the participant profile, preferred background, and genre above, then click <b>Generate Passage &amp; Questions</b>.
                The app will create a CEFR-level passage that follows those choices, then run the questionnaire part by part with adaptive follow-up questions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(section: dict):
    st.markdown(
        (
            f"<div class='section-title'>{html.escape(section['title'])}</div>"
            f"<div class='section-subtitle'>{html.escape(section['title_ko'])}</div>"
        ),
        unsafe_allow_html=True,
    )


def render_question_card(question: dict, is_complete: bool = False):
    card_classes = ["question-card"]

    if question["type"] == "Self":
        card_classes.append("question-card-self")
    elif question["type"] == "Depth 2":
        card_classes.append("question-card-depth")
    else:
        card_classes.append("question-card-character")

    if not question.get("ready_for_answer", True):
        card_classes.append("question-card-pending")

    note_text = question.get("prompt_note", "").strip()
    note_class = "question-note"
    if note_text and question.get("prompt_source") == "pending":
        note_class = "question-note question-note-muted"

    note_html = ""
    if note_text:
        note_html = f"<div class='{note_class}'>{html.escape(note_text)}</div>"

    if is_complete:
        badge_html = "<span class='question-complete-badge'>&#10003; Complete</span>"
    else:
        badge_html = "<span class='question-incomplete-badge'>&#9675; Incomplete</span>"

    st.markdown(
        (
            f"<div class='{' '.join(card_classes)}'>"
            "<div class='question-top'>"
            f"<span class='question-chip question-chip-label'>{html.escape(question['label'])}</span>"
            f"<span class='question-chip question-chip-layer'>{html.escape(question['layer'])}</span>"
            f"<span class='question-chip question-chip-type'>{html.escape(question['type'])}</span>"
            f"{badge_html}"
            "</div>"
            f"<div class='question-prompt'><span class='question-label-prefix'>{html.escape(question['label'])}.</span> {html.escape(question['prompt'])}</div>"
            f"<div class='question-ko'>한국어 해석: {html.escape(question['prompt_ko'])}</div>"
            f"{note_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_passage_panel(questionnaire: dict, student_name: str, student_number: str, cefr_level: str):
    english_sentences = "".join(
        f"<p class='passage-sentence'>{html.escape(sentence)}</p>"
        for sentence in questionnaire.get("passage_sentences", [])
    )
    korean_sentences = "".join(
        f"<p class='passage-sentence'>{html.escape(sentence)}</p>"
        for sentence in questionnaire.get("passage_ko_sentences", [])
    )

    st.markdown(
        f"""
        <div class="passage-panel">
            <div class="passage-kicker">Live Passage Panel</div>
            <div class="story-title">{html.escape(questionnaire.get("story_title", ""))}</div>
            <div class="story-title-ko">{html.escape(questionnaire.get("story_title_ko", ""))}</div>
            <div class="meta-row">
                <span class="meta-chip">Name: {html.escape(student_name)}</span>
                <span class="meta-chip">Student No: {html.escape(student_number)}</span>
                <span class="meta-chip">CEFR: {html.escape(cefr_level)}</span>
                <span class="meta-chip">Background: {html.escape(questionnaire.get("selected_background", ""))}</span>
                <span class="meta-chip">Genre: {html.escape(questionnaire.get("selected_genre", ""))}</span>
            </div>
            <div class="passage-block">
                <h4>English Passage</h4>
                {english_sentences}
            </div>
            <div class="passage-block">
                <h4>Korean Translation</h4>
                {korean_sentences}
            </div>
            <div class="status-box status-box-neutral">
                Passage stays visible here by default so participants do not need to keep scrolling up while answering.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_box(style_name: str, text: str):
    st.markdown(
        f"<div class='status-box {style_name}'>{html.escape(text)}</div>",
        unsafe_allow_html=True,
    )


def render_check_status_box(state: str, label: str, detail: str = ""):
    """
    Render a per-question Check status box.
    state: 'complete' | 'rewrite' | 'pending'
    label: short header text (e.g. 'Complete', 'Rewrite needed', 'Not checked yet')
    detail: optional longer message shown below the label
    """
    if state == "complete":
        box_class = "check-box check-box-complete"
        icon = "&#10003;"
    elif state == "rewrite":
        box_class = "check-box check-box-rewrite"
        icon = "&#9888;"
    else:
        box_class = "check-box check-box-pending"
        icon = "&#9675;"

    detail_html = (
        f"<div class='check-box-detail'>{html.escape(detail)}</div>" if detail else ""
    )
    st.markdown(
        f"<div class='{box_class}'>"
        f"<span class='check-box-icon'>{icon}</span>"
        f"<div>"
        f"<div class='check-box-header'>Check &mdash; {html.escape(label)}</div>"
        f"{detail_html}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# -----------------------------
# App bootstrap
# -----------------------------
initialize_session_state()
ensure_output_directories()
google_sheets_config = get_google_sheets_config()
llm_config = get_llm_config()
inject_custom_styles()


# -----------------------------
# UI
# -----------------------------
render_page_header()

if st.session_state.submission_complete:
    ev = st.session_state.get("last_evaluation_result", {})

    def _level_badge(level: str) -> str:
        cls = {"High": "result-level-high", "Mid": "result-level-mid"}.get(level, "result-level-low")
        return f"<span class='result-level-badge {cls}'>{html.escape(level)}</span>"

    def _sub_value_class(level: str) -> str:
        return {"High": "result-sub-high", "Mid": "result-sub-mid"}.get(level, "result-sub-low")

    def _hml_score(level: str) -> int:
        return {"High": 2, "Mid": 1, "Low": 0}.get(level, 1)

    def _avg_to_label(scores: list) -> str:
        avg = sum(scores) / len(scores) if scores else 1
        if avg >= 1.4:
            return "High"
        if avg >= 0.6:
            return "Mid"
        return "Low"

    def _overall_row(label: str, ko_text: str) -> str:
        return (
            f"<div class='result-overall-row'>"
            f"{_level_badge(label)}"
            f"<span class='result-level-label'>{html.escape(ko_text)}</span>"
            f"</div>"
        )

    def _sub_items(items: list) -> str:
        parts = []
        for name, value in items:
            vc = _sub_value_class(value)
            parts.append(
                f"<div class='result-sub-item'>"
                f"<span class='result-sub-name'>{html.escape(name)}</span>"
                f"<span class='result-sub-value {vc}'>{html.escape(value)}</span>"
                f"</div>"
            )
        return f"<div class='result-sub-grid'>{''.join(parts)}</div>"

    st.success("모든 답변이 저장되었습니다! Your responses have been saved.")

    st.markdown(
        f"<div class='result-page-header'>Your Diagnostic Results</div>"
        f"<div class='result-page-sub'>"
        f"{html.escape(st.session_state.student_name_value)} &nbsp;|&nbsp; "
        f"Student No. {html.escape(st.session_state.student_number_value)} &nbsp;|&nbsp; "
        f"CEFR {html.escape(st.session_state.cefr_level_value)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Fluency ──────────────────────────────────────────────
    st.markdown("<div class='result-section-title'>Fluency / 유창성</div>", unsafe_allow_html=True)
    fluency_overall = ev.get("Fluency", "—")
    sentence_length_level = ev.get("Fluency_sentence_length", "—")
    connector_level = ev.get("Fluency_connector_use", "—")
    structure_level = ev.get("Fluency_structure_complexity", "—")
    organization_level = ev.get("Fluency_organization", "—")
    strategy_expr_level = ev.get("Fluency_strategy_expression", "—")
    grammar_level = ev.get("Fluency_grammar", "—")
    fluency_subs = [
        ("Sentence Length", sentence_length_level),
        ("Connector Use", connector_level),
        ("Structure", structure_level),
        ("Organization", organization_level),
        ("Strategy Use", strategy_expr_level),
        ("Grammar", grammar_level),
    ]
    sl_desc = {
        "High": "문장이 충분히 길고 내용이 풍부합니다.",
        "Mid": "문장 길이가 보통 수준입니다.",
        "Low": "문장이 짧고 내용이 단순한 편입니다.",
    }.get(sentence_length_level, "")
    connector_examples = ev.get("Fluency_connector_examples", "")
    conn_ex_note = f" (사용: <b>{html.escape(connector_examples)}</b>)" if connector_examples else ""
    cn_desc = {
        "High": "because, but, so 등 다양한 연결어를 잘 활용합니다.",
        "Mid": "연결어를 일부 사용하고 있습니다.",
        "Low": "연결어 사용이 거의 나타나지 않았습니다.",
    }.get(connector_level, "")
    st_desc = {
        "High": "복문 구조를 활용하여 문장을 구성합니다.",
        "Mid": "문장 구조가 어느 정도 다양합니다.",
        "Low": "단순한 문장 구조가 주로 사용되었습니다.",
    }.get(structure_level, "")
    org_desc = {
        "High": "first, then, finally 등 구성 마커를 사용하여 내용을 잘 조직합니다.",
        "Mid": "내용 구성이 어느 정도 나타납니다.",
        "Low": "내용의 조직력이 낮게 나타났습니다.",
    }.get(organization_level, "")
    se_fl_desc = {
        "High": "I think, I feel 등 자신의 생각을 표현하는 어구를 잘 활용합니다.",
        "Mid": "전략적 표현을 일부 사용합니다.",
        "Low": "자신의 생각이나 느낌을 표현하는 어구가 적게 나타났습니다.",
    }.get(strategy_expr_level, "")
    gr_desc = {
        "High": "문법 오류가 거의 발견되지 않았습니다.",
        "Mid": "일부 문법 오류가 나타났지만 전달은 가능합니다.",
        "Low": "주어-동사 일치, 조동사 사용 등 문법 오류가 자주 나타났습니다.",
    }.get(grammar_level, "")
    st.markdown(
        f"<div class='result-card'>"
        f"<div class='result-card-title'>Overall Fluency Level</div>"
        f"<div class='result-overall-row'>"
        f"{_level_badge(fluency_overall)}"
        f"<span class='result-level-label'>전반적 유창성 수준</span>"
        f"</div>"
        f"{_sub_items(fluency_subs)}"
        f"<div style='margin-top:0.65rem;font-size:0.84rem;color:#5a7085;line-height:1.6'>"
        f"<b>Sentence Length:</b> {html.escape(sl_desc)}<br>"
        f"<b>Connector Use:</b> {html.escape(cn_desc)}{conn_ex_note}<br>"
        f"<b>Structure:</b> {html.escape(st_desc)}<br>"
        f"<b>Organization:</b> {html.escape(org_desc)}<br>"
        f"<b>Strategy Use:</b> {html.escape(se_fl_desc)}<br>"
        f"<b>Grammar:</b> {html.escape(gr_desc)}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Emotion ───────────────────────────────────────────────
    st.markdown("<div class='result-section-title'>Emotion / 감정</div>", unsafe_allow_html=True)
    fla = ev.get("FLA", "—")
    fle = ev.get("FLE", "—")
    # FLA: Low anxiety = good (invert), FLE: High enjoyment = good
    fla_inv_score = {"Low": 2, "Mid": 1, "High": 0}.get(fla, 1)
    fle_score = _hml_score(fle)
    emotion_overall = _avg_to_label([fla_inv_score, fle_score])
    emotion_subs = [
        ("Anxiety (FLA)", fla),
        ("Enjoyment (FLE)", fle),
    ]
    fla_desc = {
        "High": "외국어 사용 시 불안감이 높게 나타났습니다.",
        "Mid": "외국어 사용 시 불안과 조절이 함께 나타났습니다.",
        "Low": "외국어 사용 시 불안이 낮고 안정적입니다.",
    }.get(fla, "")
    fle_desc = {
        "High": "외국어 학습에 대한 즐거움과 흥미가 높습니다.",
        "Mid": "외국어 학습에 대해 보통 수준의 흥미를 보입니다.",
        "Low": "외국어 학습에 대한 즐거움이 낮게 나타났습니다.",
    }.get(fle, "")
    st.markdown(
        f"<div class='result-card'>"
        f"<div class='result-card-title'>Foreign Language Anxiety &amp; Enjoyment</div>"
        f"{_overall_row(emotion_overall, '전반적 감정 수준')}"
        f"{_sub_items(emotion_subs)}"
        f"<div style='margin-top:0.65rem;font-size:0.84rem;color:#5a7085;line-height:1.6'>"
        f"<b>Anxiety:</b> {html.escape(fla_desc)}<br>"
        f"<b>Enjoyment:</b> {html.escape(fle_desc)}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Cognition ─────────────────────────────────────────────
    st.markdown("<div class='result-section-title'>Cognition / 인지</div>", unsafe_allow_html=True)
    self_efficacy = ev.get("Self_efficacy", "—")
    metacognition = ev.get("Metacognition", "—")
    cognition_overall = _avg_to_label([_hml_score(self_efficacy), _hml_score(metacognition)])
    cognition_subs = [
        ("Self-Efficacy", self_efficacy),
        ("Metacognition", metacognition),
    ]
    se_desc = {
        "High": "영어 사용에 대한 자신감이 높게 나타났습니다.",
        "Mid": "영어 사용에 대한 자신감이 보통 수준입니다.",
        "Low": "영어 사용에 대한 자신감이 낮게 나타났습니다.",
    }.get(self_efficacy, "")
    mc_desc = {
        "High": "자신의 학습 상태를 잘 인식하고 스스로 조절하는 능력이 높습니다.",
        "Mid": "학습에 대한 인식은 있으나 조절 행동이 부분적으로 나타납니다.",
        "Low": "자기 인식과 조절 능력이 아직 낮은 편입니다.",
    }.get(metacognition, "")
    st.markdown(
        f"<div class='result-card'>"
        f"<div class='result-card-title'>Self-Efficacy &amp; Metacognition</div>"
        f"{_overall_row(cognition_overall, '전반적 인지 수준')}"
        f"{_sub_items(cognition_subs)}"
        f"<div style='margin-top:0.65rem;font-size:0.84rem;color:#5a7085;line-height:1.6'>"
        f"<b>Self-Efficacy:</b> {html.escape(se_desc)}<br>"
        f"<b>Metacognition:</b> {html.escape(mc_desc)}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Behavior ──────────────────────────────────────────────
    st.markdown("<div class='result-section-title'>Behavior / 행동</div>", unsafe_allow_html=True)
    wtc = ev.get("WTC", "—")
    coping = ev.get("Coping", "—")
    engagement = ev.get("Engagement", "—")
    coping_score = {"Active": 2, "Mixed": 1, "Avoidant": 0}.get(coping, 1)
    behavior_overall = _avg_to_label([_hml_score(wtc), coping_score, _hml_score(engagement)])
    behavior_subs = [
        ("WTC", wtc),
        ("Coping", coping),
        ("Engagement", engagement),
    ]
    wtc_desc = {
        "High": "영어로 적극적으로 소통하려는 의지가 높습니다.",
        "Mid": "상황에 따라 소통 의지가 달라지는 편입니다.",
        "Low": "영어 소통을 회피하거나 조용히 있는 경향이 있습니다.",
    }.get(wtc, "")
    coping_desc = {
        "Active": "어려움이 생겼을 때 연습, 질문, 준비 등 능동적으로 대처합니다.",
        "Mixed": "능동적 대처와 회피 행동이 함께 나타납니다.",
        "Avoidant": "어려움이 생겼을 때 회피하거나 포기하는 경향이 있습니다.",
    }.get(coping, "")
    eng_desc = {
        "High": "수업에 적극적으로 참여하고 상호작용하는 편입니다.",
        "Mid": "수업 참여가 제한적이거나 상황에 따라 다릅니다.",
        "Low": "수업 참여도가 낮고 수동적인 경향이 있습니다.",
    }.get(engagement, "")
    st.markdown(
        f"<div class='result-card'>"
        f"<div class='result-card-title'>Willingness to Communicate &amp; Coping</div>"
        f"{_overall_row(behavior_overall, '전반적 행동 수준')}"
        f"{_sub_items(behavior_subs)}"
        f"<div style='margin-top:0.65rem;font-size:0.84rem;color:#5a7085;line-height:1.6'>"
        f"<b>WTC:</b> {html.escape(wtc_desc)}<br>"
        f"<b>Coping:</b> {html.escape(coping_desc)}<br>"
        f"<b>Engagement:</b> {html.escape(eng_desc)}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Strategy ──────────────────────────────────────────────
    st.markdown("<div class='result-section-title'>Strategy / 전략</div>", unsafe_allow_html=True)
    strategy_type = ev.get("Strategy_type", "—")
    strategy_quality = ev.get("Strategy_quality", "—")
    strategy_quality_score = {"Effective": 2, "Limited": 1, "Avoidant": 0}.get(strategy_quality, 1)
    strategy_overall = _avg_to_label([strategy_quality_score])
    strategy_subs = [
        ("Type", strategy_type),
        ("Quality", strategy_quality),
    ]
    type_desc_map = {
        "Cognitive": "암기, 반복, 노트 정리 등 인지 전략을 주로 사용합니다.",
        "Metacognitive": "계획 세우기, 자기 점검 등 메타인지 전략을 주로 사용합니다.",
        "Affective": "긍정적 사고, 긴장 완화 등 정서 조절 전략을 주로 사용합니다.",
        "Social": "친구나 선생님에게 도움을 구하는 사회적 전략을 주로 사용합니다.",
        "Compensation": "쉬운 표현, 추측, 몸짓 등 보완 전략을 주로 사용합니다.",
        "Unclear": "특정 전략 유형이 뚜렷하게 나타나지 않았습니다.",
    }
    type_desc = next(
        (desc for key, desc in type_desc_map.items() if strategy_type.startswith(key)),
        f"{strategy_type} 전략을 사용합니다." if strategy_type not in ("—", "Unclear") else "특정 전략 유형이 뚜렷하게 나타나지 않았습니다.",
    )
    quality_desc = {
        "Effective": "전략이 목표에 잘 맞고 효과적으로 활용되고 있습니다.",
        "Limited": "전략을 사용하지만 단순하거나 반복적인 경향이 있습니다.",
        "Avoidant": "어려움을 회피하는 방식으로 대처하는 경향이 있습니다.",
    }.get(strategy_quality, "")
    st.markdown(
        f"<div class='result-card'>"
        f"<div class='result-card-title'>Learning Strategy</div>"
        f"{_overall_row(strategy_overall, '전반적 전략 수준')}"
        f"{_sub_items(strategy_subs)}"
        f"<div style='margin-top:0.65rem;font-size:0.84rem;color:#5a7085;line-height:1.6'>"
        f"<b>Type:</b> {html.escape(type_desc)}<br>"
        f"<b>Quality:</b> {html.escape(quality_desc)}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Start New Participant", use_container_width=True):
        reset_session_state()
        st.rerun()

    with st.expander("Admin info"):
        st.write(f"Submission ID: `{st.session_state.last_submission_id}`")
        st.write(f"Storage backend: `{st.session_state.last_storage_backend}`")
        st.write(f"Responses destination: `{st.session_state.last_response_file}`")
        st.write(f"Evaluations destination: `{st.session_state.last_evaluation_file}`")

    st.stop()

render_participant_header()

name_col, number_col, cefr_col = st.columns([1.1, 1.1, 0.8], gap="small")
background_col, genre_col, button_col = st.columns([1.0, 1.0, 0.9], gap="small")

with name_col:
    st.text_input(
        "Name / 이름",
        key="student_name_input",
        placeholder="e.g., Minji Kim",
    )

with number_col:
    st.text_input(
        "Student Number / 학번",
        key="student_number_input",
        placeholder="e.g., 20241234",
    )

with cefr_col:
    st.selectbox(
        "CEFR Level",
        options=CEFR_LEVEL_OPTIONS,
        key="cefr_level_input",
    )

with background_col:
    st.selectbox(
        "Preferred Background / 배경",
        options=BACKGROUND_OPTIONS,
        key="selected_background_input",
    )

with genre_col:
    st.selectbox(
        "Text Genre / 장르",
        options=GENRE_OPTIONS,
        key="selected_genre_input",
    )

with button_col:
    st.write("")
    generate_clicked = st.button("Generate Passage & Questions", use_container_width=True)

participant_error = validate_participant_inputs(
    student_name=st.session_state.student_name_input,
    student_number=st.session_state.student_number_input,
    cefr_level=st.session_state.cefr_level_input,
    selected_background=st.session_state.selected_background_input,
    selected_genre=st.session_state.selected_genre_input,
)

if generate_clicked:
    st.session_state.generation_error = ""

    if participant_error:
        st.error(participant_error)
    else:
        generation_nonce = uuid4().hex

        try:
            with st.spinner(
                "Generating the CEFR passage and interactive questionnaire. "
                "The screen is being kept in a light-style view for readability."
            ):
                questionnaire = build_generated_experiment(
                    cefr_level=st.session_state.cefr_level_input,
                    student_name=st.session_state.student_name_input.strip(),
                    student_number=st.session_state.student_number_input.strip(),
                    selected_background=st.session_state.selected_background_input,
                    selected_genre=st.session_state.selected_genre_input,
                    api_key=llm_config["api_key"],
                    model=llm_config["model"],
                    nonce=generation_nonce,
                )
        except Exception as error:
            st.session_state.generation_error = (
                "Passage and question generation failed.\n\n"
                f"Error: {format_exception_message(error)}"
            )
            st.session_state.active_questionnaire = {}
            st.session_state.saved_answers = {}
            st.session_state.current_section_index = 0
            st.session_state.materialized_questionnaire = {}
            st.session_state.materialized_signature = ""
            st.session_state.feedback_map = {}
            st.session_state.feedback_signature = ""
        else:
            clear_question_widgets()
            st.session_state.saved_answers = {}
            st.session_state.student_name_value = st.session_state.student_name_input.strip()
            st.session_state.student_number_value = st.session_state.student_number_input.strip()
            st.session_state.cefr_level_value = st.session_state.cefr_level_input
            st.session_state.selected_background_value = st.session_state.selected_background_input
            st.session_state.selected_genre_value = st.session_state.selected_genre_input
            st.session_state.active_questionnaire = questionnaire
            st.session_state.current_section_index = 0
            st.session_state.generation_nonce = generation_nonce
            st.session_state.materialized_questionnaire = {}
            st.session_state.materialized_signature = ""
            st.session_state.feedback_map = {}
            st.session_state.feedback_signature = ""
            st.session_state.generation_error = ""
            st.rerun()

if st.session_state.generation_error:
    st.error(st.session_state.generation_error)

questionnaire_ready = bool(st.session_state.active_questionnaire)

if questionnaire_ready:
    profile_changed_after_generation = (
        st.session_state.student_name_input.strip() != st.session_state.student_name_value
        or st.session_state.student_number_input.strip() != st.session_state.student_number_value
        or st.session_state.cefr_level_input != st.session_state.cefr_level_value
        or st.session_state.selected_background_input != st.session_state.selected_background_value
        or st.session_state.selected_genre_input != st.session_state.selected_genre_value
    )
    if profile_changed_after_generation:
        st.warning(
            "Participant information or generation options changed after generation. Click Generate Passage & Questions again to refresh the passage and question set."
        )

if google_sheets_config["enabled"]:
    st.success(
        "Storage mode: Google Sheets\n\n"
        f"Spreadsheet ID: `{google_sheets_config['spreadsheet_id']}`\n\n"
        f"Responses tab: `{google_sheets_config['responses_worksheet']}`\n\n"
        f"Evaluations tab: `{google_sheets_config['evaluations_worksheet']}`"
    )
else:
    st.info(
        "Storage mode: Local CSV\n\n"
        f"Current local files: `{RESPONSES_FILE}` and `{EVALUATIONS_FILE}`\n\n"
        "For deployed apps, configure Google Sheets for persistent storage because local cloud files may not be kept after restart or redeploy."
    )

if llm_config["enabled"]:
    st.success(
        "LLM mode: Active\n\n"
        f"Model: `{llm_config['model']}`\n\n"
        "The app will generate the CEFR passage, interactive questions, and personalized self follow-ups with OpenAI when possible. Basic answer usability checks are handled locally."
    )
else:
    st.warning(
        "LLM mode: Inactive\n\n"
        "OpenAI API key is not configured. Passage generation and personalized follow-up generation require OpenAI. Basic answer usability checks still work locally."
    )

if not questionnaire_ready:
    render_placeholder()
    st.stop()

base_questionnaire = st.session_state.active_questionnaire
base_questionnaire_json = json.dumps(base_questionnaire, ensure_ascii=False, sort_keys=True)
current_answers_snapshot = build_answers_snapshot(base_questionnaire)
base_answers_json = serialize_answers_snapshot(base_questionnaire, current_answers_snapshot)

materialized_signature = hashlib.sha256(
    f"{base_questionnaire_json}|{base_answers_json}|{llm_config['model']}|{llm_config['api_key']}".encode("utf-8")
).hexdigest()

if st.session_state.materialized_signature == materialized_signature:
    current_questionnaire = deepcopy(st.session_state.materialized_questionnaire)
else:
    with st.spinner(
        "Analyzing answers and preparing the next interactive question. "
        "The loading indicator is centered to keep the experiment flow clear for participants."
    ):
        try:
            current_questionnaire = materialize_questionnaire_for_answers_cached(
                questionnaire_json=base_questionnaire_json,
                answers_json=base_answers_json,
                api_key=llm_config["api_key"],
                model=llm_config["model"],
            )
        except Exception as error:
            st.error(
                "OpenAI follow-up generation failed while preparing the questionnaire.\n\n"
                f"Error: {format_exception_message(error)}"
            )
            st.stop()
    st.session_state.materialized_questionnaire = current_questionnaire
    st.session_state.materialized_signature = materialized_signature
    st.session_state.feedback_map = {}
    st.session_state.feedback_signature = ""

if st.session_state.current_section_index >= len(current_questionnaire["sections"]):
    st.session_state.current_section_index = max(0, len(current_questionnaire["sections"]) - 1)

current_section_index = st.session_state.current_section_index
current_section = current_questionnaire["sections"][current_section_index]
current_answers = {
    question["id"]: current_answers_snapshot.get(question["id"], "").strip()
    for question in get_all_questions(current_questionnaire)
}
questionnaire_json = json.dumps(current_questionnaire, ensure_ascii=False, sort_keys=True)
current_answers_json = serialize_answers_snapshot(current_questionnaire, current_answers)
feedback_signature = hashlib.sha256(
    f"{questionnaire_json}|{current_answers_json}|{llm_config['model']}|{llm_config['api_key']}".encode("utf-8")
).hexdigest()

if st.session_state.feedback_signature == feedback_signature:
    feedback_map = dict(st.session_state.feedback_map)
else:
    with st.spinner(
        "Checking the current answers and updating completion status."
    ):
        try:
            feedback_map = build_answer_feedback_map(
                questionnaire_json=questionnaire_json,
                answers_json=current_answers_json,
                api_key=llm_config["api_key"],
                model=llm_config["model"],
            )
        except Exception as error:
            st.error(
                "OpenAI answer checking failed.\n\n"
                f"Error: {format_exception_message(error)}"
            )
            st.stop()
    st.session_state.feedback_map = feedback_map
    st.session_state.feedback_signature = feedback_signature

st.caption(
    f"Question status: {current_questionnaire.get('question_generation_note', '')}"
)
st.info(
    "Answer guide: Every response must be written in English. "
    "For Q1, write at least 2 sentences and separate them with periods. The app only asks for a rewrite when the answer is blank, says I don't know, contains only symbols or punctuation, or Q1 has fewer than 2 period-based sentences. "
    "In each part, Question 1 appears first, and Question 2 is generated from the answer to Question 1. "
    "After writing an answer, click the Check button below each question to confirm it and unlock the next question.\n\n"
    "답변 가이드: 모든 답변은 영어로 작성해 주세요. Q1은 마침표(.)를 기준으로 최소 2문장 이상 작성해 주세요. 답이 비어 있거나 I don't know이거나 기호만 있거나, Q1이 2문장 미만이면 다시 쓰라는 안내가 나옵니다. "
    "각 파트에서는 1번 질문이 먼저 나오고, 2번 질문은 1번 답변을 보고 생성됩니다. "
    "답변을 작성한 뒤 각 질문 아래의 Check 버튼을 눌러 완료를 확인하고 다음 질문을 열어주세요."
)

passage_col, question_col = st.columns([0.9, 1.3], gap="large")

previous_clicked = False
next_clicked = False
submit_clicked = False

with passage_col:
    render_passage_panel(
        questionnaire=current_questionnaire,
        student_name=st.session_state.student_name_value,
        student_number=st.session_state.student_number_value,
        cefr_level=st.session_state.cefr_level_value,
    )

with question_col:
    st.markdown(
        """
        <div class="question-shell">
            <div class="panel-title">Interactive Questionnaire</div>
            <div class="panel-subtitle">
                Move one part at a time. In each part, the second question opens only after the first answer is usable.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_questions = len(get_all_questions(current_questionnaire))
    section_completed_questions = 0
    section_total_questions = len(current_section["questions"])

    render_section_header(current_section)
    st.caption(f"Part {current_section_index + 1} of {len(current_questionnaire['sections'])}")

    for question in current_section["questions"]:
        question_key = widget_key(question["id"])
        if not question.get("ready_for_answer", True):
            st.session_state[question_key] = ""
            current_answers[question["id"]] = ""
        else:
            effective_answer = current_answers.get(question["id"], "")
            if question_key not in st.session_state or (
                not str(st.session_state.get(question_key, "")).strip() and effective_answer
            ):
                st.session_state[question_key] = effective_answer
        q_feedback = feedback_map.get(question["id"])
        q_is_complete = bool(q_feedback and q_feedback.get("status") == "valid")
        render_question_card(question, is_complete=q_is_complete)

        answer = st.text_area(
            f"{question['label']}. {question['prompt']}",
            key=question_key,
            label_visibility="collapsed",
            height=170,
            disabled=not question.get("ready_for_answer", True),
            placeholder=(
                "Write a simple English answer. Short answers are okay."
                if question.get("ready_for_answer", True)
                else "The follow-up question will open after the first answer in this part is usable."
            ),
        )
        answer = answer.strip()
        current_answers[question["id"]] = answer

        if not question.get("ready_for_answer", True):
            render_check_status_box(
                "pending",
                "Waiting",
                "This follow-up will open after the previous answer can be used to create the next question.",
            )
            continue

        check_q_clicked = st.button(
            f"Check {question['label']}",
            key=f"check_btn_{question['id']}_{st.session_state.generation_nonce}_{current_section_index}",
            use_container_width=True,
            disabled=not answer,
        )

        if check_q_clicked:
            for q in current_section["questions"]:
                qk = widget_key(q["id"])
                if q.get("ready_for_answer", True):
                    set_saved_answer(q["id"], str(st.session_state.get(qk, "")).strip())
                else:
                    set_saved_answer(q["id"], "")
            st.session_state.materialized_signature = ""
            st.session_state.feedback_signature = ""
            st.rerun()

        if not answer:
            render_check_status_box(
                "pending",
                "Not answered yet",
                "Write an answer above and click the Check button to confirm completion.",
            )
            continue

        feedback = feedback_map.get(question["id"])
        if feedback and feedback["status"] == "valid":
            section_completed_questions += 1
            render_check_status_box(
                "complete",
                "Complete",
                f"{feedback['message']} / {feedback['message_ko']}",
            )
        elif feedback:
            render_check_status_box(
                "rewrite",
                "Rewrite needed",
                f"{feedback['message']} / {feedback['message_ko']}",
            )
        else:
            render_check_status_box(
                "pending",
                "Ready to check",
                "Click the Check button above to verify this answer.",
            )

    completed_questions = 0
    for question in get_all_questions(current_questionnaire):
        answer = current_answers.get(question["id"], "").strip()
        if not question.get("ready_for_answer", True) or not answer:
            continue

        feedback = feedback_map.get(question["id"])
        if feedback and feedback["status"] == "valid":
            completed_questions += 1

    section_complete = section_completed_questions == section_total_questions

    st.markdown("### Completion")
    checklist_rows = []
    for cl_section in current_questionnaire["sections"]:
        badges = []
        for cl_q in cl_section["questions"]:
            qid = cl_q["id"]
            label = cl_q.get("label", qid)
            has_answer = bool(current_answers.get(qid, "").strip())
            cl_fb = feedback_map.get(qid)
            is_valid = bool(cl_fb and cl_fb.get("status") == "valid")

            if is_valid:
                item_class = "checklist-item checklist-item-valid"
                icon = "&#10003;"
            elif has_answer:
                item_class = "checklist-item checklist-item-answered"
                icon = "&#10003;"
            else:
                item_class = "checklist-item"
                icon = "&#9675;"

            badges.append(
                f"<span class='{item_class}'>{icon} {html.escape(label)}</span>"
            )

        checklist_rows.append(
            f"<div class='checklist-row'>"
            f"<span class='checklist-section-title'>{html.escape(cl_section.get('title', ''))}</span>"
            f"<span class='checklist-questions'>{''.join(badges)}</span>"
            f"</div>"
        )

    st.markdown(
        f"<div class='checklist-grid'>{''.join(checklist_rows)}</div>",
        unsafe_allow_html=True,
    )

    if not section_complete:
        st.caption("Finish this part first. Then the Next button will be available.")

    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        previous_clicked = st.button(
            "Previous Part",
            key=f"prev_btn_{st.session_state.generation_nonce}_{current_section_index}",
            use_container_width=True,
            disabled=(current_section_index == 0),
        )
    with nav_col2:
        if current_section_index < len(current_questionnaire["sections"]) - 1:
            next_clicked = st.button(
                "Next Part",
                key=f"next_btn_{st.session_state.generation_nonce}_{current_section_index}",
                use_container_width=True,
                disabled=(not section_complete or profile_changed_after_generation),
            )
        else:
            submit_clicked = st.button(
                "Submit Responses",
                key=f"submit_btn_{st.session_state.generation_nonce}_{current_section_index}",
                use_container_width=True,
                disabled=(not section_complete or profile_changed_after_generation),
            )

def _save_section_answers():
    for question in current_section["questions"]:
        question_key = widget_key(question["id"])
        if not question.get("ready_for_answer", True):
            st.session_state[question_key] = ""
            current_answers[question["id"]] = ""
            set_saved_answer(question["id"], "")
        else:
            normalized_answer = str(st.session_state.get(question_key, "")).strip()
            current_answers[question["id"]] = normalized_answer
            set_saved_answer(question["id"], normalized_answer)


if previous_clicked:
    _save_section_answers()
    st.session_state.current_section_index = max(0, current_section_index - 1)
    st.rerun()

if next_clicked:
    _save_section_answers()
    st.session_state.current_section_index = min(
        len(current_questionnaire["sections"]) - 1,
        current_section_index + 1,
    )
    st.rerun()


if submit_clicked:
    _save_section_answers()
    if (
        st.session_state.student_name_input.strip() != st.session_state.student_name_value
        or st.session_state.student_number_input.strip() != st.session_state.student_number_value
        or st.session_state.cefr_level_input != st.session_state.cefr_level_value
        or st.session_state.selected_background_input != st.session_state.selected_background_value
        or st.session_state.selected_genre_input != st.session_state.selected_genre_value
    ):
        st.error("Participant information or generation options changed after generation. Please generate the passage and questions again before submitting.")
    else:
        try:
            validation_error = validate_submission_answers(
                questionnaire=current_questionnaire,
                answers=current_answers,
                llm_config=llm_config,
                feedback_map=feedback_map,
            )
        except Exception as error:
            st.error(
                "OpenAI answer validation failed during submission.\n\n"
                f"Error: {format_exception_message(error)}"
            )
            validation_error = "__OPENAI_VALIDATION_ERROR__"

        if validation_error == "__OPENAI_VALIDATION_ERROR__":
            pass
        elif validation_error:
            st.error(validation_error)
        else:
            submission_id = create_submission_id(
                student_name=st.session_state.student_name_value,
                student_number=st.session_state.student_number_value,
            )

            response_row = build_response_row(
                submission_id=submission_id,
                student_name=st.session_state.student_name_value,
                student_number=st.session_state.student_number_value,
                cefr_level=st.session_state.cefr_level_value,
                questionnaire=current_questionnaire,
                answers=current_answers,
            )
            evaluation_row = build_evaluation_row(
                submission_id=submission_id,
                student_name=st.session_state.student_name_value,
                student_number=st.session_state.student_number_value,
                cefr_level=st.session_state.cefr_level_value,
                questionnaire=current_questionnaire,
                answers=current_answers,
                api_key=llm_config["api_key"],
                model=llm_config["model"],
            )

            try:
                response_save_result = save_rows(
                    rows=[response_row],
                    local_file_path=RESPONSES_FILE,
                    google_worksheet_name=google_sheets_config.get("responses_worksheet", DEFAULT_RESPONSES_WORKSHEET),
                )
                evaluation_save_result = save_rows(
                    rows=[evaluation_row],
                    local_file_path=EVALUATIONS_FILE,
                    google_worksheet_name=google_sheets_config.get("evaluations_worksheet", DEFAULT_EVALUATIONS_WORKSHEET),
                )
            except Exception as error:
                st.error(
                    "Saving failed. Please check the Google Sheets configuration in Streamlit secrets.\n\n"
                    f"Error: {error}"
                )
            else:
                st.session_state.last_submission_id = submission_id
                st.session_state.last_storage_backend = response_save_result["backend"]
                st.session_state.last_evaluation_result = evaluate_state(
                    current_answers,
                    api_key=llm_config["api_key"],
                    model=llm_config["model"],
                )

                if response_save_result["backend"] == "google_sheets":
                    st.session_state.last_response_file = (
                        f"{response_save_result['spreadsheet_url']} / "
                        f"{response_save_result['worksheet_name']}"
                    )
                    st.session_state.last_evaluation_file = (
                        f"{evaluation_save_result['spreadsheet_url']} / "
                        f"{evaluation_save_result['worksheet_name']}"
                    )
                else:
                    st.session_state.last_response_file = response_save_result["file_path"]
                    st.session_state.last_evaluation_file = evaluation_save_result["file_path"]

                st.session_state.submission_complete = True
                st.rerun()

st.markdown("---")
st.caption("Prototype for CEFR-based passage generation, interactive questioning, and state evaluation")
