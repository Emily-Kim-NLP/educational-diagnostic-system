import html
import json
import os
import re
from copy import deepcopy
from datetime import datetime
from uuid import uuid4

import gspread
import pandas as pd
import streamlit as st
from openai import OpenAI


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Educational Diagnostic System",
    page_icon="📘",
    layout="centered",
)


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
DEFAULT_QUESTION_MODEL = "gpt-5-mini"

MIN_SENTENCES = 2
RECOMMENDED_SENTENCES = 3

CEFR_STORY_TEXT = """
**Frank's Last Case**

Frank was a police officer near retirement.
He felt unhappy because he did not have enough money for his future.
One day, he heard about a plan to steal a famous diamond.
Instead of stopping the criminals, Frank made a different plan.
He decided to let them steal the diamond and catch them later.
When the police stopped the criminals, Frank secretly changed the real diamond with a fake one.
In the end, the criminals were arrested, but Frank kept the real diamond for himself.
"""

CEFR_STORY_TEXT_KO = """
**Frank의 마지막 사건**

Frank는 은퇴를 앞둔 경찰관이었습니다.
그는 미래를 위한 돈이 충분하지 않아서 불행했습니다.
어느 날 그는 유명한 다이아몬드를 훔치려는 계획에 대해 듣게 되었습니다.
범죄자들을 막는 대신, Frank는 다른 계획을 세웠습니다.
그는 그들이 다이아몬드를 훔치도록 내버려 둔 뒤 나중에 그들을 잡기로 했습니다.
경찰이 범죄자들을 막았을 때, Frank는 진짜 다이아몬드를 가짜와 몰래 바꾸었습니다.
결국 범죄자들은 체포되었지만, Frank는 진짜 다이아몬드를 자신이 차지했습니다.
"""

LITERATURE_TEXT = """
**Literary Situation (Jane Austen - Pride and Prejudice)**

Elizabeth met Mr. Darcy at a dance.
She wanted to talk with him, but Darcy did not show interest.
He said she was not very interesting.
Elizabeth heard this and felt hurt and a little angry.
After that, she decided not to talk to him again.
She thought he was rude and not kind.
"""

LITERATURE_TEXT_KO = """
**문학적 상황 (제인 오스틴 - 오만과 편견)**

Elizabeth는 무도회에서 Mr. Darcy를 만났습니다.
Elizabeth는 그와 이야기하고 싶었지만, Darcy는 관심을 보이지 않았습니다.
Darcy는 Elizabeth가 별로 흥미롭지 않다고 말했습니다.
Elizabeth는 그 말을 듣고 상처를 받고 조금 화가 났습니다.
그 후 Elizabeth는 다시는 그와 이야기하지 않기로 했습니다.
Elizabeth는 Darcy가 무례하고 친절하지 않다고 생각했습니다.
"""

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
                "question_goal": "Ask the student to explain the key event, conflict, or situation from the passage in their own words.",
            }
        ],
    },
    {
        "title": "Part 2. Emotion",
        "title_ko": "2부. 감정",
        "questions": [
            {
                "id": "q2",
                "label": "Q2",
                "layer": "Emotion",
                "type": "Character",
                "question_goal": "Ask how the main character feels in the passage and why.",
            },
            {
                "id": "q3",
                "label": "Q3",
                "layer": "Emotion",
                "type": "Self",
                "depends_on": "q2",
                "question_goal": "Ask how the student would feel in a similar English-class situation.",
                "criteria_focus": "Target FLA and FLE without naming the labels. Encourage feelings such as nervous, worried, afraid, comfortable, relaxed, interested, happy, or enjoying the class.",
            },
        ],
    },
    {
        "title": "Part 3. Cognition",
        "title_ko": "3부. 생각",
        "questions": [
            {
                "id": "q4",
                "label": "Q4",
                "layer": "Cognition",
                "type": "Character",
                "question_goal": "Ask what the main character might think in that moment.",
            },
            {
                "id": "q5",
                "label": "Q5",
                "layer": "Cognition",
                "type": "Self",
                "depends_on": "q4",
                "question_goal": "Ask what the student would think in a similar English-class situation.",
                "criteria_focus": "Target self-efficacy and metacognition without naming the labels. Encourage reflection on ability, confidence, noticing problems, and managing or improving learning.",
            },
        ],
    },
    {
        "title": "Part 4. Behavior",
        "title_ko": "4부. 행동",
        "questions": [
            {
                "id": "q6",
                "label": "Q6",
                "layer": "Behavior",
                "type": "Character",
                "question_goal": "Ask what the main character could do in that situation.",
            },
            {
                "id": "q7",
                "label": "Q7",
                "layer": "Behavior",
                "type": "Self",
                "depends_on": "q6",
                "question_goal": "Ask what the student would do in a similar English-class situation.",
                "criteria_focus": "Target WTC, coping, and engagement without naming the labels. Encourage reflection on speaking, staying quiet, asking for help, preparing, practicing, avoiding, or participating.",
            },
        ],
    },
    {
        "title": "Part 5. Strategy",
        "title_ko": "5부. 전략",
        "questions": [
            {
                "id": "q8",
                "label": "Q8",
                "layer": "Strategy",
                "type": "Character",
                "question_goal": "Ask what the best way is for the main character to handle the situation and why.",
            },
            {
                "id": "q9",
                "label": "Q9",
                "layer": "Strategy",
                "type": "Self",
                "depends_on": "q8",
                "question_goal": "Ask what helps the student most in a similar English-class situation.",
                "criteria_focus": "Target Oxford-based strategy type and quality without naming the labels. Encourage strategy examples such as practice, planning, calming down, asking others, guessing, or using simple words.",
            },
        ],
    },
]

QUESTION_GENERATION_CRITERIA = """
Layer and evaluation targets:

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
    "q3": {
        "state_targets": "FLA / FLE",
        "criteria_focus": (
            "Generate a self question for English class that connects the student's character-emotion answer "
            "to their own emotional state. It should help reveal anxiety versus enjoyment without naming the labels."
        ),
        "fallback_en": (
            "Based on your answer to {source_label} about {character_focus}, when you feel something similar "
            "in English class, how do you feel? Explain whether you feel nervous, comfortable, interested, "
            "happy, or something else, and why."
        ),
        "fallback_ko": (
            "{character_focus}에 대한 {source_label} 답변을 바탕으로, 영어 수업에서 비슷한 느낌이 들 때 "
            "당신은 어떤 기분이 드나요? 긴장되는지, 편안한지, 흥미로운지, 행복한지, 또는 다른 감정이 "
            "드는지와 그 이유를 설명해 보세요."
        ),
        "note": "Personalized from your character response to check emotion in English class.",
    },
    "q5": {
        "state_targets": "Self-efficacy / Metacognition",
        "criteria_focus": (
            "Generate a self question for English class that connects the student's character-thinking answer "
            "to their own confidence, self-awareness, noticing problems, and regulation."
        ),
        "fallback_en": (
            "Based on your answer to {source_label} about {character_focus}, in a similar situation in English class, "
            "what would you think about your English ability? How would you notice your problem and try to manage it?"
        ),
        "fallback_ko": (
            "{character_focus}에 대한 {source_label} 답변을 바탕으로, 영어 수업에서 비슷한 상황이라면 "
            "자신의 영어 능력에 대해 어떤 생각이 들까요? 자신의 문제를 어떻게 알아차리고 어떻게 "
            "관리하려고 할지도 설명해 보세요."
        ),
        "note": "Personalized from your character response to check thinking in English class.",
    },
    "q7": {
        "state_targets": "WTC / Coping / Engagement",
        "criteria_focus": (
            "Generate a self question for English class that connects the student's character-behavior answer "
            "to their own willingness to communicate, coping under difficulty, and participation."
        ),
        "fallback_en": (
            "Based on your answer to {source_label} about {character_focus}, in a similar situation in English class, "
            "what would you do? Would you speak, stay quiet, ask for help, prepare, practice more, or avoid it? Explain."
        ),
        "fallback_ko": (
            "{character_focus}에 대한 {source_label} 답변을 바탕으로, 영어 수업에서 비슷한 상황이라면 "
            "어떻게 행동할까요? 말하려고 하는지, 조용히 있는지, 도움을 요청하는지, 준비하거나 더 연습하는지, "
            "또는 피하려고 하는지 설명해 보세요."
        ),
        "note": "Personalized from your character response to check behavior in English class.",
    },
    "q9": {
        "state_targets": "Strategy Type / Strategy Quality",
        "criteria_focus": (
            "Generate a self question for English class that connects the student's character-strategy answer "
            "to their own learning strategies. It should help reveal Oxford-based strategy type and quality "
            "without naming the labels."
        ),
        "fallback_en": (
            "Based on your answer to {source_label} about {character_focus}, what strategy would help you most "
            "in a similar English-class situation? Explain what you would do and how it helps you."
        ),
        "fallback_ko": (
            "{character_focus}에 대한 {source_label} 답변을 바탕으로, 영어 수업에서 비슷한 상황이라면 "
            "어떤 전략이 가장 도움이 될까요? 무엇을 할 것인지와 그것이 어떻게 도움이 되는지 설명해 보세요."
        ),
        "note": "Personalized from your character response to check strategy in English class.",
    },
}

QUESTIONNAIRE_DEFINITIONS = [
    {
        "id": "cefr_story",
        "page_title": "CEFR Story",
        "page_title_ko": "CEFR 스토리",
        "story_title": "Frank's last case",
        "character_focus": "Frank",
        "relationship_focus": "Frank and the criminals",
        "relationship_focus_ko": "Frank와 범죄자들",
        "intro": (
            "Complete the CEFR story questionnaire first. After you finish all 9 questions, "
            "click Next to move to the literature questionnaire."
        ),
        "intro_ko": (
            "먼저 CEFR 스토리 질문지를 완료하세요. 9개 문항을 모두 작성한 뒤 "
            "Next 버튼을 눌러 literature 질문지로 이동하세요."
        ),
        "text": CEFR_STORY_TEXT,
        "text_ko": CEFR_STORY_TEXT_KO,
    },
    {
        "id": "literature",
        "page_title": "Literature",
        "page_title_ko": "문학",
        "story_title": "Jane Austen - Pride and Prejudice",
        "character_focus": "Elizabeth",
        "relationship_focus": "Elizabeth and Mr. Darcy",
        "relationship_focus_ko": "Elizabeth와 Mr. Darcy",
        "intro": (
            "This is the second questionnaire. Read the literature passage and answer all 9 questions "
            "before submitting the full response."
        ),
        "intro_ko": (
            "두 번째 질문지입니다. 문학 지문을 읽고 9개 문항에 모두 답한 뒤 "
            "전체 응답을 제출하세요."
        ),
        "text": LITERATURE_TEXT,
        "text_ko": LITERATURE_TEXT_KO,
    },
]


# -----------------------------
# Helper functions
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

    return normalized


def normalize_service_account_info(service_account_info: dict) -> dict:
    normalized = dict(service_account_info)

    if "private_key" in normalized:
        normalized["private_key"] = normalize_private_key(str(normalized["private_key"]))

    return normalized


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

    return {
        "enabled": True,
        "service_account": normalize_service_account_info(google_service_account),
        "spreadsheet_id": spreadsheet_id,
        "responses_worksheet": sheet_settings.get("responses_worksheet", DEFAULT_RESPONSES_WORKSHEET),
        "evaluations_worksheet": sheet_settings.get("evaluations_worksheet", DEFAULT_EVALUATIONS_WORKSHEET),
    }


def get_question_generation_config() -> dict:
    api_key = ""
    model = DEFAULT_QUESTION_MODEL

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
        "model": model or DEFAULT_QUESTION_MODEL,
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


def get_question_slots() -> list:
    return [
        question
        for section in QUESTION_SECTION_BLUEPRINTS
        for question in section["questions"]
    ]


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


def get_question_lookup(questionnaire: dict) -> dict:
    return {question["id"]: question for question in get_all_questions(questionnaire)}


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


def build_pending_self_prompt(questionnaire: dict, question: dict, source_question: dict) -> dict:
    return {
        "prompt": (
            f"Finish {source_question['label']} first. Your personalized self question for English class "
            "will appear here after you write at least 2 sentences."
        ),
        "prompt_ko": (
            f"먼저 {source_question['label']}에 2문장 이상 답하면, 영어 수업과 연결된 맞춤형 self 질문이 "
            "여기에 나타납니다."
        ),
        "prompt_source": "pending",
        "prompt_note": (
            f"Answer {source_question['label']} with at least {MIN_SENTENCES} sentences to unlock a linked follow-up question."
        ),
        "ready_for_answer": False,
    }


def build_rule_based_personalized_self_prompt(questionnaire: dict, question: dict, source_question: dict) -> dict:
    spec = get_personalized_self_spec(question["id"])

    return {
        "prompt": spec["fallback_en"].format(
            source_label=source_question["label"],
            character_focus=questionnaire.get("character_focus", "the character"),
        ),
        "prompt_ko": spec["fallback_ko"].format(
            source_label=source_question["label"],
            character_focus=questionnaire.get("character_focus", "the character"),
        ),
        "prompt_source": "rule_based_personalized",
        "prompt_note": spec["note"],
        "ready_for_answer": True,
    }


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

    system_prompt = """
You generate one personalized self-reflection question for an English education diagnostic app.
Return valid JSON only.
The question must be clearly connected to the student's answer about the character, but it must move into the student's own experience in English class.
Use simple learner-friendly English.
Do not mention technical labels such as FLA, FLE, self-efficacy, metacognition, WTC, coping, engagement, or Oxford strategy type.
Make the Korean translation natural and faithful.
"""

    user_prompt = f"""
Passage title: {questionnaire['story_title']}
Passage:
{questionnaire['text']}

Current layer: {question['layer']}
State targets: {spec.get('state_targets', '')}
Instruction:
{spec.get('criteria_focus', '')}

Character question:
{source_question['prompt']}

Student's answer to the character question:
{source_answer}

Create one self question for English class that:
1. Feels meaningfully linked to the student's answer above.
2. Checks the student's state in English learning.
3. Sounds natural for a student questionnaire.
4. Uses 1 or 2 short sentences.
5. Does not mention the passage characters by name unless that linkage is very short and natural.

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
    api_key: str,
    model: str,
) -> dict:
    questionnaire = json.loads(questionnaire_json)
    question = json.loads(question_json)
    source_question = json.loads(source_question_json)

    if sentence_count(source_answer) < MIN_SENTENCES:
        return build_pending_self_prompt(questionnaire, question, source_question)

    fallback = build_rule_based_personalized_self_prompt(questionnaire, question, source_question)

    if not api_key:
        return fallback

    try:
        return generate_personalized_self_prompt_with_openai(
            questionnaire=questionnaire,
            question=question,
            source_question=source_question,
            source_answer=source_answer,
            api_key=api_key,
            model=model,
        )
    except Exception:
        return fallback


def build_rule_based_prompt_map(questionnaire: dict) -> dict:
    character_focus = questionnaire.get("character_focus", "the main character")
    relationship_focus = questionnaire.get("relationship_focus", character_focus)
    relationship_focus_ko = questionnaire.get("relationship_focus_ko", relationship_focus)

    return {
        "q1": {
            "prompt": f"What happens between {relationship_focus} in this passage? Explain in your own words.",
            "prompt_ko": f"이 지문에서 {relationship_focus_ko} 사이에 어떤 일이 일어나나요? 자신의 말로 설명해 보세요.",
        },
        "q2": {
            "prompt": f"How does {character_focus} feel in this situation? Why?",
            "prompt_ko": f"이 상황에서 {character_focus}는 어떻게 느끼나요? 왜 그렇게 느끼나요?",
        },
        "q3": {
            "prompt": (
                "If you were in a similar situation in English class, how would you feel? "
                "Explain whether you would feel nervous, comfortable, interested, or something else, and why."
            ),
            "prompt_ko": (
                "영어 수업에서 비슷한 상황에 있다면 어떤 기분이 들까요? "
                "긴장되는지, 편안한지, 흥미로운지, 또는 다른 감정이 드는지와 그 이유를 설명해 보세요."
            ),
        },
        "q4": {
            "prompt": f"What might {character_focus} think at this moment?",
            "prompt_ko": f"이 순간 {character_focus}는 무슨 생각을 할까요?",
        },
        "q5": {
            "prompt": (
                "In a similar situation in English class, what would you think about your English ability? "
                "How would you notice your problem and try to manage it?"
            ),
            "prompt_ko": (
                "영어 수업에서 비슷한 상황에 있다면 자신의 영어 능력에 대해 어떤 생각이 들까요? "
                "자신의 어려움을 어떻게 알아차리고 어떻게 관리하려고 할지도 함께 설명해 보세요."
            ),
        },
        "q6": {
            "prompt": f"What could {character_focus} do in this situation?",
            "prompt_ko": f"이 상황에서 {character_focus}는 무엇을 할 수 있을까요?",
        },
        "q7": {
            "prompt": (
                "In a similar situation in English class, what would you do? "
                "Would you speak, stay quiet, ask for help, practice more, or avoid the situation? Explain."
            ),
            "prompt_ko": (
                "영어 수업에서 비슷한 상황에 있다면 어떻게 행동할까요? "
                "말하려고 하는지, 조용히 있는지, 도움을 요청하는지, 더 연습하는지, 또는 피하려고 하는지 설명해 보세요."
            ),
        },
        "q8": {
            "prompt": f"What is the best way for {character_focus} to handle this situation? Why?",
            "prompt_ko": f"{character_focus}가 이 상황을 가장 잘 다루는 방법은 무엇일까요? 왜 그렇게 생각하나요?",
        },
        "q9": {
            "prompt": (
                "What strategy would help you the most in a similar situation in English class? "
                "Explain what you would do and how it helps you."
            ),
            "prompt_ko": (
                "영어 수업에서 비슷한 상황이 있을 때 어떤 전략이 가장 도움이 될까요? "
                "무엇을 할 것인지와 그것이 어떻게 도움이 되는지 설명해 보세요."
            ),
        },
    }


def apply_prompt_map_to_sections(prompt_map: dict) -> list:
    sections = deepcopy(QUESTION_SECTION_BLUEPRINTS)
    for section in sections:
        for question in section["questions"]:
            generated = prompt_map[question["id"]]
            question["prompt"] = generated["prompt"]
            question["prompt_ko"] = generated["prompt_ko"]
    return sections


def generate_prompt_map_with_openai(questionnaire: dict, api_key: str, model: str) -> dict:
    client = get_openai_client(api_key)
    question_slot_instructions = build_question_slot_instructions()

    system_prompt = """
You design bilingual questionnaire prompts for an English-learning diagnostic app.
Return valid JSON only.
Create exactly 9 prompts with ids q1 to q9.
Use short, clear English suitable for learners.
Make each Korean translation natural and faithful.
Keep self questions explicitly connected to English class.
Do not show technical labels such as FLA, FLE, self-efficacy, metacognition, WTC, coping, engagement, or Oxford strategy type in the student-facing questions.
Make the questions passage-specific and distinct.
"""

    user_prompt = f"""
Passage title: {questionnaire['story_title']}
Main character focus: {questionnaire.get('character_focus', '')}
Relationship focus: {questionnaire.get('relationship_focus', '')}

Passage in English:
{questionnaire['text']}

Passage in Korean:
{questionnaire['text_ko']}

Evaluation criteria:
{QUESTION_GENERATION_CRITERIA}

Question slots:
{question_slot_instructions}

Return JSON only in this format:
{{
  "questions": [
    {{"id": "q1", "prompt": "....", "prompt_ko": "...."}},
    {{"id": "q2", "prompt": "....", "prompt_ko": "...."}},
    {{"id": "q3", "prompt": "....", "prompt_ko": "...."}},
    {{"id": "q4", "prompt": "....", "prompt_ko": "...."}},
    {{"id": "q5", "prompt": "....", "prompt_ko": "...."}},
    {{"id": "q6", "prompt": "....", "prompt_ko": "...."}},
    {{"id": "q7", "prompt": "....", "prompt_ko": "...."}},
    {{"id": "q8", "prompt": "....", "prompt_ko": "...."}},
    {{"id": "q9", "prompt": "....", "prompt_ko": "...."}}
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
    return normalize_prompt_map(payload)


@st.cache_data(show_spinner=False, ttl=86400)
def build_generated_questionnaire(questionnaire_json: str, api_key: str, model: str) -> dict:
    questionnaire = json.loads(questionnaire_json)
    fallback_prompt_map = build_rule_based_prompt_map(questionnaire)
    fallback_sections = apply_prompt_map_to_sections(fallback_prompt_map)

    if not api_key:
        return {
            "sections": fallback_sections,
            "question_source": "rule_based",
            "question_model": "",
            "question_generation_note": "OpenAI API key is not configured.",
        }

    try:
        prompt_map = generate_prompt_map_with_openai(
            questionnaire=questionnaire,
            api_key=api_key,
            model=model,
        )
        return {
            "sections": apply_prompt_map_to_sections(prompt_map),
            "question_source": "openai",
            "question_model": model,
            "question_generation_note": "Questions were generated from the passage with OpenAI.",
        }
    except Exception:
        return {
            "sections": fallback_sections,
            "question_source": "rule_based",
            "question_model": model,
            "question_generation_note": "OpenAI question generation failed, so the app is using the built-in fallback questions.",
        }


def build_questionnaires(question_generation_config: dict) -> list:
    questionnaires = []

    for questionnaire_definition in QUESTIONNAIRE_DEFINITIONS:
        generated = build_generated_questionnaire(
            questionnaire_json=json.dumps(questionnaire_definition, ensure_ascii=False, sort_keys=True),
            api_key=question_generation_config["api_key"],
            model=question_generation_config["model"],
        )
        questionnaire = dict(questionnaire_definition)
        questionnaire["sections"] = generated["sections"]
        questionnaire["question_source"] = generated["question_source"]
        questionnaire["question_model"] = generated["question_model"]
        questionnaire["question_generation_note"] = generated["question_generation_note"]
        questionnaires.append(questionnaire)

    return questionnaires


def materialize_questionnaire_for_answers(
    questionnaire: dict,
    answers: dict,
    question_generation_config: dict,
) -> dict:
    materialized = deepcopy(questionnaire)
    question_lookup = get_question_lookup(materialized)
    default_prompt_source = materialized.get("question_source", "rule_based")

    for question in get_all_questions(materialized):
        question["prompt_source"] = default_prompt_source
        question["prompt_note"] = ""
        question["ready_for_answer"] = True

        if question["type"] != "Self" or not question.get("depends_on"):
            continue

        source_question = question_lookup[question["depends_on"]]
        source_answer = answers.get(source_question["id"], "").strip()
        personalized = build_personalized_self_prompt(
            questionnaire_json=json.dumps(questionnaire, ensure_ascii=False, sort_keys=True),
            question_json=json.dumps(question, ensure_ascii=False, sort_keys=True),
            source_question_json=json.dumps(source_question, ensure_ascii=False, sort_keys=True),
            source_answer=source_answer,
            api_key=question_generation_config["api_key"],
            model=question_generation_config["model"],
        )
        question.update(personalized)

    return materialized


def get_all_questions(questionnaire: dict) -> list:
    return [
        question
        for section in questionnaire["sections"]
        for question in section["questions"]
    ]


def widget_key(questionnaire_id: str, question_id: str) -> str:
    return f"{questionnaire_id}_{question_id}"


def completion_key(questionnaire_id: str, question_id: str) -> str:
    return f"{widget_key(questionnaire_id, question_id)}_complete"


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.strip().split())


def sentence_count(text: str) -> int:
    if not text:
        return 0
    sentences = re.split(r"\.+", text.strip())
    sentences = [sentence for sentence in sentences if sentence.strip()]
    return len(sentences)


def sanitize_filename_component(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "-", name.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "student"


def get_student_file_paths(student_name: str) -> tuple:
    filename = f"{sanitize_filename_component(student_name)}.csv"
    response_file = os.path.join(BY_STUDENT_RESPONSES_DIR, filename)
    evaluation_file = os.path.join(BY_STUDENT_EVALUATIONS_DIR, filename)
    return response_file, evaluation_file


def update_question_completion(questionnaire_id: str, question_id: str):
    key = widget_key(questionnaire_id, question_id)
    answer = st.session_state.get(key, "")
    st.session_state[completion_key(questionnaire_id, question_id)] = sentence_count(answer) >= MIN_SENTENCES


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


def fluency_label(text: str) -> str:
    wc = word_count(text)
    sc = sentence_count(text)
    connectors = detect_connectors(text)

    if wc >= 18 and sc >= 2 and len(connectors) >= 2:
        return "High"
    if wc >= 8 and sc >= 2 and len(connectors) >= 1:
        return "Mid"
    return "Low"


def contains_any(text: str, keywords: list) -> bool:
    return any(keyword in text for keyword in keywords)


def count_matches(text: str, keywords: list) -> int:
    return sum(1 for keyword in keywords if keyword in text)


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


def evaluate_state(answers: dict) -> dict:
    fla, fle = classify_fla_fle(answers.get("q3", ""))
    self_efficacy, metacognition = classify_self_efficacy_metacognition(answers.get("q5", ""))
    wtc, coping, engagement = classify_behavior(answers.get("q7", ""))
    strategy_type, strategy_quality = classify_strategy(answers.get("q9", ""))

    return {
        "evaluation_method": "criteria_rules_v1",
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


def append_rows(file_path: str, rows: list):
    ensure_output_directories()
    df_new = pd.DataFrame(rows)

    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(file_path, index=False, encoding="utf-8-sig")


def create_submission_id(student_name: str, participant_id: str) -> str:
    source_value = participant_id.strip() or student_name.strip()
    safe_value = re.sub(r"[^A-Za-z0-9_-]+", "-", source_value)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_value}_{timestamp}_{uuid4().hex[:6]}"


def build_response_row(
    submission_id: str,
    student_name: str,
    participant_id: str,
    questionnaire: dict,
    step_order: int,
    answers: dict,
) -> dict:
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "submission_id": submission_id,
        "student_name": student_name,
        "participant_id": participant_id,
        "questionnaire_id": questionnaire["id"],
        "questionnaire_title": questionnaire["page_title"],
        "questionnaire_title_ko": questionnaire["page_title_ko"],
        "story_title": questionnaire["story_title"],
        "sequence_order": step_order,
        "question_source": questionnaire.get("question_source", "rule_based"),
        "question_model": questionnaire.get("question_model", ""),
        "total_questions": len(get_all_questions(questionnaire)),
        "minimum_sentences_required": MIN_SENTENCES,
        "recommended_sentences": RECOMMENDED_SENTENCES,
        "sentence_count_rule": "A sentence is counted only when it ends with a period (.)",
    }

    for question in get_all_questions(questionnaire):
        answer = answers.get(question["id"], "").strip()
        row[f"{question['id']}_layer"] = question["layer"]
        row[f"{question['id']}_type"] = question["type"]
        row[f"{question['id']}_prompt"] = question["prompt"]
        row[f"{question['id']}_prompt_ko"] = question["prompt_ko"]
        row[f"{question['id']}_prompt_source"] = question.get("prompt_source", questionnaire.get("question_source", "rule_based"))
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
    participant_id: str,
    questionnaire: dict,
    step_order: int,
    answers: dict,
) -> dict:
    evaluation = evaluate_state(answers)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "submission_id": submission_id,
        "student_name": student_name,
        "participant_id": participant_id,
        "questionnaire_id": questionnaire["id"],
        "questionnaire_title": questionnaire["page_title"],
        "questionnaire_title_ko": questionnaire["page_title_ko"],
        "story_title": questionnaire["story_title"],
        "sequence_order": step_order,
        "question_source": questionnaire.get("question_source", "rule_based"),
        "question_model": questionnaire.get("question_model", ""),
        "evaluation_method": evaluation["evaluation_method"],
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
        "emotion_self_response": answers.get("q3", "").strip(),
        "cognition_self_response": answers.get("q5", "").strip(),
        "behavior_self_response": answers.get("q7", "").strip(),
        "strategy_self_response": answers.get("q9", "").strip(),
    }


def validate_answers(student_name: str, questionnaire: dict, answers: dict, require_student_name: bool) -> str:
    if require_student_name and not student_name.strip():
        return "Please enter Student Name / 학생 이름을 입력해주세요."

    if any(not answers.get(question["id"], "").strip() for question in get_all_questions(questionnaire)):
        return f"Please answer all 9 questions in {questionnaire['page_title']} / 모든 9개 문항에 답해주세요."

    if any(sentence_count(answer) < MIN_SENTENCES for answer in answers.values()):
        return (
            f"Please write at least {MIN_SENTENCES} sentences in English for each answer, "
            "and end each sentence with a period (.). / 각 답변은 영어로 최소 2문장 이상 작성하고, "
            "문장 끝에 반드시 마침표(.)를 넣어주세요."
        )

    return ""


def initialize_session_state():
    defaults = {
        "current_questionnaire_index": 0,
        "saved_questionnaire_answers": {},
        "student_name_value": "",
        "student_name_input": "",
        "participant_id_value": "",
        "participant_id_input": "",
        "submission_complete": False,
        "last_submission_id": "",
        "last_response_file": "",
        "last_evaluation_file": "",
        "last_storage_backend": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session_state():
    keys_to_clear = [
        "current_questionnaire_index",
        "saved_questionnaire_answers",
        "student_name_value",
        "student_name_input",
        "participant_id_value",
        "participant_id_input",
        "submission_complete",
        "last_submission_id",
        "last_response_file",
        "last_evaluation_file",
        "last_storage_backend",
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)

    for questionnaire in QUESTIONNAIRES:
        for question in get_all_questions(questionnaire):
            question_key = widget_key(questionnaire["id"], question["id"])
            st.session_state.pop(question_key, None)
            st.session_state.pop(completion_key(questionnaire["id"], question["id"]), None)


def initialize_questionnaire_widgets(questionnaire: dict):
    saved_answers = st.session_state.saved_questionnaire_answers.get(questionnaire["id"], {})

    for question in get_all_questions(questionnaire):
        key = widget_key(questionnaire["id"], question["id"])
        if key not in st.session_state:
            st.session_state[key] = saved_answers.get(question["id"], "")
        completion_state_key = completion_key(questionnaire["id"], question["id"])
        if completion_state_key not in st.session_state:
            st.session_state[completion_state_key] = sentence_count(st.session_state[key]) >= MIN_SENTENCES


def inject_custom_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(241, 244, 255, 0.95), transparent 38%),
                linear-gradient(180deg, #f7f3ea 0%, #fbfcfe 42%, #f5f8fc 100%);
        }

        .section-title {
            margin-top: 1.1rem;
            margin-bottom: 0.25rem;
            font-size: 1.25rem;
            font-weight: 800;
            color: #17263a;
            letter-spacing: -0.02em;
        }

        .section-subtitle {
            margin-bottom: 0.75rem;
            color: #53657a;
            font-size: 0.94rem;
        }

        .question-card {
            border-radius: 20px;
            padding: 1rem 1.05rem 0.95rem 1.05rem;
            margin: 0.7rem 0 0.45rem 0;
            border: 1px solid #d5dfeb;
            box-shadow: 0 14px 30px rgba(21, 43, 68, 0.08);
        }

        .question-card-character {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border-left: 7px solid #2f6fed;
        }

        .question-card-self {
            background: linear-gradient(180deg, #fffef8 0%, #fff8ea 100%);
            border-left: 7px solid #d97706;
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
            font-size: 0.73rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .question-chip-label {
            background: #12263f;
            color: #ffffff;
        }

        .question-chip-layer {
            background: #dce9ff;
            color: #1b4fb8;
        }

        .question-chip-type {
            background: #fff0d6;
            color: #9a5a00;
        }

        .question-prompt {
            font-size: 1.18rem;
            line-height: 1.58;
            font-weight: 800;
            color: #13263b;
            margin-bottom: 0.65rem;
        }

        .question-ko {
            font-size: 0.98rem;
            line-height: 1.62;
            color: #4d6177;
            margin-bottom: 0.55rem;
        }

        .question-note {
            font-size: 0.9rem;
            line-height: 1.55;
            color: #6a4b0a;
            background: rgba(255, 244, 214, 0.7);
            border-radius: 12px;
            padding: 0.6rem 0.75rem;
            border: 1px solid rgba(217, 119, 6, 0.18);
        }

        .question-note-muted {
            color: #56697d;
            background: rgba(235, 242, 248, 0.8);
            border: 1px solid rgba(123, 141, 162, 0.18);
        }

        .stTextArea textarea {
            font-size: 1.02rem !important;
            line-height: 1.72 !important;
            background: #fffdf8 !important;
            border-radius: 16px !important;
            border: 1.5px solid #c8d5e4 !important;
            box-shadow: inset 0 1px 3px rgba(15, 23, 42, 0.03);
        }

        .stTextArea textarea:focus {
            border-color: #2f6fed !important;
            box-shadow: 0 0 0 0.18rem rgba(47, 111, 237, 0.14) !important;
        }
        </style>
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


def render_question_card(question: dict):
    card_classes = ["question-card"]
    if question["type"] == "Self":
        card_classes.append("question-card-self")
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

    st.markdown(
        (
            f"<div class='{' '.join(card_classes)}'>"
            "<div class='question-top'>"
            f"<span class='question-chip question-chip-label'>{html.escape(question['label'])}</span>"
            f"<span class='question-chip question-chip-layer'>{html.escape(question['layer'])}</span>"
            f"<span class='question-chip question-chip-type'>{html.escape(question['type'])}</span>"
            "</div>"
            f"<div class='question-prompt'>{html.escape(question['prompt'])}</div>"
            f"<div class='question-ko'>한국어 해석: {html.escape(question['prompt_ko'])}</div>"
            f"{note_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


# -----------------------------
# Session defaults
# -----------------------------
initialize_session_state()
ensure_output_directories()
google_sheets_config = get_google_sheets_config()
question_generation_config = get_question_generation_config()

with st.spinner("Preparing passage-based questions..."):
    QUESTIONNAIRES = build_questionnaires(question_generation_config)

inject_custom_styles()


# -----------------------------
# UI
# -----------------------------
st.title("Educational Diagnostic System")
st.subheader("Bilingual two-step questionnaire for student state evaluation")

if st.session_state.submission_complete:
    st.success("All responses and state evaluations have been saved.")
    st.write(f"Student Name / 학생 이름: `{st.session_state.student_name_value}`")
    if st.session_state.participant_id_value:
        st.write(f"Participant ID / 참가자 ID: `{st.session_state.participant_id_value}`")
    st.write(f"Submission ID: `{st.session_state.last_submission_id}`")
    st.write(f"Storage backend: `{st.session_state.last_storage_backend}`")
    st.write(f"Responses destination: `{st.session_state.last_response_file}`")
    st.write(f"Evaluations destination: `{st.session_state.last_evaluation_file}`")

    if st.button("Start New Participant", use_container_width=True):
        reset_session_state()
        st.rerun()

    st.markdown("---")
    st.caption("Ready for the next participant.")
    st.stop()

if st.session_state.current_questionnaire_index >= len(QUESTIONNAIRES):
    st.session_state.current_questionnaire_index = 0

current_index = st.session_state.current_questionnaire_index
current_questionnaire_base = QUESTIONNAIRES[current_index]
initialize_questionnaire_widgets(current_questionnaire_base)

if current_index > 0 and not st.session_state.student_name_value:
    st.session_state.current_questionnaire_index = 0
    st.rerun()

current_answers_snapshot = {
    question["id"]: st.session_state.get(widget_key(current_questionnaire_base["id"], question["id"]), "").strip()
    for question in get_all_questions(current_questionnaire_base)
}
current_questionnaire = materialize_questionnaire_for_answers(
    questionnaire=current_questionnaire_base,
    answers=current_answers_snapshot,
    question_generation_config=question_generation_config,
)

st.caption(f"Step {current_index + 1} of {len(QUESTIONNAIRES)}")
st.progress((current_index + 1) / len(QUESTIONNAIRES))

if google_sheets_config["enabled"]:
    st.success(
        "Storage mode: Google Sheets\n\n"
        f"Spreadsheet ID: `{google_sheets_config['spreadsheet_id']}`\n\n"
        f"Worksheets: `{google_sheets_config['responses_worksheet']}` and `{google_sheets_config['evaluations_worksheet']}`"
    )
else:
    st.warning(
        "Storage mode: Local CSV fallback. Google Sheets is not configured yet.\n\n"
        f"Current local files: `{RESPONSES_FILE}` and `{EVALUATIONS_FILE}`"
    )

if question_generation_config["enabled"]:
    st.success(
        "Question mode: AI-personalized self follow-up questions are active.\n\n"
        f"Model: `{question_generation_config.get('model', DEFAULT_QUESTION_MODEL)}`"
    )
else:
    st.info(
        "Question mode: Built-in character questions and template-based self follow-up questions are active.\n\n"
        "Add `OPENAI_API_KEY` or `[openai].api_key` in Streamlit secrets to turn the self follow-up questions into AI-personalized prompts."
    )
st.caption(f"Base passage-question status: {current_questionnaire_base.get('question_generation_note', '')}")
st.caption(
    f"Self questions refresh after you complete the character question above them with at least {MIN_SENTENCES} sentences."
)

st.info(
    "This activity checks how students respond to a CEFR-level story and a literature passage. "
    "Please answer every question in English. Use a period (.) at the end of each sentence. "
    f"Each answer must contain at least {MIN_SENTENCES} sentences, and {RECOMMENDED_SENTENCES} or more sentences are recommended.\n\n"
    "이 활동은 학생이 영어 이야기와 문학 지문에 어떻게 반응하는지 확인하기 위한 것입니다. "
    "모든 답변은 영어로 작성하세요. 각 문장 끝에는 반드시 마침표(.)를 찍어야 문장으로 계산됩니다. "
    f"각 답변은 최소 {MIN_SENTENCES}문장 이상, 가능하면 {RECOMMENDED_SENTENCES}문장 이상 작성하세요."
)

st.markdown(f"## {current_questionnaire['page_title']}")
st.write(current_questionnaire["intro"])
st.caption(current_questionnaire["intro_ko"])

if current_index == 0:
    student_name = st.text_input(
        "Student Name / 학생 이름",
        key="student_name_input",
        placeholder="e.g., Minji Kim",
    )
    participant_id = st.text_input(
        "Participant ID / 참가자 ID (optional)",
        key="participant_id_input",
        placeholder="e.g., P001",
    )
else:
    student_name = st.session_state.student_name_value
    participant_id = st.session_state.participant_id_value
    st.text_input(
        "Student Name / 학생 이름",
        value=student_name,
        disabled=True,
    )
    st.text_input(
        "Participant ID / 참가자 ID",
        value=participant_id,
        disabled=True,
    )

st.markdown("### Passage")
st.markdown(current_questionnaire["text"])
st.markdown("#### Korean Translation")
st.markdown(current_questionnaire["text_ko"])

st.info(
    "Sentence guide: Only sentences ending with a period (.) are counted. "
    "If you do not use a period, the sentence will not be counted.\n\n"
    "문장 작성 가이드: 문장 끝에 마침표(.)가 있어야 문장 수로 계산됩니다. "
    "마침표를 쓰지 않으면 문장으로 계산되지 않습니다."
)

current_answers = {
    question["id"]: st.session_state.get(widget_key(current_questionnaire["id"], question["id"]), "").strip()
    for question in get_all_questions(current_questionnaire)
}

completed_questions = 0
total_questions = len(get_all_questions(current_questionnaire))

for section in current_questionnaire["sections"]:
    render_section_header(section)

    for question in section["questions"]:
        question_key = widget_key(current_questionnaire["id"], question["id"])
        render_question_card(question)
        answer = st.text_area(
            f"{question['label']}. {question['prompt']}",
            key=question_key,
            label_visibility="collapsed",
            height=170,
            disabled=not question.get("ready_for_answer", True),
            placeholder=(
                "Write at least 2 sentences in English. Use a period (.) at the end of each sentence."
                if question.get("ready_for_answer", True)
                else f"Finish {question.get('depends_on', 'the previous question')} first to unlock this personalized self question."
            ),
            on_change=update_question_completion,
            args=(current_questionnaire["id"], question["id"]),
        )
        current_answers[question["id"]] = answer.strip()

        counted_sentences = sentence_count(answer)
        is_complete = question.get("ready_for_answer", True) and counted_sentences >= MIN_SENTENCES
        if is_complete:
            completed_questions += 1

        question_completion_key = completion_key(current_questionnaire["id"], question["id"])
        st.session_state[question_completion_key] = is_complete

        status_col, detail_col = st.columns([1, 4])
        with status_col:
            st.checkbox("Completed", key=question_completion_key, disabled=True)
        with detail_col:
            if answer.strip():
                if is_complete:
                    st.markdown(f":green[Completed. Counted sentences: {counted_sentences}.]")
                else:
                    st.caption(
                        (
                            f"Counted sentences: {counted_sentences}. "
                            "Write at least 2 sentences and use a period (.) after each sentence."
                            if question.get("ready_for_answer", True)
                            else f"This answer box will open after {question.get('depends_on', 'the previous question')} is completed."
                        )
                    )
            else:
                st.caption(
                    (
                        "Counted sentences: 0. Start writing in English and end each sentence with a period (.)."
                        if question.get("ready_for_answer", True)
                        else f"Waiting for {question.get('depends_on', 'the previous question')} so this personalized follow-up can be generated."
                    )
                )

st.markdown("### Completion")
st.caption(f"Completed questions: {completed_questions} / {total_questions}")
st.progress(completed_questions / total_questions if total_questions else 0.0)

back_clicked = False
next_clicked = False
submit_clicked = False

if current_index == 0:
    next_clicked = st.button("Next: Literature / 다음: 문학", use_container_width=True)
else:
    button_col1, button_col2 = st.columns(2)
    with button_col1:
        back_clicked = st.button("Back to CEFR Story / 이전 단계", use_container_width=True)
    with button_col2:
        submit_clicked = st.button("Submit All Responses / 전체 제출", use_container_width=True)


# -----------------------------
# Process submission
# -----------------------------
if back_clicked:
    saved_answers = dict(st.session_state.saved_questionnaire_answers)
    saved_answers[current_questionnaire["id"]] = current_answers
    st.session_state.saved_questionnaire_answers = saved_answers
    st.session_state.current_questionnaire_index = 0
    st.rerun()

if next_clicked or submit_clicked:
    student_name_value = student_name.strip()
    participant_id_value = participant_id.strip()
    error_message = validate_answers(
        student_name=student_name_value,
        questionnaire=current_questionnaire,
        answers=current_answers,
        require_student_name=(current_index == 0),
    )

    if error_message:
        st.error(error_message)
    else:
        saved_answers = dict(st.session_state.saved_questionnaire_answers)
        saved_answers[current_questionnaire["id"]] = current_answers
        st.session_state.saved_questionnaire_answers = saved_answers

        if current_index == 0:
            st.session_state.student_name_value = student_name_value
            st.session_state.participant_id_value = participant_id_value
            st.session_state.current_questionnaire_index = 1
            st.rerun()

        submission_id = create_submission_id(
            student_name=st.session_state.student_name_value,
            participant_id=st.session_state.participant_id_value,
        )
        response_rows = []
        evaluation_rows = []

        for step_order, questionnaire in enumerate(QUESTIONNAIRES, start=1):
            answers = st.session_state.saved_questionnaire_answers.get(questionnaire["id"], {})
            materialized_questionnaire = materialize_questionnaire_for_answers(
                questionnaire=questionnaire,
                answers=answers,
                question_generation_config=question_generation_config,
            )
            response_rows.append(
                build_response_row(
                    submission_id=submission_id,
                    student_name=st.session_state.student_name_value,
                    participant_id=st.session_state.participant_id_value,
                    questionnaire=materialized_questionnaire,
                    step_order=step_order,
                    answers=answers,
                )
            )
            evaluation_rows.append(
                build_evaluation_row(
                    submission_id=submission_id,
                    student_name=st.session_state.student_name_value,
                    participant_id=st.session_state.participant_id_value,
                    questionnaire=materialized_questionnaire,
                    step_order=step_order,
                    answers=answers,
                )
            )

        try:
            response_save_result = save_rows(
                rows=response_rows,
                local_file_path=RESPONSES_FILE,
                google_worksheet_name=google_sheets_config.get("responses_worksheet", DEFAULT_RESPONSES_WORKSHEET),
            )
            evaluation_save_result = save_rows(
                rows=evaluation_rows,
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
st.caption("Prototype for CEFR story and literature-based state evaluation")
