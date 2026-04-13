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
    page_title="CEFR Interactive Diagnostic",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed",
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
DEFAULT_LLM_MODEL = "gpt-5-mini"

MIN_SENTENCES = 2
RECOMMENDED_SENTENCES = 3
CEFR_LEVEL_OPTIONS = ["A1", "A2", "B1", "B2", "C1"]

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
                "question_goal": "Ask the learner to explain the main situation or event from the passage in their own words.",
            },
            {
                "id": "q2",
                "label": "Q2",
                "layer": "Understanding",
                "type": "Depth 2",
                "question_goal": "Ask a deeper inference or evidence question about why the event matters, what causes it, or which detail is most important.",
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
                "question_goal": "Ask how the learner would feel in a similar English-learning situation.",
                "criteria_focus": "Target FLA and FLE without naming those labels. Encourage feelings such as nervous, worried, afraid, comfortable, relaxed, interested, happy, or enjoying the class.",
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
                "question_goal": "Ask what the learner would think about their own English ability and how they would notice or manage difficulty.",
                "criteria_focus": "Target self-efficacy and metacognition without naming the labels. Encourage reflection on ability, confidence, noticing problems, and managing or improving learning.",
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
                "question_goal": "Ask what the learner would do in a similar English-learning situation.",
                "criteria_focus": "Target WTC, coping, and engagement without naming the labels. Encourage reflection on speaking, staying quiet, asking for help, preparing, practicing, avoiding, or participating.",
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
                "question_goal": "Ask what strategy would help the learner most in a similar English-learning situation.",
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
    "q4": {
        "state_targets": "FLA / FLE",
        "criteria_focus": (
            "Generate a self question for English class that connects the learner's character-emotion answer "
            "to their own emotional state. It should help reveal anxiety versus enjoyment without naming the labels."
        ),
        "fallback_en": (
            "In a similar situation in English class, how would you feel? "
            "Would you feel nervous, comfortable, interested, happy, or something else? Explain why."
        ),
        "fallback_ko": (
            "영어 수업에서 비슷한 상황이라면 어떤 기분이 들까요? "
            "긴장되는지, 편안한지, 흥미로운지, 행복한지, 또는 다른 감정이 드는지와 그 이유를 설명해 보세요."
        ),
        "note": "This follow-up is linked to your answer in Q3 and checks your English-class emotion state.",
    },
    "q6": {
        "state_targets": "Self-efficacy / Metacognition",
        "criteria_focus": (
            "Generate a self question for English class that connects the learner's character-thinking answer "
            "to their own confidence, self-awareness, noticing problems, and regulation."
        ),
        "fallback_en": (
            "In a similar situation in English class, what would you think about your English ability? "
            "How would you notice your problem and try to manage it?"
        ),
        "fallback_ko": (
            "영어 수업에서 비슷한 상황이라면 자신의 영어 능력에 대해 어떤 생각이 들까요? "
            "자신의 문제를 어떻게 알아차리고 어떻게 관리하려고 할지도 설명해 보세요."
        ),
        "note": "This follow-up is linked to your answer in Q5 and checks your English-class thinking state.",
    },
    "q8": {
        "state_targets": "WTC / Coping / Engagement",
        "criteria_focus": (
            "Generate a self question for English class that connects the learner's character-behavior answer "
            "to their own willingness to communicate, coping under difficulty, and participation."
        ),
        "fallback_en": (
            "In a similar situation in English class, what would you do? "
            "Would you speak, stay quiet, ask for help, prepare more, practice more, or avoid it? Explain."
        ),
        "fallback_ko": (
            "영어 수업에서 비슷한 상황이라면 어떻게 행동할까요? "
            "말하려고 하는지, 조용히 있는지, 도움을 요청하는지, 더 준비하거나 연습하는지, 또는 피하려고 하는지 설명해 보세요."
        ),
        "note": "This follow-up is linked to your answer in Q7 and checks your English-class behavior state.",
    },
    "q10": {
        "state_targets": "Strategy Type / Strategy Quality",
        "criteria_focus": (
            "Generate a self question for English class that connects the learner's character-strategy answer "
            "to their own learning strategies. It should help reveal strategy type and quality without naming the labels."
        ),
        "fallback_en": (
            "In a similar situation in English class, what strategy would help you most? "
            "Explain what you would do and how it helps you."
        ),
        "fallback_ko": (
            "영어 수업에서 비슷한 상황이라면 어떤 전략이 가장 도움이 될까요? "
            "무엇을 할 것인지와 그것이 어떻게 도움이 되는지 설명해 보세요."
        ),
        "note": "This follow-up is linked to your answer in Q9 and checks your English-class strategy state.",
    },
}

FALLBACK_PASSAGES_BY_LEVEL = {
    "A1": {
        "story_title": "Mina's English Turn",
        "story_title_ko": "Mina의 영어 발표 차례",
        "character_focus": "Mina",
        "relationship_focus": "Mina and her English class",
        "relationship_focus_ko": "Mina와 영어 수업",
        "passage_sentences": [
            "Mina is in English class.",
            "Her teacher asks her to read a short answer.",
            "Mina feels nervous because many students are looking at her.",
            "She remembers the words, but her voice becomes small.",
            "Her friend smiles and tells her that she can do it.",
            "Mina must decide if she will speak clearly or stay quiet.",
        ],
        "passage_ko_sentences": [
            "Mina는 영어 수업에 있습니다.",
            "선생님은 Mina에게 짧은 답을 읽어 보라고 합니다.",
            "많은 학생들이 자신을 보고 있어서 Mina는 긴장합니다.",
            "단어는 기억나지만 목소리가 작아집니다.",
            "친구는 미소를 지으며 할 수 있다고 말해 줍니다.",
            "Mina는 분명하게 말할지 조용히 있을지 결정해야 합니다.",
        ],
    },
    "A2": {
        "story_title": "Jisoo's Pair Talk",
        "story_title_ko": "Jisoo의 짝 활동 대화",
        "character_focus": "Jisoo",
        "relationship_focus": "Jisoo and her partner",
        "relationship_focus_ko": "Jisoo와 짝 친구",
        "passage_sentences": [
            "Jisoo has a pair-speaking activity in English class.",
            "She knows the topic, but she worries about making mistakes.",
            "Her partner starts speaking quickly and asks an extra question.",
            "Jisoo understands only part of it, so she feels pressure.",
            "She thinks about asking for repetition, using simple words, or staying silent.",
            "The teacher is walking closer, and Jisoo has to respond soon.",
        ],
        "passage_ko_sentences": [
            "Jisoo는 영어 수업에서 짝 말하기 활동을 합니다.",
            "주제는 알고 있지만 실수할까 봐 걱정합니다.",
            "짝 친구는 빠르게 말하기 시작하고 추가 질문도 합니다.",
            "Jisoo는 그 말의 일부만 이해해서 압박을 느낍니다.",
            "다시 말해 달라고 하거나 쉬운 표현을 쓰거나 조용히 있을지를 생각합니다.",
            "선생님이 가까이 오고 있어서 Jisoo는 곧 대답해야 합니다.",
        ],
    },
    "B1": {
        "story_title": "Daniel's Group Presentation",
        "story_title_ko": "Daniel의 모둠 발표",
        "character_focus": "Daniel",
        "relationship_focus": "Daniel and his group",
        "relationship_focus_ko": "Daniel과 모둠 친구들",
        "passage_sentences": [
            "Daniel is preparing a short group presentation in English class.",
            "He practiced his part at home, but he still worries that his pronunciation is weak.",
            "During practice time, one group member suggests adding a harder example.",
            "Daniel thinks the new example sounds better, but he is not sure he can explain it clearly.",
            "He notices that he speaks less whenever the task becomes difficult.",
            "Now he must decide how to help the group without hiding behind the others.",
        ],
        "passage_ko_sentences": [
            "Daniel은 영어 수업에서 짧은 모둠 발표를 준비하고 있습니다.",
            "집에서 자신의 부분을 연습했지만 발음이 약하다고 여전히 걱정합니다.",
            "연습 시간에 한 모둠 친구가 더 어려운 예시를 넣자고 제안합니다.",
            "Daniel은 그 예시가 더 좋다고 생각하지만 자신이 그것을 분명하게 설명할 수 있을지는 확신하지 못합니다.",
            "과제가 어려워질수록 자신이 말을 덜 하게 된다는 점도 알아차립니다.",
            "이제 그는 다른 친구들 뒤에 숨지 않으면서 모둠에 어떻게 기여할지 결정해야 합니다.",
        ],
    },
    "B2": {
        "story_title": "Hana's Debate Decision",
        "story_title_ko": "Hana의 토론 선택",
        "character_focus": "Hana",
        "relationship_focus": "Hana and the debate team",
        "relationship_focus_ko": "Hana와 토론 팀",
        "passage_sentences": [
            "Hana joins an English debate activity about school rules.",
            "She understands the topic well and has several ideas, but she hesitates because confident speakers often talk first.",
            "When the discussion begins, another student presents one of Hana's ideas before she can say it.",
            "Hana feels frustrated and starts to doubt whether her English is strong enough to add something new.",
            "At the same time, she knows that careful listening and quick note-taking could help her re-enter the discussion.",
            "She has only a short moment to decide whether to speak up, ask a question, or stay in the background.",
        ],
        "passage_ko_sentences": [
            "Hana는 학교 규칙에 대한 영어 토론 활동에 참여합니다.",
            "주제를 잘 이해하고 여러 생각도 있지만 자신감 있는 학생들이 먼저 말하는 경우가 많아서 망설입니다.",
            "토론이 시작되자 다른 학생이 Hana가 말하려던 생각 중 하나를 먼저 발표합니다.",
            "Hana는 답답함을 느끼고 자신의 영어가 새로운 의견을 더할 만큼 충분한지 의심하기 시작합니다.",
            "동시에 신중하게 듣고 빠르게 메모하는 것이 다시 대화에 들어가는 데 도움이 될 수 있다는 점도 알고 있습니다.",
            "그녀는 말할지, 질문할지, 아니면 뒤로 물러날지 아주 짧은 시간 안에 결정해야 합니다.",
        ],
    },
    "C1": {
        "story_title": "Yuna's Seminar Response",
        "story_title_ko": "Yuna의 세미나 응답",
        "character_focus": "Yuna",
        "relationship_focus": "Yuna and the seminar group",
        "relationship_focus_ko": "Yuna와 세미나 그룹",
        "passage_sentences": [
            "Yuna is taking part in an advanced English seminar that values spontaneous discussion.",
            "She has read the article carefully and formed a nuanced opinion, yet she worries that her response may sound less sophisticated than the comments made by fluent classmates.",
            "When the instructor asks for immediate reactions, the room becomes briefly silent before several students begin speaking at once.",
            "Yuna recognizes that waiting too long may make her seem passive, but speaking too quickly could weaken the quality of her point.",
            "She starts weighing whether to summarize the article, challenge one idea, or build on a classmate's comment with a precise example.",
            "Her next choice will shape both how others see her and how confidently she joins future discussions.",
        ],
        "passage_ko_sentences": [
            "Yuna는 즉흥적인 토론을 중요하게 여기는 고급 영어 세미나에 참여하고 있습니다.",
            "글을 꼼꼼하게 읽고 섬세한 의견도 만들었지만, 유창한 친구들의 발언보다 자신의 응답이 덜 세련되게 들릴까 걱정합니다.",
            "교수가 즉각적인 반응을 요청하자 교실은 잠시 조용해졌다가 여러 학생이 동시에 말하기 시작합니다.",
            "Yuna는 너무 오래 기다리면 수동적으로 보일 수 있지만 너무 빨리 말하면 자신의 의견의 질이 약해질 수 있다는 점을 알고 있습니다.",
            "그녀는 글을 요약할지, 한 가지 생각에 이의를 제기할지, 아니면 정확한 예시로 친구의 의견을 확장할지를 저울질하기 시작합니다.",
            "다음 선택은 다른 사람들이 자신을 어떻게 보는지와 앞으로 토론에 얼마나 자신감 있게 참여하는지를 모두 좌우하게 됩니다.",
        ],
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


def format_passage_markdown(title: str, sentences: list) -> str:
    return "\n".join([f"**{title}**", "", *sentences])


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
    if len(sentences) not in {5, 6}:
        raise ValueError(f"{key} must contain 5 or 6 sentences.")

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


def build_rule_based_prompt_map(questionnaire: dict) -> dict:
    character_focus = questionnaire.get("character_focus", "the main character")
    relationship_focus = questionnaire.get("relationship_focus", character_focus)
    relationship_focus_ko = questionnaire.get("relationship_focus_ko", relationship_focus)

    return {
        "q1": {
            "prompt": f"What happens between {relationship_focus} in this passage? Explain the situation in your own words.",
            "prompt_ko": f"이 지문에서 {relationship_focus_ko} 사이에 어떤 일이 일어나나요? 자신의 말로 상황을 설명해 보세요.",
        },
        "q2": {
            "prompt": "Which detail is the most important for understanding this situation? Explain why that detail matters.",
            "prompt_ko": "이 상황을 이해하는 데 가장 중요한 세부 내용은 무엇인가요? 왜 중요한지도 설명해 보세요.",
        },
        "q3": {
            "prompt": f"How does {character_focus} feel in this situation? What detail in the passage shows that feeling?",
            "prompt_ko": f"이 상황에서 {character_focus}는 어떻게 느끼나요? 그 감정을 보여 주는 지문 속 단서도 함께 설명해 보세요.",
        },
        "q4": {
            "prompt": (
                "In a similar situation in English class, how would you feel? "
                "Would you feel nervous, comfortable, interested, happy, or something else? Explain why."
            ),
            "prompt_ko": (
                "영어 수업에서 비슷한 상황이라면 어떤 기분이 들까요? "
                "긴장되는지, 편안한지, 흥미로운지, 행복한지, 또는 다른 감정이 드는지와 그 이유를 설명해 보세요."
            ),
        },
        "q5": {
            "prompt": f"What might {character_focus} think or worry about at this moment? Explain your idea.",
            "prompt_ko": f"이 순간 {character_focus}는 어떤 생각이나 걱정을 할까요? 그렇게 생각한 이유도 설명해 보세요.",
        },
        "q6": {
            "prompt": (
                "In a similar situation in English class, what would you think about your English ability? "
                "How would you notice your problem and try to manage it?"
            ),
            "prompt_ko": (
                "영어 수업에서 비슷한 상황이라면 자신의 영어 능력에 대해 어떤 생각이 들까요? "
                "자신의 문제를 어떻게 알아차리고 관리하려고 할지도 설명해 보세요."
            ),
        },
        "q7": {
            "prompt": f"What could {character_focus} do next? Which action seems most likely or most helpful?",
            "prompt_ko": f"이 다음에 {character_focus}는 무엇을 할 수 있을까요? 가장 가능성이 크거나 가장 도움이 되는 행동도 설명해 보세요.",
        },
        "q8": {
            "prompt": (
                "In a similar situation in English class, what would you do? "
                "Would you speak, stay quiet, ask for help, prepare more, practice more, or avoid it? Explain."
            ),
            "prompt_ko": (
                "영어 수업에서 비슷한 상황이라면 어떻게 행동할까요? "
                "말하려고 하는지, 조용히 있는지, 도움을 요청하는지, 더 준비하거나 연습하는지, 또는 피하려고 하는지 설명해 보세요."
            ),
        },
        "q9": {
            "prompt": f"What is the best strategy for {character_focus} in this situation? Why would it work?",
            "prompt_ko": f"이 상황에서 {character_focus}에게 가장 좋은 전략은 무엇일까요? 왜 효과가 있을지도 설명해 보세요.",
        },
        "q10": {
            "prompt": (
                "In a similar situation in English class, what strategy would help you most? "
                "Explain what you would do and how it helps you."
            ),
            "prompt_ko": (
                "영어 수업에서 비슷한 상황이라면 어떤 전략이 가장 도움이 될까요? "
                "무엇을 할 것인지와 그것이 어떻게 도움이 되는지 설명해 보세요."
            ),
        },
    }


def build_fallback_experiment(cefr_level: str, reason: str) -> dict:
    template = FALLBACK_PASSAGES_BY_LEVEL.get(cefr_level, FALLBACK_PASSAGES_BY_LEVEL["A2"])
    questionnaire = dict(template)
    questionnaire["id"] = "cefr_dynamic"
    questionnaire["page_title"] = "CEFR Interactive Diagnostic"
    questionnaire["page_title_ko"] = "CEFR 상호작용 진단"
    questionnaire["intro"] = (
        "Read the passage on the left and answer every question on the right in English. "
        "Each answer must include at least 2 complete sentences with periods."
    )
    questionnaire["intro_ko"] = (
        "왼쪽 지문을 읽고 오른쪽 질문에 모두 영어로 답하세요. "
        "각 답변은 마침표가 있는 완전한 영어 문장 최소 2개 이상이어야 합니다."
    )
    questionnaire["cefr_level"] = cefr_level
    questionnaire["text"] = format_passage_markdown(template["story_title"], template["passage_sentences"])
    questionnaire["text_ko"] = format_passage_markdown(template["story_title_ko"], template["passage_ko_sentences"])
    questionnaire["passage_plain"] = " ".join(template["passage_sentences"])
    questionnaire["passage_ko_plain"] = " ".join(template["passage_ko_sentences"])
    questionnaire["passage_sentences"] = list(template["passage_sentences"])
    questionnaire["passage_ko_sentences"] = list(template["passage_ko_sentences"])
    questionnaire["sections"] = apply_prompt_map_to_sections(build_rule_based_prompt_map(questionnaire))
    questionnaire["question_source"] = "rule_based"
    questionnaire["question_model"] = ""
    questionnaire["question_generation_note"] = reason
    return questionnaire


def normalize_experiment_payload(payload: dict, cefr_level: str) -> dict:
    story_title = str(payload.get("story_title", "")).strip()
    story_title_ko = str(payload.get("story_title_ko", "")).strip()
    character_focus = str(payload.get("character_focus", "")).strip()
    relationship_focus = str(payload.get("relationship_focus", "")).strip()
    relationship_focus_ko = str(payload.get("relationship_focus_ko", "")).strip()

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
        "character_focus": character_focus,
        "relationship_focus": relationship_focus,
        "relationship_focus_ko": relationship_focus_ko,
        "intro": (
            "Read the CEFR-level passage on the left and answer every question on the right in English. "
            "Each answer must include at least 2 complete sentences with periods."
        ),
        "intro_ko": (
            "왼쪽 CEFR 수준 지문을 읽고 오른쪽 질문에 모두 영어로 답하세요. "
            "각 답변은 마침표가 있는 완전한 영어 문장 최소 2개 이상이어야 합니다."
        ),
        "cefr_level": cefr_level,
        "passage_sentences": passage_sentences,
        "passage_ko_sentences": passage_ko_sentences,
        "text": format_passage_markdown(story_title, passage_sentences),
        "text_ko": format_passage_markdown(story_title_ko, passage_ko_sentences),
        "passage_plain": " ".join(passage_sentences),
        "passage_ko_plain": " ".join(passage_ko_sentences),
        "sections": apply_prompt_map_to_sections(prompt_map),
    }

    return questionnaire


def generate_experiment_with_openai(cefr_level: str, api_key: str, model: str) -> dict:
    client = get_openai_client(api_key)
    question_slot_instructions = build_question_slot_instructions()

    system_prompt = """
You design one bilingual CEFR-based reading-and-response experiment for an English-education diagnostic app.
Return valid JSON only.

Requirements:
- Create one short passage with exactly 5 or 6 English sentences.
- Match the CEFR level requested by the user.
- Use one clear main character and one meaningful classroom, speaking, presentation, discussion, or communication situation.
- Make the situation rich enough to support emotion, cognition, behavior, and strategy evaluation.
- Create exactly 10 student-facing questions with ids q1 to q10.
- Make the questions interactive, specific, and learner-friendly.
- Include depth 2 by making the second question in each section meaningfully deeper than the first.
- Keep self questions explicitly connected to English class.
- Do not use technical labels such as FLA, FLE, self-efficacy, metacognition, WTC, coping, engagement, or Oxford strategy type in the student-facing questions.
- Make the Korean translations natural and faithful.
"""

    user_prompt = f"""
Target CEFR level: {cefr_level}

Evaluation criteria:
{QUESTION_GENERATION_CRITERIA}

Question slots:
{question_slot_instructions}

Return JSON only in this format:
{{
  "story_title": "...",
  "story_title_ko": "...",
  "character_focus": "...",
  "relationship_focus": "...",
  "relationship_focus_ko": "...",
  "passage_sentences": ["...", "...", "...", "...", "..."],
  "passage_ko_sentences": ["...", "...", "...", "...", "..."],
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
    questionnaire["question_source"] = "openai"
    questionnaire["question_model"] = model
    questionnaire["question_generation_note"] = "Passage and interactive questions were generated with OpenAI."
    return questionnaire


@st.cache_data(show_spinner=False, ttl=86400)
def build_generated_experiment(cefr_level: str, api_key: str, model: str, nonce: str) -> dict:
    if not api_key:
        return build_fallback_experiment(
            cefr_level=cefr_level,
            reason="OpenAI API key is not configured, so the app is using a built-in fallback passage and question set.",
        )

    try:
        return generate_experiment_with_openai(
            cefr_level=cefr_level,
            api_key=api_key,
            model=model,
        )
    except Exception:
        return build_fallback_experiment(
            cefr_level=cefr_level,
            reason="OpenAI generation failed, so the app is using a built-in fallback passage and question set.",
        )


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
    sentences = re.split(r"\.+", text.strip())
    sentences = [sentence for sentence in sentences if sentence.strip()]
    return len(sentences)


def contains_any(text: str, keywords: list) -> bool:
    return any(keyword in text for keyword in keywords)


def count_matches(text: str, keywords: list) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def build_rule_based_answer_feedback(question: dict, answer: str) -> dict:
    stripped = answer.strip()
    lower = stripped.lower()

    if not stripped:
        return {
            "status": "rewrite",
            "message": "This answer is incomplete. Please write at least 2 complete English sentences and answer the question directly.",
            "message_ko": "이 답변은 불완전합니다. 영어 완전한 문장 2개 이상으로 질문에 직접 답하도록 다시 작성해 주세요.",
        }

    invalid_patterns = [
        "i don't know",
        "i do not know",
        "dont know",
        "idk",
        "not sure",
        "no idea",
        "maybe",
        "i guess",
        "what?",
        "huh",
    ]

    if sentence_count(stripped) < MIN_SENTENCES:
        return {
            "status": "rewrite",
            "message": "This answer is incomplete. Please rewrite it in at least 2 complete English sentences with periods.",
            "message_ko": "이 답변은 불완전합니다. 마침표가 있는 영어 완전한 문장 2개 이상으로 다시 작성해 주세요.",
        }

    if word_count(stripped) < 6 or contains_any(lower, invalid_patterns):
        return {
            "status": "rewrite",
            "message": "This answer looks incomplete or unclear. Please rewrite it in complete English sentences and answer the question more clearly.",
            "message_ko": "이 답변은 불완전하거나 모호해 보입니다. 영어 완전한 문장으로 질문에 더 분명하게 답하도록 다시 작성해 주세요.",
        }

    if stripped.endswith("?"):
        return {
            "status": "rewrite",
            "message": "This answer looks incomplete because it reads like a question. Please rewrite it as complete English sentences.",
            "message_ko": "이 답변은 질문처럼 보여서 불완전합니다. 영어 완전한 문장으로 다시 작성해 주세요.",
        }

    return {
        "status": "valid",
        "message": "Good. This answer is clear enough to continue.",
        "message_ko": "좋습니다. 이 답변은 다음 단계로 진행할 만큼 충분히 분명합니다.",
    }


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
- Mark status as "rewrite" if the answer is off-topic, too vague, says I don't know, is incomplete, asks a question back, or does not contain at least 2 complete English sentences.
- Mark status as "valid" only if the answer directly addresses the question and is clear enough to continue.
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


@st.cache_data(show_spinner=False, ttl=86400)
def get_answer_feedback(
    questionnaire_json: str,
    question_json: str,
    answer: str,
    api_key: str,
    model: str,
) -> dict:
    questionnaire = json.loads(questionnaire_json)
    question = json.loads(question_json)
    fallback = build_rule_based_answer_feedback(question, answer)

    if fallback["status"] == "rewrite" or not api_key:
        return fallback

    try:
        return generate_answer_feedback_with_openai(
            questionnaire=questionnaire,
            question=question,
            answer=answer,
            api_key=api_key,
            model=model,
        )
    except Exception:
        return fallback


def build_pending_self_prompt(
    question: dict,
    source_question: dict,
    source_feedback: dict,
) -> dict:
    reason_en = source_feedback.get(
        "message",
        "The previous answer needs to be rewritten in complete English sentences.",
    )
    reason_ko = source_feedback.get(
        "message_ko",
        "이전 답변을 영어 완전한 문장으로 다시 작성해야 합니다.",
    )

    return {
        "prompt": question.get("base_prompt", question.get("prompt", "")),
        "prompt_ko": question.get("base_prompt_ko", question.get("prompt_ko", "")),
        "prompt_source": "pending",
        "prompt_note": (
            f"Rewrite {source_question['label']} first. {reason_en}\n\n"
            f"{source_question['label']}을 먼저 다시 작성해 주세요. {reason_ko}"
        ),
        "ready_for_answer": False,
    }


def build_rule_based_personalized_self_prompt(question: dict) -> dict:
    spec = get_personalized_self_spec(question["id"])

    return {
        "prompt": spec["fallback_en"],
        "prompt_ko": spec["fallback_ko"],
        "prompt_source": "rule_based_personalized",
        "prompt_note": spec["note"],
        "ready_for_answer": True,
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

    system_prompt = """
You generate one personalized self-reflection question for an English-education diagnostic app.
Return valid JSON only.
The question must connect to the learner's answer about the character, but it must move into the learner's own experience in English class.
Use simple, learner-friendly English.
Do not mention technical labels such as FLA, FLE, self-efficacy, metacognition, WTC, coping, engagement, or strategy type.
Make the Korean translation natural and faithful.
"""

    user_prompt = f"""
Passage title: {questionnaire['story_title']}
Passage:
{questionnaire['passage_plain']}

Current layer: {question['layer']}
State targets: {spec.get('state_targets', '')}
Instruction:
{spec.get('criteria_focus', '')}

Base self question:
{question.get('base_prompt', question.get('prompt', ''))}

Character question:
{source_question['prompt']}

Learner's answer to the character question:
{source_answer}

Create one self question for English class that:
1. Feels clearly linked to the learner's answer above.
2. Still helps evaluate the learner's state in English learning.
3. Sounds natural for a student questionnaire.
4. Uses 1 or 2 short sentences.

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

    if sentence_count(source_answer) < MIN_SENTENCES or source_feedback.get("status") != "valid":
        return build_pending_self_prompt(question, source_question, source_feedback)

    fallback = build_rule_based_personalized_self_prompt(question)

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
        question["prompt_source"] = materialized.get("question_source", "rule_based")
        question["prompt_note"] = ""
        question["ready_for_answer"] = True

        if question["type"] != "Self" or not question.get("depends_on"):
            continue

        source_question = question_lookup[question["depends_on"]]
        source_answer = answers.get(source_question["id"], "").strip()
        source_feedback = get_answer_feedback(
            questionnaire_json=questionnaire_json,
            question_json=json.dumps(source_question, ensure_ascii=False, sort_keys=True),
            answer=source_answer,
            api_key=llm_config["api_key"],
            model=llm_config["model"],
        )

        personalized = build_personalized_self_prompt(
            questionnaire_json=questionnaire_json,
            question_json=json.dumps(question, ensure_ascii=False, sort_keys=True),
            source_question_json=json.dumps(source_question, ensure_ascii=False, sort_keys=True),
            source_answer=source_answer,
            source_feedback_json=json.dumps(source_feedback, ensure_ascii=False, sort_keys=True),
            api_key=llm_config["api_key"],
            model=llm_config["model"],
        )
        question.update(personalized)

    return materialized


# -----------------------------
# Evaluation helpers
# -----------------------------
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
    fla, fle = classify_fla_fle(answers.get("q4", ""))
    self_efficacy, metacognition = classify_self_efficacy_metacognition(answers.get("q6", ""))
    wtc, coping, engagement = classify_behavior(answers.get("q8", ""))
    strategy_type, strategy_quality = classify_strategy(answers.get("q10", ""))

    return {
        "evaluation_method": "criteria_rules_v2_dynamic_cefr",
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
        "questionnaire_id": questionnaire["id"],
        "questionnaire_title": questionnaire["page_title"],
        "questionnaire_title_ko": questionnaire["page_title_ko"],
        "story_title": questionnaire["story_title"],
        "story_title_ko": questionnaire.get("story_title_ko", ""),
        "passage_text": questionnaire.get("passage_plain", ""),
        "passage_text_ko": questionnaire.get("passage_ko_plain", ""),
        "question_source": questionnaire.get("question_source", "rule_based"),
        "question_model": questionnaire.get("question_model", ""),
        "question_generation_note": questionnaire.get("question_generation_note", ""),
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
    student_number: str,
    cefr_level: str,
    questionnaire: dict,
    answers: dict,
) -> dict:
    evaluation = evaluate_state(answers)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "submission_id": submission_id,
        "student_name": student_name,
        "student_number": student_number,
        "cefr_level": cefr_level,
        "questionnaire_id": questionnaire["id"],
        "questionnaire_title": questionnaire["page_title"],
        "questionnaire_title_ko": questionnaire["page_title_ko"],
        "story_title": questionnaire["story_title"],
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
        "emotion_self_response": answers.get("q4", "").strip(),
        "cognition_self_response": answers.get("q6", "").strip(),
        "behavior_self_response": answers.get("q8", "").strip(),
        "strategy_self_response": answers.get("q10", "").strip(),
    }


def validate_participant_inputs(student_name: str, student_number: str, cefr_level: str) -> str:
    if not student_name.strip():
        return "Please enter Name / 이름을 입력해주세요."
    if not student_number.strip():
        return "Please enter Student Number / 학번을 입력해주세요."
    if cefr_level not in CEFR_LEVEL_OPTIONS:
        return "Please choose a CEFR level / CEFR 레벨을 선택해주세요."
    return ""


def validate_submission_answers(questionnaire: dict, answers: dict, llm_config: dict) -> str:
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
        "student_name_value": "",
        "student_number_value": "",
        "cefr_level_value": "",
        "active_questionnaire": {},
        "generation_nonce": "",
        "submission_complete": False,
        "last_submission_id": "",
        "last_response_file": "",
        "last_evaluation_file": "",
        "last_storage_backend": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_question_widgets():
    for question in get_question_slots():
        st.session_state.pop(widget_key(question["id"]), None)


def reset_session_state():
    clear_question_widgets()
    keys_to_clear = [
        "student_name_input",
        "student_number_input",
        "cefr_level_input",
        "student_name_value",
        "student_number_value",
        "cefr_level_value",
        "active_questionnaire",
        "generation_nonce",
        "submission_complete",
        "last_submission_id",
        "last_response_file",
        "last_evaluation_file",
        "last_storage_backend",
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

        .placeholder-shell {
            border-radius: 28px;
            padding: 1.4rem 1.2rem;
            background: rgba(255, 251, 245, 0.92);
            border: 1px solid #eadfcf;
            box-shadow: 0 22px 48px rgba(18, 38, 58, 0.08);
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header():
    st.markdown(
        """
        <div class="hero-title">CEFR Interactive Diagnostic</div>
        <div class="hero-subtitle">
            Generate one CEFR-level passage with interactive state-evaluation questions, keep the passage visible on the left,
            and guide participants to answer in complete English sentences.
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
                이름, 학번, CEFR 레벨을 먼저 입력한 뒤 지문과 질문을 생성하세요. 생성 이후 CEFR을 바꾸면 다시 생성해야 새 실험 세트가 반영됩니다.
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
                Enter the participant profile above, then click <b>Generate Passage &amp; Questions</b>.
                The app will create a CEFR-level passage, interactive prompts, and state-linked follow-up questions.
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


def render_question_card(question: dict):
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
    st.success("All responses and evaluations have been saved.")
    st.write(f"Name / 이름: `{st.session_state.student_name_value}`")
    st.write(f"Student Number / 학번: `{st.session_state.student_number_value}`")
    st.write(f"CEFR Level: `{st.session_state.cefr_level_value}`")
    st.write(f"Submission ID: `{st.session_state.last_submission_id}`")
    st.write(f"Storage backend: `{st.session_state.last_storage_backend}`")
    st.write(f"Responses destination: `{st.session_state.last_response_file}`")
    st.write(f"Evaluations destination: `{st.session_state.last_evaluation_file}`")

    if st.button("Start New Participant", use_container_width=True):
        reset_session_state()
        st.rerun()

    st.stop()

render_participant_header()

name_col, number_col, cefr_col, button_col = st.columns([1.1, 1.1, 0.8, 0.9], gap="small")

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

with button_col:
    st.write("")
    generate_clicked = st.button("Generate Passage & Questions", use_container_width=True)

participant_error = validate_participant_inputs(
    student_name=st.session_state.student_name_input,
    student_number=st.session_state.student_number_input,
    cefr_level=st.session_state.cefr_level_input,
)

if generate_clicked:
    if participant_error:
        st.error(participant_error)
    else:
        generation_nonce = uuid4().hex
        with st.spinner(
            "Generating the CEFR passage and interactive questionnaire. "
            "The screen is being kept in a light-style view for readability."
        ):
            questionnaire = build_generated_experiment(
                cefr_level=st.session_state.cefr_level_input,
                api_key=llm_config["api_key"],
                model=llm_config["model"],
                nonce=generation_nonce,
            )

        clear_question_widgets()
        st.session_state.student_name_value = st.session_state.student_name_input.strip()
        st.session_state.student_number_value = st.session_state.student_number_input.strip()
        st.session_state.cefr_level_value = st.session_state.cefr_level_input
        st.session_state.active_questionnaire = questionnaire
        st.session_state.generation_nonce = generation_nonce
        st.rerun()

questionnaire_ready = bool(st.session_state.active_questionnaire)

if questionnaire_ready:
    profile_changed_after_generation = (
        st.session_state.student_name_input.strip() != st.session_state.student_name_value
        or st.session_state.student_number_input.strip() != st.session_state.student_number_value
        or st.session_state.cefr_level_input != st.session_state.cefr_level_value
    )
    if profile_changed_after_generation:
        st.warning(
            "Participant information changed after generation. Click Generate Passage & Questions again to refresh the passage and question set."
        )

if google_sheets_config["enabled"]:
    st.success(
        "Storage mode: Google Sheets\n\n"
        f"Spreadsheet ID: `{google_sheets_config['spreadsheet_id']}`\n\n"
        f"Worksheets: `{google_sheets_config['responses_worksheet']}` and `{google_sheets_config['evaluations_worksheet']}`"
    )
else:
    st.info(
        "Storage mode: Local CSV fallback\n\n"
        f"Current local files: `{RESPONSES_FILE}` and `{EVALUATIONS_FILE}`\n\n"
        "For deployed apps, configure Google Sheets for persistent storage because local cloud files may not be kept after restart or redeploy."
    )

if llm_config["enabled"]:
    st.success(
        "LLM mode: Active\n\n"
        f"Model: `{llm_config['model']}`\n\n"
        "The app will generate the CEFR passage, interactive questions, answer checks, and personalized self follow-ups with OpenAI when possible."
    )
else:
    st.warning(
        "LLM mode: Fallback\n\n"
        "OpenAI API key is not configured, so the app will use the built-in fallback passage, built-in question set, and rule-based answer checks."
    )

if not questionnaire_ready:
    render_placeholder()
    st.stop()

base_questionnaire = st.session_state.active_questionnaire
current_answers_snapshot = {
    question["id"]: st.session_state.get(widget_key(question["id"]), "").strip()
    for question in get_all_questions(base_questionnaire)
}

with st.spinner(
    "Analyzing answers and preparing the next interactive question. "
    "The loading indicator is centered to keep the experiment flow clear for participants."
):
    current_questionnaire = materialize_questionnaire_for_answers(
        questionnaire=base_questionnaire,
        answers=current_answers_snapshot,
        llm_config=llm_config,
    )

st.caption(
    f"Question status: {current_questionnaire.get('question_generation_note', '')}"
)
st.info(
    "Answer guide: Every response must be written in English. "
    "Each answer needs at least 2 complete sentences with periods. "
    "If the answer is too vague, says I don't know, or does not fit the question, the app will ask the participant to rewrite it as incomplete.\n\n"
    "답변 가이드: 모든 답변은 영어로 작성해야 하며, 마침표가 있는 완전한 영어 문장 최소 2개가 필요합니다. "
    "답변이 너무 모호하거나 I don't know와 비슷하거나 질문에 맞지 않으면 불완전한 답변으로 판단되어 다시 작성하라는 안내가 나옵니다."
)

passage_col, question_col = st.columns([0.9, 1.3], gap="large")

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
                Answer every question in English. The self questions are personalized after the related character answer becomes clear enough.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    completed_questions = 0
    total_questions = len(get_all_questions(current_questionnaire))
    questionnaire_json = json.dumps(current_questionnaire, ensure_ascii=False, sort_keys=True)
    current_answers = {}

    for section in current_questionnaire["sections"]:
        render_section_header(section)

        for question in section["questions"]:
            question_key = widget_key(question["id"])
            if not question.get("ready_for_answer", True):
                st.session_state[question_key] = ""
            render_question_card(question)

            answer = st.text_area(
                f"{question['label']}. {question['prompt']}",
                key=question_key,
                label_visibility="collapsed",
                height=170,
                disabled=not question.get("ready_for_answer", True),
                placeholder=(
                    "Write at least 2 complete English sentences. Use a period (.) at the end of each sentence."
                    if question.get("ready_for_answer", True)
                    else "The previous answer must be rewritten clearly before this follow-up opens."
                ),
            )
            answer = answer.strip()
            current_answers[question["id"]] = answer

            if not question.get("ready_for_answer", True):
                render_status_box(
                    "status-box-neutral",
                    "This follow-up will open after the previous answer is clear enough and written in complete English sentences.",
                )
                continue

            if not answer:
                render_status_box(
                    "status-box-neutral",
                    "Counted sentences: 0. Start writing in English and end each sentence with a period (.).",
                )
                continue

            feedback = get_answer_feedback(
                questionnaire_json=questionnaire_json,
                question_json=json.dumps(question, ensure_ascii=False, sort_keys=True),
                answer=answer,
                api_key=llm_config["api_key"],
                model=llm_config["model"],
            )
            counted_sentences = sentence_count(answer)

            if feedback["status"] == "valid":
                completed_questions += 1
                render_status_box(
                    "status-box-valid",
                    (
                        f"Completed. Counted sentences: {counted_sentences}. "
                        f"{feedback['message']} / {feedback['message_ko']}"
                    ),
                )
            else:
                render_status_box(
                    "status-box-rewrite",
                    (
                        f"Counted sentences: {counted_sentences}. "
                        f"{feedback['message']} / {feedback['message_ko']}"
                    ),
                )

    st.markdown("### Completion")
    st.caption(f"Completed questions: {completed_questions} / {total_questions}")
    st.progress(completed_questions / total_questions if total_questions else 0.0)

    submit_clicked = st.button("Submit Responses", use_container_width=True)


if submit_clicked:
    if (
        st.session_state.student_name_input.strip() != st.session_state.student_name_value
        or st.session_state.student_number_input.strip() != st.session_state.student_number_value
        or st.session_state.cefr_level_input != st.session_state.cefr_level_value
    ):
        st.error("Participant information changed after generation. Please generate the passage and questions again before submitting.")
    else:
        validation_error = validate_submission_answers(
            questionnaire=current_questionnaire,
            answers=current_answers,
            llm_config=llm_config,
        )

        if validation_error:
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
