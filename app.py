import html
import hashlib
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
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
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


class PromptFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def load_prompt_template(*path_parts: str, **context) -> str:
    prompt_path = os.path.join(PROMPTS_DIR, *path_parts)
    with open(prompt_path, "r", encoding="utf-8") as prompt_file:
        template = prompt_file.read().strip()
    return template.format_map(PromptFormatDict(context))


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
                "type": "Depth 2",
                "depends_on": "q1",
                "question_goal": "Ask a deeper inference or evidence question about why the situation matters, what causes the turning point, or which detail is most important.",
                "criteria_focus": "Generate a deeper follow-up question based on the learner's first understanding answer. Ask for a reason, key evidence, cause, turning point, or important detail from the passage.",
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
                "question_goal": "Ask how the main character feels in the situation, and make the learner identify emotional evidence linked to anxiety, motivation, or enjoyment without using technical labels.",
            },
            {
                "id": "q4",
                "label": "Q4",
                "layer": "Emotion",
                "type": "Self",
                "depends_on": "q3",
                "question_goal": "Ask how the learner would feel in a similar situation in their own life, in a way that can reveal anxiety, motivation, and enjoyment.",
                "criteria_focus": "Target the learner's emotional state without naming technical labels. The question should create room to reveal anxiety, motivation, and enjoyment through ideas such as nervousness, fear of mistakes, willingness to keep trying, personal goals, classroom enjoyment, sense of achievement, or feeling supported by others.",
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
                "question_goal": "Ask what the main character might think, believe, or worry about in that moment, so the learner can infer confidence, self-regulation, and beliefs about the causes of success or difficulty.",
            },
            {
                "id": "q6",
                "label": "Q6",
                "layer": "Cognition",
                "type": "Self",
                "depends_on": "q5",
                "question_goal": "Ask what the learner would think about their own ability, how they would notice or manage difficulty, and how they would explain the cause of success or difficulty in a similar situation.",
                "criteria_focus": "Target self-efficacy, metacognitive regulation, and attribution-beliefs without naming technical labels. Encourage reflection on confidence, planning, noticing problems, checking understanding, choosing strategies, and explaining why success or difficulty happened.",
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
                "question_goal": "Ask what the best strategy is for the character and why it would work in that situation, with room to reveal a specific Oxford strategy type and the quality of the strategy choice.",
            },
            {
                "id": "q10",
                "label": "Q10",
                "layer": "Strategy",
                "type": "Self",
                "depends_on": "q9",
                "question_goal": "Ask what strategy would help the learner most in a similar situation in their own life, with room to reveal a specific Oxford strategy type and the quality of the strategy choice.",
                "criteria_focus": "Target Oxford's six strategy types and strategy quality without naming technical labels. Encourage answers that can reveal memory, compensation, cognitive, metacognitive, social, or affective strategies, and whether the strategy is effective, limited, or avoidant in that situation.",
            },
        ],
    },
]

QUESTION_GENERATION_CRITERIA = """
Layer and evaluation targets:

Fluency
- Linguistic Fluency
  Low: very short, hesitant, or fragment-like responses with frequent pauses in idea flow and little forward development.
  Mid: understandable connected response with some continuation of ideas, though pacing may still feel stop-and-go or uneven.
  High: smooth and sustained response that develops ideas with little strain, keeps moving forward, and sounds relatively easy to follow.
- Complexity
  Low: mostly isolated words, short phrases, or very simple sentence patterns with little expansion.
  Mid: some sentence expansion and combination of ideas using basic clauses, connectors, or supporting detail.
  High: varied and well-developed sentence structures with clear clause combination, explanation, and more elaborated expression.
- Accuracy
  Low: frequent grammar, word form, or sentence errors that often interfere with clarity.
  Mid: noticeable errors are present, but the main meaning is usually clear and the learner can still communicate the answer.
  High: generally controlled grammar and sentence formation with only minor errors that do not reduce clarity.
- Lexical Diversity
  Low: very limited vocabulary with heavy repetition of the same basic words or expressions.
  Mid: some range of familiar vocabulary with occasional variation beyond the most basic repeated words.
  High: a wider and more purposeful range of vocabulary with flexible word choice that supports explanation and nuance.
- Cohesion
  Low: ideas are loosely connected or disconnected, with little use of linking language or logical ordering.
  Mid: ideas are connected with basic linking words such as and, because, but, so, when, if, or then, though organization may still be simple.
  High: ideas are clearly connected and organized with effective linking, sequencing, and logical progression across the response.
- Observable features to support fluency judgment:
  Linguistic fluency: continuity of response, amount of development, and whether the answer moves forward without collapsing into fragments.
  Complexity: use of clauses, sentence expansion, combined ideas, and explanation beyond a single short statement.
  Accuracy: control of grammar, sentence form, agreement, tense, and whether errors interfere with understanding.
  Lexical diversity: range of vocabulary, variation in word choice, and reduced dependence on repeated basic expressions.
  Cohesion: connector use, sequencing, reference between ideas, and overall logical flow such as first, next, then, later, finally, so, or in the end.

Emotion
- Anxiety: Foreign Language Classroom Anxiety (외국어 학습 불안)
  Representative scale: FLCAS (Foreign Language Classroom Anxiety Scale)
  Key indicators: communication apprehension, test anxiety, fear of negative evaluation.
  High: strong anxiety expressions such as nervous, afraid, worried about mistakes, avoiding speaking, or fear of classmates' or teacher's judgment.
  Mid: anxiety is present but partly managed, such as nervous but try, worried but prepare, or afraid yet still attempt to respond.
  Low: little anxiety is shown; the learner sounds comfortable, relaxed, or not especially worried about speaking or evaluation.
- Motivation: L2 Motivational Self System (제2언어 동기 자아 체계)
  Representative scale: L2MSS
  Key indicators: ideal L2 self, ought-to L2 self, L2 learning experience.
  High: clear future-oriented motivation, personal goals, or meaningful reasons for learning, such as wanting to become a capable English user or valuing the learning experience.
  Mid: some motivation is visible, but goals or reasons are limited, externally driven, or not well elaborated.
  Low: weak motivation, little personal investment, or no clear reason for continued effort.
- Enjoyment: Foreign Language Enjoyment (외국어 즐거움)
  Representative scale: FLE Scale
  Key indicators: classroom joy, sense of achievement, bonding with teacher or peers.
  High: clear enjoyment or positive activation such as fun, interest, pride, achievement, or feeling supported in class.
  Mid: mixed positive and neutral emotion such as okay, sometimes enjoyable, or partly interested.
  Low: low enjoyment or negative feeling such as boring, uncomfortable, not enjoyable, or emotionally disengaged.

Cognition
- Self-efficacy: Language Learning Self-Efficacy (자기 인식)
  Representative scale: ASLSE
  Key indicators: task performance confidence.
  High: strong confidence about completing the task or communicating meaning, such as I can do this, I can explain it, or I can speak well enough.
  Mid: partial or conditional confidence such as I can, but only with help, simple English, or extra time.
  Low: weak confidence and expectation of failure such as I cannot do it, I will fail, or I cannot express my ideas.
- Metacognition: Metacognition / Self-regulation (자기 조절)
  Representative scales: MAI / SILL
  Key indicators: planning, monitoring, strategy selection.
  High: clear evidence of planning, checking understanding, noticing problems, and choosing a strategy to improve performance.
  Mid: some awareness of learning process is present, but regulation is partial, vague, or inconsistent.
  Low: little evidence of planning, monitoring, self-checking, or deliberate strategy choice.
- Attribution / Beliefs: Attribution / Beliefs (원인 인식)
  Representative scale: Attribution Scale
  Key indicators: interpretation of success and failure causes.
  High: the learner clearly explains causes of success or difficulty in a thoughtful way, linking outcomes to effort, strategy use, preparation, task conditions, or controllable factors.
  Mid: the learner gives a simple or partly developed cause, but the interpretation is limited or only somewhat reflective.
  Low: the learner shows little causal reflection, gives no reason, or uses very fixed or unexamined beliefs about success and failure.

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
- Strategy Type (Oxford's six language learning strategies)
  Memory: storing and retrieving language through review, grouping, imagery, association, or repetition.
  Compensation: overcoming knowledge gaps by guessing, paraphrasing, using gestures, or using simpler language.
  Cognitive: practicing, analyzing, summarizing, note-taking, repeating, or manipulating language directly.
  Metacognitive: planning, monitoring, organizing, evaluating, and managing learning.
  Social: asking questions, cooperating with others, and learning through interaction.
  Affective: controlling feelings, reducing anxiety, encouraging oneself, and managing emotions.
- Strategy Quality
  Effective: strategy matches the goal and shows regulation.
  Limited: strategy is simple, narrow, repetitive, or only partly matched to the task.
  Avoidant: response relies mainly on withdrawal or avoidance instead of a helpful learning strategy.
"""

PERSONALIZED_SELF_QUESTION_SPECS = {
    "q2": {
        "state_targets": "Understanding depth / evidence",
        "criteria_focus": (
            "Generate a deeper understanding follow-up question based on the learner's answer to Q1. "
            "Ask the learner to explain an important detail, cause, turning point, reason, or evidence from the passage."
        ),
        "note": "This follow-up is linked to your answer in Q1 and asks for deeper understanding.",
    },
    "q4": {
        "state_targets": "Anxiety / Motivation / Enjoyment",
        "criteria_focus": (
            "Generate a self question for a similar real-life situation that connects the learner's character-emotion answer "
            "to their own emotional state. It should help reveal anxiety, motivation, and enjoyment without naming technical labels."
        ),
        "note": "This follow-up is linked to your answer in Q3 and checks your emotion in a similar situation.",
    },
    "q6": {
        "state_targets": "Self-efficacy / Metacognition / Attribution-Beliefs",
        "criteria_focus": (
            "Generate a self question for a similar real-life situation that connects the learner's character-thinking answer "
            "to their own confidence, self-awareness, planning, noticing problems, regulation, and beliefs about why success or difficulty happens."
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
        "state_targets": "Oxford Strategy Type / Strategy Quality",
        "criteria_focus": (
            "Generate a self question for a similar real-life situation that connects the learner's character-strategy answer "
            "to their own learning strategies. It should help reveal one or more Oxford strategy types and the quality of the strategy choice without naming the labels."
        ),
        "note": "This follow-up is linked to your answer in Q9 and checks your strategy use in a similar situation.",
    },
}

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

    system_prompt = load_prompt_template("generate_experiment", "system.txt")
    user_prompt = load_prompt_template(
        "generate_experiment",
        "user.txt",
        cefr_level=cefr_level,
        student_name=student_name,
        student_number=student_number,
        variation_seed=variation_seed,
        selected_background=selected_background,
        selected_genre=selected_genre,
        question_generation_criteria=QUESTION_GENERATION_CRITERIA,
        question_slot_instructions=question_slot_instructions,
    )

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
        raise ValueError("OpenAI API key is required to generate the passage and questions.")

    try:
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
    except Exception as exc:
        raise RuntimeError(
            "OpenAI generation failed. Please check the API key, model setting, and prompt output format, then try again."
        ) from exc


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


def build_basic_answer_feedback(answer: str) -> dict:
    stripped = answer.strip()
    lower = stripped.lower()

    if not stripped:
        return {
            "status": "rewrite",
            "message": "This answer is incomplete. Please write something for this question.",
            "message_ko": "이 답변은 불완전합니다. 이 질문에 대해 내용을 적어 주세요.",
        }

    if not re.search(r"[A-Za-z]", stripped):
        return {
            "status": "rewrite",
            "message": "This answer is incomplete. Please rewrite it with a simple English answer.",
            "message_ko": "이 답변은 불완전합니다. 간단한 영어 답변으로 다시 작성해 주세요.",
        }

    idk_patterns = [
        "i don't know",
        "i do not know",
        "dont know",
        "idk",
        "no idea",
    ]
    weird_short_patterns = [
        "?",
        "??",
        "...",
        "..",
        "what",
        "huh",
    ]

    if contains_any(lower, idk_patterns):
        return {
            "status": "rewrite",
            "message": "This answer is incomplete because it says I don't know. Please try to answer in simple English.",
            "message_ko": "이 답변은 I don't know라고 되어 있어서 불완전합니다. 쉬운 영어로라도 답해 주세요.",
        }

    if lower in weird_short_patterns or (word_count(stripped) == 1 and len(stripped) <= 2):
        return {
            "status": "rewrite",
            "message": "This answer looks too unclear to use. Please rewrite it with a simple English answer.",
            "message_ko": "이 답변은 사용하기에 너무 모호합니다. 쉬운 영어 답변으로 다시 작성해 주세요.",
        }

    return {
        "status": "valid",
        "message": "Recorded. You can continue.",
        "message_ko": "기록되었습니다. 계속 진행할 수 있습니다.",
    }


def build_unavailable_answer_feedback() -> dict:
    return {
        "status": "valid",
        "message": "LLM feedback is temporarily unavailable. Your answer is still recorded.",
        "message_ko": "LLM 피드백을 일시적으로 사용할 수 없지만 답변은 기록됩니다.",
    }


def generate_answer_feedback_with_openai(
    questionnaire: dict,
    question: dict,
    answer: str,
    api_key: str,
    model: str,
) -> dict:
    client = get_openai_client(api_key)

    system_prompt = load_prompt_template("answer_feedback", "system.txt")
    user_prompt = load_prompt_template(
        "answer_feedback",
        "user.txt",
        story_title=questionnaire["story_title"],
        passage_plain=questionnaire["passage_plain"],
        question_layer=question["layer"],
        question_type=question["type"],
        question_prompt=question["prompt"],
        answer=answer,
    )

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
    basic_feedback = build_basic_answer_feedback(answer)

    if basic_feedback["status"] == "rewrite":
        return basic_feedback

    if not api_key:
        return build_unavailable_answer_feedback()

    try:
        return generate_answer_feedback_with_openai(
            questionnaire=questionnaire,
            question=question,
            answer=answer,
            api_key=api_key,
            model=model,
        )
    except Exception:
        return build_unavailable_answer_feedback()


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


def build_base_personalized_self_prompt(question: dict) -> dict:
    spec = get_personalized_self_spec(question["id"])

    return {
        "prompt": question.get("base_prompt", question.get("prompt", "")),
        "prompt_ko": question.get("base_prompt_ko", question.get("prompt_ko", "")),
        "prompt_source": "base_prompt",
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

    system_prompt = load_prompt_template("personalized_self_prompt", "system.txt")
    user_prompt = load_prompt_template(
        "personalized_self_prompt",
        "user.txt",
        story_title=questionnaire["story_title"],
        passage_plain=questionnaire["passage_plain"],
        question_layer=question["layer"],
        question_type=question["type"],
        state_targets=spec.get("state_targets", ""),
        criteria_focus=spec.get("criteria_focus", ""),
        base_prompt=question.get("base_prompt", question.get("prompt", "")),
        source_question_prompt=source_question["prompt"],
        source_answer=source_answer,
    )

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
        return build_base_personalized_self_prompt(question)

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
        return build_base_personalized_self_prompt(question)


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

        feedback_map[question["id"]] = get_answer_feedback(
            questionnaire_json=questionnaire_json,
            question_json=json.dumps(question, ensure_ascii=False, sort_keys=True),
            answer=answer,
            api_key=api_key,
            model=model,
        )

    return feedback_map


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


def analyze_fluency_features(text: str) -> dict:
    wc = word_count(text)
    sc = sentence_count(text)
    connectors = detect_connectors(text)
    structure_markers = detect_structure_markers(text)
    organization_markers = detect_organization_markers(text)
    strategy_expressions = detect_strategy_expressions(text)

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
        "overall_label": overall_label,
    }


def fluency_label(text: str) -> str:
    return analyze_fluency_features(text)["overall_label"]


def evaluate_fluency(answers: dict) -> dict:
    analyses = [
        analyze_fluency_features(answer)
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
            "Fluency_state": (
                "Fluency: Low / Sentence length: Low / Connector use: Low / "
                "Structure complexity: Low / Organization: Low / Strategy expression: Low"
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

    averaged_non_low = sum(
        label != "Low"
        for label in [
            sentence_length_label,
            connector_label,
            structure_label,
            organization_label,
            strategy_label,
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
        "Fluency_state": (
            f"Fluency: {overall_label} / Sentence length: {sentence_length_label} / "
            f"Connector use: {connector_label} / Structure complexity: {structure_label} / "
            f"Organization: {organization_label} / Strategy expression: {strategy_label}"
        ),
        "Fluency_connector_examples": ", ".join(connector_examples),
        "Fluency_feature_note": (
            "Based on sentence length, connector use, structure complexity, "
            "organization, and strategy expression across the learner's responses."
        ),
    }


def classify_emotion(text: str) -> tuple:
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
        "mistake",
        "judge",
        "ashamed",
        "fear",
        "negative evaluation",
    ]
    anxiety_control_keywords = [
        "but i try",
        "but i still try",
        "try",
        "prepare",
        "practice",
        "check",
        "manage",
        "calm down",
    ]
    calm_keywords = ["comfortable", "relaxed", "calm", "fine", "okay"]

    motivation_high_keywords = [
        "my goal",
        "i want to improve",
        "i want to get better",
        "i want to speak better",
        "for my future",
        "for my dream",
        "important for me",
        "i really want",
        "i want to use english well",
        "i want to learn more",
        "i keep trying",
    ]
    motivation_mid_keywords = [
        "i want to",
        "i need to",
        "i should",
        "i have to",
        "i try",
        "i will try",
        "it is useful",
        "it helps me",
    ]
    motivation_low_keywords = [
        "i do not want",
        "i don't want",
        "no reason",
        "not important",
        "i do not care",
        "i don't care",
        "give up",
        "not interested",
    ]

    enjoyment_high_keywords = [
        "enjoy",
        "fun",
        "interesting",
        "happy",
        "excited",
        "proud",
        "achievement",
        "i like",
        "i love",
        "supported",
    ]
    enjoyment_mid_keywords = [
        "okay",
        "sometimes enjoyable",
        "sometimes fun",
        "a little fun",
        "not bad",
        "sometimes interesting",
    ]
    enjoyment_low_keywords = [
        "boring",
        "do not like",
        "don't like",
        "not interesting",
        "hate",
        "uncomfortable",
        "not enjoyable",
    ]

    anxiety_score = count_matches(lower_text, anxiety_keywords)
    has_avoidance = contains_any(lower_text, ["avoid", "stay quiet", "silent", "do not speak", "don't speak"])
    has_anxiety_control = anxiety_score > 0 and contains_any(lower_text, anxiety_control_keywords)

    if anxiety_score >= 2 or (anxiety_score >= 1 and has_avoidance):
        anxiety = "High"
    elif anxiety_score >= 1 or has_anxiety_control:
        anxiety = "Mid"
    elif contains_any(lower_text, calm_keywords):
        anxiety = "Low"
    else:
        anxiety = "Low"

    if contains_any(lower_text, motivation_low_keywords):
        motivation = "Low"
    elif count_matches(lower_text, motivation_high_keywords) >= 1 or (contains_any(lower_text, ["goal", "future", "dream"]) and contains_any(lower_text, ["improve", "better", "learn"])):
        motivation = "High"
    elif contains_any(lower_text, motivation_mid_keywords):
        motivation = "Mid"
    else:
        motivation = "Low"

    enjoyment_score = count_matches(lower_text, enjoyment_high_keywords)
    if enjoyment_score >= 2:
        enjoyment = "High"
    elif enjoyment_score >= 1 or contains_any(lower_text, enjoyment_mid_keywords):
        enjoyment = "Mid"
    elif contains_any(lower_text, enjoyment_low_keywords) or anxiety_score >= 1:
        enjoyment = "Low"
    else:
        enjoyment = "Low"

    return anxiety, motivation, enjoyment


def classify_cognition(text: str) -> tuple:
    lower_text = text.lower()

    high_selfeff = [
        "i can speak well",
        "i am confident",
        "i can do it",
        "i can answer",
        "i can explain",
        "i can speak in english",
        "i can handle it",
    ]
    mid_selfeff = [
        "i can, but",
        "i can but",
        "simple english",
        "a little confident",
        "sometimes i can",
        "i can try",
        "if i prepare",
        "with help",
    ]
    low_selfeff = [
        "i cannot",
        "i can't",
        "i am not good",
        "i will make mistakes",
        "i am bad",
        "too difficult",
        "hard for me",
        "i am not good at english",
        "i cannot explain",
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
        "i notice",
        "i choose",
        "i will use",
    ]
    meta_mid = [
        "i know",
        "i think i need",
        "sometimes",
        "maybe",
        "i should",
        "i need to",
    ]

    attribution_high_keywords = [
        "because i practiced",
        "because i prepare",
        "because i prepared",
        "because i studied",
        "because of my effort",
        "because of my strategy",
        "because i did not prepare",
        "because i was nervous",
        "because the task was difficult",
        "because i used",
        "the reason is",
    ]
    attribution_mid_keywords = [
        "because",
        "so",
        "that is why",
        "the reason",
        "if i practice",
        "if i prepare",
    ]
    attribution_low_keywords = [
        "i don't know why",
        "just because",
        "no reason",
    ]

    if contains_any(lower_text, low_selfeff):
        self_efficacy = "Low"
    elif contains_any(lower_text, high_selfeff):
        self_efficacy = "High"
    elif contains_any(lower_text, mid_selfeff):
        self_efficacy = "Mid"
    else:
        self_efficacy = "Mid"

    if count_matches(lower_text, meta_high) >= 2 or contains_any(lower_text, ["plan", "check", "monitor", "review"]) and contains_any(lower_text, ["problem", "mistake", "improve"]):
        metacognition = "High"
    elif contains_any(lower_text, meta_high) or contains_any(lower_text, meta_mid):
        metacognition = "Mid"
    else:
        metacognition = "Low"

    if contains_any(lower_text, attribution_low_keywords):
        attribution_beliefs = "Low"
    elif count_matches(lower_text, attribution_high_keywords) >= 1 or (contains_any(lower_text, ["because", "reason", "why"]) and contains_any(lower_text, ["practice", "prepare", "effort", "strategy", "mistake", "difficult"])):
        attribution_beliefs = "High"
    elif contains_any(lower_text, attribution_mid_keywords):
        attribution_beliefs = "Mid"
    else:
        attribution_beliefs = "Low"

    return self_efficacy, metacognition, attribution_beliefs


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

    strategy_keywords = {
        "Memory": ["memorize", "memory", "repeat", "review words", "group words", "association", "image"],
        "Compensation": ["guess", "gesture", "simple words", "easy words", "use other words", "paraphrase"],
        "Cognitive": ["practice", "write", "note", "read again", "summarize", "analyze", "translate"],
        "Metacognitive": ["plan", "check", "monitor", "prepare", "set a goal", "review", "organize", "evaluate"],
        "Social": ["ask friends", "ask teacher", "talk with friends", "study group", "peer", "ask questions", "work with others"],
        "Affective": ["calm", "relax", "positive", "encourage", "breathe", "feel better", "reduce anxiety"],
    }
    avoidant_keywords = ["avoid", "give up", "stay quiet", "do nothing", "ignore"]
    limited_keywords = ["only memorize", "just memorize", "just repeat only"]
    effective_keywords = ["helps", "help me", "useful", "work well", "improve", "better"]

    scores = {
        strategy_name: count_matches(lower_text, keywords)
        for strategy_name, keywords in strategy_keywords.items()
    }
    max_score = max(scores.values()) if scores else 0

    if max_score <= 0:
        strategy_type = "Unclear"
        matched_types = []
    else:
        matched_types = [name for name, score in scores.items() if score == max_score and score > 0]
        priority = ["Metacognitive", "Cognitive", "Compensation", "Social", "Affective", "Memory"]
        strategy_type = next(name for name in priority if name in matched_types)

    if contains_any(lower_text, avoidant_keywords):
        strategy_quality = "Avoidant"
    elif strategy_type == "Unclear":
        strategy_quality = "Limited"
    elif contains_any(lower_text, limited_keywords):
        strategy_quality = "Limited"
    elif len(matched_types) >= 2 or contains_any(lower_text, effective_keywords) or contains_any(lower_text, ["because it helps", "so i can", "to improve"]):
        strategy_quality = "Effective"
    else:
        strategy_quality = "Limited"

    return strategy_type, strategy_quality


def evaluate_state(answers: dict) -> dict:
    fluency = evaluate_fluency(answers)
    anxiety, motivation, enjoyment = classify_emotion(answers.get("q4", ""))
    self_efficacy, metacognition, attribution_beliefs = classify_cognition(answers.get("q6", ""))
    wtc, coping, engagement = classify_behavior(answers.get("q8", ""))
    strategy_type, strategy_quality = classify_strategy(answers.get("q10", ""))

    return {
        "evaluation_method": "criteria_rules_v4_dynamic_cefr",
        "Fluency": fluency["Fluency"],
        "Fluency_sentence_length": fluency["Fluency_sentence_length"],
        "Fluency_connector_use": fluency["Fluency_connector_use"],
        "Fluency_structure_complexity": fluency["Fluency_structure_complexity"],
        "Fluency_organization": fluency["Fluency_organization"],
        "Fluency_strategy_expression": fluency["Fluency_strategy_expression"],
        "Fluency_state": fluency["Fluency_state"],
        "Fluency_connector_examples": fluency["Fluency_connector_examples"],
        "Fluency_feature_note": fluency["Fluency_feature_note"],
        "FLA": anxiety,
        "Motivation": motivation,
        "FLE": enjoyment,
        "Emotion_state": f"Anxiety: {anxiety} / Motivation: {motivation} / Enjoyment: {enjoyment}",
        "Self_efficacy": self_efficacy,
        "Metacognition": metacognition,
        "Attribution_beliefs": attribution_beliefs,
        "Cognition_state": f"Self-efficacy: {self_efficacy} / Metacognition: {metacognition} / Attribution-Beliefs: {attribution_beliefs}",
        "WTC": wtc,
        "Coping": coping,
        "Engagement": engagement,
        "Behavior_state": f"WTC: {wtc} / Coping: {coping} / Engagement: {engagement}",
        "Strategy_type": strategy_type,
        "Strategy_quality": strategy_quality,
        "Strategy_state": f"Oxford Strategy Type: {strategy_type} / Strategy Quality: {strategy_quality}",
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
) -> dict:
    evaluation = evaluate_state(answers)

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
        "Motivation": evaluation["Motivation"],
        "FLE": evaluation["FLE"],
        "Emotion_state": evaluation["Emotion_state"],
        "cognition_layer": "Cognition",
        "Self_efficacy": evaluation["Self_efficacy"],
        "Metacognition": evaluation["Metacognition"],
        "Attribution_beliefs": evaluation["Attribution_beliefs"],
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


def build_answers_snapshot(questionnaire: dict) -> dict:
    saved_answers = get_saved_answers()
    snapshot = {}

    for question in get_all_questions(questionnaire):
        snapshot[question["id"]] = st.session_state.get(
            widget_key(question["id"]),
            saved_answers.get(question["id"], ""),
        ).strip()

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
    st.write(f"Background / 배경: `{st.session_state.selected_background_value}`")
    st.write(f"Genre / 장르: `{st.session_state.selected_genre_value}`")
    st.write(f"Submission ID: `{st.session_state.last_submission_id}`")
    st.write(f"Storage backend: `{st.session_state.last_storage_backend}`")
    st.write(f"Responses destination: `{st.session_state.last_response_file}`")
    st.write(f"Evaluations destination: `{st.session_state.last_evaluation_file}`")

    if st.button("Start New Participant", use_container_width=True):
        reset_session_state()
        st.rerun()

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
        except Exception as exc:
            st.error(str(exc))
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
            st.rerun()

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
        f"Worksheets: `{google_sheets_config['responses_worksheet']}` and `{google_sheets_config['evaluations_worksheet']}`"
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
        "The app will generate the CEFR passage, interactive questions, answer checks, and personalized self follow-ups with OpenAI when possible."
    )
else:
    st.warning(
        "LLM mode: Inactive\n\n"
        "OpenAI API key is not configured. Passage generation, answer feedback, and personalized follow-up generation require OpenAI to be enabled."
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
        current_questionnaire = materialize_questionnaire_for_answers_cached(
            questionnaire_json=base_questionnaire_json,
            answers_json=base_answers_json,
            api_key=llm_config["api_key"],
            model=llm_config["model"],
        )
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
        feedback_map = build_answer_feedback_map(
            questionnaire_json=questionnaire_json,
            answers_json=current_answers_json,
            api_key=llm_config["api_key"],
            model=llm_config["model"],
        )
    st.session_state.feedback_map = feedback_map
    st.session_state.feedback_signature = feedback_signature

st.caption(
    f"Question status: {current_questionnaire.get('question_generation_note', '')}"
)
st.info(
    "Answer guide: Every response must be written in English. "
    "Short answers are allowed. The app only asks for a rewrite when the answer is empty, says I don't know, or is too unclear to use. "
    "In each part, Question 1 appears first, and Question 2 is generated from the answer to Question 1. "
    "After editing a part, click Check This Part to refresh the feedback and unlock the next follow-up question.\n\n"
    "답변 가이드: 짧은 답변도 가능합니다. 답이 비어 있거나 I don't know이거나 사용하기 어려울 만큼 이상한 경우에만 다시 쓰라는 안내가 나옵니다. "
    "각 파트에서는 1번 질문이 먼저 나오고, 2번 질문은 1번 답변을 보고 생성됩니다. "
    "답을 수정한 뒤에는 Check This Part 버튼을 눌러 피드백과 다음 질문을 새로 반영해 주세요."
)

passage_col, question_col = st.columns([0.9, 1.3], gap="large")

previous_clicked = False
check_part_clicked = False
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

    with st.form(
        key=f"question_section_form_{st.session_state.generation_nonce}_{current_section_index}",
        clear_on_submit=False,
    ):
        render_section_header(current_section)
        st.caption(f"Part {current_section_index + 1} of {len(current_questionnaire['sections'])}")

        for question in current_section["questions"]:
            question_key = widget_key(question["id"])
            if not question.get("ready_for_answer", True):
                st.session_state[question_key] = ""
                current_answers[question["id"]] = ""
            elif question_key not in st.session_state:
                st.session_state[question_key] = current_answers.get(question["id"], "")
            render_question_card(question)

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
                render_status_box(
                    "status-box-neutral",
                    "This follow-up will open after the previous answer can be used to create the next question.",
                )
                continue

            if not answer:
                render_status_box(
                    "status-box-neutral",
                    "Write an answer for this question, then click Check This Part.",
                )
                continue

            feedback = feedback_map.get(question["id"])
            if feedback and feedback["status"] == "valid":
                section_completed_questions += 1
                render_status_box(
                    "status-box-valid",
                    f"Complete. {feedback['message']} / {feedback['message_ko']}",
                )
            elif feedback:
                render_status_box(
                    "status-box-rewrite",
                    f"{feedback['message']} / {feedback['message_ko']}",
                )
            else:
                render_status_box(
                    "status-box-neutral",
                    "Click Check This Part to refresh the answer check.",
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
        st.caption(f"Current part: {section_completed_questions} / {section_total_questions}")
        st.progress(section_completed_questions / section_total_questions if section_total_questions else 0.0)
        st.caption(f"Overall: {completed_questions} / {total_questions}")
        st.progress(completed_questions / total_questions if total_questions else 0.0)

        if not section_complete:
            st.caption("Finish this part first. Then the Next button will be available.")

        nav_col1, nav_col2, nav_col3 = st.columns(3)
        with nav_col1:
            previous_clicked = st.form_submit_button(
                "Previous Part",
                use_container_width=True,
                disabled=(current_section_index == 0),
            )
        with nav_col2:
            check_part_clicked = st.form_submit_button(
                "Check This Part",
                use_container_width=True,
            )
        with nav_col3:
            if current_section_index < len(current_questionnaire["sections"]) - 1:
                next_clicked = st.form_submit_button(
                    "Next Part",
                    use_container_width=True,
                    disabled=(not section_complete or profile_changed_after_generation),
                )
            else:
                submit_clicked = st.form_submit_button(
                    "Submit Responses",
                    use_container_width=True,
                    disabled=(not section_complete or profile_changed_after_generation),
                )

form_submitted = previous_clicked or check_part_clicked or next_clicked or submit_clicked

if form_submitted:
    for question in current_section["questions"]:
        question_key = widget_key(question["id"])
        if not question.get("ready_for_answer", True):
            st.session_state[question_key] = ""
            current_answers[question["id"]] = ""
            set_saved_answer(question["id"], "")
            continue

        normalized_answer = st.session_state.get(question_key, "").strip()
        st.session_state[question_key] = normalized_answer
        current_answers[question["id"]] = normalized_answer
        set_saved_answer(question["id"], normalized_answer)


if previous_clicked:
    st.session_state.current_section_index = max(0, current_section_index - 1)
    st.rerun()

if next_clicked:
    st.session_state.current_section_index = min(
        len(current_questionnaire["sections"]) - 1,
        current_section_index + 1,
    )
    st.rerun()


if submit_clicked:
    if (
        st.session_state.student_name_input.strip() != st.session_state.student_name_value
        or st.session_state.student_number_input.strip() != st.session_state.student_number_value
        or st.session_state.cefr_level_input != st.session_state.cefr_level_value
        or st.session_state.selected_background_input != st.session_state.selected_background_value
        or st.session_state.selected_genre_input != st.session_state.selected_genre_value
    ):
        st.error("Participant information or generation options changed after generation. Please generate the passage and questions again before submitting.")
    else:
        validation_error = validate_submission_answers(
            questionnaire=current_questionnaire,
            answers=current_answers,
            llm_config=llm_config,
            feedback_map=feedback_map,
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
