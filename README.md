# CEFR Interactive Diagnostic

A Streamlit app for CEFR-based passage generation, interactive questionnaire delivery, answer-quality checking, and state evaluation for English education experiments.

## What This App Does

- Collects participant name, student number, CEFR level, preferred background, and text genre.
- Generates a CEFR-matched passage with exactly 10 sentences.
- Generates 10 interactive questions across understanding, emotion, cognition, behavior, and strategy.
- Personalizes self questions based on previous character-response answers.
- Prompts the participant to rewrite incomplete, vague, or `I don't know` style answers.
- Evaluates fluency, emotion, cognition, behavior, and strategy signals.
- Saves responses and evaluation results to Google Sheets or local CSV.

## Run Locally

```bash
cd edu_ui
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Use `app.py` as the Streamlit entrypoint when deploying this repository or this `edu_ui` folder as the app root.

Recommended deployment target:

- Streamlit Community Cloud

Recommended app settings:

- Main file path: `app.py`
- Branch: `main`
- Python dependencies: `requirements.txt`

## Required and Recommended Secrets

### OpenAI

Strongly recommended for the intended deployed behavior.

With OpenAI configured, the app will:

- Generate the CEFR passage
- Generate the interactive question set
- Check whether answers are incomplete or need rewriting
- Generate personalized self follow-up questions
- Better reflect the selected background and genre in generated passages

Without OpenAI, the app will fall back to built-in passages, question templates, and rule-based answer checks.
The fallback path also supports the selected background and genre and keeps the 10-sentence passage format.

### Google Sheets

Strongly recommended for deployed apps.

If Google Sheets is not configured, the app falls back to local CSV files under `outputs/`. That is acceptable for local testing, but it is not reliable for cloud deployment because local app storage may be ephemeral.

Use the example file below when setting deployment secrets:

- `.streamlit/secrets.example.toml`

See also:

- `GOOGLE_SHEETS_SETUP.md`

## Deployment Checklist

1. Push this repo to GitHub.
2. Create or open the Streamlit app connected to the repo.
3. Set the entrypoint to `app.py`.
4. Paste secrets from `.streamlit/secrets.example.toml` into the deployment secrets manager.
5. Add an OpenAI API key.
6. Add Google Sheets service account and sheet settings if you need persistent response storage.
7. Redeploy or reboot the app.
