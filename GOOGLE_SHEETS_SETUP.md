# Google Sheets Setup

This app can save responses and evaluations directly to Google Sheets.

## 1. Create a Google service account

1. Open Google Cloud Console.
2. Enable the Google Sheets API for your project.
3. Create a service account.
4. Generate a JSON key.

## 2. Share your Google Sheet

1. Create a Google Sheet for the project.
2. Copy the spreadsheet ID from the URL.
3. Share the sheet with the service account email as an editor.

## 3. Add Streamlit secrets

In Streamlit Community Cloud:

1. Open your app.
2. Go to `Manage app`.
3. Open `Secrets`.
4. Paste the values from `.streamlit/secrets.example.toml`.
5. Replace the example values with your real service account JSON fields and spreadsheet ID.

The app supports both of these secrets formats:

1. Recommended:
   - `[google_service_account]`
   - `[google_sheets]`
2. Compatible alternative:
   - `[connections.gsheets]`

For the spreadsheet target, you can use either:
- `spreadsheet_id = "your-sheet-id"`
- `spreadsheet_url = "https://docs.google.com/spreadsheets/d/.../edit"`

## 4. Worksheets used by the app

- `questionnaire_responses`
- `state_evaluations`

The app will create these worksheets automatically if they do not exist.
