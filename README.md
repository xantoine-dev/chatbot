# 💬 Chatbot template

A simple Streamlit app that shows how to build a chatbot using OpenAI's GPT-3.5.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

Environment / Gemini notes
--------------------------

This project can use either OpenAI or Google Gemini (via the `google-genai` client). To use Gemini:

1. Install the dependencies (the `google-genai` package is listed in `requirements.txt`).

2. Provide your Gemini API key via the `GEMINI_API_KEY` environment variable or paste it into the app when selecting the "Gemini (Google)" provider.

Example (Linux/macOS):

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
streamlit run streamlit_app.py
```

Or create a local `.env` file (copy `.env.example`) and set the key there if you use a dotenv loader.

