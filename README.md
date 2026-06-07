# Agro Sandesh Article Writer

Streamlit app for researching current agriculture topics in Gujarat and drafting Gujarati extension articles with a Swaminathan-inspired, farmer-centric voice.

## Local run

```powershell
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

Put your Gemini API key in `.env` as `GEMINI_API_KEY`, or paste it in the app sidebar.

## Streamlit Community Cloud

Use `app.py` as the app entry point and add `GEMINI_API_KEY` in the app secrets/settings.
