# Agro Sandesh Gujarati Agriculture Article Writer

A Streamlit app that drafts, reviews, and polishes Gujarati agricultural
magazine articles using a multi-AI workflow (Perplexity/Gemini for research,
Gemini for Gujarati drafting, OpenAI/Gemini for quality review).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Provide API keys via environment variables or `.streamlit/secrets.toml`:

- `GEMINI_API_KEY` (required)
- `PERPLEXITY_API_KEY` (for deep research)
- `OPENAI_API_KEY` (for quality review)

## Trusted chemical / recommendation sources

The app can ground its chemical advice in two official sources so the article
never invents pesticide doses:

### 1. PPQS / CIB&RC label claims
In the **PPQS / CIB&RC Label Claim Checker** you can either:
- Click **Fetch Major Uses document list from PPQS website** to load the latest
  Major Uses PDFs directly from
  [ppqs.gov.in](https://ppqs.gov.in/divisions/cib-rc/major-uses-of-pesticides),
  or
- Upload the PDF manually.

Search by crop and pest; the best label-claim molecules are auto-selected and
fed to the article as the only allowed chemicals.

### 2. Gujarat University (AGRESCO) recommendations
`agresco_recommendations.json` holds the "Recommendations for Farming Community"
extracted from Gujarat Combined AGRESCO proceedings. In the app, the **Gujarat
University Recommendations (AGRESCO)** panel matches these official recommendations
to your crop/problem and shares them with the article as trusted guidance.

#### Adding more years of proceedings

1. Put every proceedings PDF (2011 to date) in one folder.
2. Run:

   ```bash
   python extract_agresco_recommendations.py path/to/folder agresco_recommendations.json
   ```

3. Commit the updated `agresco_recommendations.json`. The full PDFs are **not**
   committed — only the compact extracted recommendations.
