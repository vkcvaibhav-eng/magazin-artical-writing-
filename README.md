# Agro Sandesh Gujarati Agriculture Article Writer

A Streamlit app that drafts, reviews, and polishes Gujarati agricultural
magazine articles using a multi-AI workflow (Perplexity/Gemini for research,
Gemini for Gujarati drafting, OpenAI/Gemini for quality review).

## Krushi Prabhat output

- Article-length choices include 700, 800, 900, and 1000 words (plus the existing
  longer formats for other magazines).
- Select **700 words** for an official Krushi Prabhat submission; the publication
  notice sets 700 words as the maximum. The longer choices are working-draft formats.
- Final articles show a word-count and Gujarati Unicode compliance check.
- Downloaded Word files are editable DOCX files using **Nirmala UI** with Gujarati
  (`gu-IN`) Unicode language metadata.

## District-first crop and pest evidence

Topic discovery is now district-first. The user selects one of Gujarat's 34
districts (including Vav-Tharad), and may supply an actual sowing/transplanting
date, a known crop stage, and a field/weather observation. Before suggesting a
topic, deep research must produce a `DISTRICT_CROP_EVIDENCE` section and verify:

1. crop presence from Gujarat Directorate of Economics & Statistics records;
2. sowing/phenology from the ICAR-CRIDA district plan or a current official advisory;
3. current weather from IMD district agromet information;
4. pest occurrence from SAU/KVK/NPSS surveillance or advisory evidence.

Every topic is labelled **Seasonal possibility**, **Pest watch**, or
**Confirmed alert**. Weather can raise monitoring priority but cannot, by itself,
confirm a pest outbreak. If an official crop record is unavailable, the app asks
the research model to state a data gap rather than inventing crop area or rank.
The parsed government-recorded crop shortlist is selectable before the user chooses
an article topic.
Krushi Go-Vidya is retained only as a secondary editorial comparison, not the
primary crop calendar. Older Vav-Tharad crop records are treated as a clearly
labelled Banaskantha legacy baseline rather than current district totals.

## Topic-based management evidence

After selecting a suggested topic in any writing tab, the app now shows:

1. **PPQS / CIB&RC Label Claim Checker** — search and select only the
   label-claim pesticides allowed in that topic's article.
2. **Gujarat University Recommendations (AGRESCO)** — search and select the
   official university recommendations to include.

The structured topic list supplies English crop and pest search terms. Evidence
choices are stored separately for each tab and reset automatically when its
selected topic changes.

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
