import os
import re
import zipfile
from datetime import datetime
from html import escape
from io import BytesIO

import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

REGIONS = [
    "South Gujarat",
    "Whole Gujarat",
    "India with Gujarat relevance",
]

SUBJECT_AREAS = [
    "Agricultural acarology",
    "Agricultural entomology",
    "Mite pests in crops",
    "Insect pest management",
    "Integrated pest management and natural enemies",
    "Climate-linked pest outbreak",
]

ARTICLE_LENGTHS = [
    "1000 words",
    "1200 words",
    "1500 words",
]

MAGAZINE_OPTIONS = [
    "Krushi Vigyan",
    "Krushi Go-Vidya",
    "Krushi Jivan",
    "Krishi Jagran Gujarati",
    "Krushi Prabhat",
    "Agro Sandesh",
    "Gujarati farmer magazine",
    "Gujarati long-form agricultural magazine",
]

MAGAZINE_STYLE_NOTES = {
    "Krishi Jagran Gujarati": (
        "Digital Gujarati agriculture news/explainer style. Use a strong clickable "
        "title, short intro, current relevance, simple explanation, practical "
        "farmer benefit, 4-6 subheadings, and an active timely tone. Avoid thesis "
        "style and slow academic introductions."
    ),
    "Krushi Go-Vidya": (
        "University extension advisory style. Keep the tone scientific, trustworthy, "
        "and farmer-useful. Include crop-stage relevance, symptoms or observations, "
        "simple scientific reason, locally applicable recommendations, precautions, "
        "and local university/KVK verification where useful."
    ),
    "Krushi Jivan": (
        "Scientist-to-farmer Gujarati monthly magazine style. Explain latest research, "
        "new technology, nutrient management, plant protection, soil, water, dairy, "
        "or broad farmer education in a balanced, credible, non-promotional voice."
    ),
    "Krushi Prabhat": (
        "Daily agriculture newspaper style. Keep the article shorter, timely, direct, "
        "and news-oriented. Start with the main point, then farmer relevance, "
        "region/crop connection, and immediate practical advisory. Avoid long background."
    ),
    "Krushi Vigyan": (
        "Practical field-solution Gujarati magazine style. Begin with a field problem, "
        "explain the cause, give crop-stage-wise practical solutions, include farmer "
        "benefit, and keep the tone scientific but directly useful."
    ),
    "Agro Sandesh": (
        "Farmer-centric Gujarati agriculture magazine style with practical extension "
        "guidance, simple science, local relevance, and a hopeful field-oriented voice."
    ),
    "Gujarati farmer magazine": (
        "General Gujarati farmer magazine style. Keep it simple, practical, field-based, "
        "and useful for farmers, extension workers, agriculture students, and growers."
    ),
    "Gujarati long-form agricultural magazine": (
        "Long-form Gujarati agricultural feature style. Use scene, observation, simple "
        "science, practical meaning, and polished magazine flow."
    ),
}

PROVIDER_GEMINI = "Gemini"
PROVIDER_PERPLEXITY = "Perplexity"
PROVIDER_OPENAI = "OpenAI"

PROVIDER_KEY_ENV = {
    PROVIDER_GEMINI: "GEMINI_API_KEY",
    PROVIDER_PERPLEXITY: "PERPLEXITY_API_KEY",
    PROVIDER_OPENAI: "OPENAI_API_KEY",
}


st.set_page_config(
    page_title="Agro Sandesh Article Writer",
    page_icon="AS",
    layout="wide",
)


def config_value(name: str, default: str = "") -> str:
    env_value = os.getenv(name, "").strip()
    if env_value:
        return env_value

    try:
        secret_value = st.secrets.get(name, "")
    except Exception:
        secret_value = ""

    return str(secret_value or default).strip()


def get_api_keys() -> dict[str, str]:
    return {
        PROVIDER_GEMINI: config_value("GEMINI_API_KEY"),
        PROVIDER_PERPLEXITY: config_value("PERPLEXITY_API_KEY"),
        PROVIDER_OPENAI: config_value("OPENAI_API_KEY"),
    }


def missing_api_keys(selected_providers: list[str], api_keys: dict[str, str]) -> list[str]:
    missing = []
    for provider in [PROVIDER_GEMINI, PROVIDER_PERPLEXITY, PROVIDER_OPENAI]:
        if provider in selected_providers and not api_keys.get(provider):
            missing.append(f"{provider} ({PROVIDER_KEY_ENV[provider]})")
    return missing


def build_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def get_attr(obj, *names, default=None):
    for name in names:
        if obj is None:
            continue
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def extract_grounding_sources(response) -> list[dict[str, str]]:
    sources = []
    seen = set()

    candidates = get_attr(response, "candidates", default=[]) or []
    if not candidates:
        return sources

    metadata = get_attr(candidates[0], "grounding_metadata", "groundingMetadata")
    chunks = get_attr(metadata, "grounding_chunks", "groundingChunks", default=[]) or []

    for chunk in chunks:
        web = get_attr(chunk, "web", default={}) or {}
        title = get_attr(web, "title", default="Source")
        uri = get_attr(web, "uri", default="")
        if uri and uri not in seen:
            seen.add(uri)
            sources.append({"title": title or "Source", "uri": uri})

    return sources


def extract_perplexity_sources(data: dict) -> list[dict[str, str]]:
    sources = []
    seen = set()

    for result in data.get("search_results") or []:
        uri = result.get("url") or ""
        title = result.get("title") or uri or "Source"
        if uri and uri not in seen:
            seen.add(uri)
            sources.append({"title": title, "uri": uri})

    for uri in data.get("citations") or []:
        if uri and uri not in seen:
            seen.add(uri)
            sources.append({"title": uri, "uri": uri})

    return sources


def extract_openai_text(data: dict) -> str:
    if data.get("output_text"):
        return data["output_text"]

    text_parts = []
    for item in data.get("output") or []:
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                text_parts.append(content["text"])
    return "\n".join(text_parts)


def raise_for_api_error(response: requests.Response, provider: str) -> None:
    if response.ok:
        return

    try:
        detail = response.json()
    except ValueError:
        detail = response.text

    raise RuntimeError(f"{provider} API error {response.status_code}: {detail}")


def generate_gemini_text(
    client: genai.Client,
    model: str,
    prompt: str,
    *,
    use_search: bool,
    temperature: float,
):
    tools = []
    if use_search:
        tools.append(types.Tool(google_search=types.GoogleSearch()))

    config = types.GenerateContentConfig(
        tools=tools or None,
        temperature=temperature,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text or "", extract_grounding_sources(response)


def generate_perplexity_text(
    api_key: str,
    model: str,
    prompt: str,
    *,
    temperature: float,
):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if "reasoning" in model:
        payload["reasoning_effort"] = "medium"

    response = requests.post(
        "https://api.perplexity.ai/v1/sonar",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    raise_for_api_error(response, PROVIDER_PERPLEXITY)
    data = response.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return text, extract_perplexity_sources(data)


def generate_openai_text(
    api_key: str,
    model: str,
    prompt: str,
    *,
    temperature: float,
):
    payload = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
    }

    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    raise_for_api_error(response, PROVIDER_OPENAI)
    return extract_openai_text(response.json()), []


def generate_text(
    client: genai.Client,
    model: str,
    prompt: str,
    *,
    use_search: bool,
    temperature: float,
    provider: str = PROVIDER_GEMINI,
    api_keys: dict[str, str] = None,
):
    if provider == PROVIDER_GEMINI:
        return generate_gemini_text(
            client,
            model,
            prompt,
            use_search=use_search,
            temperature=temperature,
        )

    api_keys = api_keys or {}
    if provider == PROVIDER_PERPLEXITY:
        return generate_perplexity_text(
            api_keys.get(PROVIDER_PERPLEXITY, ""),
            model,
            prompt,
            temperature=temperature,
        )

    if provider == PROVIDER_OPENAI:
        return generate_openai_text(
            api_keys.get(PROVIDER_OPENAI, ""),
            model,
            prompt,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported AI provider: {provider}")


def topic_research_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
) -> str:
    return f"""
You are an agricultural research assistant for Gujarati agriculture magazines.

Use Google Search grounding to identify current, prevailing, and seasonally relevant
agriculture article topics for {month} in {region}.

Subject focus: {subject_area}
Crop focus, if any: {crop_focus or "No specific crop focus"}

Research priorities:
- South Gujarat and Gujarat agriculture
- Agricultural acarology and agricultural entomology
- Current pest and mite problems
- Seasonal crop stage and month-wise agricultural activity
- Weather-linked pest and mite risk
- Natural enemies, IPM, monitoring, and preventive action
- Official advisories, agricultural university/KVK guidance, research sources,
  and current web context where useful
- Practical advisory value for farmers
- Relevance to this month
- Suitability for an agricultural magazine article

First create a deep research pack using multiple search angles:
1. Current pest/mite relevance
2. Crop stage and seasonal activity
3. Month/weather connection
4. Gujarat and South Gujarat relevance
5. Field observations farmers may recognize
6. Scientific background in simple language
7. Natural enemies and integrated management
8. Farmer benefit and practical relevance

Return 10 topic options. For each option include:
1. Gujarati title
2. English explanation
3. Why it matters in {month}
4. Region relevance
5. Farmer benefit
6. Suitability score out of 10

Do not select the final topic automatically. The user will manually choose which
topic to write. After the 10 topic options, provide a useful research note pack
for each option so the user can compare and choose:
- Why now
- Regional/crop relevance
- Field observations
- Scientific background
- Practical management
- Farmer benefits
- Reference quality notes with source labels: official, university/KVK,
  government, research, news, or general web
- Caution notes such as "verify locally", "use cautiously", or "avoid
  overclaiming" where needed

Write clearly. Do not invent local outbreaks or official advisories. If evidence is
uncertain, say so and suggest field verification with local agricultural university,
KVK, or extension officers.
Do not write like a research paper. The research notes are for article support;
do not suggest inline citations or an academic reference section for the article.
Do not write a final recommendation such as "best topic", "selected topic", or
"write this topic"; keep the choice open for the user.
""".strip()


def article_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    article_length: str,
    target_magazine: str,
    selected_topic: str,
) -> str:
    return f"""
Write a full Gujarati agricultural extension article for {target_magazine}.

Target magazine personality:
{magazine_style_note(target_magazine)}

Important authorship instruction:
- Do not claim that Dr. M. S. Swaminathan wrote the article.
- Do not write in first person as Dr. Swaminathan.
- Use an original Gujarati voice inspired by his public communication values:
  scientific temper, farmer welfare, practical field wisdom, sustainability,
  productivity, ecological care, and hope for small and progressive farmers.

Writing architecture:
1. Begin with a real field situation observed by farmers, not a definition.
2. Explain why the issue matters economically and practically.
3. Give the scientific reason in simple farmer-friendly language.
4. Explain technical words immediately after using them.
5. Connect every scientific fact with a farmer outcome.
6. Use cause, effect, consequence, and solution as hidden thinking logic only.
7. Keep paragraphs focused on one central idea.
8. Every recommendation must naturally include what farmers should do, why it
   matters, and how it improves yield, quality, cost, risk, sustainability, or
   profit. Do this inside flowing paragraphs, not as a question-answer list.
9. Make farmers, crops, productivity, quality, profitability, and sustainability
   the main subjects of sentences.
10. Include field observations and practical examples from Indian agriculture,
    especially Gujarat or South Gujarat when relevant.
11. Avoid thesis style, literature review style, political language, and excessive jargon.
12. Avoid unsafe pesticide dosage claims unless clearly supported. When mentioning
    chemical control, advise farmers to follow label recommendations and local
    agricultural university or KVK guidance.
13. End with a positive, practical takeaway message.

Preferred flow:
Farmer problem -> scientific reason -> practical solution -> farmer benefit.

Style rule:
- Keep this flow invisible to the reader. Do not print labels such as
  "શું કરવું?", "શા માટે?", "લાભ", "મુખ્ય કારણ", "અસર", "પરિણામ",
  "ઉકેલ", "સમસ્યા", or similar checklist headings.
- Do not use bold/italic label blocks inside the article.
- Use normal Gujarati magazine paragraphs and a few natural subheadings only
  when they improve reading flow.
- The reader should feel the logic through the paragraph rhythm, not see the
  planning structure printed on the page.

Soft evidence guidance:
- Use the research notes and reference quality labels as gentle guardrails.
- Soften risky, overconfident, or locally uncertain statements.
- Do not demand a source for every sentence.
- Do not add inline citations, reference lists, or academic evidence language.
- Preserve farmer usefulness, magazine rhythm, and natural Gujarati prose.

Target publication: {target_magazine}
Language: Gujarati
Length: {article_length}
Region: {region}
Month: {month}
Subject area: {subject_area}
Crop focus: {crop_focus or "No specific crop focus"}

Selected topic and research notes:
{selected_topic}

Write the complete article with a suitable Gujarati title.
""".strip()


def review_prompt(article: str, target_magazine: str = "selected Gujarati agriculture magazine") -> str:
    return f"""
Review the following Gujarati agriculture article for {target_magazine}.

Target magazine personality:
{magazine_style_note(target_magazine)}

Check:
1. Is the opening farmer-oriented?
2. Is the science explained in simple language?
3. Are recommendations practical and actionable?
4. Does every recommendation explain farmer benefit?
5. Is the tone farmer-centric, trustworthy, evidence-based, and hopeful?
6. Does it avoid research paper, thesis, and review article style?
7. Is it suitable for farmers, extension workers, agriculture students, and progressive growers?
8. Is the Gujarati language clear and natural?
9. Are any claims risky, unsupported, or too broad?
10. Does it avoid repeated label-style blocks such as "શું કરવું?",
    "શા માટે?", "લાભ", "મુખ્ય કારણ", "અસર", "પરિણામ", and "ઉકેલ"?
11. Give a rating out of 10.

Then provide specific improvements and rewrite only weak paragraphs if needed.

Article:
{article}
""".strip()


def rewrite_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    article_length: str,
    target_magazine: str,
    selected_topic: str,
    article: str,
) -> str:
    return f"""
Rewrite the following Gujarati agriculture article into a stronger magazine-quality
article for {target_magazine}.

Target magazine personality:
{magazine_style_note(target_magazine)}

Important authorship instruction:
- Do not claim that Dr. M. S. Swaminathan wrote the article.
- Do not write in first person as Dr. Swaminathan.
- Use an original Gujarati extension-writing voice inspired by his public values:
  farmer welfare, scientific temper, field wisdom, sustainability, productivity,
  practical hope, and respect for small and progressive farmers.

Rewrite goals:
1. Make the opening more field-based and farmer-oriented.
2. Improve the flow: farmer problem -> scientific reason -> practical solution -> benefit.
   Keep this flow invisible and express it through natural Gujarati paragraphs.
3. Make each recommendation clearer, more practical, and linked to farmer profit,
   quality, yield, cost reduction, risk reduction, or long-term crop health.
4. Remove thesis-style language, repetition, and heavy jargon.
5. Explain technical terms immediately in simple Gujarati.
6. Keep scientific accuracy. Do not invent official advisories, pesticide doses,
   outbreak claims, or names of sources.
7. When chemical control is mentioned, keep it cautious: follow label, local
   agricultural university, KVK, or extension officer guidance.
8. Keep the tone practical, trustworthy, hopeful, and suitable for farmers,
   extension workers, agriculture students, and progressive growers.
9. Remove direct checklist labels and rewrite those ideas into paragraph rhythm.
   Do not print labels such as "શું કરવું?", "શા માટે?", "લાભ",
   "મુખ્ય કારણ", "અસર", "પરિણામ", "ઉકેલ", "સમસ્યા", or similar
   planning headings.
10. Do not use bold or italic marker labels inside the article. Use only natural
    Gujarati magazine prose with occasional reader-friendly subheadings.

Target publication: {target_magazine}
Language: Gujarati
Length: {article_length}
Month: {month}
Region: {region}
Subject area: {subject_area}
Crop focus: {crop_focus or "No specific crop focus"}

Selected topic and research notes:
{selected_topic}

Draft article:
{article}

Return only the rewritten Gujarati article with a suitable title. Do not include
editor notes before or after the article.
""".strip()


def final_editor_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    article_length: str,
    target_magazine: str,
    selected_topic: str,
    article: str,
) -> str:
    return f"""
Act as the final Gujarati magazine editor for {target_magazine}.

Target magazine personality:
{magazine_style_note(target_magazine)}

Final editorial standard:
- Do not claim that Dr. M. S. Swaminathan wrote the article.
- Keep an original voice inspired by his farmer-centric scientific communication.
- Make the final article publication-ready for a Gujarati agriculture magazine.

Final checks to apply silently:
1. Strong Gujarati title.
2. Farmer-oriented first paragraph.
3. Clear seasonal and regional relevance.
4. Simple scientific explanation.
5. Practical recommendations written in connected magazine prose, not repeated
   question-answer or checklist blocks.
6. Every recommendation explains farmer benefit inside normal paragraphs.
7. Good magazine flow with readable paragraphs and useful subheadings.
8. No research-paper style headings.
9. No unsafe pesticide dosage claims.
10. No unsupported outbreak or official-advisory claims.
11. Natural Gujarati language, polished grammar, and no unnecessary English.
12. Positive practical takeaway at the end.
13. Remove direct structural labels such as "શું કરવું?", "શા માટે?",
    "લાભ", "મુખ્ય કારણ", "અસર", "પરિણામ", "ઉકેલ", "સમસ્યા", and
    similar checklist words when they are used as headings.
14. Remove bold/italic label formatting and weave those ideas into smooth
    Gujarati magazine paragraphs.

Soft evidence guidance:
- Use the research notes and reference quality labels as gentle guardrails.
- Soften risky, overconfident, or locally uncertain statements.
- Do not demand a source for every sentence.
- Do not add inline citations, reference lists, or academic evidence language.
- Preserve farmer usefulness, magazine rhythm, and natural Gujarati prose.

Target publication: {target_magazine}
Language: Gujarati
Length: {article_length}
Month: {month}
Region: {region}
Subject area: {subject_area}
Crop focus: {crop_focus or "No specific crop focus"}

Selected topic and research notes:
{selected_topic}

Article to finalize:
{article}

Return only the final magazine-ready Gujarati article. Do not include score,
checklist, comments, or editor notes.
""".strip()


def story_research_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    topic_hint: str,
) -> str:
    return f"""
You are a senior agricultural research assistant for Gujarati agriculture magazines.

Use Google Search grounding to research a current, seasonally relevant Gujarati
agriculture article topic. The final article will use a human-centered field
opening, Swaminathan-inspired farmer welfare science, and practical extension
recommendations.

Research assignment:
- Month: {month}
- Region: {region}
- Subject area: {subject_area}
- Crop: {crop_focus or "No specific crop"}
- Topic hint: {topic_hint or "Find current candidate topics; user will choose manually"}

Research priorities:
- Current and prevailing crop problems
- Agricultural acarology and agricultural entomology relevance
- South Gujarat and Gujarat farming conditions
- Crop stage, weather influence, and seasonal activity
- Farmer observations and field-level symptoms
- Scientific reason behind the problem
- Integrated management, natural enemies, monitoring, and preventive action
- Official advisories, agricultural university/KVK guidance, research sources,
  and current web context where useful
- Practical value for farmers, extension workers, agriculture students, rural
  youth, and farm advisors

Build a deep research pack using several search angles before presenting topic
options:
- Current pest/mite or crop problem relevance
- Month, weather, and crop-stage connection
- Gujarat and South Gujarat field context
- Farmer-recognizable observations for a story opening
- Science that can be explained simply after the field situation
- Natural enemies, IPM, monitoring, and practical decision support
- Farmer benefit: yield, quality, cost reduction, sustainability, and profit

Return 5 to 8 Gujarati article topic options. Do not choose a final topic.
For each option include:
1. Gujarati article topic.
2. Why this topic is relevant now.
3. Region and crop relevance.
4. Field observations farmers may recognize.
5. Simple scientific explanation.
6. Practical management points.
7. Farmer benefits: yield, quality, cost reduction, profitability, and
   long-term crop health.
8. Cautions: uncertain claims, pesticide safety, and need for local verification.
9. Source-backed notes that can guide the article.
10. Reference quality notes. For important sources, label the source type as
    official, university/KVK, government, research, news, or general web. Briefly
    say what each source is useful for and where to use caution.

Do not invent official outbreaks, advisories, pesticide doses, or local claims.
When evidence is uncertain, say that field verification with local agricultural
university, KVK, or extension officers is needed.
Do not make the research feel like a literature review. The references should
strengthen the story and practical guidance while keeping the final article
magazine-like and citation-free.
Do not write a final recommendation such as "best topic", "selected topic", or
"write this topic"; keep the choice open for the user.
""".strip()


def story_article_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    article_length: str,
    target_magazine: str,
    topic_hint: str,
    research_notes: str,
) -> str:
    return f"""
Write a Gujarati {target_magazine} article using the following editorial blend:

Target magazine personality:
{magazine_style_note(target_magazine)}

- 20 percent human-centered rural storytelling: real field situations, farmer
  observations, simple vivid descriptions, and a curiosity-building opening.
  Do not imitate any living writer's exact wording or private style.
- 70 percent Swaminathan-inspired agricultural communication values: science
  linked with farmer welfare, practical solutions, scientific accuracy,
  sustainability, productivity, profitability, and a positive hopeful tone.
- 10 percent agricultural extension specialist: field observations,
  crop-specific recommendations, integrated management, region-specific
  advisories, and practical decision support.

Important authorship instruction:
- Do not claim that Dr. M. S. Swaminathan or any journalist wrote the article.
- Use an original Gujarati voice suitable for {target_magazine}.

Target audience:
Farmers, progressive growers, extension workers, agriculture students, rural
youth, and farm advisors.

Article requirements:
- Target publication: {target_magazine}
- Language: Gujarati
- Length: {article_length}
- Region: {region}
- Month: {month}
- Subject area: {subject_area}
- Crop: {crop_focus or "No specific crop"}
- Topic hint: {topic_hint or "Use the selected research topic"}

Opening requirement:
The first 150 to 250 words must not begin with definitions, statistics, or
technical terms. Begin with a farmer observation, field visit, seasonal
challenge, orchard or field experience, crop situation, or real-world problem.
The reader should feel: "I have seen this in my own field."

Article rhythm:
- Move gradually from field observation to scientific explanation.
- Every paragraph should connect problem, importance, simple science,
  practical solution, and farmer benefit.
- Use cause, effect, consequence, solution, and benefit as hidden writing logic.
- Do not print that planning structure as labels.
- Technical terms must be explained immediately in farmer-friendly Gujarati.
- Include field observations: seasonal trends, weather influence, crop stage,
  farmer practices, pest or mite behaviour, and natural enemies.
- Recommendations must naturally explain what to do, why it matters, and how it
  benefits the farmer.
- Use concepts naturally: crop health, yield improvement, quality improvement,
  timely monitoring, sustainable management, integrated management, natural
  enemies, preventive action, profitability, cost reduction, informed decision
  making, and long-term crop health.

Avoid:
- Literature review, research paper, thesis style, excessive statistics, long
  technical paragraphs, policy discussion, political commentary, and government
  programme discussion.
- Headings like Introduction, Materials and Methods, Results, Discussion, and
  Conclusion.
- Repeated label blocks such as direct "what to do", "why", "benefit",
  "main reason", "effect", "result", or "solution" headings.
- Unsafe pesticide dosage claims. When chemical control is mentioned, advise
  farmers to follow label recommendations and local agricultural university,
  KVK, or extension officer guidance.

Ending:
End with practical confidence: the problem is manageable, farmers can act,
science provides solutions, and timely field decisions improve outcomes.

Research notes and sources:
{research_notes}

Return only the complete Gujarati article with a suitable Gujarati title.
""".strip()


def story_rewrite_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    article_length: str,
    target_magazine: str,
    topic_hint: str,
    research_notes: str,
    article: str,
) -> str:
    return f"""
Rewrite the following Gujarati article into a stronger {target_magazine} magazine
article using the story + science + extension workflow.

Target magazine personality:
{magazine_style_note(target_magazine)}

Keep the same facts and topic, but improve:
1. Human-centered field opening.
2. Gradual transition from farmer observation to simple science.
3. Swaminathan-inspired farmer welfare, scientific accuracy, sustainability,
   productivity, profitability, and hope.
4. Practical extension advice for Gujarat farmers.
5. Natural paragraph rhythm instead of checklist labels.
6. Technical terms explained immediately in farmer-friendly Gujarati.
7. Recommendations that naturally include action, reason, and benefit.
8. Final takeaway that gives confidence to farmers.

Do not claim that Dr. M. S. Swaminathan or any journalist wrote the article.
Do not imitate any living writer's exact wording or private style. Use an
original Gujarati magazine voice.

Remove:
- Research paper style.
- Repeated "what/why/benefit" blocks.
- Direct "main reason/effect/result/solution" label headings.
- Unsupported outbreak claims, official advisories, and unsafe pesticide doses.

Target publication: {target_magazine}
Language: Gujarati
Length: {article_length}
Month: {month}
Region: {region}
Subject area: {subject_area}
Crop: {crop_focus or "No specific crop"}
Topic hint: {topic_hint or "Use the selected research topic"}

Research notes:
{research_notes}

Draft article:
{article}

Return only the rewritten Gujarati article with a suitable Gujarati title.
""".strip()


def story_final_editor_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    article_length: str,
    target_magazine: str,
    topic_hint: str,
    research_notes: str,
    article: str,
) -> str:
    return f"""
Act as the final Gujarati magazine editor for {target_magazine}.

Target magazine personality:
{magazine_style_note(target_magazine)}

Finalize the article below using the attached story + science + extension
standard:
- Human-centered field opening.
- Swaminathan-inspired farmer welfare and scientific temper.
- Practical extension decision support.
- Clear source-aware scientific accuracy.
- Natural Gujarati magazine paragraphs.

Final checks to apply silently:
1. The article starts with a farmer situation, not a definition.
2. The science is simplified and connected with farmer relevance.
3. Field observations are realistic and not exaggerated.
4. Recommendations explain practical benefit without checklist labels.
5. It includes crop health, timely monitoring, integrated management, natural
   enemies, sustainability, cost reduction, yield, quality, and profitability
   where relevant.
6. It avoids research paper style, political commentary, and policy discussion.
7. It avoids unsupported advisories, outbreak claims, and unsafe pesticide doses.
8. It removes direct "what to do/why/benefit/main reason/effect/result/solution"
   label blocks.
9. It ends with confidence and practical hope.

Soft evidence guidance:
- Use the research notes and reference quality labels as gentle guardrails.
- Soften risky, overconfident, or locally uncertain statements.
- Do not demand a source for every sentence.
- Do not add inline citations, reference lists, or academic evidence language.
- Preserve storytelling, farmer usefulness, and magazine rhythm.

Target publication: {target_magazine}
Language: Gujarati
Length: {article_length}
Month: {month}
Region: {region}
Subject area: {subject_area}
Crop: {crop_focus or "No specific crop"}
Topic hint: {topic_hint or "Use the selected research topic"}

Research notes:
{research_notes}

Article to finalize:
{article}

Return only the final magazine-ready Gujarati article. Do not include editor
notes, score, checklist, or comments.
""".strip()


def farm_wisdom_research_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    topic_hint: str,
    season_context: str,
    target_magazine: str,
) -> str:
    return f"""
You are an agricultural research assistant for Gujarati farmer-oriented magazines.

Use Google Search grounding to research a current and seasonally relevant topic
for an observation-first agricultural article. The final article will sound like
an experienced farmer-scientist sharing practical wisdom with fellow farmers.

Research assignment:
- Target magazine: {target_magazine}
- Month: {month}
- Season/context: {season_context or month}
- Region: {region}
- Subject area: {subject_area}
- Crop: {crop_focus or "No specific crop"}
- Topic hint: {topic_hint or "Find current candidate topics; user will choose manually"}

Research priorities:
- Current and prevailing crop, pest, mite, weather, or field observation issues
- Agricultural acarology and entomology relevance when useful
- Gujarat and South Gujarat farming realities
- Seasonal field conditions, crop stage, weather, soil, dust, irrigation, and
  farmer habits
- Practical observations farmers may recognize
- Scientific explanation behind the observation, written later in simple language
- Natural enemies, balance, patience, timely observation, and practical wisdom
- Official advisories, agricultural university/KVK guidance, research sources,
  and current web context where useful
- Farmer benefit through better observation, lower cost, better decisions,
  crop health, yield, quality, and profitability

Build a deep research pack using several search angles before presenting topic
options:
- Current pest/mite, crop, weather, or field-observation relevance
- Month, season, crop stage, and weather connection
- Gujarat and South Gujarat farming reality
- Farm habits, orchard/field scenes, soil, dust, moisture, and natural balance
- Scientific explanation that can emerge from observation
- Natural enemies, IPM, patient monitoring, and practical wisdom
- Farmer benefit through better observation and wiser decisions

Return 5 to 8 Gujarati article topic options. Do not choose a final topic.
For each option include:
1. Gujarati article topic.
2. Why this topic is relevant for the selected season/month.
3. Region and crop relevance.
4. Field/orchard/village observations that can open the article.
5. Questions that can create curiosity for readers.
6. Simple scientific explanation.
7. Practical lessons and farmer benefits.
8. Cautions about uncertain claims, pesticide safety, and local verification.
9. Source-backed research notes for writing the article.
10. Reference quality notes. For important sources, label the source type as
    official, university/KVK, government, research, news, or general web. Briefly
    say what each source is useful for and where to use caution.

Do not invent official outbreaks, advisories, pesticide doses, or local claims.
When evidence is uncertain, say field verification with local agricultural
university, KVK, or extension officers is needed.
Do not make the research feel like an academic review. The references should
quietly support a thoughtful farmer-scientist conversation, not turn the article
into a cited report.
Do not write a final recommendation such as "best topic", "selected topic", or
"write this topic"; keep the choice open for the user.
""".strip()


def farm_wisdom_article_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    article_length: str,
    topic_hint: str,
    season_context: str,
    target_magazine: str,
    research_notes: str,
) -> str:
    return f"""
Write a Gujarati agricultural magazine article using an original farmer-scientist
observation voice inspired by rural essay writing and practical farm wisdom.

Important authorship instruction:
- Do not claim that Gene Logsdon or any named writer wrote the article.
- Do not imitate any writer's exact wording.
- Write in an original Gujarati voice suitable for farmer-oriented magazines.

Core writing philosophy:
- Do not write like a scientist, university professor, research paper author,
  extension bulletin writer, or technical report writer.
- Write like an experienced farmer-scientist who has spent years walking through
  fields, orchards, villages, and farms and is sharing thoughtful practical
  wisdom with fellow farmers.
- The article should feel like a conversation, not a lecture.
- Readers should feel: "This writer understands farming."

Article purpose:
Help readers observe better, think differently, understand causes, appreciate
farming realities, and make wiser decisions. Knowledge should emerge naturally
through storytelling, observation, reflection, and simple explanation.

Article requirements:
- Target magazine: {target_magazine}
- Target magazine personality: {magazine_style_note(target_magazine)}
- Language: Gujarati
- Length: {article_length}
- Month: {month}
- Season/context: {season_context or month}
- Region: {region}
- Subject area: {subject_area}
- Crop: {crop_focus or "No specific crop"}
- Topic hint: {topic_hint or "Use the selected research topic"}

Opening style:
- Never begin with definitions, scientific facts, statistics, research findings,
  or technical terms.
- Begin with a seasonal observation, field situation, orchard experience, farmer
  habit, village reality, or crop condition.
- The opening should create recognition: the reader should feel that they have
  seen this in their own field.

Section rhythm:
- Every section should begin with observation.
- Use this hidden rhythm: observation -> reflection -> explanation -> practical
  lesson.
- Use questions naturally to create curiosity: why does this happen, what
  changes between seasons, why do some farms suffer more, what is nature showing
  us?
- Use science only after observation and reflection. Science should support the
  story; the story should not feel like decoration for science.

Tone:
Thoughtful, calm, wise, observational, practical, respectful, and lived-in.
Avoid urgent, fear-based, academic, or marketing-style language.

Technical information rule:
- Scientific information must appear naturally.
- Explain technical terms immediately in simple farmer language.
- Avoid heavy taxonomy, long technical paragraphs, and disconnected facts.

Recommendation style:
- Do not command farmers.
- Advice should sound like practical wisdom from field experience.
- Prefer gentle sentence forms such as "regular observation often prevents a
  bigger problem" or "looking under leaves can save unnecessary spray cost."
- When chemical control is mentioned, stay cautious: follow label
  recommendations and local agricultural university, KVK, or extension officer
  guidance.

Use naturally:
Observation, experience, season, nature, balance, habit, field, orchard, soil,
weather, common sense, careful observation, patience, understanding, practical
wisdom, timely monitoring, natural enemies, crop health, quality, yield, and
profitability.

Avoid excessive use of:
Management, control, technology, intervention, protocol, recommendation, and
treatment.

Ending style:
End with reflection and wisdom, not a formal conclusion. The reader should leave
with the feeling: "I understand this better now."

Avoid:
- Research paper style, thesis style, extension bulletin style, policy
  discussion, political commentary, and government programme discussion.
- Headings like Introduction, Materials and Methods, Results, Discussion, and
  Conclusion.
- Repeated label blocks like "what to do", "why", "benefit", "main reason",
  "effect", "result", or "solution".
- Unsupported outbreak claims, official advisories, and unsafe pesticide doses.

Research notes and sources:
{research_notes}

Return only the complete Gujarati article with a suitable Gujarati title.
""".strip()


def farm_wisdom_rewrite_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    article_length: str,
    topic_hint: str,
    season_context: str,
    target_magazine: str,
    research_notes: str,
    article: str,
) -> str:
    return f"""
Rewrite the following Gujarati article into a stronger observation-first farm
wisdom article for a Gujarati agricultural magazine.

Important authorship instruction:
- Do not claim that Gene Logsdon or any named writer wrote the article.
- Do not imitate exact wording. Use an original Gujarati voice.

Improve the article so it feels like:
- An experienced farmer-scientist speaking with fellow farmers.
- A thoughtful conversation, not a lecture.
- Observation first, then question, then explanation, then practical lesson.
- Calm, wise, lived-in, respectful, and practical.

Rewrite goals:
1. Start with a recognizable field, orchard, village, season, soil, weather, or
   crop observation.
2. Create curiosity with natural questions.
3. Bring science gradually and simply.
4. Make technical words farmer-friendly.
5. Replace commands with practical wisdom.
6. Remove research-paper, extension-bulletin, and checklist-label style.
7. Keep farmer benefit visible through better observation, reduced cost, crop
   health, yield, quality, profitability, and wiser decisions.
8. Preserve scientific accuracy and source-aware caution.
9. Avoid unsupported outbreak claims, official advisories, and pesticide doses.

Target magazine: {target_magazine}
Target magazine personality:
{magazine_style_note(target_magazine)}
Language: Gujarati
Length: {article_length}
Month: {month}
Season/context: {season_context or month}
Region: {region}
Subject area: {subject_area}
Crop: {crop_focus or "No specific crop"}
Topic hint: {topic_hint or "Use the selected research topic"}

Research notes:
{research_notes}

Draft article:
{article}

Return only the rewritten Gujarati article with a suitable Gujarati title.
""".strip()


def farm_wisdom_final_editor_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    article_length: str,
    topic_hint: str,
    season_context: str,
    target_magazine: str,
    research_notes: str,
    article: str,
) -> str:
    return f"""
Act as the final Gujarati magazine editor for {target_magazine}.

Finalize the article below into a polished observation-first agricultural
magazine article.

Final editorial standard:
- Original Gujarati farmer-scientist voice.
- Thoughtful conversation, not lecture.
- Observation -> question -> explanation -> reflection -> practical lesson.
- Science appears naturally and supports the reader's understanding.
- Advice sounds like wisdom, not orders.
- The article feels lived-in and enjoyable for farmers to read.

Final checks to apply silently:
1. Does it begin with observation, not definition?
2. Does it create recognition and curiosity?
3. Is the science simplified and naturally introduced?
4. Does it sound like field experience rather than university notes?
5. Does it contain practical wisdom and farmer benefit?
6. Does it avoid fear-based tone and chemical-first thinking?
7. Does it avoid research paper, thesis, and extension bulletin style?
8. Does it avoid unsupported advisories, outbreak claims, and unsafe pesticide
   doses?
9. Does it avoid checklist-label blocks?
10. Does the ending leave the reader with reflection and confidence?

Soft evidence guidance:
- Use the research notes and reference quality labels as gentle guardrails.
- Soften risky, overconfident, or locally uncertain statements.
- Do not demand a source for every sentence.
- Do not add inline citations, reference lists, or academic evidence language.
- Preserve the farmer-scientist conversation and lived-in magazine voice.

Target magazine: {target_magazine}
Target magazine personality:
{magazine_style_note(target_magazine)}
Language: Gujarati
Length: {article_length}
Month: {month}
Season/context: {season_context or month}
Region: {region}
Subject area: {subject_area}
Crop: {crop_focus or "No specific crop"}
Topic hint: {topic_hint or "Use the selected research topic"}

Research notes:
{research_notes}

Article to finalize:
{article}

Return only the final magazine-ready Gujarati article. Do not include score,
checklist, editor notes, or comments.
""".strip()


def field_discovery_research_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    topic_hint: str,
    season_context: str,
    target_magazine: str,
) -> str:
    return f"""
You are an agricultural research assistant for Gujarati long-form agricultural
magazines.

Use Google Search grounding to research a current, seasonally relevant topic
for a scene-based agricultural feature article. The final article will feel like
a journey of discovery through a field, orchard, season, weather change, crop
condition, farmer observation, and practical understanding.

Research assignment:
- Target magazine: {target_magazine}
- Month: {month}
- Season/context: {season_context or month}
- Region: {region}
- Subject area: {subject_area}
- Crop: {crop_focus or "No specific crop"}
- Topic hint: {topic_hint or "Find current candidate topics; user will choose manually"}

Research priorities:
- Current crop, pest, mite, weather, field, orchard, or seasonal observation
  issues relevant to farmers
- Agricultural acarology and entomology relevance when useful
- Gujarat and South Gujarat farming conditions
- Visual scene details: light, weather, crop appearance, leaf condition, dust,
  humidity, dry winds, seasonal transition, farmer activity, and field texture
- Observations that can create curiosity before explanation
- Hidden causes behind visible crop symptoms
- Scientific understanding that can emerge gradually after observation
- Practical meaning for farmers: observation, monitoring, natural enemies,
  timely decision-making, lower cost, crop health, quality, yield, and profit
- Official advisories, agricultural university/KVK guidance, research sources,
  and current web context where useful

Build a deep research pack using several search angles before presenting topic
options:
- Current pest/mite, crop, weather, or field-scene relevance
- Month, season, crop stage, and weather connection
- Gujarat and South Gujarat field/orchard context
- Visual clues that can carry the opening scene
- Observations and questions that delay discovery naturally
- Scientific explanation that can appear after curiosity is built
- Natural enemies, IPM, monitoring, and practical meaning
- Farmer benefit through observation, timely decisions, quality, yield, and profit

Return 5 to 8 Gujarati article topic options. Do not choose a final topic.
For each option include:
1. Gujarati article topic.
2. Why this topic is relevant for the selected season/month.
3. Region and crop relevance.
4. Scene details that can open the article.
5. Observations and curiosity-building questions.
6. Clues that can lead to delayed discovery.
7. Simple scientific explanation.
8. Practical meaning and farmer benefit.
9. Cautions about uncertain claims, pesticide safety, and local verification.
10. Source-backed notes for article writing.
11. Reference quality notes. For important sources, label the source type as
    official, university/KVK, government, research, news, or general web. Briefly
    say what each source is useful for and where to use caution.

Do not invent official outbreaks, advisories, pesticide doses, or local claims.
When evidence is uncertain, say field verification with local agricultural
university, KVK, or extension officers is needed.
Do not make the research feel like a technical literature review. The references
should quietly strengthen the scene, discovery, and practical meaning while the
final article remains citation-free and magazine-like.
Do not write a final recommendation such as "best topic", "selected topic", or
"write this topic"; keep the choice open for the user.
""".strip()


def field_discovery_article_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    article_length: str,
    topic_hint: str,
    season_context: str,
    target_magazine: str,
    research_notes: str,
) -> str:
    return f"""
Write a Gujarati agricultural magazine feature using an original field-discovery
voice inspired by careful observation of farm life and seasons.

Important authorship instruction:
- Do not claim that Kristin Kimball or any named writer wrote the article.
- Do not imitate any writer's exact wording.
- Use an original Gujarati narrative voice suitable for long-form agricultural
  magazines.

Core writing philosophy:
- Do not write like a scientist presenting facts, a professor teaching a lesson,
  a technical expert giving recommendations, a research paper author, or an
  extension bulletin writer.
- Write like a thoughtful observer of farm life who discovers agricultural
  knowledge through seasons, fields, orchards, crops, farmers, weather, and
  everyday experiences.
- The article should feel like a journey of discovery. Readers should feel they
  are walking through the field with the writer.

Primary objective:
Help readers notice things they normally overlook, become curious, discover
hidden causes, understand farming more deeply, and appreciate the connection
between weather, crops, pests, and people.

Article requirements:
- Target magazine: {target_magazine}
- Target magazine personality: {magazine_style_note(target_magazine)}
- Language: Gujarati
- Length: {article_length}
- Month: {month}
- Season/context: {season_context or month}
- Region: {region}
- Subject area: {subject_area}
- Crop: {crop_focus or "No specific crop"}
- Topic hint: {topic_hint or "Use the selected research topic"}

Article architecture:
Scene -> observation -> curiosity -> discovery -> scientific understanding ->
practical meaning -> reflection.

Opening section:
- The first 200 to 300 words must contain a season, a place, a crop, an
  observation, and a feeling of curiosity.
- Do not begin with definitions, statistics, research findings,
  recommendations, or technical explanations.
- The opening must create a visual image. Readers should be able to see the
  field, orchard, weather, crop, or farmer activity.

Scene building:
Use real-feeling details such as light, weather, field condition, crop
appearance, seasonal changes, farmer activity, morning dew, dry winds, dusty
leaves, bright sunlight, changing leaf colour, quiet orchards, or seasonal
transition when relevant.

Observation density:
Every paragraph should include at least one observation. Invite readers to look
more carefully at their own fields.

Curiosity and delayed discovery:
- Frequently create questions, but do not answer immediately.
- Build a path: observation -> additional observation -> question -> more clues
  -> discovery -> simple scientific explanation.
- Science should appear only after readers are emotionally invested.
- Use transitions such as "a closer look revealed", "only later did it become
  clear", "the explanation lies in", and "what seemed mysterious became easier
  to understand" in natural Gujarati.

Sentence rhythm:
Alternate short, medium, and longer descriptive sentences. Use short impactful
sentences when a discovery or reflection becomes clear.

Subject selection:
Avoid making pests the main subject. Prefer season, weather, field, crop, tree,
farmer, orchard, and landscape as the actors.

Practical recommendation style:
- Recommendations should arise naturally from understanding.
- Avoid command-heavy writing such as "farmers should spray".
- Practical meaning should feel earned by the observations.
- When chemical control is mentioned, stay cautious: follow label
  recommendations and local agricultural university, KVK, or extension officer
  guidance.

Language style:
Descriptive, reflective, thoughtful, observational, narrative, natural, and
readable.

Avoid:
- Bullet-point writing, extension bulletin style, instruction-heavy writing,
  textbook language, research paper style, policy discussion, and political
  commentary.
- Headings like Introduction, Materials and Methods, Results, Discussion, and
  Conclusion.
- Repeated label blocks like "what to do", "why", "benefit", "main reason",
  "effect", "result", or "solution".
- Unsupported outbreak claims, official advisories, and unsafe pesticide doses.

Ending style:
End with reflection, a lesson learned, deeper understanding, renewed
appreciation for observation, and a hopeful outlook. The ending should leave
readers thinking.

Research notes and sources:
{research_notes}

Return only the complete Gujarati article with a suitable Gujarati title.
""".strip()


def field_discovery_rewrite_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    article_length: str,
    topic_hint: str,
    season_context: str,
    target_magazine: str,
    research_notes: str,
    article: str,
) -> str:
    return f"""
Rewrite the following Gujarati article into a stronger scene-based field
discovery feature for a Gujarati agricultural magazine.

Important authorship instruction:
- Do not claim that Kristin Kimball or any named writer wrote the article.
- Do not imitate exact wording. Use an original Gujarati voice.

Improve the article so it feels like:
- A journey through a field, orchard, season, crop condition, and farmer
  observation.
- Scene -> observation -> curiosity -> discovery -> scientific understanding ->
  practical meaning -> reflection.
- Science delayed until the reader has seen the clues.
- A magazine feature rather than an advisory article.

Rewrite goals:
1. Begin with a vivid scene: season, place, crop, observation, and curiosity.
2. Add observation density to every paragraph.
3. Use questions to create curiosity, then reveal science gradually.
4. Make the environment, crop, farmer, orchard, field, and weather the main
   subjects more often than pests.
5. Keep science simple and naturally introduced.
6. Make recommendations arise from understanding, not command style.
7. Remove bullet-point, textbook, extension-bulletin, and checklist-label style.
8. Preserve scientific accuracy and source-aware caution.
9. Avoid unsupported outbreak claims, official advisories, and pesticide doses.
10. End with reflection and renewed appreciation for careful observation.

Target magazine: {target_magazine}
Target magazine personality:
{magazine_style_note(target_magazine)}
Language: Gujarati
Length: {article_length}
Month: {month}
Season/context: {season_context or month}
Region: {region}
Subject area: {subject_area}
Crop: {crop_focus or "No specific crop"}
Topic hint: {topic_hint or "Use the selected research topic"}

Research notes:
{research_notes}

Draft article:
{article}

Return only the rewritten Gujarati article with a suitable Gujarati title.
""".strip()


def field_discovery_final_editor_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    article_length: str,
    topic_hint: str,
    season_context: str,
    target_magazine: str,
    research_notes: str,
    article: str,
) -> str:
    return f"""
Act as the final Gujarati magazine editor for {target_magazine}.

Finalize the article below into a polished field-discovery agricultural feature.

Final editorial standard:
- Original Gujarati long-form magazine voice.
- The article begins with a scene and creates visual imagination.
- Readers feel they are walking through the field with the writer.
- Curiosity builds before scientific explanation.
- Science appears as discovery, not lecture.
- Observations are frequent and practical meaning feels earned.
- The ending leaves readers thinking and encourages them to observe their own
  fields more carefully.

Final checks to apply silently:
1. Does the article begin with a scene?
2. Can readers visualize the situation?
3. Is curiosity created before explanation?
4. Is science delayed until discovery?
5. Does every paragraph contain observation?
6. Does it feel like a journey, not an advisory bulletin?
7. Is it enjoyable even without recommendations?
8. Does it avoid unsupported advisories, outbreak claims, and unsafe pesticide
   doses?
9. Does it avoid bullet points, checklist labels, and technical-report style?
10. Does the ending give reflection, understanding, and hope?

Soft evidence guidance:
- Use the research notes and reference quality labels as gentle guardrails.
- Soften risky, overconfident, or locally uncertain statements.
- Do not demand a source for every sentence.
- Do not add inline citations, reference lists, or academic evidence language.
- Preserve the field-discovery journey and reflective magazine voice.

Target magazine: {target_magazine}
Target magazine personality:
{magazine_style_note(target_magazine)}
Language: Gujarati
Length: {article_length}
Month: {month}
Season/context: {season_context or month}
Region: {region}
Subject area: {subject_area}
Crop: {crop_focus or "No specific crop"}
Topic hint: {topic_hint or "Use the selected research topic"}

Research notes:
{research_notes}

Article to finalize:
{article}

Return only the final magazine-ready Gujarati article. Do not include score,
checklist, editor notes, or comments.
""".strip()


def markdown_to_docx_blocks(text: str) -> list[tuple[str, str]]:
    blocks = []
    pending = []

    def flush_pending() -> None:
        if pending:
            blocks.append(("Normal", " ".join(pending).strip()))
            pending.clear()

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            flush_pending()
            continue

        if line.startswith("#"):
            flush_pending()
            title = line.lstrip("#").strip()
            if title:
                style = "Title" if not blocks else "Heading1"
                blocks.append((style, title))
            continue

        if line.startswith(("- ", "* ")) or re.match(r"^\d+\.\s+", line):
            flush_pending()
            item = re.sub(r"^([-*]|\d+\.)\s+", "", line).strip()
            blocks.append(("ListParagraph", item))
            continue

        pending.append(line)

    flush_pending()
    return blocks or [("Normal", text.strip() or "")]


def docx_paragraph(style: str, text: str) -> str:
    style_xml = ""
    if style:
        style_xml = f'<w:pPr><w:pStyle w:val="{escape(style)}"/></w:pPr>'

    if style == "ListParagraph":
        text = f"- {text}"

    return (
        "<w:p>"
        f"{style_xml}"
        "<w:r>"
        '<w:rPr><w:rFonts w:ascii="Nirmala UI" w:hAnsi="Nirmala UI" '
        'w:cs="Nirmala UI"/></w:rPr>'
        f'<w:t xml:space="preserve">{escape(text)}</w:t>'
        "</w:r>"
        "</w:p>"
    )


def make_docx(article: str) -> bytes:
    document_body = "".join(
        docx_paragraph(style, text) for style, text in markdown_to_docx_blocks(article)
    )

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {document_body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>"""

    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr><w:rFonts w:ascii="Nirmala UI" w:hAnsi="Nirmala UI" w:cs="Nirmala UI"/><w:sz w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:after="240"/></w:pPr>
    <w:rPr><w:b/><w:rFonts w:ascii="Nirmala UI" w:hAnsi="Nirmala UI" w:cs="Nirmala UI"/><w:sz w:val="36"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr>
    <w:rPr><w:b/><w:rFonts w:ascii="Nirmala UI" w:hAnsi="Nirmala UI" w:cs="Nirmala UI"/><w:sz w:val="28"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph">
    <w:name w:val="List Paragraph"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:ind w:left="720"/></w:pPr>
  </w:style>
</w:styles>"""

    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    doc_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types_xml)
        docx.writestr("_rels/.rels", rels_xml)
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", styles_xml)
        docx.writestr("word/_rels/document.xml.rels", doc_rels_xml)

    return buffer.getvalue()


def render_sources(title: str, sources: list[dict[str, str]]) -> None:
    if not sources:
        return

    with st.expander(title, expanded=False):
        for index, source in enumerate(sources, start=1):
            st.markdown(f"{index}. [{source['title']}]({source['uri']})")


def selected_topic_context(topic: str, research_notes: str) -> str:
    topic = (topic or "").strip()
    research_notes = (research_notes or "").strip()
    if research_notes:
        return f"Manually selected topic:\n{topic}\n\nResearch notes:\n{research_notes}"
    return f"Manually selected topic:\n{topic}"


def magazine_style_note(target_magazine: str) -> str:
    return MAGAZINE_STYLE_NOTES.get(
        target_magazine,
        MAGAZINE_STYLE_NOTES["Gujarati farmer magazine"],
    )


def has_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def recommend_target_magazine(
    topic: str,
    subject_area: str = "",
    research_notes: str = "",
    fallback: str = "Krushi Vigyan",
) -> str:
    text = " ".join([topic or "", subject_area or "", research_notes or ""]).lower()
    scores = {magazine: 0 for magazine in MAGAZINE_OPTIONS}
    scores[fallback if fallback in scores else "Krushi Vigyan"] += 1

    if has_any(text, ["mite", "acarology", "ipm", "pest", "disease", "thrips", "whitefly", "nematode", "mealybug", "fruit fly", "crop protection"]):
        scores["Krushi Vigyan"] += 5
        scores["Krushi Go-Vidya"] += 3
        scores["Krushi Jivan"] += 2

    if has_any(text, ["university", "kvk", "recommendation", "advisory", "agromet", "crop stage", "extension", "natural farming", "training"]):
        scores["Krushi Go-Vidya"] += 5

    if has_any(text, ["fertilizer", "nutrient", "nutrition", "soil", "micronutrient", "water soluble", "research", "technology", "dairy", "animal husbandry", "water recharge", "farm forestry"]):
        scores["Krushi Jivan"] += 5

    if has_any(text, ["news", "scheme", "yojana", "market", "commodity", "success story", "progressive farmer", "iot", "drone", "machinery", "explainer", "current"]):
        scores["Krishi Jagran Gujarati"] += 5

    if has_any(text, ["today", "daily", "mandi", "price", "rainfall", "rain", "weather update", "subsidy", "local event", "alert", "urgent"]):
        scores["Krushi Prabhat"] += 5
        scores["Krishi Jagran Gujarati"] += 2

    ranking = [
        "Krushi Vigyan",
        "Krushi Go-Vidya",
        "Krushi Jivan",
        "Krishi Jagran Gujarati",
        "Krushi Prabhat",
        "Agro Sandesh",
        "Gujarati farmer magazine",
        "Gujarati long-form agricultural magazine",
    ]
    return max(ranking, key=lambda magazine: scores.get(magazine, 0))


def target_magazine_selector(
    key: str,
    topic: str,
    subject_area: str,
    research_notes: str,
    fallback: str = "Krushi Vigyan",
):
    if not topic.strip():
        st.caption("Type your selected topic to get a target magazine suggestion.")
        return None

    suggested_magazine = recommend_target_magazine(
        topic,
        subject_area,
        research_notes,
        fallback,
    )
    st.caption(
        f"Suggested target magazine: {suggested_magazine}. "
        f"{magazine_style_note(suggested_magazine)}"
    )
    suggestion_key = f"{key}_suggested"
    current_magazine = st.session_state.get(key)
    previous_suggestion = st.session_state.get(suggestion_key)
    if current_magazine is None or current_magazine == previous_suggestion:
        st.session_state[key] = suggested_magazine
    st.session_state[suggestion_key] = suggested_magazine

    current_index = MAGAZINE_OPTIONS.index(
        st.session_state.get(key, suggested_magazine)
        if st.session_state.get(key, suggested_magazine) in MAGAZINE_OPTIONS
        else suggested_magazine
    )
    return st.selectbox(
        "Target magazine personality",
        MAGAZINE_OPTIONS,
        index=current_index,
        key=key,
    )


def main() -> None:
    st.title("Agro Sandesh Gujarati Agriculture Article Writer")
    st.caption(
        "Multi-AI workflow: Perplexity for research, Gemini for Gujarati drafting, "
        "and OpenAI for strict quality review."
    )

    with st.sidebar:
        api_keys = get_api_keys()

        st.header("AI routing")
        st.caption("API keys are loaded from Streamlit secrets or environment variables.")
        research_provider = st.selectbox(
            "Deep research provider",
            [PROVIDER_PERPLEXITY, PROVIDER_GEMINI],
            index=0,
        )
        research_model = st.text_input(
            "Research model",
            value=(
                config_value("PERPLEXITY_MODEL", "sonar-reasoning-pro")
                if research_provider == PROVIDER_PERPLEXITY
                else config_value("GEMINI_RESEARCH_MODEL", "gemini-3.1-pro-preview")
            ),
        )
        article_model = st.text_input(
            "Gemini article model",
            value=config_value("GEMINI_ARTICLE_MODEL", "gemini-3.1-pro-preview"),
        )
        review_provider = st.selectbox(
            "Quality review provider",
            [PROVIDER_OPENAI, PROVIDER_GEMINI],
            index=0,
        )
        review_model = st.text_input(
            "Review model",
            value=(
                config_value("OPENAI_REVIEW_MODEL", "gpt-4o")
                if review_provider == PROVIDER_OPENAI
                else article_model
            ),
        )

        st.header("Writing settings")
        model = article_model
        temperature = st.slider("Creativity", 0.1, 1.0, 0.7, 0.1)
        use_search_for_article = st.checkbox(
            "Use Google Search while writing article",
            value=True,
        )

    selected_providers = [PROVIDER_GEMINI, research_provider, review_provider]
    missing_keys = missing_api_keys(selected_providers, api_keys)
    if missing_keys:
        st.warning(
            "Add these API keys to Streamlit secrets/settings or environment variables: "
            + ", ".join(missing_keys)
        )
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        month = st.selectbox(
            "Month",
            MONTHS,
            index=datetime.now().month - 1,
        )
    with col2:
        region = st.selectbox("Region", REGIONS)

    subject_area = st.selectbox("Subject area", SUBJECT_AREAS)
    crop_focus = st.text_input(
        "Crop focus optional",
        placeholder="Example: mango, okra, sugarcane, fruit crops, vegetables",
    )
    article_length = st.selectbox("Article length", ARTICLE_LENGTHS, index=1)

    client = build_client(api_keys[PROVIDER_GEMINI])
    tab_classic, tab_story, tab_farm_wisdom, tab_field_discovery = st.tabs(
        [
            "Tab 1: Swaminathan Workflow",
            "Tab 2: Story + Science Prompt",
            "Tab 3: Farm Wisdom Prompt",
            "Tab 4: Field Discovery Prompt",
        ]
    )

    with tab_classic:
        st.subheader("Current Workflow")
        st.write(
            "Use this tab for the original topic discovery, Gujarati article draft, "
            "Swaminathan-inspired rewrite, final editor check, and Word download."
        )

        if st.button("Deep research and references", type="primary", key="classic_find_topics"):
            with st.spinner("Researching current and seasonally relevant topics..."):
                prompt = topic_research_prompt(month, region, subject_area, crop_focus)
                topics, sources = generate_text(
                    client,
                    research_model,
                    prompt,
                    use_search=research_provider == PROVIDER_GEMINI,
                    temperature=0.45,
                    provider=research_provider,
                    api_keys=api_keys,
                )
                st.session_state["topics"] = topics
                st.session_state["topic_sources"] = sources
                st.session_state.pop("classic_manual_topic", None)
                st.session_state.pop("classic_target_magazine", None)

        if "topics" in st.session_state:
            st.subheader("Suggested topics")
            st.markdown(st.session_state["topics"])
            render_sources("Research sources", st.session_state.get("topic_sources", []))

            selected_topic_title = st.text_input(
                "Manually selected topic for writing",
                placeholder="Type or paste the Gujarati topic you want to write.",
                key="classic_manual_topic",
            )
            selected_topic_notes = st.text_area(
                "Research notes to use for the selected topic",
                value=st.session_state["topics"],
                height=260,
                key="classic_selected_topic_notes",
            )
            selected_target_magazine = target_magazine_selector(
                "classic_target_magazine",
                selected_topic_title,
                subject_area,
                selected_topic_notes,
                "Krushi Vigyan",
            )

            if st.button("Use this research to write article", key="classic_write_article"):
                if not selected_topic_title.strip():
                    st.warning("Please manually select or type one topic before writing.")
                elif not selected_target_magazine:
                    st.warning("Please select the target magazine personality before writing.")
                else:
                    selected_topic = selected_topic_context(
                        selected_topic_title,
                        selected_topic_notes,
                    )
                    with st.spinner("Writing the Gujarati article draft..."):
                        prompt = article_prompt(
                            month,
                            region,
                            subject_area,
                            crop_focus,
                            article_length,
                            selected_target_magazine,
                            selected_topic,
                        )
                        article, sources = generate_text(
                            client,
                            model,
                            prompt,
                            use_search=use_search_for_article,
                            temperature=temperature,
                        )
                        st.session_state["article"] = article
                        st.session_state["article_sources"] = sources
                        st.session_state["selected_topic"] = selected_topic
                        st.session_state["selected_target_magazine"] = selected_target_magazine
                        st.session_state.pop("rewritten_article", None)
                        st.session_state.pop("final_article", None)
                        st.session_state.pop("review", None)

        if "article" in st.session_state:
            st.subheader("Step 1: Gujarati article draft")
            draft_article = st.text_area(
                "Draft article",
                value=st.session_state["article"],
                height=420,
                key="classic_draft_article",
            )
            st.session_state["article"] = draft_article
            render_sources("Article grounding sources", st.session_state.get("article_sources", []))

            st.download_button(
                "Download draft as TXT",
                data=draft_article,
                file_name="agro_sandesh_draft_article.txt",
                mime="text/plain",
                key="classic_download_draft",
            )

            col_review, col_rewrite = st.columns(2)
            with col_review:
                review_clicked = st.button("Review draft quality", key="classic_review_draft")
            with col_rewrite:
                rewrite_clicked = st.button(
                    "Rewrite in Swaminathan-inspired style",
                    key="classic_rewrite_article",
                )

            if review_clicked:
                with st.spinner("Reviewing article quality..."):
                    review, _ = generate_text(
                        client,
                        review_model,
                        review_prompt(
                            draft_article,
                            st.session_state.get("selected_target_magazine", "Agro Sandesh"),
                        ),
                        use_search=False,
                        temperature=0.25,
                        provider=review_provider,
                        api_keys=api_keys,
                    )
                    st.session_state["review"] = review

            if rewrite_clicked:
                with st.spinner("Rewriting the article with stronger farmer-centric flow..."):
                    rewrite, _ = generate_text(
                        client,
                        model,
                        rewrite_prompt(
                            month,
                            region,
                            subject_area,
                            crop_focus,
                            article_length,
                            st.session_state.get("selected_target_magazine", "Agro Sandesh"),
                            st.session_state.get("selected_topic", ""),
                            draft_article,
                        ),
                        use_search=False,
                        temperature=0.55,
                    )
                    st.session_state["rewritten_article"] = rewrite
                    st.session_state.pop("final_article", None)

        if "review" in st.session_state:
            st.subheader("Article review")
            st.markdown(st.session_state["review"])

        if "rewritten_article" in st.session_state:
            st.subheader("Step 2: Swaminathan-inspired rewrite")
            rewritten_article = st.text_area(
                "Improved article",
                value=st.session_state["rewritten_article"],
                height=460,
                key="classic_rewritten_article",
            )
            st.session_state["rewritten_article"] = rewritten_article

            st.download_button(
                "Download rewritten article as TXT",
                data=rewritten_article,
                file_name="agro_sandesh_rewritten_article.txt",
                mime="text/plain",
                key="classic_download_rewrite",
            )

            st.caption("Soft evidence check included in editor pass.")

            if st.button(
                "Final editor check and make magazine article",
                type="primary",
                key="classic_final_editor",
            ):
                with st.spinner("Final editor is polishing the magazine-ready version..."):
                    final_article, _ = generate_text(
                        client,
                        model,
                        final_editor_prompt(
                            month,
                            region,
                            subject_area,
                            crop_focus,
                            article_length,
                            st.session_state.get("selected_target_magazine", "Agro Sandesh"),
                            st.session_state.get("selected_topic", ""),
                            rewritten_article,
                        ),
                        use_search=False,
                        temperature=0.35,
                    )
                    st.session_state["final_article"] = final_article

        if "final_article" in st.session_state:
            st.subheader("Step 3: Final magazine-ready article")
            final_article = st.text_area(
                "Final article for magazine",
                value=st.session_state["final_article"],
                height=520,
                key="classic_final_article",
            )
            st.session_state["final_article"] = final_article

            col_txt, col_docx = st.columns(2)
            with col_txt:
                st.download_button(
                    "Download final article as TXT",
                    data=final_article,
                    file_name="agro_sandesh_final_article.txt",
                    mime="text/plain",
                    key="classic_download_final_txt",
                )
            with col_docx:
                st.download_button(
                    "Download final article as Word DOCX",
                    data=make_docx(final_article),
                    file_name="agro_sandesh_final_article.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="classic_download_final_docx",
                )

    with tab_story:
        st.subheader("Story + Science Prompt Workflow")
        st.write(
            "This tab adds your attached prompt style: field-story opening, "
            "science linked with farmer welfare, extension recommendations, "
            "research sources, rewrite, final editor check, and Word download."
        )

        story_col1, story_col2 = st.columns(2)
        with story_col1:
            story_topic_hint = st.text_input(
                "Topic optional",
                placeholder="Example: red spider mite in vegetables, mango mites, sugarcane mites",
                key="story_topic_hint",
            )
        with story_col2:
            story_crop_focus = st.text_input(
                "Crop for Tab 2",
                value=crop_focus,
                placeholder="Example: mango, okra, brinjal, cotton, vegetables",
                key="story_crop_focus",
            )

        if st.button(
            "Deep research and references for Tab 2",
            type="primary",
            key="story_research_button",
        ):
            with st.spinner("Researching story-style topic, field context, and references..."):
                prompt = story_research_prompt(
                    month,
                    region,
                    subject_area,
                    story_crop_focus,
                    story_topic_hint,
                )
                research, sources = generate_text(
                    client,
                    research_model,
                    prompt,
                    use_search=research_provider == PROVIDER_GEMINI,
                    temperature=0.35,
                    provider=research_provider,
                    api_keys=api_keys,
                )
                st.session_state["story_research"] = research
                st.session_state["story_sources"] = sources
                st.session_state["story_saved_topic_hint"] = story_topic_hint
                st.session_state["story_saved_crop_focus"] = story_crop_focus
                st.session_state.pop("story_manual_topic", None)
                st.session_state.pop("story_target_magazine", None)
                st.session_state.pop("story_article", None)
                st.session_state.pop("story_rewritten_article", None)
                st.session_state.pop("story_final_article", None)
                st.session_state.pop("story_review", None)

        if "story_research" in st.session_state:
            st.subheader("Tab 2 research notes")
            st.markdown(st.session_state["story_research"])
            render_sources("Tab 2 research sources", st.session_state.get("story_sources", []))

            story_selected_topic = st.text_input(
                "Manually selected topic for Tab 2",
                placeholder="Type or paste the topic you choose from the research options.",
                key="story_manual_topic",
            )
            story_research_notes = st.text_area(
                "Selected research notes for Tab 2",
                value=st.session_state["story_research"],
                height=300,
                key="story_research_notes",
            )
            story_target_magazine = target_magazine_selector(
                "story_target_magazine",
                story_selected_topic,
                subject_area,
                story_research_notes,
                "Krushi Vigyan",
            )

            if st.button("Use this research to write story + science article", key="story_write_article"):
                if not story_selected_topic.strip():
                    st.warning("Please manually select or type one Tab 2 topic before writing.")
                elif not story_target_magazine:
                    st.warning("Please select the target magazine personality before writing.")
                else:
                    story_selected_context = selected_topic_context(
                        story_selected_topic,
                        story_research_notes,
                    )
                    with st.spinner("Writing the article using the attached prompt style..."):
                        prompt = story_article_prompt(
                            month,
                            region,
                            subject_area,
                            st.session_state.get("story_saved_crop_focus", story_crop_focus),
                            article_length,
                            story_target_magazine,
                            story_selected_topic,
                            story_selected_context,
                        )
                        article, sources = generate_text(
                            client,
                            model,
                            prompt,
                            use_search=use_search_for_article,
                            temperature=temperature,
                        )
                        st.session_state["story_article"] = article
                        st.session_state["story_article_sources"] = sources
                        st.session_state["story_selected_topic"] = story_selected_topic
                        st.session_state["story_selected_target_magazine"] = story_target_magazine
                        st.session_state["story_research_notes_saved"] = story_selected_context
                        st.session_state.pop("story_rewritten_article", None)
                        st.session_state.pop("story_final_article", None)
                        st.session_state.pop("story_review", None)

        if "story_article" in st.session_state:
            st.subheader("Tab 2 Step 1: Story + science draft")
            story_draft = st.text_area(
                "Tab 2 draft article",
                value=st.session_state["story_article"],
                height=440,
                key="story_draft_article",
            )
            st.session_state["story_article"] = story_draft
            render_sources(
                "Tab 2 article grounding sources",
                st.session_state.get("story_article_sources", []),
            )

            st.download_button(
                "Download Tab 2 draft as TXT",
                data=story_draft,
                file_name="agro_sandesh_story_science_draft.txt",
                mime="text/plain",
                key="story_download_draft",
            )

            story_review_col, story_rewrite_col = st.columns(2)
            with story_review_col:
                story_review_clicked = st.button(
                    "Review Tab 2 draft quality",
                    key="story_review_draft",
                )
            with story_rewrite_col:
                story_rewrite_clicked = st.button(
                    "Rewrite with story + science style",
                    key="story_rewrite_button",
                )

            if story_review_clicked:
                with st.spinner("Reviewing Tab 2 article quality..."):
                    review, _ = generate_text(
                        client,
                        review_model,
                        review_prompt(
                            story_draft,
                            st.session_state.get("story_selected_target_magazine", "Agro Sandesh"),
                        ),
                        use_search=False,
                        temperature=0.25,
                        provider=review_provider,
                        api_keys=api_keys,
                    )
                    st.session_state["story_review"] = review

            if story_rewrite_clicked:
                with st.spinner("Rewriting with the attached prompt style..."):
                    rewrite, _ = generate_text(
                        client,
                        model,
                        story_rewrite_prompt(
                            month,
                            region,
                            subject_area,
                            st.session_state.get("story_saved_crop_focus", story_crop_focus),
                            article_length,
                            st.session_state.get("story_selected_target_magazine", "Agro Sandesh"),
                            st.session_state.get("story_selected_topic", ""),
                            st.session_state.get("story_research_notes_saved", ""),
                            story_draft,
                        ),
                        use_search=False,
                        temperature=0.45,
                    )
                    st.session_state["story_rewritten_article"] = rewrite
                    st.session_state.pop("story_final_article", None)

        if "story_review" in st.session_state:
            st.subheader("Tab 2 article review")
            st.markdown(st.session_state["story_review"])

        if "story_rewritten_article" in st.session_state:
            st.subheader("Tab 2 Step 2: Story + science rewrite")
            story_rewrite = st.text_area(
                "Tab 2 improved article",
                value=st.session_state["story_rewritten_article"],
                height=480,
                key="story_rewritten_text",
            )
            st.session_state["story_rewritten_article"] = story_rewrite

            st.download_button(
                "Download Tab 2 rewritten article as TXT",
                data=story_rewrite,
                file_name="agro_sandesh_story_science_rewrite.txt",
                mime="text/plain",
                key="story_download_rewrite",
            )

            st.caption("Soft evidence check included in editor pass.")

            if st.button(
                "Final editor check for Tab 2 magazine article",
                type="primary",
                key="story_final_editor_button",
            ):
                with st.spinner("Final editor is polishing the Tab 2 article..."):
                    final_article, _ = generate_text(
                        client,
                        model,
                        story_final_editor_prompt(
                            month,
                            region,
                            subject_area,
                            st.session_state.get("story_saved_crop_focus", story_crop_focus),
                            article_length,
                            st.session_state.get("story_selected_target_magazine", "Agro Sandesh"),
                            st.session_state.get("story_selected_topic", ""),
                            st.session_state.get("story_research_notes_saved", ""),
                            story_rewrite,
                        ),
                        use_search=False,
                        temperature=0.3,
                    )
                    st.session_state["story_final_article"] = final_article

        if "story_final_article" in st.session_state:
            st.subheader("Tab 2 Step 3: Final magazine-ready article")
            story_final = st.text_area(
                "Tab 2 final article for magazine",
                value=st.session_state["story_final_article"],
                height=540,
                key="story_final_text",
            )
            st.session_state["story_final_article"] = story_final

            story_txt_col, story_docx_col = st.columns(2)
            with story_txt_col:
                st.download_button(
                    "Download Tab 2 final article as TXT",
                    data=story_final,
                    file_name="agro_sandesh_story_science_final.txt",
                    mime="text/plain",
                    key="story_download_final_txt",
                )
            with story_docx_col:
                st.download_button(
                    "Download Tab 2 final article as Word DOCX",
                    data=make_docx(story_final),
                    file_name="agro_sandesh_story_science_final.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="story_download_final_docx",
                )

    with tab_farm_wisdom:
        st.subheader("Farm Wisdom Observation Prompt Workflow")
        st.write(
            "This tab adds the new master prompt style: observation first, "
            "farmer-scientist conversation, curiosity, reflection, practical "
            "wisdom, source-backed research, final editor check, and Word download."
        )

        wisdom_col1, wisdom_col2 = st.columns(2)
        with wisdom_col1:
            wisdom_topic_hint = st.text_input(
                "Topic optional for Tab 3",
                placeholder="Example: mites after dry weather, dusty leaves, orchard observation",
                key="wisdom_topic_hint",
            )
        with wisdom_col2:
            wisdom_crop_focus = st.text_input(
                "Crop for Tab 3",
                value=crop_focus,
                placeholder="Example: mango, sapota, okra, cotton, vegetables",
                key="wisdom_crop_focus",
            )

        wisdom_col3, wisdom_col4 = st.columns(2)
        with wisdom_col3:
            wisdom_season_context = st.text_input(
                "Season or field context",
                value=month,
                placeholder="Example: early monsoon, summer, post-rain humid weather",
                key="wisdom_season_context",
            )
        with wisdom_col4:
            wisdom_target_magazine = st.selectbox(
                "Initial target magazine for Tab 3 research",
                MAGAZINE_OPTIONS,
                index=0,
                key="wisdom_target_magazine",
            )

        if st.button(
            "Deep research and references for Tab 3",
            type="primary",
            key="wisdom_research_button",
        ):
            with st.spinner("Researching observation-first topic, field context, and references..."):
                prompt = farm_wisdom_research_prompt(
                    month,
                    region,
                    subject_area,
                    wisdom_crop_focus,
                    wisdom_topic_hint,
                    wisdom_season_context,
                    wisdom_target_magazine,
                )
                research, sources = generate_text(
                    client,
                    research_model,
                    prompt,
                    use_search=research_provider == PROVIDER_GEMINI,
                    temperature=0.35,
                    provider=research_provider,
                    api_keys=api_keys,
                )
                st.session_state["wisdom_research"] = research
                st.session_state["wisdom_sources"] = sources
                st.session_state["wisdom_saved_topic_hint"] = wisdom_topic_hint
                st.session_state["wisdom_saved_crop_focus"] = wisdom_crop_focus
                st.session_state["wisdom_saved_season_context"] = wisdom_season_context
                st.session_state["wisdom_saved_target_magazine"] = wisdom_target_magazine
                st.session_state.pop("wisdom_manual_topic", None)
                st.session_state.pop("wisdom_article_target_magazine", None)
                st.session_state.pop("wisdom_article", None)
                st.session_state.pop("wisdom_rewritten_article", None)
                st.session_state.pop("wisdom_final_article", None)
                st.session_state.pop("wisdom_review", None)

        if "wisdom_research" in st.session_state:
            st.subheader("Tab 3 research notes")
            st.markdown(st.session_state["wisdom_research"])
            render_sources("Tab 3 research sources", st.session_state.get("wisdom_sources", []))

            wisdom_selected_topic = st.text_input(
                "Manually selected topic for Tab 3",
                placeholder="Type or paste the topic you choose from the research options.",
                key="wisdom_manual_topic",
            )
            wisdom_research_notes = st.text_area(
                "Selected research notes for Tab 3",
                value=st.session_state["wisdom_research"],
                height=300,
                key="wisdom_research_notes",
            )
            wisdom_article_target_magazine = target_magazine_selector(
                "wisdom_article_target_magazine",
                wisdom_selected_topic,
                subject_area,
                wisdom_research_notes,
                st.session_state.get("wisdom_saved_target_magazine", wisdom_target_magazine),
            )

            if st.button("Use this research to write farm wisdom article", key="wisdom_write_article"):
                if not wisdom_selected_topic.strip():
                    st.warning("Please manually select or type one Tab 3 topic before writing.")
                elif not wisdom_article_target_magazine:
                    st.warning("Please select the target magazine personality before writing.")
                else:
                    wisdom_selected_context = selected_topic_context(
                        wisdom_selected_topic,
                        wisdom_research_notes,
                    )
                    with st.spinner("Writing the article using the observation-first master prompt..."):
                        prompt = farm_wisdom_article_prompt(
                            month,
                            region,
                            subject_area,
                            st.session_state.get("wisdom_saved_crop_focus", wisdom_crop_focus),
                            article_length,
                            wisdom_selected_topic,
                            st.session_state.get("wisdom_saved_season_context", wisdom_season_context),
                            wisdom_article_target_magazine,
                            wisdom_selected_context,
                        )
                        article, sources = generate_text(
                            client,
                            model,
                            prompt,
                            use_search=use_search_for_article,
                            temperature=temperature,
                        )
                        st.session_state["wisdom_article"] = article
                        st.session_state["wisdom_article_sources"] = sources
                        st.session_state["wisdom_selected_topic"] = wisdom_selected_topic
                        st.session_state["wisdom_selected_target_magazine"] = wisdom_article_target_magazine
                        st.session_state["wisdom_research_notes_saved"] = wisdom_selected_context
                        st.session_state.pop("wisdom_rewritten_article", None)
                        st.session_state.pop("wisdom_final_article", None)
                        st.session_state.pop("wisdom_review", None)

        if "wisdom_article" in st.session_state:
            st.subheader("Tab 3 Step 1: Farm wisdom draft")
            wisdom_draft = st.text_area(
                "Tab 3 draft article",
                value=st.session_state["wisdom_article"],
                height=440,
                key="wisdom_draft_article",
            )
            st.session_state["wisdom_article"] = wisdom_draft
            render_sources(
                "Tab 3 article grounding sources",
                st.session_state.get("wisdom_article_sources", []),
            )

            st.download_button(
                "Download Tab 3 draft as TXT",
                data=wisdom_draft,
                file_name="agri_farm_wisdom_draft.txt",
                mime="text/plain",
                key="wisdom_download_draft",
            )

            wisdom_review_col, wisdom_rewrite_col = st.columns(2)
            with wisdom_review_col:
                wisdom_review_clicked = st.button(
                    "Review Tab 3 draft quality",
                    key="wisdom_review_draft",
                )
            with wisdom_rewrite_col:
                wisdom_rewrite_clicked = st.button(
                    "Rewrite with farm wisdom style",
                    key="wisdom_rewrite_button",
                )

            if wisdom_review_clicked:
                with st.spinner("Reviewing Tab 3 article quality..."):
                    review, _ = generate_text(
                        client,
                        review_model,
                        review_prompt(
                            wisdom_draft,
                            st.session_state.get("wisdom_selected_target_magazine", "Krushi Vigyan"),
                        ),
                        use_search=False,
                        temperature=0.25,
                        provider=review_provider,
                        api_keys=api_keys,
                    )
                    st.session_state["wisdom_review"] = review

            if wisdom_rewrite_clicked:
                with st.spinner("Rewriting with the observation-first master prompt..."):
                    rewrite, _ = generate_text(
                        client,
                        model,
                        farm_wisdom_rewrite_prompt(
                            month,
                            region,
                            subject_area,
                            st.session_state.get("wisdom_saved_crop_focus", wisdom_crop_focus),
                            article_length,
                            st.session_state.get("wisdom_selected_topic", ""),
                            st.session_state.get("wisdom_saved_season_context", wisdom_season_context),
                            st.session_state.get("wisdom_selected_target_magazine", "Krushi Vigyan"),
                            st.session_state.get("wisdom_research_notes_saved", ""),
                            wisdom_draft,
                        ),
                        use_search=False,
                        temperature=0.45,
                    )
                    st.session_state["wisdom_rewritten_article"] = rewrite
                    st.session_state.pop("wisdom_final_article", None)

        if "wisdom_review" in st.session_state:
            st.subheader("Tab 3 article review")
            st.markdown(st.session_state["wisdom_review"])

        if "wisdom_rewritten_article" in st.session_state:
            st.subheader("Tab 3 Step 2: Farm wisdom rewrite")
            wisdom_rewrite = st.text_area(
                "Tab 3 improved article",
                value=st.session_state["wisdom_rewritten_article"],
                height=480,
                key="wisdom_rewritten_text",
            )
            st.session_state["wisdom_rewritten_article"] = wisdom_rewrite

            st.download_button(
                "Download Tab 3 rewritten article as TXT",
                data=wisdom_rewrite,
                file_name="agri_farm_wisdom_rewrite.txt",
                mime="text/plain",
                key="wisdom_download_rewrite",
            )

            st.caption("Soft evidence check included in editor pass.")

            if st.button(
                "Final editor check for Tab 3 magazine article",
                type="primary",
                key="wisdom_final_editor_button",
            ):
                with st.spinner("Final editor is polishing the Tab 3 article..."):
                    final_article, _ = generate_text(
                        client,
                        model,
                        farm_wisdom_final_editor_prompt(
                            month,
                            region,
                            subject_area,
                            st.session_state.get("wisdom_saved_crop_focus", wisdom_crop_focus),
                            article_length,
                            st.session_state.get("wisdom_selected_topic", ""),
                            st.session_state.get("wisdom_saved_season_context", wisdom_season_context),
                            st.session_state.get("wisdom_selected_target_magazine", "Krushi Vigyan"),
                            st.session_state.get("wisdom_research_notes_saved", ""),
                            wisdom_rewrite,
                        ),
                        use_search=False,
                        temperature=0.3,
                    )
                    st.session_state["wisdom_final_article"] = final_article

        if "wisdom_final_article" in st.session_state:
            st.subheader("Tab 3 Step 3: Final magazine-ready article")
            wisdom_final = st.text_area(
                "Tab 3 final article for magazine",
                value=st.session_state["wisdom_final_article"],
                height=540,
                key="wisdom_final_text",
            )
            st.session_state["wisdom_final_article"] = wisdom_final

            wisdom_txt_col, wisdom_docx_col = st.columns(2)
            with wisdom_txt_col:
                st.download_button(
                    "Download Tab 3 final article as TXT",
                    data=wisdom_final,
                    file_name="agri_farm_wisdom_final.txt",
                    mime="text/plain",
                    key="wisdom_download_final_txt",
                )
            with wisdom_docx_col:
                st.download_button(
                    "Download Tab 3 final article as Word DOCX",
                    data=make_docx(wisdom_final),
                    file_name="agri_farm_wisdom_final.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="wisdom_download_final_docx",
                )

    with tab_field_discovery:
        st.subheader("Field Discovery Prompt Workflow")
        st.write(
            "This tab adds the new master prompt style: scene first, visual "
            "observation, curiosity, delayed discovery, science as understanding, "
            "practical meaning, source-backed research, final editor check, and Word download."
        )

        discovery_col1, discovery_col2 = st.columns(2)
        with discovery_col1:
            discovery_topic_hint = st.text_input(
                "Topic optional for Tab 4",
                placeholder="Example: orchard after dry wind, leaves changing colour, mites after dust",
                key="discovery_topic_hint",
            )
        with discovery_col2:
            discovery_crop_focus = st.text_input(
                "Crop for Tab 4",
                value=crop_focus,
                placeholder="Example: mango, sapota, okra, cotton, vegetables",
                key="discovery_crop_focus",
            )

        discovery_col3, discovery_col4 = st.columns(2)
        with discovery_col3:
            discovery_season_context = st.text_input(
                "Season or scene context",
                value=month,
                placeholder="Example: early monsoon morning, summer dry spell, post-rain humidity",
                key="discovery_season_context",
            )
        with discovery_col4:
            discovery_target_magazine = st.selectbox(
                "Initial target magazine for Tab 4 research",
                MAGAZINE_OPTIONS,
                index=0,
                key="discovery_target_magazine",
            )

        if st.button(
            "Deep research and references for Tab 4",
            type="primary",
            key="discovery_research_button",
        ):
            with st.spinner("Researching scene-based topic, observations, and references..."):
                prompt = field_discovery_research_prompt(
                    month,
                    region,
                    subject_area,
                    discovery_crop_focus,
                    discovery_topic_hint,
                    discovery_season_context,
                    discovery_target_magazine,
                )
                research, sources = generate_text(
                    client,
                    research_model,
                    prompt,
                    use_search=research_provider == PROVIDER_GEMINI,
                    temperature=0.35,
                    provider=research_provider,
                    api_keys=api_keys,
                )
                st.session_state["discovery_research"] = research
                st.session_state["discovery_sources"] = sources
                st.session_state["discovery_saved_topic_hint"] = discovery_topic_hint
                st.session_state["discovery_saved_crop_focus"] = discovery_crop_focus
                st.session_state["discovery_saved_season_context"] = discovery_season_context
                st.session_state["discovery_saved_target_magazine"] = discovery_target_magazine
                st.session_state.pop("discovery_manual_topic", None)
                st.session_state.pop("discovery_article_target_magazine", None)
                st.session_state.pop("discovery_article", None)
                st.session_state.pop("discovery_rewritten_article", None)
                st.session_state.pop("discovery_final_article", None)
                st.session_state.pop("discovery_review", None)

        if "discovery_research" in st.session_state:
            st.subheader("Tab 4 research notes")
            st.markdown(st.session_state["discovery_research"])
            render_sources("Tab 4 research sources", st.session_state.get("discovery_sources", []))

            discovery_selected_topic = st.text_input(
                "Manually selected topic for Tab 4",
                placeholder="Type or paste the topic you choose from the research options.",
                key="discovery_manual_topic",
            )
            discovery_research_notes = st.text_area(
                "Selected research notes for Tab 4",
                value=st.session_state["discovery_research"],
                height=300,
                key="discovery_research_notes",
            )
            discovery_article_target_magazine = target_magazine_selector(
                "discovery_article_target_magazine",
                discovery_selected_topic,
                subject_area,
                discovery_research_notes,
                st.session_state.get("discovery_saved_target_magazine", discovery_target_magazine),
            )

            if st.button("Use this research to write field discovery article", key="discovery_write_article"):
                if not discovery_selected_topic.strip():
                    st.warning("Please manually select or type one Tab 4 topic before writing.")
                elif not discovery_article_target_magazine:
                    st.warning("Please select the target magazine personality before writing.")
                else:
                    discovery_selected_context = selected_topic_context(
                        discovery_selected_topic,
                        discovery_research_notes,
                    )
                    with st.spinner("Writing the article using the field-discovery master prompt..."):
                        prompt = field_discovery_article_prompt(
                            month,
                            region,
                            subject_area,
                            st.session_state.get("discovery_saved_crop_focus", discovery_crop_focus),
                            article_length,
                            discovery_selected_topic,
                            st.session_state.get("discovery_saved_season_context", discovery_season_context),
                            discovery_article_target_magazine,
                            discovery_selected_context,
                        )
                        article, sources = generate_text(
                            client,
                            model,
                            prompt,
                            use_search=use_search_for_article,
                            temperature=temperature,
                        )
                        st.session_state["discovery_article"] = article
                        st.session_state["discovery_article_sources"] = sources
                        st.session_state["discovery_selected_topic"] = discovery_selected_topic
                        st.session_state["discovery_selected_target_magazine"] = discovery_article_target_magazine
                        st.session_state["discovery_research_notes_saved"] = discovery_selected_context
                        st.session_state.pop("discovery_rewritten_article", None)
                        st.session_state.pop("discovery_final_article", None)
                        st.session_state.pop("discovery_review", None)

        if "discovery_article" in st.session_state:
            st.subheader("Tab 4 Step 1: Field discovery draft")
            discovery_draft = st.text_area(
                "Tab 4 draft article",
                value=st.session_state["discovery_article"],
                height=440,
                key="discovery_draft_article",
            )
            st.session_state["discovery_article"] = discovery_draft
            render_sources(
                "Tab 4 article grounding sources",
                st.session_state.get("discovery_article_sources", []),
            )

            st.download_button(
                "Download Tab 4 draft as TXT",
                data=discovery_draft,
                file_name="agri_field_discovery_draft.txt",
                mime="text/plain",
                key="discovery_download_draft",
            )

            discovery_review_col, discovery_rewrite_col = st.columns(2)
            with discovery_review_col:
                discovery_review_clicked = st.button(
                    "Review Tab 4 draft quality",
                    key="discovery_review_draft",
                )
            with discovery_rewrite_col:
                discovery_rewrite_clicked = st.button(
                    "Rewrite with field discovery style",
                    key="discovery_rewrite_button",
                )

            if discovery_review_clicked:
                with st.spinner("Reviewing Tab 4 article quality..."):
                    review, _ = generate_text(
                        client,
                        review_model,
                        review_prompt(
                            discovery_draft,
                            st.session_state.get("discovery_selected_target_magazine", "Krushi Vigyan"),
                        ),
                        use_search=False,
                        temperature=0.25,
                        provider=review_provider,
                        api_keys=api_keys,
                    )
                    st.session_state["discovery_review"] = review

            if discovery_rewrite_clicked:
                with st.spinner("Rewriting with the field-discovery master prompt..."):
                    rewrite, _ = generate_text(
                        client,
                        model,
                        field_discovery_rewrite_prompt(
                            month,
                            region,
                            subject_area,
                            st.session_state.get("discovery_saved_crop_focus", discovery_crop_focus),
                            article_length,
                            st.session_state.get("discovery_selected_topic", ""),
                            st.session_state.get("discovery_saved_season_context", discovery_season_context),
                            st.session_state.get("discovery_selected_target_magazine", "Krushi Vigyan"),
                            st.session_state.get("discovery_research_notes_saved", ""),
                            discovery_draft,
                        ),
                        use_search=False,
                        temperature=0.45,
                    )
                    st.session_state["discovery_rewritten_article"] = rewrite
                    st.session_state.pop("discovery_final_article", None)

        if "discovery_review" in st.session_state:
            st.subheader("Tab 4 article review")
            st.markdown(st.session_state["discovery_review"])

        if "discovery_rewritten_article" in st.session_state:
            st.subheader("Tab 4 Step 2: Field discovery rewrite")
            discovery_rewrite = st.text_area(
                "Tab 4 improved article",
                value=st.session_state["discovery_rewritten_article"],
                height=480,
                key="discovery_rewritten_text",
            )
            st.session_state["discovery_rewritten_article"] = discovery_rewrite

            st.download_button(
                "Download Tab 4 rewritten article as TXT",
                data=discovery_rewrite,
                file_name="agri_field_discovery_rewrite.txt",
                mime="text/plain",
                key="discovery_download_rewrite",
            )

            st.caption("Soft evidence check included in editor pass.")

            if st.button(
                "Final editor check for Tab 4 magazine article",
                type="primary",
                key="discovery_final_editor_button",
            ):
                with st.spinner("Final editor is polishing the Tab 4 article..."):
                    final_article, _ = generate_text(
                        client,
                        model,
                        field_discovery_final_editor_prompt(
                            month,
                            region,
                            subject_area,
                            st.session_state.get("discovery_saved_crop_focus", discovery_crop_focus),
                            article_length,
                            st.session_state.get("discovery_selected_topic", ""),
                            st.session_state.get("discovery_saved_season_context", discovery_season_context),
                            st.session_state.get("discovery_selected_target_magazine", "Krushi Vigyan"),
                            st.session_state.get("discovery_research_notes_saved", ""),
                            discovery_rewrite,
                        ),
                        use_search=False,
                        temperature=0.3,
                    )
                    st.session_state["discovery_final_article"] = final_article

        if "discovery_final_article" in st.session_state:
            st.subheader("Tab 4 Step 3: Final magazine-ready article")
            discovery_final = st.text_area(
                "Tab 4 final article for magazine",
                value=st.session_state["discovery_final_article"],
                height=540,
                key="discovery_final_text",
            )
            st.session_state["discovery_final_article"] = discovery_final

            discovery_txt_col, discovery_docx_col = st.columns(2)
            with discovery_txt_col:
                st.download_button(
                    "Download Tab 4 final article as TXT",
                    data=discovery_final,
                    file_name="agri_field_discovery_final.txt",
                    mime="text/plain",
                    key="discovery_download_final_txt",
                )
            with discovery_docx_col:
                st.download_button(
                    "Download Tab 4 final article as Word DOCX",
                    data=make_docx(discovery_final),
                    file_name="agri_field_discovery_final.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="discovery_download_final_docx",
                )


if __name__ == "__main__":
    main()
