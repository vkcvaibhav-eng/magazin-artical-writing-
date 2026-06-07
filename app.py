import os
from datetime import datetime

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


st.set_page_config(
    page_title="Agro Sandesh Article Writer",
    page_icon="AS",
    layout="wide",
)


def get_api_key() -> str:
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    entered_key = st.sidebar.text_input(
        "Gemini API key",
        value=env_key,
        type="password",
        help="Use GEMINI_API_KEY in .env locally, or add it in Streamlit secrets/settings when hosted.",
    ).strip()
    return entered_key or env_key


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


def generate_text(
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


def topic_research_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
) -> str:
    return f"""
You are an agricultural research assistant for Agro Sandesh magazine.

Use Google Search grounding to identify current, prevailing, and seasonally relevant
agriculture article topics for {month} in {region}.

Subject focus: {subject_area}
Crop focus, if any: {crop_focus or "No specific crop focus"}

Research priorities:
- South Gujarat and Gujarat agriculture
- Agricultural acarology and agricultural entomology
- Current pest and mite problems
- Seasonal crop stage and weather-linked pest risk
- Practical advisory value for farmers
- Relevance to this month
- Suitability for an agricultural magazine article

Return 10 topic options. For each option include:
1. Gujarati title
2. English explanation
3. Why it matters in {month}
4. Region relevance
5. Farmer benefit
6. Suitability score out of 10

Then select the single best topic and explain why it is the strongest choice.

Write clearly. Do not invent local outbreaks or official advisories. If evidence is
uncertain, say so and suggest field verification with local agricultural university,
KVK, or extension officers.
""".strip()


def article_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    article_length: str,
    selected_topic: str,
) -> str:
    return f"""
Write a full Gujarati agricultural extension article for Agro Sandesh.

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
6. Build logical chains: cause to effect to consequence to solution.
7. Keep paragraphs focused on one central idea.
8. Every recommendation must answer:
   - What should farmers do?
   - Why should they do it?
   - How does it improve yield, quality, cost, risk, sustainability, or profit?
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
Farmer problem -> Scientific reason -> Practical solution -> Farmer benefit.

Target publication: Agro Sandesh
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


def review_prompt(article: str) -> str:
    return f"""
Review the following Gujarati agriculture article for Agro Sandesh.

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
10. Give a rating out of 10.

Then provide specific improvements and rewrite only weak paragraphs if needed.

Article:
{article}
""".strip()


def render_sources(title: str, sources: list[dict[str, str]]) -> None:
    if not sources:
        return

    with st.expander(title, expanded=False):
        for index, source in enumerate(sources, start=1):
            st.markdown(f"{index}. [{source['title']}]({source['uri']})")


def main() -> None:
    st.title("Agro Sandesh Gujarati Agriculture Article Writer")
    st.caption(
        "Research current Gujarat agriculture topics with Gemini Google Search grounding, "
        "then draft a farmer-centric Gujarati article."
    )

    api_key = get_api_key()

    with st.sidebar:
        st.header("Settings")
        model = st.text_input("Gemini model", value="gemini-3.5-flash")
        temperature = st.slider("Creativity", 0.1, 1.0, 0.7, 0.1)
        use_search_for_article = st.checkbox(
            "Use Google Search while writing article",
            value=True,
        )

    if not api_key:
        st.warning("Enter your Gemini API key in the sidebar to continue.")
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

    client = build_client(api_key)

    if st.button("Find latest topics", type="primary"):
        with st.spinner("Researching current and seasonally relevant topics..."):
            prompt = topic_research_prompt(month, region, subject_area, crop_focus)
            topics, sources = generate_text(
                client,
                model,
                prompt,
                use_search=True,
                temperature=0.45,
            )
            st.session_state["topics"] = topics
            st.session_state["topic_sources"] = sources

    if "topics" in st.session_state:
        st.subheader("Suggested topics")
        st.markdown(st.session_state["topics"])
        render_sources("Research sources", st.session_state.get("topic_sources", []))

        selected_topic = st.text_area(
            "Selected topic and notes",
            value=st.session_state["topics"],
            height=260,
        )

        if st.button("Write Gujarati article"):
            with st.spinner("Writing the Gujarati article draft..."):
                prompt = article_prompt(
                    month,
                    region,
                    subject_area,
                    crop_focus,
                    article_length,
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

    if "article" in st.session_state:
        st.subheader("Gujarati article draft")
        st.markdown(st.session_state["article"])
        render_sources("Article grounding sources", st.session_state.get("article_sources", []))

        st.download_button(
            "Download article as TXT",
            data=st.session_state["article"],
            file_name="agro_sandesh_gujarati_article.txt",
            mime="text/plain",
        )

        if st.button("Review article quality"):
            with st.spinner("Reviewing article quality..."):
                review, _ = generate_text(
                    client,
                    model,
                    review_prompt(st.session_state["article"]),
                    use_search=False,
                    temperature=0.25,
                )
                st.session_state["review"] = review

    if "review" in st.session_state:
        st.subheader("Article review")
        st.markdown(st.session_state["review"])


if __name__ == "__main__":
    main()
