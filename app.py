import os
import re
import zipfile
from datetime import datetime
from html import escape
from io import BytesIO

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


def rewrite_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    article_length: str,
    selected_topic: str,
    article: str,
) -> str:
    return f"""
Rewrite the following Gujarati agriculture article into a stronger magazine-quality
Agro Sandesh article.

Important authorship instruction:
- Do not claim that Dr. M. S. Swaminathan wrote the article.
- Do not write in first person as Dr. Swaminathan.
- Use an original Gujarati extension-writing voice inspired by his public values:
  farmer welfare, scientific temper, field wisdom, sustainability, productivity,
  practical hope, and respect for small and progressive farmers.

Rewrite goals:
1. Make the opening more field-based and farmer-oriented.
2. Improve the flow: farmer problem -> scientific reason -> practical solution -> benefit.
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

Target publication: Agro Sandesh
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
    selected_topic: str,
    article: str,
) -> str:
    return f"""
Act as the final Gujarati magazine editor for Agro Sandesh.

Final editorial standard:
- Do not claim that Dr. M. S. Swaminathan wrote the article.
- Keep an original voice inspired by his farmer-centric scientific communication.
- Make the final article publication-ready for a Gujarati agriculture magazine.

Final checks to apply silently:
1. Strong Gujarati title.
2. Farmer-oriented first paragraph.
3. Clear seasonal and regional relevance.
4. Simple scientific explanation.
5. Practical step-by-step recommendations.
6. Every recommendation explains farmer benefit.
7. Good magazine flow with readable paragraphs and useful subheadings.
8. No research-paper style headings.
9. No unsafe pesticide dosage claims.
10. No unsupported outbreak or official-advisory claims.
11. Natural Gujarati language, polished grammar, and no unnecessary English.
12. Positive practical takeaway at the end.

Target publication: Agro Sandesh
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
                st.session_state["selected_topic"] = selected_topic
                st.session_state.pop("rewritten_article", None)
                st.session_state.pop("final_article", None)
                st.session_state.pop("review", None)

    if "article" in st.session_state:
        st.subheader("Step 1: Gujarati article draft")
        draft_article = st.text_area(
            "Draft article",
            value=st.session_state["article"],
            height=420,
        )
        st.session_state["article"] = draft_article
        render_sources("Article grounding sources", st.session_state.get("article_sources", []))

        st.download_button(
            "Download draft as TXT",
            data=draft_article,
            file_name="agro_sandesh_draft_article.txt",
            mime="text/plain",
        )

        col_review, col_rewrite = st.columns(2)

        with col_review:
            review_clicked = st.button("Review draft quality")

        with col_rewrite:
            rewrite_clicked = st.button("Rewrite in Swaminathan-inspired style")

        if review_clicked:
            with st.spinner("Reviewing article quality..."):
                review, _ = generate_text(
                    client,
                    model,
                    review_prompt(draft_article),
                    use_search=False,
                    temperature=0.25,
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
        )
        st.session_state["rewritten_article"] = rewritten_article

        st.download_button(
            "Download rewritten article as TXT",
            data=rewritten_article,
            file_name="agro_sandesh_rewritten_article.txt",
            mime="text/plain",
        )

        if st.button("Final editor check and make magazine article", type="primary"):
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
        )
        st.session_state["final_article"] = final_article

        col_txt, col_docx = st.columns(2)
        with col_txt:
            st.download_button(
                "Download final article as TXT",
                data=final_article,
                file_name="agro_sandesh_final_article.txt",
                mime="text/plain",
            )
        with col_docx:
            st.download_button(
                "Download final article as Word DOCX",
                data=make_docx(final_article),
                file_name="agro_sandesh_final_article.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )


if __name__ == "__main__":
    main()
