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

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None


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
    "Whole Gujarat",
    "South Gujarat",
    "Saurashtra",
    "Central Gujarat (Middle Gujarat)",
    "North Gujarat",
    "Kutch (Kachchh)",
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
    page_icon="🌾",
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
        "https://api.perplexity.ai/chat/completions",
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
    # Reasoning models prepend a <think>...</think> block; keep only the answer.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
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
    }
    # Reasoning models (o-series, gpt-5.x) reject the temperature parameter.
    if not re.match(r"^(o\d|gpt-5)", model):
        payload["temperature"] = temperature

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


def safe_generate_text(*args, **kwargs):
    try:
        return generate_text(*args, **kwargs)
    except Exception as exc:
        st.error(f"AI request failed: {exc}")
        st.stop()


PPQS_LABEL_COLUMNS = [
    "source_file",
    "source_page",
    "pesticide_name",
    "formulation",
    "crop",
    "pest",
    "ai_dose_per_ha",
    "formulation_dose_per_ha",
    "dilution_water_l_per_ha",
    "waiting_period_days",
    "use_type",
    "dose_per_10_litre",
    "remarks",
]


def clean_ppqs_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[–—−]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_crop_name(text: str) -> str:
    text = clean_ppqs_text(text).lower()
    text = re.sub(r"[^a-z0-9\s/&,-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_pest_name(text: str) -> str:
    text = clean_ppqs_text(text).lower()
    text = re.sub(r"[^a-z0-9\s/&,-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _empty_ppqs_df() -> "pd.DataFrame":
    return pd.DataFrame(columns=PPQS_LABEL_COLUMNS)


def _require_ppqs_dependencies() -> None:
    missing = []
    if pd is None:
        missing.append("pandas")
    if pdfplumber is None:
        missing.append("pdfplumber")
    if fuzz is None:
        missing.append("rapidfuzz")
    if missing:
        raise ImportError("Install missing packages: " + ", ".join(missing))


def _clean_cell(cell) -> str:
    if cell is None:
        return ""
    return clean_ppqs_text(str(cell))


def _looks_like_pesticide_heading(text: str) -> bool:
    text = clean_ppqs_text(text)
    if not text:
        return False
    if re.search(r"\b(crop|pest|dose|dilution|waiting|formulation)\b", text, re.IGNORECASE):
        return False
    return bool(
        re.search(
            r"\d+(?:\.\d+)?\s*%\s*(?:SC|EC|SP|WP|WG|SG|SL|GR|FS|DS|OD|ME|CS|EW|DP|ULV|WDG)\b",
            text,
            re.IGNORECASE,
        )
    )


def _split_pesticide_heading(text: str) -> tuple[str, str]:
    text = clean_ppqs_text(text)
    match = re.search(
        r"(.+?)\s+(\d+(?:\.\d+)?\s*%\s*(?:SC|EC|SP|WP|WG|SG|SL|GR|FS|DS|OD|ME|CS|EW|DP|ULV|WDG)\b.*)",
        text,
        re.IGNORECASE,
    )
    if match:
        return clean_ppqs_text(match.group(1)), clean_ppqs_text(match.group(2))
    return text, ""


def _is_header_row(cells: list[str]) -> bool:
    joined = " ".join(cells).lower()
    return "crop" in joined and any(word in joined for word in ["pest", "dose", "dilution", "waiting"])


def _invalid_water_volume(text: str) -> bool:
    value = clean_ppqs_text(text).lower()
    if not value or value in {"-", "na", "n/a", "nil"}:
        return True
    return any(
        phrase in value
        for phrase in [
            "not required",
            "broadcast",
            "seed dresser",
            "seed treatment",
            "dry seed",
            "not applicable",
        ]
    )


def _numeric_range(text: str):
    text = clean_ppqs_text(text).replace(",", "")
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return None
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return min(numbers[0], numbers[1]), max(numbers[0], numbers[1])


def _dose_unit(text: str) -> str:
    value = clean_ppqs_text(text).lower()
    if re.search(r"\bml\b|m\.l\.|litre|liter", value):
        return "ml"
    if re.search(r"\bkg\b|kilogram", value):
        return "kg"
    if re.search(r"\bg\b|\bgm\b|gram", value):
        return "g"
    return ""


def calculate_dose_per_10_litres(formulation_dose, water_volume) -> str:
    dose_text = clean_ppqs_text(formulation_dose)
    water_text = clean_ppqs_text(water_volume)
    if _invalid_water_volume(water_text):
        return "Not applicable / cannot calculate from label water volume"

    unit = _dose_unit(dose_text)
    if unit not in {"g", "ml"}:
        return "Not applicable / cannot calculate from label dose unit"

    dose_range = _numeric_range(dose_text)
    water_range = _numeric_range(water_text)
    if not dose_range or not water_range or water_range[0] <= 0 or water_range[1] <= 0:
        return "Not applicable / cannot calculate from label water volume"

    low = (dose_range[0] / water_range[1]) * 10
    high = (dose_range[1] / water_range[0]) * 10
    if abs(low - high) < 0.0001:
        return f"{low:.1f} {unit} / 10 L water"
    return f"{low:.1f}-{high:.1f} {unit} / 10 L water"


def _detect_use_type(row_text: str, formulation: str, formulation_dose: str, water_volume: str) -> str:
    text = " ".join([row_text, formulation, formulation_dose, water_volume]).lower()
    if any(token in text for token in ["seed treatment", "seed dresser", " g/kg", " ml/kg", "kg seed", " ds", " fs"]):
        return "Seed treatment"
    if any(token in text for token in ["broadcast", "whorl", "soil", "bait", "fumigation", "burrow", "granule"]):
        return "Granule / broadcast / soil application"
    if not _invalid_water_volume(water_volume):
        return "Foliar spray"
    return "Other / manual verification"


def _label_claim_row(
    *,
    source_file: str,
    source_page: int,
    pesticide_name: str,
    formulation: str,
    crop: str,
    pest: str,
    ai_dose: str,
    formulation_dose: str,
    water_volume: str,
    waiting_period: str,
    raw_text: str,
    remarks: str = "",
) -> dict[str, str]:
    use_type = _detect_use_type(raw_text, formulation, formulation_dose, water_volume)
    dose_per_10_litre = (
        calculate_dose_per_10_litres(formulation_dose, water_volume)
        if use_type == "Foliar spray"
        else "Not applicable / cannot calculate from label water volume"
    )
    if not crop or not pest or not pesticide_name:
        remarks = "; ".join(filter(None, [remarks, "Needs manual verification"]))
    return {
        "source_file": source_file,
        "source_page": source_page,
        "pesticide_name": pesticide_name,
        "formulation": formulation,
        "crop": crop,
        "pest": pest,
        "ai_dose_per_ha": ai_dose,
        "formulation_dose_per_ha": formulation_dose,
        "dilution_water_l_per_ha": water_volume,
        "waiting_period_days": waiting_period,
        "use_type": use_type,
        "dose_per_10_litre": dose_per_10_litre,
        "remarks": remarks,
    }


def _row_from_cells(cells: list[str], current_pesticide: dict, source_file: str, page_num: int, current_crop: str):
    padded = (cells + [""] * 7)[:7]
    crop = padded[0] or current_crop
    pest = padded[1]
    ai_dose = padded[2]
    formulation_dose = padded[3]
    water_volume = padded[4]
    waiting_period = padded[5]
    raw_text = " | ".join(cells)
    row = _label_claim_row(
        source_file=source_file,
        source_page=page_num,
        pesticide_name=current_pesticide.get("name", ""),
        formulation=current_pesticide.get("formulation", ""),
        crop=crop,
        pest=pest,
        ai_dose=ai_dose,
        formulation_dose=formulation_dose,
        water_volume=water_volume,
        waiting_period=waiting_period,
        raw_text=raw_text,
        remarks="Needs manual verification" if len([cell for cell in cells if cell]) < 5 else "",
    )
    return row, crop


def extract_label_claim_rows_from_pdf(uploaded_file) -> "pd.DataFrame":
    source_file = getattr(uploaded_file, "name", "uploaded_ppqs_major_uses.pdf")
    return extract_label_claim_rows_from_bytes(uploaded_file.getvalue(), source_file)


def extract_label_claim_rows_from_bytes(pdf_bytes: bytes, source_file: str) -> "pd.DataFrame":
    _require_ppqs_dependencies()
    rows = []

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        current_pesticide = {"name": "", "formulation": ""}
        current_crop = ""
        for page_index, page in enumerate(pdf.pages, start=1):
            page_tables = page.extract_tables() or []
            for table in page_tables:
                for raw_row in table or []:
                    cells = [_clean_cell(cell) for cell in (raw_row or [])]
                    non_empty = [cell for cell in cells if cell]
                    if not non_empty:
                        continue
                    joined = " ".join(non_empty)
                    if _looks_like_pesticide_heading(joined):
                        name, formulation = _split_pesticide_heading(joined)
                        current_pesticide = {"name": name, "formulation": formulation}
                        current_crop = ""
                        continue
                    if _is_header_row(non_empty) or not current_pesticide.get("name"):
                        continue
                    row, current_crop = _row_from_cells(
                        cells,
                        current_pesticide,
                        source_file,
                        page_index,
                        current_crop,
                    )
                    rows.append(row)

            text = page.extract_text() or ""
            for raw_line in text.splitlines():
                line = clean_ppqs_text(raw_line)
                if not line:
                    continue
                if _looks_like_pesticide_heading(line):
                    name, formulation = _split_pesticide_heading(line)
                    current_pesticide = {"name": name, "formulation": formulation}
                    continue
                if not current_pesticide.get("name"):
                    continue
                parts = [clean_ppqs_text(part) for part in re.split(r"\s{2,}|\t+", line) if clean_ppqs_text(part)]
                if len(parts) >= 5 and not _is_header_row(parts):
                    row, current_crop = _row_from_cells(
                        parts,
                        current_pesticide,
                        source_file,
                        page_index,
                        current_crop,
                    )
                    row["remarks"] = "; ".join(filter(None, [row["remarks"], "Text extraction row - verify against PDF"]))
                    rows.append(row)

    if not rows:
        return _empty_ppqs_df()

    df = pd.DataFrame(rows)
    for column in PPQS_LABEL_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[PPQS_LABEL_COLUMNS].drop_duplicates().reset_index(drop=True)


def parse_ppqs_pdf(uploaded_file) -> "pd.DataFrame":
    return extract_label_claim_rows_from_pdf(uploaded_file)


PPQS_MAJOR_USES_PAGE = "https://ppqs.gov.in/divisions/cib-rc/major-uses-of-pesticides"
PPQS_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
}


def _ppqs_get(url: str, timeout: int) -> requests.Response:
    try:
        response = requests.get(url, headers=PPQS_REQUEST_HEADERS, timeout=timeout)
    except requests.exceptions.SSLError:
        # Some government servers ship incomplete certificate chains.
        response = requests.get(
            url, headers=PPQS_REQUEST_HEADERS, timeout=timeout, verify=False
        )
    response.raise_for_status()
    return response


def _ppqs_absolute_url(href: str) -> str:
    return href if href.lower().startswith("http") else "https://ppqs.gov.in" + href


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_ppqs_document_list() -> list[dict[str, str]]:
    html = _ppqs_get(PPQS_MAJOR_USES_PAGE, timeout=60).text
    documents = []
    seen = set()

    # The download links sit in table rows whose text carries the document title.
    for row_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", html, re.IGNORECASE | re.DOTALL):
        row_html = row_match.group(1)
        link = re.search(r'href="([^"]+\.pdf[^"]*)"', row_html, re.IGNORECASE)
        if not link:
            continue
        url = _ppqs_absolute_url(link.group(1))
        text = re.sub(r"<[^>]+>", " ", row_html)
        text = re.sub(r"\s+", " ", text).strip()
        title = re.sub(r"\b(download|view)\b", "", text, flags=re.IGNORECASE)
        title = re.sub(r"^\s*\d+\s*[.)]?\s*", "", title).strip(" -|:")
        if not title:
            title = url.rsplit("/", 1)[-1]
        if "major uses" not in title.lower() and "mup" not in url.lower():
            continue
        if url not in seen:
            seen.add(url)
            documents.append({"title": title, "url": url})

    if documents:
        return documents

    # Fallback if the page stops using tables: take every PDF link, name by file.
    for match in re.finditer(r'href="([^"]+\.pdf[^"]*)"', html, re.IGNORECASE):
        url = _ppqs_absolute_url(match.group(1))
        if "mup" not in url.lower():
            continue
        if url not in seen:
            seen.add(url)
            documents.append({"title": url.rsplit("/", 1)[-1], "url": url})
    return documents


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=6)
def download_and_parse_ppqs_pdf(url: str, title: str) -> "pd.DataFrame":
    pdf_bytes = _ppqs_get(url, timeout=300).content
    source_name = title or url.rsplit("/", 1)[-1]
    return extract_label_claim_rows_from_bytes(pdf_bytes, source_name)


PPQS_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppqs_label_cache.json")


def load_ppqs_label_cache():
    """Return (dataframe, meta) from the saved PPQS label cache, or (None, {}).

    The cache is a speed/offline fallback; the live PPQS fetch stays the source
    of truth and refreshes it. meta carries the 'fetched' date and document list.
    """
    try:
        import json

        with open(PPQS_CACHE_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, ValueError):
        return None, {}

    rows = data.get("rows") or []
    meta = {"fetched": data.get("fetched", ""), "documents": data.get("documents", [])}
    if pd is None or not rows:
        return None, meta

    df = pd.DataFrame(rows)
    for column in PPQS_LABEL_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[PPQS_LABEL_COLUMNS], meta


def save_ppqs_label_cache(df, documents: list[str]) -> str:
    """Persist parsed label rows so later runs load instantly / survive an outage."""
    if pd is None or df is None or df.empty:
        return ""
    try:
        import datetime
        import json

        payload = {
            "fetched": datetime.date.today().isoformat(),
            "documents": documents,
            "columns": PPQS_LABEL_COLUMNS,
            "rows": df.to_dict("records"),
        }
        with open(PPQS_CACHE_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        return payload["fetched"]
    except Exception:
        return ""


def _keyword_overlap(query: str, value: str) -> bool:
    query_tokens = {token for token in normalize_pest_name(query).split() if len(token) >= 3}
    value_tokens = {token for token in normalize_pest_name(value).split() if len(token) >= 3}
    return bool(query_tokens and query_tokens.intersection(value_tokens))


def search_label_claims(df, crop_query, pest_query) -> "pd.DataFrame":
    _require_ppqs_dependencies()
    if df is None or df.empty:
        return _empty_ppqs_df()

    crop_norm = normalize_crop_name(crop_query)
    pest_norm = normalize_pest_name(pest_query)
    work = df.copy()
    work["_crop_norm"] = work["crop"].map(normalize_crop_name)
    work["_pest_norm"] = work["pest"].map(normalize_pest_name)

    matches = []
    for index, row in work.iterrows():
        row_crop = row["_crop_norm"]
        row_pest = row["_pest_norm"]
        exact_crop = bool(
            crop_norm and row_crop and (crop_norm == row_crop or crop_norm in row_crop or row_crop in crop_norm)
        )
        exact_pest = bool(
            pest_norm and row_pest and (pest_norm == row_pest or pest_norm in row_pest or row_pest in pest_norm)
        )
        fuzzy_crop = bool(crop_norm and fuzz and fuzz.partial_ratio(crop_norm, row_crop) >= 82)
        fuzzy_pest = bool(pest_norm and fuzz and fuzz.partial_ratio(pest_norm, row_pest) >= 78)
        pest_overlap = bool(pest_norm and _keyword_overlap(pest_norm, row_pest))

        rank = None
        match_type = ""
        if exact_crop and exact_pest:
            rank, match_type = 1, "exact crop + exact pest"
        elif exact_crop and (fuzzy_pest or pest_overlap):
            rank, match_type = 2, "exact crop + fuzzy/keyword pest"
        elif (exact_crop or fuzzy_crop) and (exact_pest or fuzzy_pest or pest_overlap):
            rank, match_type = 3, "fuzzy crop + fuzzy/keyword pest"
        elif exact_crop or fuzzy_crop:
            rank, match_type = 4, "crop-only match - verify pest manually"
        elif not crop_norm and (exact_pest or fuzzy_pest or pest_overlap):
            rank, match_type = 5, "pest-only match - verify crop manually"

        if rank:
            item = row.drop(labels=["_crop_norm", "_pest_norm"]).to_dict()
            item["match_type"] = match_type
            item["_match_rank"] = rank
            matches.append(item)

    if not matches:
        return _empty_ppqs_df()

    result = pd.DataFrame(matches).sort_values(["_match_rank", "crop", "pest", "pesticide_name"])
    result = result.drop(columns=["_match_rank"])
    return result.reset_index(drop=True)


def auto_select_label_claims(matched_df, limit: int = 4) -> list[int]:
    """Pick the best label-claim rows so the user gets a safe default selection.

    Prefers exact crop+pest matches, single-molecule products, rows with a
    calculable spray dose and waiting period, and skips rows flagged for
    manual verification. Returns at most `limit` rows, one per pesticide.
    """
    if matched_df is None or matched_df.empty:
        return []

    scored = []
    for index, row in matched_df.iterrows():
        name = clean_ppqs_text(row.get("pesticide_name", ""))
        if not name:
            continue
        match_type = str(row.get("match_type", "")).lower()
        remarks = str(row.get("remarks", "")).lower()
        dose10 = str(row.get("dose_per_10_litre", "")).lower()
        waiting = clean_ppqs_text(row.get("waiting_period_days", ""))

        score = 0
        if match_type.startswith("exact crop + exact pest"):
            score += 100
        elif match_type.startswith("exact crop"):
            score += 60
        elif match_type.startswith("fuzzy crop"):
            score += 30
        else:
            score += 10
        if "+" not in name:
            score += 25
        if dose10 and not dose10.startswith("not applicable"):
            score += 20
        if waiting and waiting != "-":
            score += 10
        if "manual verification" in remarks or "verify against pdf" in remarks:
            score -= 40
        scored.append((score, index, name.lower()))

    scored.sort(key=lambda item: (-item[0], item[1]))

    selected = []
    seen_names = set()
    for score, index, name in scored:
        if score < 100 or name in seen_names:
            continue
        seen_names.add(name)
        selected.append(index)
        if len(selected) >= limit:
            break

    if not selected:
        for score, index, name in scored:
            if name in seen_names:
                continue
            seen_names.add(name)
            selected.append(index)
            if len(selected) >= 2:
                break
    return selected


AGRESCO_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agresco_recommendations.json")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_agresco_recommendations() -> list[dict]:
    """Load the pre-extracted Gujarat SAU (AGRESCO) farmer recommendations."""
    try:
        import json

        with open(AGRESCO_JSON_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, ValueError):
        return []


def _agresco_haystack(rec: dict) -> str:
    return " ".join(
        [
            rec.get("title", ""),
            rec.get("crop", ""),
            rec.get("pest", ""),
            rec.get("recommendation_en", ""),
        ]
    ).lower()


def search_agresco_recommendations(records, crop, pest, limit: int = 4) -> list[dict]:
    """Rank official AGRESCO recommendations by relevance to crop and pest."""
    crop_norm = normalize_crop_name(crop)
    pest_norm = normalize_pest_name(pest)
    crop_tokens = [tok for tok in crop_norm.split() if len(tok) >= 3]
    pest_tokens = [tok for tok in pest_norm.split() if len(tok) >= 3]
    if not crop_tokens and not pest_tokens:
        return []

    # Prefer specific words (e.g. "armyworm") over generic ones (e.g. "fall")
    # so an exact pest match outranks an incidental word hit.
    pest_significant = [tok for tok in pest_tokens if len(tok) >= 5] or pest_tokens

    scored = []
    for rec in records or []:
        hay = _agresco_haystack(rec)
        if not hay.strip():
            continue
        crop_hit = any(tok in hay for tok in crop_tokens)
        pest_hit = any(tok in hay for tok in pest_significant)
        fuzzy = 0
        if fuzz and not (crop_hit and pest_hit):
            target = " ".join(crop_tokens + pest_tokens)
            if target:
                fuzzy = fuzz.partial_ratio(target, hay)

        score = 0
        if crop_hit and pest_hit:
            score = 100
        elif crop_hit and not pest_tokens:
            score = 70
        elif pest_hit and not crop_tokens:
            score = 60
        elif crop_hit or pest_hit:
            score = 45
        elif fuzzy >= 80:
            score = 30
        if score:
            # Prefer plant protection / horticulture entries with a dose.
            if re.search(r"\d", rec.get("recommendation_en", "")):
                score += 5
            scored.append((score, rec))

    scored.sort(key=lambda item: -item[0])
    return [rec for _, rec in scored[:limit]]


def format_agresco_for_prompt(selected) -> str:
    records = list(selected or [])
    if not records:
        return ""

    lines = [
        "OFFICIAL_GUJARAT_UNIVERSITY_RECOMMENDATIONS (AGRESCO):",
        "These are approved farmer recommendations from Gujarat State Agricultural",
        "Universities (AGRESCO proceedings). They are verified and may be used and",
        "referred to in the article. Use the English recommendation for technical",
        "accuracy and write natural fresh Gujarati; do not copy raw extracted text.",
    ]
    for index, rec in enumerate(records, start=1):
        meta = ", ".join(
            filter(
                None,
                [
                    rec.get("year", ""),
                    rec.get("university", ""),
                    rec.get("section", ""),
                ],
            )
        )
        recommendation = rec.get("recommendation_en", "") or rec.get("title", "")
        lines.append(
            "\n".join(
                [
                    f"{index}. [{meta}] {rec.get('title', '')}".rstrip(),
                    f"   Recommendation: {recommendation}",
                    f"   Source: {rec.get('source_file', '')}, page {rec.get('source_page', '')}",
                ]
            )
        )
    return "\n".join(lines).strip()


def with_reference_recommendations(context: str, agresco_block: str) -> str:
    """Append official AGRESCO recommendations to the article research context."""
    context = context or ""
    if not agresco_block:
        return context
    return f"{context}\n\n{agresco_block}".strip()


def render_agresco_recommendation_helper(crop_default: str = "", pest_default: str = "") -> str:
    records = load_agresco_recommendations()
    st.session_state.setdefault("agresco_block", "")

    with st.expander("Gujarat University Recommendations (AGRESCO)", expanded=False):
        if not records:
            st.info(
                "No AGRESCO recommendations file found. Add "
                "agresco_recommendations.json to the app to enable official "
                "Gujarat university recommendations."
            )
            st.session_state["agresco_block"] = ""
            return ""

        years = sorted({rec.get("year", "") for rec in records if rec.get("year")})
        st.caption(
            f"{len(records)} official Gujarat SAU farmer recommendations loaded"
            + (f" (years: {', '.join(years)})." if years else ".")
        )
        col_crop, col_pest = st.columns(2)
        with col_crop:
            crop_query = st.text_input(
                "Crop for university recommendation search",
                value=crop_default,
                key="agresco_crop_query",
            )
        with col_pest:
            pest_query = st.text_input(
                "Pest / problem for university recommendation search",
                value=pest_default,
                placeholder="Example: whitefly, fruit borer, mites, wilt",
                key="agresco_pest_query",
            )

        if st.button("Find official university recommendations", key="agresco_search"):
            matches = search_agresco_recommendations(records, crop_query, pest_query)
            st.session_state["agresco_matches"] = matches
            st.session_state["agresco_block"] = format_agresco_for_prompt(matches)

        matches = st.session_state.get("agresco_matches", [])
        if matches:
            st.success(
                f"{len(matches)} official recommendation(s) will be shared with the "
                "article as trusted Gujarat university guidance."
            )
            for rec in matches:
                meta = ", ".join(
                    filter(None, [rec.get("year", ""), rec.get("university", ""), rec.get("section", "")])
                )
                st.markdown(f"**{rec.get('title', '')}**  \n*{meta} — page {rec.get('source_page', '')}*")
                if rec.get("recommendation_en"):
                    st.write(rec["recommendation_en"])
                if rec.get("recommendation_gu"):
                    st.caption("Gujarati (raw extract for reference): " + rec["recommendation_gu"])
        elif "agresco_matches" in st.session_state:
            st.info(
                "No matching university recommendation found for this crop/problem. "
                "The article will still use your other research."
            )

    return st.session_state.get("agresco_block", "")


def format_verified_chemicals_for_prompt(selected_rows) -> str:
    if selected_rows is None:
        return ""
    if pd is not None and isinstance(selected_rows, pd.DataFrame):
        records = selected_rows.to_dict("records")
    else:
        records = list(selected_rows or [])
    if not records:
        return ""

    lines = [
        "Use only these user-selected PPQS/CIB&RC label-claim chemical rows.",
        "Do not add any other chemical pesticide or dose.",
    ]
    for index, row in enumerate(records, start=1):
        lines.append(
            "\n".join(
                [
                    f"{index}. Pesticide: {row.get('pesticide_name', '')}",
                    f"   Formulation: {row.get('formulation', '')}",
                    f"   Crop: {row.get('crop', '')}",
                    f"   Pest: {row.get('pest', '')}",
                    f"   Use type: {row.get('use_type', '')}",
                    f"   Label a.i. dose/ha: {row.get('ai_dose_per_ha', '')}",
                    f"   Label formulation dose/ha: {row.get('formulation_dose_per_ha', '')}",
                    f"   Label dilution water L/ha: {row.get('dilution_water_l_per_ha', '')}",
                    f"   Calculated dose per 10 L: {row.get('dose_per_10_litre', '')}",
                    f"   Waiting period days: {row.get('waiting_period_days', '')}",
                    f"   Source: {row.get('source_file', '')}, page {row.get('source_page', '')}",
                    f"   Remarks: {row.get('remarks', '')}",
                ]
            )
        )
    return "\n".join(lines).strip()


def verified_chemicals_prompt_section(verified_label_claim_chemicals: str = "") -> str:
    verified = (verified_label_claim_chemicals or "").strip()
    return f"""
VERIFIED_LABEL_CLAIM_CHEMICALS:
{verified}

Strict chemical control rule:
- Use chemical control only from VERIFIED_LABEL_CLAIM_CHEMICALS.
- Do not add any pesticide, formulation, dose, waiting period, water quantity, or
  seed-treatment recommendation from memory, web search, or deep research.
- If a chemical from research notes is not present in the verified PPQS/CIB&RC
  label-claim list for the same crop and pest, exclude it.
- If VERIFIED_LABEL_CLAIM_CHEMICALS is empty, do not write chemical pesticide
  recommendation. Write only monitoring, cultural, mechanical, biological and
  IPM guidance, and advise farmers to confirm chemical control from the latest
  CIB&RC label claim and local agricultural university/KVK.
- For foliar spray, write the calculated dose per 10 litres only if it is
  present in the verified block.
- For seed treatment, write the label dose separately and do not convert it to
  10 litres.
""".strip()


# Internal dose conversion examples:
# calculate_dose_per_10_litres("500 ml/ha", "500 L/ha") -> "10.0 ml / 10 L water"
# calculate_dose_per_10_litres("250 g/ha", "500 L/ha") -> "5.0 g / 10 L water"
# Seed treatment g/kg seed and broadcast/NA water are handled as non-foliar rows.


def current_problem_research_guide(month: str, region: str) -> str:
    current_date = datetime.now().strftime("%d %B %Y")
    return f"""
Current-problem discovery rules:
- Current date for research context: {current_date}.
- Treat this as real farmer-problem discovery for {month}, not generic topic brainstorming.
- Search current web context first: IMD/agromet advisories, Gujarat agricultural
  university/KVK advisories, state agriculture advisories, recent agriculture news,
  rainfall/monsoon updates, mandi/market/input reports, and credible farmer-facing
  digital agriculture sources.
- If using trends, social posts, YouTube, or local media signals, use them only as
  weak signals and corroborate with official, university/KVK, weather, market, or
  multiple news sources.
- For "Whole Gujarat", compare South Gujarat, Saurashtra, Central/Middle Gujarat,
  North Gujarat, and Kutch separately before ranking topics.
- For a selected sub-region such as {region}, focus on that sub-region's crops,
  districts, rainfall pattern, crop stage, pest/disease pressure, irrigation stress,
  soil/salinity/wind/dust issues, market pressure, and farmer-visible symptoms.
- Do not suggest random evergreen topics such as generic IPM, generic nutrient
  management, or generic technology unless there is current regional evidence that
  farmers are facing that problem now.
- Prefer topics where a farmer can immediately say: "This is happening in my field
  or village this month."
- Rank topics by farmer urgency, evidence strength, regional specificity, seasonal
  timing, magazine usefulness, and safety of recommendations.

At the top of the response, include this exact structured section so the app can
make topic selection easy:
TOPIC_OPTIONS
TOPIC 1 | Gujarati title | Region/sub-region | Main crop | Current farmer problem | Evidence confidence /10
TOPIC 2 | Gujarati title | Region/sub-region | Main crop | Current farmer problem | Evidence confidence /10
Continue for 8 to 10 topics.

After TOPIC_OPTIONS, provide a ranked evidence pack. For every topic include:
- Current farmer problem being addressed
- Region/sub-region and important districts if known
- Crop stage or seasonal timing
- Field symptoms farmers may recognize
- Why this is a current {month} problem, not a random topic
- Source signal summary: official, university/KVK, government, weather, market,
  news, farmer trend, or general web
- Caution: what must be locally verified before publication
""".strip()


def topic_research_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    manual_title: str = "",
    search_details: str = "",
) -> str:
    return f"""
You are an agricultural research assistant for Gujarati agriculture magazines.

Use Google Search grounding to identify current, prevailing, and seasonally relevant
agriculture article topics for {month} in {region}.

Subject focus: {subject_area}
Crop focus, if any: {crop_focus or "No specific crop focus"}

{manual_search_context(manual_title, search_details)}

{current_problem_research_guide(month, region)}

Research priorities:
- Selected Gujarat region and sub-region-specific farmer problems
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
4. Selected Gujarat region and sub-region relevance
5. Field observations farmers may recognize
6. Scientific background in simple language
7. Natural enemies and integrated management
8. Farmer benefit and practical relevance

Return 8 to 10 topic options using the required TOPIC_OPTIONS format above.
For each topic, keep the Gujarati title specific to a real current farmer
problem, crop, and Gujarat region/sub-region.

Do not select the final article topic automatically. The user will choose from
the ranked suggested topics in the app. After the topic options, provide a useful
research note pack for each option so the user can compare and choose:
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
    verified_label_claim_chemicals: str = "",
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

{verified_chemicals_prompt_section(verified_label_claim_chemicals)}

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
    verified_label_claim_chemicals: str = "",
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

{verified_chemicals_prompt_section(verified_label_claim_chemicals)}

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
    verified_label_claim_chemicals: str = "",
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

{verified_chemicals_prompt_section(verified_label_claim_chemicals)}

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
    search_details: str = "",
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
- Topic hint: {topic_hint or "Find current ranked topic options; user will choose from suggestions"}

{manual_search_context(topic_hint, search_details)}

{current_problem_research_guide(month, region)}

Research priorities:
- Current and prevailing crop problems
- Agricultural acarology and agricultural entomology relevance
- Selected Gujarat region and sub-region-specific farming conditions
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
- Selected Gujarat region and sub-region field context
- Farmer-recognizable observations for a story opening
- Science that can be explained simply after the field situation
- Natural enemies, IPM, monitoring, and practical decision support
- Farmer benefit: yield, quality, cost reduction, sustainability, and profit

Return 8 to 10 Gujarati article topic options using the required TOPIC_OPTIONS
format above. Do not choose a final topic. Each topic must be a real current
farmer problem, not a general evergreen theme.

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
    verified_label_claim_chemicals: str = "",
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

{verified_chemicals_prompt_section(verified_label_claim_chemicals)}

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
    verified_label_claim_chemicals: str = "",
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

{verified_chemicals_prompt_section(verified_label_claim_chemicals)}

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
    verified_label_claim_chemicals: str = "",
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

{verified_chemicals_prompt_section(verified_label_claim_chemicals)}

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
    search_details: str = "",
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
- Topic hint: {topic_hint or "Find current ranked topic options; user will choose from suggestions"}

{manual_search_context(topic_hint, search_details)}

Tab 4 magazine requirement:
- The selected target magazine is only a publication/personality reference.
- If the selected target is Agro Sandesh, do not use generic Agro Sandesh house
  style; use the full field-discovery magazine feature style below.
- If the selected target is Krushi Prabhat, do not use daily newspaper style.
- If the selected target is Krishi Jagran Gujarati, do not use fast digital
  news/explainer style.
- Do not use daily newspaper, short news, alert, or breaking-news style.
- Research should support a full Gujarati magazine feature with scene,
  observation, discovery, reflection, farmer meaning, and practical depth.

{current_problem_research_guide(month, region)}

Research priorities:
- Current and prevailing crop, pest, mite, weather, or field observation issues
- Agricultural acarology and entomology relevance when useful
- Selected Gujarat region and sub-region farming realities
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
- Selected Gujarat region and sub-region farming reality
- Farm habits, orchard/field scenes, soil, dust, moisture, and natural balance
- Scientific explanation that can emerge from observation
- Natural enemies, IPM, patient monitoring, and practical wisdom
- Farmer benefit through better observation and wiser decisions

Return 8 to 10 Gujarati article topic options using the required TOPIC_OPTIONS
format above. Do not choose a final topic. Each topic must be a real current
farmer problem, not a general evergreen theme.

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
    verified_label_claim_chemicals: str = "",
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

{verified_chemicals_prompt_section(verified_label_claim_chemicals)}

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
    verified_label_claim_chemicals: str = "",
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

{verified_chemicals_prompt_section(verified_label_claim_chemicals)}

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
    verified_label_claim_chemicals: str = "",
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

{verified_chemicals_prompt_section(verified_label_claim_chemicals)}

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
    search_details: str = "",
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
- Topic hint: {topic_hint or "Find current ranked topic options; user will choose from suggestions"}

{manual_search_context(topic_hint, search_details)}

{current_problem_research_guide(month, region)}

Research priorities:
- Current crop, pest, mite, weather, field, orchard, or seasonal observation
  issues relevant to farmers
- Agricultural acarology and entomology relevance when useful
- Selected Gujarat region and sub-region farming conditions
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
- Selected Gujarat region and sub-region field/orchard context
- Visual clues that can carry the opening scene
- Observations and questions that delay discovery naturally
- Scientific explanation that can appear after curiosity is built
- Natural enemies, IPM, monitoring, and practical meaning
- Farmer benefit through observation, timely decisions, quality, yield, and profit

Return 8 to 10 Gujarati article topic options using the required TOPIC_OPTIONS
format above. Do not choose a final topic. Each topic must be a real current
farmer problem, not a general evergreen theme.

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
    verified_label_claim_chemicals: str = "",
) -> str:
    return f"""
Write a Gujarati agricultural magazine feature using an original field-discovery
voice inspired by careful observation of farm life and seasons.

Tab 4 magazine requirement:
- The selected target magazine is only a publication/personality reference.
- If the selected target is Agro Sandesh, do not use generic Agro Sandesh house
  style; use the full field-discovery magazine feature style below.
- If the selected target is Krushi Prabhat, do not use daily newspaper style.
- If the selected target is Krishi Jagran Gujarati, do not use fast digital
  news/explainer style.
- Do not write like a daily newspaper, news alert, short advisory, or digital
  news explainer.
- Write a full magazine article/feature with narrative depth, scene, observation,
  discovery, simple science, farmer meaning, and reflective ending.
- Keep all field-discovery style rules below.

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

{verified_chemicals_prompt_section(verified_label_claim_chemicals)}

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
    verified_label_claim_chemicals: str = "",
) -> str:
    return f"""
Rewrite the following Gujarati article into a stronger scene-based field
discovery feature for a Gujarati agricultural magazine.

Tab 4 magazine requirement:
- The selected target magazine is only a publication/personality reference.
- If the selected target is Agro Sandesh, remove generic Agro Sandesh house style.
- If the selected target is Krushi Prabhat, remove daily newspaper style.
- If the selected target is Krishi Jagran Gujarati, remove fast digital
  news/explainer style.
- Remove daily newspaper, short news, alert, or report-like structure.
- Make it a full magazine feature with narrative flow, observation, discovery,
  simple science, farmer meaning, and reflective ending.

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

{verified_chemicals_prompt_section(verified_label_claim_chemicals)}

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
    verified_label_claim_chemicals: str = "",
) -> str:
    return f"""
Act as the final Gujarati magazine editor for {target_magazine}.

Finalize the article below into a polished field-discovery agricultural feature.

Tab 4 magazine requirement:
- The selected target magazine is only a publication/personality reference.
- If the selected target is Agro Sandesh, final article must not read like
  generic Agro Sandesh house style.
- If the selected target is Krushi Prabhat, final article must not read like a
  daily newspaper.
- If the selected target is Krishi Jagran Gujarati, final article must not read
  like a fast digital news/explainer.
- Final article must not read like a short news report, alert, or fast digital
  explainer.
- Final article must feel like a complete Gujarati magazine feature.

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

{verified_chemicals_prompt_section(verified_label_claim_chemicals)}

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


def farmer_engagement_research_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    topic_hint: str,
    season_context: str,
    target_magazine: str,
    search_details: str = "",
) -> str:
    return f"""
You are an agricultural research assistant for Gujarati farmer-oriented
magazine articles.

Research current, seasonally relevant farmer problems for a new article style:
Farmer Hook + Field Story + Simple Science + Practical Benefit.

Research assignment:
- Target magazine: {target_magazine}
- Month: {month}
- Season/context: {season_context or month}
- Region: {region}
- Subject area: {subject_area}
- Crop: {crop_focus or "No specific crop"}
- Topic hint: {topic_hint or "Find current ranked topic options; user will choose from suggestions"}

{manual_search_context(topic_hint, search_details)}

{current_problem_research_guide(month, region)}

Research priorities:
- Problems farmers are visibly facing now in the selected Gujarat region
- Field scenes that create recognition: "this is happening in my field"
- Crop stage, weather, irrigation, soil, dust, pest symptoms, disease symptoms,
  nutrient signs, natural enemies, farmer habits, and market/quality pressure
- Simple science that can explain the hidden cause after curiosity is built
- Practical action that links naturally to cost saving, yield, quality, market
  value, reduced pesticide misuse, natural enemy protection, and crop health
- Official, university/KVK, weather, market, research, news, and farmer-trend
  signals where useful

Return 8 to 10 Gujarati article topic options using the required TOPIC_OPTIONS
format above. Do not choose a final topic. Each topic must address a current
farmer problem and must be suitable for a farmer-engaging magazine article.

After TOPIC_OPTIONS, include a ranked evidence pack with field observations,
farmer questions, hidden cause, simple science angle, practical benefit, source
signal type, and caution notes for every topic.
""".strip()


def farmer_engagement_article_prompt(
    month: str,
    region: str,
    subject_area: str,
    crop_focus: str,
    article_length: str,
    topic_hint: str,
    season_context: str,
    target_magazine: str,
    research_notes: str,
    verified_label_claim_chemicals: str = "",
) -> str:
    return f"""
Write a Gujarati agricultural magazine article in the farmer-engagement style:
Farmer Hook + Field Story + Simple Science + Practical Benefit.

Target magazine: {target_magazine}
Target magazine personality:
{magazine_style_note(target_magazine)}

Author background:
Agricultural Entomology / Agricultural Acarology scientist with field experience
in Gujarat. Do not imitate any named author exactly. Use an original Gujarati
agricultural voice.

Core purpose:
The farmer should feel: "This is about my own field." The article should help
farmers recognize the problem, become curious, understand the cause, trust the
science, remember the advice, and feel confident to act.

Style blend:
- 25 percent farmer story: real farmer situation, field visit, village
  experience, crop observation, or seasonal challenge.
- 20 percent field observation: weather, crop stage, leaf colour, soil, dust,
  irrigation, pest symptoms, farmer habit, and natural enemies.
- 20 percent simple science: explain the cause in simple Gujarati only after
  field observation and farmer doubt.
- 25 percent practical solution: useful actions in natural prose, not bulletin
  orders, with farmer benefit attached to every recommendation.
- 10 percent hopeful reflection: end with wisdom, confidence, and practical hope.

Hidden article architecture:
Farmer hook -> field scene -> visible problem -> farmer question -> hidden cause
-> simple scientific explanation -> practical field wisdom -> step-by-step
advisory in natural prose -> farmer benefit -> reflective hopeful ending.
Do not print these as labels.

Opening rule:
The first 200 words must not begin with a definition, scientific name,
statistics, research result, pesticide recommendation, economic threshold, or
technical explanation. Begin with a farmer walking in the field, a grower
noticing a crop change, a seasonal problem, field visit, orchard observation,
common farmer doubt, or visual symptom.

Reader recognition and curiosity:
- Include sentences that make farmers feel recognition: many farmers see this,
  at first it looks ordinary, this scene is familiar in the field, but a closer
  look shows something different.
- Use natural questions, but do not answer immediately. Add one more observation
  before explaining the science.

Science placement:
Science must not come first. Use this order: field observation, farmer doubt,
hidden cause, simple science. Explain technical terms immediately in simple
farmer language.

Recommendation style:
Do not write as orders. Avoid harsh command tone. Write as farmer wisdom.
Every recommendation must naturally answer what to do, why to do it, and how the
farmer benefits through cost saving, yield, quality, market value, reduced
unnecessary pesticide use, natural enemy protection, long-term crop health, or
better decisions.

Paragraph style:
Each paragraph should have one central idea and normally 70 to 120 words. Use
mixed sentence rhythm: short for attention, medium for observation, longer for
explanation, short for impact.

Must avoid:
Thesis style, literature review style, research paper style, excessive English,
too many pesticide names, fear-based writing, political discussion, government
scheme discussion, academic references, copied author voice, unsupported
outbreak claims, and unsafe pesticide dosage.

{verified_chemicals_prompt_section(verified_label_claim_chemicals)}

Special ending boxes:
At the end, add two short reader-friendly boxes:
1. ખેડૂત માટે 5 યાદ રાખવા જેવી વાતો
2. આ ભૂલો ટાળો
Keep both boxes short, practical, and easy to remember.

Target details:
- Language: Gujarati
- Length: {article_length}
- Month: {month}
- Season/context: {season_context or month}
- Region: {region}
- Subject area: {subject_area}
- Crop: {crop_focus or "No specific crop"}
- Topic: {topic_hint or "Use the selected research topic"}

Research notes:
{research_notes}

Return only the complete Gujarati article with a suitable Gujarati title.
""".strip()


def farmer_engagement_rewrite_prompt(
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
    verified_label_claim_chemicals: str = "",
) -> str:
    return f"""
Rewrite the Gujarati article into a stronger farmer-engagement magazine article:
Farmer Hook + Field Story + Simple Science + Practical Benefit.

Keep the facts and selected topic, but improve:
1. Farmer hook and first-paragraph recognition.
2. Field scene, crop observation, weather/season details, and farmer doubt.
3. Curiosity before science.
4. Simple science after hidden cause.
5. Practical advice written as farmer wisdom, not bulletin orders.
6. Every recommendation linked to farmer benefit.
7. Conversational but scientific Gujarati.
8. Pesticide and outbreak safety.
9. Two short ending boxes: "ખેડૂત માટે 5 યાદ રાખવા જેવી વાતો" and "આ ભૂલો ટાળો".
10. Reflective hopeful ending.

Avoid thesis style, literature review style, report-like ending, fear-based
writing, unsafe pesticide dosage, unsupported local claims, and copied author
voice.

{verified_chemicals_prompt_section(verified_label_claim_chemicals)}

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
Topic: {topic_hint or "Use the selected research topic"}

Research notes:
{research_notes}

Draft article:
{article}

Return only the rewritten Gujarati article with a suitable Gujarati title.
""".strip()


def farmer_engagement_final_editor_prompt(
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
    verified_label_claim_chemicals: str = "",
) -> str:
    return f"""
Act as the final Gujarati magazine editor for {target_magazine}.

Finalize the article into a polished farmer-engagement magazine article:
Farmer Hook + Field Story + Simple Science + Practical Benefit.

Final checks to apply silently:
1. Does the article start with a farmer situation, not science?
2. Does the first paragraph create recognition?
3. Is curiosity created before explanation?
4. Is science simple and useful?
5. Is the tone conversational but still scientific?
6. Does every recommendation show farmer benefit?
7. Are pesticide claims safe and cautious?
8. Does it feel suitable for Gujarati farmer magazines such as Krushi Jivan,
   Krushi Go-Vidya, Krushi Vigyan, or long-form farmer magazines?
9. Would a farmer enjoy reading it fully?
10. Does it end with practical wisdom?
11. Are the two boxes present and short:
    "ખેડૂત માટે 5 યાદ રાખવા જેવી વાતો" and "આ ભૂલો ટાળો"?

{verified_chemicals_prompt_section(verified_label_claim_chemicals)}

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
Topic: {topic_hint or "Use the selected research topic"}

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


def manual_search_context(manual_title: str, search_details: str) -> str:
    manual_title = (manual_title or "").strip()
    search_details = (search_details or "").strip()
    if not manual_title and not search_details:
        return ""

    lines = ["Manual title and search guidance from user:"]
    if manual_title:
        lines.append(f"- Gujarati article title typed by user: {manual_title}")
    if search_details:
        lines.append(f"- Extra details to guide search: {search_details}")
    lines.extend(
        [
            "- Use this input to shape search queries, source choice, and topic ranking.",
            "- If a manual title is given, keep research focused on it and include it,",
            "  or a close evidence-backed refinement of it, as TOPIC 1.",
            "- Do not invent evidence to fit the manual title; mention uncertainty and",
            "  local verification needs when support is weak.",
        ]
    )
    return "\n".join(lines)


def manual_topic_inputs(prefix: str) -> tuple[str, str]:
    manual_title = st.text_input(
        "Manual Gujarati article title optional",
        placeholder="Type the article title in Gujarati, or leave blank for topic suggestions.",
        key=f"{prefix}_manual_title",
    )
    search_details = st.text_area(
        "Extra details to guide search optional",
        placeholder=(
            "Crop, pest or disease, district, season, symptoms, farmer question, "
            "market issue, source clue, or any point the article must cover."
        ),
        height=100,
        key=f"{prefix}_search_details",
    )
    return manual_title, search_details


def selected_topic_context(
    topic: str,
    research_notes: str,
    manual_title: str = "",
    search_details: str = "",
) -> str:
    topic = (topic or "").strip()
    research_notes = (research_notes or "").strip()
    manual_title = (manual_title or "").strip()
    search_details = (search_details or "").strip()
    parts = [f"Selected article topic:\n{topic}"]
    if manual_title or search_details:
        manual_parts = []
        if manual_title:
            manual_parts.append(f"Manual Gujarati title from user:\n{manual_title}")
        if search_details:
            manual_parts.append(f"Extra user details for search and article:\n{search_details}")
        parts.append("\n\n".join(manual_parts))
    if research_notes:
        parts.append(f"Research notes:\n{research_notes}")
    return "\n\n".join(parts)


def clean_topic_option(option: str) -> str:
    option = re.sub(r"[*_`#]+", "", option or "").strip()
    option = re.sub(r"\s+", " ", option)
    return option.strip(" -|:")


def extract_suggested_topics(research_notes: str) -> list[str]:
    topics = []
    seen = set()

    for raw_line in (research_notes or "").splitlines():
        line = clean_topic_option(raw_line)
        if not line:
            continue

        match = re.match(r"^TOPIC\s*\d+\s*(?:[:|\-–—]\s*)?(.*)$", line, re.IGNORECASE)
        if match:
            candidate = clean_topic_option(match.group(1))
        else:
            title_match = re.search(
                r"(?:Gujarati title|Gujarati article topic|article topic)\s*[:\-–—]\s*(.+)$",
                line,
                re.IGNORECASE,
            )
            candidate = clean_topic_option(title_match.group(1)) if title_match else ""

        if not candidate:
            continue
        if len(candidate) > 260:
            candidate = candidate[:257].rstrip() + "..."
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            topics.append(candidate)

    return topics


def suggested_topic_selector(
    label: str,
    key: str,
    research_notes: str,
    manual_title: str = "",
) -> str:
    topics = extract_suggested_topics(research_notes)
    manual_title = clean_topic_option(manual_title)
    if manual_title:
        manual_key = manual_title.lower()
        topics = [manual_title] + [
            topic for topic in topics if clean_topic_option(topic).lower() != manual_key
        ]
    if topics:
        return st.selectbox(label, topics, key=key)

    st.warning(
        "The research response did not include a readable TOPIC_OPTIONS section. "
        "Run research again, or paste one topic as a fallback."
    )
    return st.text_input(
        label,
        value=manual_title,
        placeholder="Fallback: paste one topic from the research response.",
        key=key,
    )


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
    magazine_options: list[str] = None,
):
    if not topic.strip():
        st.caption("Select a suggested topic to get a target magazine suggestion.")
        return None

    magazine_options = magazine_options or MAGAZINE_OPTIONS
    suggested_magazine = recommend_target_magazine(
        topic,
        subject_area,
        research_notes,
        fallback,
    )
    if suggested_magazine not in magazine_options:
        suggested_magazine = fallback if fallback in magazine_options else magazine_options[0]

    st.caption(
        f"Suggested target magazine: {suggested_magazine}. "
        f"{magazine_style_note(suggested_magazine)}"
    )
    suggestion_key = f"{key}_suggested"
    current_magazine = st.session_state.get(key)
    previous_suggestion = st.session_state.get(suggestion_key)
    if current_magazine is None or current_magazine == previous_suggestion:
        st.session_state[key] = suggested_magazine
    elif current_magazine not in magazine_options:
        st.session_state[key] = suggested_magazine
    st.session_state[suggestion_key] = suggested_magazine

    return st.selectbox(
        "Target magazine personality",
        magazine_options,
        key=key,
    )


def render_ppqs_label_claim_checker(crop_default: str = "") -> str:
    st.session_state.setdefault("verified_label_claim_chemicals", "")
    missing = []
    if pd is None:
        missing.append("pandas")
    if pdfplumber is None:
        missing.append("pdfplumber")
    if fuzz is None:
        missing.append("rapidfuzz")

    if pd is not None:
        st.session_state.setdefault("ppqs_label_df", _empty_ppqs_df())
        st.session_state.setdefault("ppqs_matched_df", _empty_ppqs_df())
        st.session_state.setdefault("ppqs_selected_rows", _empty_ppqs_df())
    st.session_state.setdefault("ppqs_search_has_run", False)
    if "ppqs_crop_query" not in st.session_state:
        st.session_state["ppqs_crop_query"] = crop_default or ""

    with st.expander("PPQS / CIB&RC Label Claim Checker", expanded=False):
        st.warning(
            "Only selected label-claim pesticides will be used in the article. "
            "AI will not add other chemicals."
        )

        if missing:
            st.info(
                "Install missing packages before using the checker: "
                + ", ".join(missing)
            )
            return st.session_state.get("verified_label_claim_chemicals", "")

        # Hybrid: load the saved label cache instantly on first entry, so the
        # checker works with no wait and even if ppqs.gov.in is unreachable.
        if not st.session_state.get("ppqs_cache_loaded"):
            st.session_state["ppqs_cache_loaded"] = True
            cached_df, cache_meta = load_ppqs_label_cache()
            current = st.session_state.get("ppqs_label_df")
            if cached_df is not None and (not isinstance(current, pd.DataFrame) or current.empty):
                st.session_state["ppqs_label_df"] = cached_df
                st.session_state["ppqs_data_as_of"] = cache_meta.get("fetched", "")
                st.session_state["ppqs_data_source"] = "saved"

        data_as_of = st.session_state.get("ppqs_data_as_of", "")
        data_source = st.session_state.get("ppqs_data_source", "")
        if data_source == "saved" and data_as_of:
            st.caption(
                f"Using saved label data as of {data_as_of}. Refresh below to pull "
                "the latest quarter from ppqs.gov.in."
            )
        elif data_source == "live" and data_as_of:
            st.caption(f"Using freshly downloaded label data (as of {data_as_of}).")

        st.markdown("**Option 1: Load directly from ppqs.gov.in**")
        if st.button("Fetch Major Uses document list from PPQS website", key="ppqs_fetch_list"):
            try:
                with st.spinner("Reading the PPQS Major Uses page..."):
                    st.session_state["ppqs_web_docs"] = fetch_ppqs_document_list()
                if not st.session_state["ppqs_web_docs"]:
                    st.warning(
                        "No PDF links were found on the PPQS page. The page layout "
                        "may have changed; upload the PDF manually below."
                    )
            except Exception as exc:
                st.session_state["ppqs_web_docs"] = []
                st.error(f"Could not read the PPQS website: {exc}")

        web_docs = st.session_state.get("ppqs_web_docs") or []
        if web_docs:
            doc_titles = [doc["title"] for doc in web_docs]
            default_docs = [
                title
                for title in doc_titles
                if "insecticide" in title.lower() and "bio" not in title.lower()
            ][:1]
            selected_doc_titles = st.multiselect(
                "PPQS documents to load (insecticides is usually enough)",
                doc_titles,
                default=default_docs,
                key="ppqs_web_doc_choice",
            )
            if st.button("Download and parse selected PPQS documents", key="ppqs_web_parse"):
                if not selected_doc_titles:
                    st.info("Select at least one PPQS document to download.")
                else:
                    frames = []
                    errors = []
                    for doc in web_docs:
                        if doc["title"] not in selected_doc_titles:
                            continue
                        try:
                            with st.spinner(
                                f"Downloading and parsing: {doc['title']} "
                                "(large PDFs can take a few minutes)..."
                            ):
                                frames.append(
                                    download_and_parse_ppqs_pdf(doc["url"], doc["title"])
                                )
                        except Exception as exc:
                            errors.append(f"{doc['title']}: {exc}")
                    for error in errors:
                        st.error(f"Could not load {error}")

                    if frames:
                        parsed_df = (
                            pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
                        )
                        st.session_state["ppqs_label_df"] = parsed_df
                        st.session_state["ppqs_matched_df"] = _empty_ppqs_df()
                        st.session_state["ppqs_selected_rows"] = _empty_ppqs_df()
                        st.session_state["verified_label_claim_chemicals"] = ""
                        st.session_state["ppqs_selected_indices"] = []
                        st.session_state["ppqs_search_has_run"] = False
                        # Refresh the on-disk cache so later runs load instantly.
                        saved_date = save_ppqs_label_cache(parsed_df, selected_doc_titles)
                        st.session_state["ppqs_data_as_of"] = saved_date
                        st.session_state["ppqs_data_source"] = "live"
                        if parsed_df.empty:
                            st.warning(
                                "No label-claim rows were extracted from the downloaded "
                                "PDF(s). Try the manual upload with a cleaner copy."
                            )
                        else:
                            st.success(
                                f"Parsed {len(parsed_df)} label-claim rows from "
                                f"{len(frames)} PPQS document(s)."
                            )
                    else:
                        # Live fetch failed entirely: keep whatever saved data we have.
                        existing = st.session_state.get("ppqs_label_df")
                        if isinstance(existing, pd.DataFrame) and not existing.empty:
                            st.warning(
                                "Could not download from ppqs.gov.in right now. Keeping "
                                "the saved label data"
                                + (
                                    f" (as of {st.session_state.get('ppqs_data_as_of', '')})."
                                    if st.session_state.get("ppqs_data_as_of")
                                    else "."
                                )
                            )

        st.markdown("**Option 2: Upload the PDF manually**")
        uploaded_file = st.file_uploader(
            "Upload latest PPQS/CIB&RC Major Uses PDF",
            type=["pdf"],
            key="ppqs_pdf_upload",
        )
        crop_query = st.text_input(
            "Crop name for label claim search",
            key="ppqs_crop_query",
        )
        pest_query = st.text_input(
            "Pest name for label claim search",
            placeholder="Example: thrips, fruit borer, mites, whitefly",
            key="ppqs_pest_query",
        )

        parse_clicked = st.button(
            "Parse / Update Label Claim Database",
            key="ppqs_parse_button",
        )
        if parse_clicked:
            if uploaded_file is None:
                st.info("Upload the latest PPQS/CIB&RC Major Uses PDF first.")
            else:
                try:
                    with st.spinner("Parsing PPQS/CIB&RC label-claim PDF..."):
                        parsed_df = parse_ppqs_pdf(uploaded_file)
                    st.session_state["ppqs_label_df"] = parsed_df
                    st.session_state["ppqs_matched_df"] = _empty_ppqs_df()
                    st.session_state["ppqs_selected_rows"] = _empty_ppqs_df()
                    st.session_state["verified_label_claim_chemicals"] = ""
                    st.session_state["ppqs_selected_indices"] = []
                    st.session_state["ppqs_search_has_run"] = False
                    if parsed_df.empty:
                        st.warning(
                            "No label-claim rows were extracted. The PDF may need "
                            "manual verification or a cleaner text/table version."
                        )
                    else:
                        st.success(f"Parsed {len(parsed_df)} label-claim rows.")
                except Exception as exc:
                    st.session_state["ppqs_label_df"] = _empty_ppqs_df()
                    st.session_state["ppqs_matched_df"] = _empty_ppqs_df()
                    st.session_state["ppqs_selected_rows"] = _empty_ppqs_df()
                    st.session_state["verified_label_claim_chemicals"] = ""
                    st.error(f"Could not parse the PPQS PDF: {exc}")

        label_df = st.session_state.get("ppqs_label_df", _empty_ppqs_df())
        if isinstance(label_df, pd.DataFrame) and not label_df.empty:
            st.caption(f"Current parsed PPQS database: {len(label_df)} rows.")
        elif uploaded_file is None:
            st.info(
                "Load the Major Uses list from the PPQS website above, or upload "
                "and parse the PDF manually, to enable chemical verification."
            )

        search_clicked = st.button(
            "Search Label Claim Pesticides",
            key="ppqs_search_button",
        )
        if search_clicked:
            if not isinstance(label_df, pd.DataFrame) or label_df.empty:
                st.info("Parse the PPQS/CIB&RC PDF before searching.")
            elif not crop_query.strip() and not pest_query.strip():
                st.warning("Enter at least a crop name or pest name for label-claim search.")
            else:
                try:
                    matched_df = search_label_claims(label_df, crop_query, pest_query)
                    st.session_state["ppqs_matched_df"] = matched_df
                    st.session_state["ppqs_selected_rows"] = _empty_ppqs_df()
                    st.session_state["verified_label_claim_chemicals"] = ""
                    st.session_state["ppqs_selected_indices"] = auto_select_label_claims(matched_df)
                    st.session_state["ppqs_search_has_run"] = True
                except Exception as exc:
                    st.session_state["ppqs_matched_df"] = _empty_ppqs_df()
                    st.session_state["verified_label_claim_chemicals"] = ""
                    st.error(f"Could not search label-claim rows: {exc}")

        matched_df = st.session_state.get("ppqs_matched_df", _empty_ppqs_df())
        if isinstance(matched_df, pd.DataFrame) and not matched_df.empty:
            display_columns = [
                column
                for column in PPQS_LABEL_COLUMNS + ["match_type"]
                if column in matched_df.columns
            ]
            st.dataframe(
                matched_df[display_columns],
                hide_index=True,
                use_container_width=True,
            )
            st.download_button(
                "Download matched label-claim CSV",
                data=matched_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="ppqs_label_claim_matches.csv",
                mime="text/csv",
                key="ppqs_download_matches",
            )

            if "match_type" in matched_df.columns and matched_df["match_type"].str.contains("-only", case=False, na=False).any():
                st.warning("Some results are crop-only or pest-only matches. Verify them manually before selecting.")
            if "remarks" in matched_df.columns and matched_df["remarks"].str.contains("Needs manual verification", case=False, na=False).any():
                st.warning("Some extracted rows need manual verification against the source PDF.")

            options = matched_df.index.tolist()
            current_selection = st.session_state.get("ppqs_selected_indices", [])
            st.session_state["ppqs_selected_indices"] = [
                index for index in current_selection if index in options
            ]

            def label_claim_option(index: int) -> str:
                row = matched_df.loc[index]
                dose = row.get("dose_per_10_litre", "")
                label_dose = row.get("formulation_dose_per_ha", "")
                return (
                    f"{row.get('pesticide_name', '')} {row.get('formulation', '')} | "
                    f"{row.get('crop', '')} | {row.get('pest', '')} | "
                    f"{dose or label_dose} | page {row.get('source_page', '')}"
                )

            st.caption(
                "The best label-claim matches are auto-selected for you. "
                "Keep them as they are, or add/remove pesticides below."
            )
            selected_indices = st.multiselect(
                "Select pesticides allowed for the article",
                options=options,
                format_func=label_claim_option,
                key="ppqs_selected_indices",
            )
            selected_df = (
                matched_df.loc[selected_indices].reset_index(drop=True)
                if selected_indices
                else _empty_ppqs_df()
            )
            st.session_state["ppqs_selected_rows"] = selected_df
            st.session_state["verified_label_claim_chemicals"] = (
                format_verified_chemicals_for_prompt(selected_df)
            )
            if selected_indices:
                st.success(f"{len(selected_indices)} verified label-claim row(s) selected for article prompts.")
            else:
                st.info("No label-claim pesticide selected. Chemical recommendations will be excluded.")
        elif st.session_state.get("ppqs_search_has_run"):
            st.warning(
                "No matching label-claim pesticide found in uploaded PPQS PDF for "
                "this crop-pest query. Chemical recommendation will be excluded "
                "unless manually verified."
            )

    return st.session_state.get("verified_label_claim_chemicals", "")


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
    verified_label_claim_chemicals = render_ppqs_label_claim_checker(crop_focus)
    agresco_block = render_agresco_recommendation_helper(crop_focus)

    client = build_client(api_keys[PROVIDER_GEMINI])
    tab_classic, tab_story, tab_farm_wisdom, tab_field_discovery, tab_farmer_engagement = st.tabs(
        [
            "Tab 1: Swaminathan Workflow",
            "Tab 2: Story + Science Prompt",
            "Tab 3: Farm Wisdom Prompt",
            "Tab 4: Field Discovery Prompt",
            "Tab 5: Farmer Engagement Prompt",
        ]
    )

    with tab_classic:
        st.subheader("Current Workflow")
        st.write(
            "Use this tab for the original topic discovery, Gujarati article draft, "
            "Swaminathan-inspired rewrite, final editor check, and Word download."
        )
        classic_manual_title, classic_search_details = manual_topic_inputs("classic")

        if st.button("Deep research and references", type="primary", key="classic_find_topics"):
            with st.spinner("Researching current and seasonally relevant topics..."):
                prompt = topic_research_prompt(
                    month,
                    region,
                    subject_area,
                    crop_focus,
                    classic_manual_title,
                    classic_search_details,
                )
                topics, sources = safe_generate_text(
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
                st.session_state["classic_saved_manual_title"] = classic_manual_title
                st.session_state["classic_saved_search_details"] = classic_search_details
                st.session_state.pop("classic_topic_choice", None)
                st.session_state.pop("classic_target_magazine", None)

        if "topics" in st.session_state:
            st.subheader("Suggested topics")
            st.markdown(st.session_state["topics"])
            render_sources("Research sources", st.session_state.get("topic_sources", []))

            selected_topic_title = suggested_topic_selector(
                "Select one current farmer-problem topic for writing",
                "classic_topic_choice",
                st.session_state["topics"],
                st.session_state.get("classic_saved_manual_title", ""),
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
                    st.warning("Please select one suggested topic before writing.")
                elif not selected_target_magazine:
                    st.warning("Please select the target magazine personality before writing.")
                else:
                    selected_topic = selected_topic_context(
                        selected_topic_title,
                        selected_topic_notes,
                        st.session_state.get("classic_saved_manual_title", ""),
                        st.session_state.get("classic_saved_search_details", ""),
                    )
                    selected_topic = with_reference_recommendations(selected_topic, agresco_block)
                    with st.spinner("Writing the Gujarati article draft..."):
                        prompt = article_prompt(
                            month,
                            region,
                            subject_area,
                            crop_focus,
                            article_length,
                            selected_target_magazine,
                            selected_topic,
                            verified_label_claim_chemicals=verified_label_claim_chemicals,
                        )
                        article, sources = safe_generate_text(
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
                    review, _ = safe_generate_text(
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
                    rewrite, _ = safe_generate_text(
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
                            verified_label_claim_chemicals=verified_label_claim_chemicals,
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
                    final_article, _ = safe_generate_text(
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
                            verified_label_claim_chemicals=verified_label_claim_chemicals,
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
                "Manual Gujarati article title optional for Tab 2",
                placeholder="Type the article title in Gujarati, or leave blank for topic suggestions.",
                key="story_topic_hint",
            )
        with story_col2:
            story_crop_focus = st.text_input(
                "Crop for Tab 2",
                value=crop_focus,
                placeholder="Example: mango, okra, brinjal, cotton, vegetables",
                key="story_crop_focus",
            )
        story_search_details = st.text_area(
            "Extra details to guide Tab 2 search optional",
            placeholder=(
                "Crop, pest or disease, district, season, symptoms, farmer question, "
                "source clue, or article points to include."
            ),
            height=100,
            key="story_search_details",
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
                    story_search_details,
                )
                research, sources = safe_generate_text(
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
                st.session_state["story_saved_search_details"] = story_search_details
                st.session_state["story_saved_crop_focus"] = story_crop_focus
                st.session_state.pop("story_topic_choice", None)
                st.session_state.pop("story_target_magazine", None)
                st.session_state.pop("story_article", None)
                st.session_state.pop("story_rewritten_article", None)
                st.session_state.pop("story_final_article", None)
                st.session_state.pop("story_review", None)

        if "story_research" in st.session_state:
            st.subheader("Tab 2 research notes")
            st.markdown(st.session_state["story_research"])
            render_sources("Tab 2 research sources", st.session_state.get("story_sources", []))

            story_selected_topic = suggested_topic_selector(
                "Select one current farmer-problem topic for Tab 2",
                "story_topic_choice",
                st.session_state["story_research"],
                st.session_state.get("story_saved_topic_hint", ""),
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
                    st.warning("Please select one suggested Tab 2 topic before writing.")
                elif not story_target_magazine:
                    st.warning("Please select the target magazine personality before writing.")
                else:
                    story_selected_context = selected_topic_context(
                        story_selected_topic,
                        story_research_notes,
                        st.session_state.get("story_saved_topic_hint", ""),
                        st.session_state.get("story_saved_search_details", ""),
                    )
                    story_selected_context = with_reference_recommendations(
                        story_selected_context, agresco_block
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
                            verified_label_claim_chemicals=verified_label_claim_chemicals,
                        )
                        article, sources = safe_generate_text(
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
                    review, _ = safe_generate_text(
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
                    rewrite, _ = safe_generate_text(
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
                            verified_label_claim_chemicals=verified_label_claim_chemicals,
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
                    final_article, _ = safe_generate_text(
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
                            verified_label_claim_chemicals=verified_label_claim_chemicals,
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
                "Manual Gujarati article title optional for Tab 3",
                placeholder="Type the article title in Gujarati, or leave blank for topic suggestions.",
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
        wisdom_search_details = st.text_area(
            "Extra details to guide Tab 3 search optional",
            placeholder=(
                "Crop, pest or disease, district, season, symptoms, farmer question, "
                "source clue, or article points to include."
            ),
            height=100,
            key="wisdom_search_details",
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
                    wisdom_search_details,
                )
                research, sources = safe_generate_text(
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
                st.session_state["wisdom_saved_search_details"] = wisdom_search_details
                st.session_state["wisdom_saved_crop_focus"] = wisdom_crop_focus
                st.session_state["wisdom_saved_season_context"] = wisdom_season_context
                st.session_state["wisdom_saved_target_magazine"] = wisdom_target_magazine
                st.session_state.pop("wisdom_topic_choice", None)
                st.session_state.pop("wisdom_article_target_magazine", None)
                st.session_state.pop("wisdom_article", None)
                st.session_state.pop("wisdom_rewritten_article", None)
                st.session_state.pop("wisdom_final_article", None)
                st.session_state.pop("wisdom_review", None)

        if "wisdom_research" in st.session_state:
            st.subheader("Tab 3 research notes")
            st.markdown(st.session_state["wisdom_research"])
            render_sources("Tab 3 research sources", st.session_state.get("wisdom_sources", []))

            wisdom_selected_topic = suggested_topic_selector(
                "Select one current farmer-problem topic for Tab 3",
                "wisdom_topic_choice",
                st.session_state["wisdom_research"],
                st.session_state.get("wisdom_saved_topic_hint", ""),
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
                    st.warning("Please select one suggested Tab 3 topic before writing.")
                elif not wisdom_article_target_magazine:
                    st.warning("Please select the target magazine personality before writing.")
                else:
                    wisdom_selected_context = selected_topic_context(
                        wisdom_selected_topic,
                        wisdom_research_notes,
                        st.session_state.get("wisdom_saved_topic_hint", ""),
                        st.session_state.get("wisdom_saved_search_details", ""),
                    )
                    wisdom_selected_context = with_reference_recommendations(
                        wisdom_selected_context, agresco_block
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
                            verified_label_claim_chemicals=verified_label_claim_chemicals,
                        )
                        article, sources = safe_generate_text(
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
                    review, _ = safe_generate_text(
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
                    rewrite, _ = safe_generate_text(
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
                            verified_label_claim_chemicals=verified_label_claim_chemicals,
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
                    final_article, _ = safe_generate_text(
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
                            verified_label_claim_chemicals=verified_label_claim_chemicals,
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
                "Manual Gujarati article title optional for Tab 4",
                placeholder="Type the article title in Gujarati, or leave blank for topic suggestions.",
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
        discovery_search_details = st.text_area(
            "Extra details to guide Tab 4 search optional",
            placeholder=(
                "Crop, pest or disease, district, season, symptoms, farmer question, "
                "source clue, or article points to include."
            ),
            height=100,
            key="discovery_search_details",
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
                    discovery_search_details,
                )
                research, sources = safe_generate_text(
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
                st.session_state["discovery_saved_search_details"] = discovery_search_details
                st.session_state["discovery_saved_crop_focus"] = discovery_crop_focus
                st.session_state["discovery_saved_season_context"] = discovery_season_context
                st.session_state["discovery_saved_target_magazine"] = discovery_target_magazine
                st.session_state.pop("discovery_topic_choice", None)
                st.session_state.pop("discovery_article_target_magazine", None)
                st.session_state.pop("discovery_article", None)
                st.session_state.pop("discovery_rewritten_article", None)
                st.session_state.pop("discovery_final_article", None)
                st.session_state.pop("discovery_review", None)

        if "discovery_research" in st.session_state:
            st.subheader("Tab 4 research notes")
            st.markdown(st.session_state["discovery_research"])
            render_sources("Tab 4 research sources", st.session_state.get("discovery_sources", []))

            discovery_selected_topic = suggested_topic_selector(
                "Select one current farmer-problem topic for Tab 4",
                "discovery_topic_choice",
                st.session_state["discovery_research"],
                st.session_state.get("discovery_saved_topic_hint", ""),
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
                    st.warning("Please select one suggested Tab 4 topic before writing.")
                elif not discovery_article_target_magazine:
                    st.warning("Please select the target magazine personality before writing.")
                else:
                    discovery_selected_context = selected_topic_context(
                        discovery_selected_topic,
                        discovery_research_notes,
                        st.session_state.get("discovery_saved_topic_hint", ""),
                        st.session_state.get("discovery_saved_search_details", ""),
                    )
                    discovery_selected_context = with_reference_recommendations(
                        discovery_selected_context, agresco_block
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
                            verified_label_claim_chemicals=verified_label_claim_chemicals,
                        )
                        article, sources = safe_generate_text(
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
                    review, _ = safe_generate_text(
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
                    rewrite, _ = safe_generate_text(
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
                            verified_label_claim_chemicals=verified_label_claim_chemicals,
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
                    final_article, _ = safe_generate_text(
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
                            verified_label_claim_chemicals=verified_label_claim_chemicals,
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

    with tab_farmer_engagement:
        st.subheader("Farmer Engagement Prompt Workflow")
        st.write(
            "This tab uses the new farmer-engagement style: farmer hook, field story, "
            "simple science, practical benefit, memory boxes, and hopeful reflection."
        )

        engagement_magazine_options = [
            "Krushi Jivan",
            "Krushi Go-Vidya",
            "Krushi Vigyan",
            "Gujarati farmer magazine",
            "Gujarati long-form agricultural magazine",
        ]

        engagement_col1, engagement_col2 = st.columns(2)
        with engagement_col1:
            engagement_topic_hint = st.text_input(
                "Manual Gujarati article title optional for Tab 5",
                placeholder="Type the article title in Gujarati, or leave blank for topic suggestions.",
                key="engagement_topic_hint",
            )
        with engagement_col2:
            engagement_crop_focus = st.text_input(
                "Crop for Tab 5",
                value=crop_focus,
                placeholder="Example: mango, chilli, cotton, okra, vegetables",
                key="engagement_crop_focus",
            )

        engagement_col3, engagement_col4 = st.columns(2)
        with engagement_col3:
            engagement_season_context = st.text_input(
                "Season or field context for Tab 5",
                value=month,
                placeholder="Example: early monsoon, dry spell, humid post-rain weather",
                key="engagement_season_context",
            )
        with engagement_col4:
            engagement_target_magazine = st.selectbox(
                "Initial target magazine for Tab 5 research",
                engagement_magazine_options,
                index=2,
                key="engagement_target_magazine",
            )
        engagement_search_details = st.text_area(
            "Extra details to guide Tab 5 search optional",
            placeholder=(
                "Crop, pest or disease, district, season, symptoms, farmer question, "
                "source clue, or article points to include."
            ),
            height=100,
            key="engagement_search_details",
        )

        if st.button(
            "Deep research and references for Tab 5",
            type="primary",
            key="engagement_research_button",
        ):
            with st.spinner("Researching farmer-engaging current problems and references..."):
                prompt = farmer_engagement_research_prompt(
                    month,
                    region,
                    subject_area,
                    engagement_crop_focus,
                    engagement_topic_hint,
                    engagement_season_context,
                    engagement_target_magazine,
                    engagement_search_details,
                )
                research, sources = safe_generate_text(
                    client,
                    research_model,
                    prompt,
                    use_search=research_provider == PROVIDER_GEMINI,
                    temperature=0.35,
                    provider=research_provider,
                    api_keys=api_keys,
                )
                st.session_state["engagement_research"] = research
                st.session_state["engagement_sources"] = sources
                st.session_state["engagement_saved_topic_hint"] = engagement_topic_hint
                st.session_state["engagement_saved_search_details"] = engagement_search_details
                st.session_state["engagement_saved_crop_focus"] = engagement_crop_focus
                st.session_state["engagement_saved_season_context"] = engagement_season_context
                st.session_state["engagement_saved_target_magazine"] = engagement_target_magazine
                st.session_state.pop("engagement_topic_choice", None)
                st.session_state.pop("engagement_article_target_magazine", None)
                st.session_state.pop("engagement_article", None)
                st.session_state.pop("engagement_rewritten_article", None)
                st.session_state.pop("engagement_final_article", None)
                st.session_state.pop("engagement_review", None)

        if "engagement_research" in st.session_state:
            st.subheader("Tab 5 research notes")
            st.markdown(st.session_state["engagement_research"])
            render_sources("Tab 5 research sources", st.session_state.get("engagement_sources", []))

            engagement_selected_topic = suggested_topic_selector(
                "Select one current farmer-problem topic for Tab 5",
                "engagement_topic_choice",
                st.session_state["engagement_research"],
                st.session_state.get("engagement_saved_topic_hint", ""),
            )
            engagement_research_notes = st.text_area(
                "Selected research notes for Tab 5",
                value=st.session_state["engagement_research"],
                height=300,
                key="engagement_research_notes",
            )
            engagement_article_target_magazine = target_magazine_selector(
                "engagement_article_target_magazine",
                engagement_selected_topic,
                subject_area,
                engagement_research_notes,
                st.session_state.get("engagement_saved_target_magazine", engagement_target_magazine),
                engagement_magazine_options,
            )

            if st.button(
                "Use this research to write farmer-engagement article",
                key="engagement_write_article",
            ):
                if not engagement_selected_topic.strip():
                    st.warning("Please select one suggested Tab 5 topic before writing.")
                elif not engagement_article_target_magazine:
                    st.warning("Please select the target magazine personality before writing.")
                else:
                    engagement_selected_context = selected_topic_context(
                        engagement_selected_topic,
                        engagement_research_notes,
                        st.session_state.get("engagement_saved_topic_hint", ""),
                        st.session_state.get("engagement_saved_search_details", ""),
                    )
                    engagement_selected_context = with_reference_recommendations(
                        engagement_selected_context, agresco_block
                    )
                    with st.spinner("Writing the farmer-engagement article..."):
                        prompt = farmer_engagement_article_prompt(
                            month,
                            region,
                            subject_area,
                            st.session_state.get("engagement_saved_crop_focus", engagement_crop_focus),
                            article_length,
                            engagement_selected_topic,
                            st.session_state.get("engagement_saved_season_context", engagement_season_context),
                            engagement_article_target_magazine,
                            engagement_selected_context,
                            verified_label_claim_chemicals=verified_label_claim_chemicals,
                        )
                        article, sources = safe_generate_text(
                            client,
                            model,
                            prompt,
                            use_search=use_search_for_article,
                            temperature=temperature,
                        )
                        st.session_state["engagement_article"] = article
                        st.session_state["engagement_article_sources"] = sources
                        st.session_state["engagement_selected_topic"] = engagement_selected_topic
                        st.session_state["engagement_selected_target_magazine"] = engagement_article_target_magazine
                        st.session_state["engagement_research_notes_saved"] = engagement_selected_context
                        st.session_state.pop("engagement_rewritten_article", None)
                        st.session_state.pop("engagement_final_article", None)
                        st.session_state.pop("engagement_review", None)

        if "engagement_article" in st.session_state:
            st.subheader("Tab 5 Step 1: Farmer-engagement draft")
            engagement_draft = st.text_area(
                "Tab 5 draft article",
                value=st.session_state["engagement_article"],
                height=440,
                key="engagement_draft_article",
            )
            st.session_state["engagement_article"] = engagement_draft
            render_sources(
                "Tab 5 article grounding sources",
                st.session_state.get("engagement_article_sources", []),
            )

            st.download_button(
                "Download Tab 5 draft as TXT",
                data=engagement_draft,
                file_name="agri_farmer_engagement_draft.txt",
                mime="text/plain",
                key="engagement_download_draft",
            )

            engagement_review_col, engagement_rewrite_col = st.columns(2)
            with engagement_review_col:
                engagement_review_clicked = st.button(
                    "Review Tab 5 draft quality",
                    key="engagement_review_draft",
                )
            with engagement_rewrite_col:
                engagement_rewrite_clicked = st.button(
                    "Rewrite with farmer-engagement style",
                    key="engagement_rewrite_button",
                )

            if engagement_review_clicked:
                with st.spinner("Reviewing Tab 5 article quality..."):
                    review, _ = safe_generate_text(
                        client,
                        review_model,
                        review_prompt(
                            engagement_draft,
                            st.session_state.get("engagement_selected_target_magazine", "Krushi Vigyan"),
                        ),
                        use_search=False,
                        temperature=0.25,
                        provider=review_provider,
                        api_keys=api_keys,
                    )
                    st.session_state["engagement_review"] = review

            if engagement_rewrite_clicked:
                with st.spinner("Rewriting with the farmer-engagement master prompt..."):
                    rewrite, _ = safe_generate_text(
                        client,
                        model,
                        farmer_engagement_rewrite_prompt(
                            month,
                            region,
                            subject_area,
                            st.session_state.get("engagement_saved_crop_focus", engagement_crop_focus),
                            article_length,
                            st.session_state.get("engagement_selected_topic", ""),
                            st.session_state.get("engagement_saved_season_context", engagement_season_context),
                            st.session_state.get("engagement_selected_target_magazine", "Krushi Vigyan"),
                            st.session_state.get("engagement_research_notes_saved", ""),
                            engagement_draft,
                            verified_label_claim_chemicals=verified_label_claim_chemicals,
                        ),
                        use_search=False,
                        temperature=0.45,
                    )
                    st.session_state["engagement_rewritten_article"] = rewrite
                    st.session_state.pop("engagement_final_article", None)

        if "engagement_review" in st.session_state:
            st.subheader("Tab 5 article review")
            st.markdown(st.session_state["engagement_review"])

        if "engagement_rewritten_article" in st.session_state:
            st.subheader("Tab 5 Step 2: Farmer-engagement rewrite")
            engagement_rewrite = st.text_area(
                "Tab 5 improved article",
                value=st.session_state["engagement_rewritten_article"],
                height=480,
                key="engagement_rewritten_text",
            )
            st.session_state["engagement_rewritten_article"] = engagement_rewrite

            st.download_button(
                "Download Tab 5 rewritten article as TXT",
                data=engagement_rewrite,
                file_name="agri_farmer_engagement_rewrite.txt",
                mime="text/plain",
                key="engagement_download_rewrite",
            )

            st.caption("Soft evidence check included in editor pass.")

            if st.button(
                "Final editor check for Tab 5 magazine article",
                type="primary",
                key="engagement_final_editor_button",
            ):
                with st.spinner("Final editor is polishing the Tab 5 article..."):
                    final_article, _ = safe_generate_text(
                        client,
                        model,
                        farmer_engagement_final_editor_prompt(
                            month,
                            region,
                            subject_area,
                            st.session_state.get("engagement_saved_crop_focus", engagement_crop_focus),
                            article_length,
                            st.session_state.get("engagement_selected_topic", ""),
                            st.session_state.get("engagement_saved_season_context", engagement_season_context),
                            st.session_state.get("engagement_selected_target_magazine", "Krushi Vigyan"),
                            st.session_state.get("engagement_research_notes_saved", ""),
                            engagement_rewrite,
                            verified_label_claim_chemicals=verified_label_claim_chemicals,
                        ),
                        use_search=False,
                        temperature=0.3,
                    )
                    st.session_state["engagement_final_article"] = final_article

        if "engagement_final_article" in st.session_state:
            st.subheader("Tab 5 Step 3: Final magazine-ready article")
            engagement_final = st.text_area(
                "Tab 5 final article for magazine",
                value=st.session_state["engagement_final_article"],
                height=540,
                key="engagement_final_text",
            )
            st.session_state["engagement_final_article"] = engagement_final

            engagement_txt_col, engagement_docx_col = st.columns(2)
            with engagement_txt_col:
                st.download_button(
                    "Download Tab 5 final article as TXT",
                    data=engagement_final,
                    file_name="agri_farmer_engagement_final.txt",
                    mime="text/plain",
                    key="engagement_download_final_txt",
                )
            with engagement_docx_col:
                st.download_button(
                    "Download Tab 5 final article as Word DOCX",
                    data=make_docx(engagement_final),
                    file_name="agri_farmer_engagement_final.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="engagement_download_final_docx",
                )


if __name__ == "__main__":
    main()
