"""Extract only the 'Recommendations for Farming Community' entries from Gujarat
Combined AGRESCO proceedings PDFs into a single compact JSON file.

Usage:
    python extract_agresco_recommendations.py <folder_with_pdfs> [output.json]

Drop every proceedings PDF (2011 to date) into one folder and run this script.
It writes agresco_recommendations.json, which the Streamlit app loads to match
official Gujarat SAU recommendations to the article's crop and pest.
"""

import json
import os
import re
import sys

import pdfplumber

SECTION_NAMES = {
    "1": "Crop Improvement",
    "2": "Crop Production",
    "3": "Plant Protection",
    "4": "Horticulture & Agro-Forestry",
    "5": "Agricultural Engineering / FPT / PHT / Food Tech.",
    "6": "Basic Science & Humanities",
    "7": "Social Science",
}

UNIVERSITY_LINE = re.compile(r"^[A-Z][A-Z .,&'()-]*AGRICULTURAL\s+UNIVERSITY\b", re.MULTILINE)
# Any four-part clause number at the start of a line, e.g. 21.3.1.1
ANY_REC_NO = re.compile(r"(?m)^\s*(\d{1,2})\.(\d{1,2})\.(\d{1,2})\.(\d{1,3})\s+")
GUJARATI = re.compile(r"[઀-૿]")
BODY_CUE = re.compile(
    r"\b(Farmers?|The farmers|Growers?|It is recommended|Recommended|"
    r"To manage|To control|To minimi|For the management|For control|"
    r"Spray|Apply|Seed treatment|Application of|Use of)\b"
)
CIBRC_CUT = re.compile(r"(CIBRC\s*Format|CIB&RC\s*Format|સીઆઇબીઆરસી|Annexure)", re.IGNORECASE)
ACTION_NOTE = re.compile(r"\(\s*Action\s*:.*?\)", re.IGNORECASE | re.DOTALL)
COMMITTEE_NOTE = re.compile(
    r"^(Approved|Accepted|Recommended for|Merged|Deferred|Rejected)\b.*?"
    r"(suggestion|following|community)?", re.IGNORECASE)


def year_from_name(name: str) -> str:
    years = re.findall(r"(20\d{2})", name)
    return years[-1] if years else ""


def meeting_from_name(name: str) -> str:
    match = re.search(r"(\d{1,2})\s*(?:st|nd|rd|th)?\s*(?:proceeding|meeting|agresco)", name, re.IGNORECASE)
    return match.group(1) if match else ""


def clean(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"Page\s*\|\s*\d+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_en_gu(block: str) -> tuple[str, str]:
    en_lines, gu_lines = [], []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        if GUJARATI.search(line):
            gu_lines.append(line)
        else:
            en_lines.append(line)
    return clean(" ".join(en_lines)), clean(" ".join(gu_lines))


def parse_recommendation(rec_no: str, body: str, university: str, year: str,
                         meeting: str, page: int, source_file: str) -> dict:
    section = SECTION_NAMES.get(rec_no.split(".")[1], "")
    # The meeting number is the reliable year anchor: the 7th Combined AGRESCO
    # was 2011, and meetings run one per year, so year = meeting number + 2004.
    meeting_num = rec_no.split(".")[0]
    if meeting_num.isdigit() and 6 < int(meeting_num) < 40:
        meeting = meeting_num
        year = str(int(meeting_num) + 2004)
    # Trim everything from the CIB&RC dose table / ingredient tables onward.
    cut = CIBRC_CUT.search(body)
    core = body[: cut.start()] if cut else body

    lines = [ln.strip() for ln in core.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not COMMITTEE_NOTE.match(ln)]
    title_lines, body_lines, in_body = [], [], False
    for line in lines:
        if not in_body and (BODY_CUE.search(line) or GUJARATI.search(line)):
            in_body = True
        (body_lines if in_body else title_lines).append(line)
    if not body_lines:  # no cue found: first line is the title, rest is body
        title_lines, body_lines = lines[:1], lines[1:]

    title = clean(" ".join(title_lines))
    text_en, text_gu = split_en_gu("\n".join(body_lines))

    # When the recommendation ran onto the title line, English body stays empty;
    # split the title at the first recommendation cue and move the tail to body.
    if not text_en:
        cue = BODY_CUE.search(title)
        if cue and cue.start() > 0:
            text_en = clean(title[cue.start():])
            title = clean(title[: cue.start()])

    # Pull crop/pest from the CIB&RC table header row if present.
    crop, pest = "", ""
    if cut:
        table = body[cut.end():]
        row = re.search(r"20\d{2}[-/]?\d{0,2}\s+([A-Za-z][A-Za-z ]+?)\s{2,}([A-Za-z][A-Za-z ,.-]+)", table)
        if row:
            crop = clean(row.group(1))
            pest = clean(row.group(2))

    return {
        "rec_no": rec_no,
        "year": year,
        "meeting": meeting,
        "section": section,
        "university": university,
        "title": title,
        "crop": crop,
        "pest": pest,
        "recommendation_en": text_en,
        "recommendation_gu": text_gu,
        "source_file": source_file,
        "source_page": page,
    }


def extract_from_pdf(path: str) -> list[dict]:
    source_file = os.path.basename(path)
    year = year_from_name(source_file)
    meeting = meeting_from_name(source_file)

    page_texts = []
    with pdfplumber.open(path) as pdf:
        if not year:
            first = " ".join((pdf.pages[i].extract_text() or "") for i in range(min(3, len(pdf.pages))))
            year = year_from_name(first)
        for page in pdf.pages:
            page_texts.append(page.extract_text() or "")

    # Map each character offset in the joined text back to a page number.
    joined = ""
    offsets = []  # (start_offset, page_number)
    for i, text in enumerate(page_texts):
        offsets.append((len(joined), i + 1))
        joined += text + "\n"

    def page_at(offset: int) -> int:
        page = 1
        for start, num in offsets:
            if start <= offset:
                page = num
            else:
                break
        return page

    # Farmer recommendations are numbered <meeting>.<section>.1.<n>; the third
    # segment "1" marks the 'Recommendations for Farming Community' subsection.
    # (".2" = scientific community, ".3" = new technical programmes.)
    records = []
    seen = set()
    matches = list(ANY_REC_NO.finditer(joined))
    for idx, match in enumerate(matches):
        if match.group(3) != "1":
            continue
        rec_no = f"{match.group(1)}.{match.group(2)}.{match.group(3)}.{match.group(4)}"
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else min(body_start + 8000, len(joined))
        body = joined[body_start:body_end]

        university = ""
        unis = list(UNIVERSITY_LINE.finditer(joined[:match.start()]))
        if unis:
            university = clean(unis[-1].group(0))

        page = page_at(match.start())
        rec = parse_recommendation(rec_no, body, university, year, meeting, page, source_file)

        key = (rec_no, rec["title"][:40])
        if key in seen:
            continue
        seen.add(key)
        if len(rec["recommendation_en"]) + len(rec["recommendation_gu"]) >= 25:
            records.append(rec)
    return records


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Args ending in .json set the output; everything else is an input PDF or
    # a folder of PDFs. This lets you pass a whole folder or a specific list.
    output = "agresco_recommendations.json"
    inputs = []
    for arg in sys.argv[1:]:
        if arg.lower().endswith(".json"):
            output = arg
        else:
            inputs.append(arg)

    pdf_paths = []
    for item in inputs:
        if os.path.isfile(item):
            pdf_paths.append(item)
        elif os.path.isdir(item):
            for name in sorted(os.listdir(item)):
                if name.lower().endswith(".pdf"):
                    pdf_paths.append(os.path.join(item, name))

    all_records = []
    for path in pdf_paths:
        print(f"Parsing {os.path.basename(path)} ...")
        try:
            recs = extract_from_pdf(path)
            print(f"  -> {len(recs)} farmer recommendations")
            all_records.extend(recs)
        except Exception as exc:  # keep going on a bad file
            print(f"  !! failed: {exc}")

    with open(output, "w", encoding="utf-8") as handle:
        json.dump(all_records, handle, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(all_records)} recommendations to {output}")


if __name__ == "__main__":
    main()
