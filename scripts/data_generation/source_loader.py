"""Phase 1: Load and preprocess 6 JSON source files (Markdown text)."""

import json
import os
import re
import chardet
from data_generation.config import SOURCE_FILES, GBK_ENCODED_FILES, CLAUSES_OUTPUT


def detect_and_read(filepath):
    """Detect encoding and read file, returning Unicode text.

    Uses chardet for encoding detection, then validates by checking
    for Chinese characters anywhere in the decoded text (not just
    the first 500 chars — some files have long ASCII headers).
    """
    with open(filepath, "rb") as f:
        raw = f.read()

    # Use chardet for primary detection
    try:
        detected = chardet.detect(raw)
        encoding = detected.get("encoding", "utf-8") or "utf-8"
        confidence = detected.get("confidence", 0)
        if confidence > 0.7 and encoding.lower() not in ("ascii", "iso-8859-1", "latin-1"):
            try:
                text = raw.decode(encoding)
                # Scan full text for Chinese chars (some files have long ASCII headers)
                if _has_chinese(text):
                    return text
            except (UnicodeDecodeError, LookupError):
                pass
    except Exception:
        pass

    # Try UTF-8 (scan full text, not just first 500 chars)
    try:
        text = raw.decode("utf-8")
        if _has_chinese(text):
            return text
    except UnicodeDecodeError:
        pass

    # Try GBK/GB18030 for files that might be legacy encoded
    for enc in ("gb18030", "gbk"):
        try:
            text = raw.decode(enc, errors="replace")
            if _has_chinese(text):
                return text
        except (UnicodeDecodeError, LookupError):
            pass

    return raw.decode("utf-8", errors="replace")


def _has_chinese(text, sample_size=3000):
    """Check if text contains Chinese characters.

    Scans up to sample_size characters across the full text,
    not just the beginning, to handle files with long headers.
    """
    step = max(len(text) // 10, 500)
    for start in range(0, min(len(text), len(text)), step):
        chunk = text[start:start + 500]
        if any('一' <= c <= '鿿' for c in chunk):
            return True
    return False


def derive_title(filepath):
    """Derive a human-readable title from the filename."""
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)[0]
    name = name.replace("+", " ")
    return name


def clean_text(text):
    """Remove OCR noise, normalize formatting, and desensitize."""
    # Remove common OCR garbage patterns
    text = re.sub(r"[犌犅]{2,}", "", text)
    text = re.sub(r"[-]", "", text)  # Private Use Area chars

    # Normalize LaTeX escaping
    text = re.sub(r"\\\\<", "<", text)
    text = re.sub(r"\\\\-\\\\->", "→", text)
    text = re.sub(r"\\\\frac", r"\\frac", text)
    text = re.sub(r"\\\\times", r"\\times", text)
    text = re.sub(r"\$([^$]*?)\\\\_([^$]*?)\$", r"$\1_\2$", text)

    # Normalize whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r" {3,}", "  ", text)

    # Desensitize: anonymize real entities before they enter the pipeline
    text = desensitize(text)

    return text


def desensitize(text):
    """Anonymize sensitive named entities in design/review documents.

    Replaces real company names, place names, substation names, line numbers,
    and project names with generic identifiers. Maintains consistent mapping
    within a single document so the same entity always gets the same pseudonym.
    """
    # ── Step 1: Fixed-pattern bulk replacement ──
    # Company/institution names
    text = re.sub(r'国家电网(?:有限)?(?:公司|有限公司)', '某电网公司', text)
    text = re.sub(r'南方电网(?:有限)?(?:公司|有限公司)', '某电网公司', text)
    text = re.sub(r'国家电网\b(?!工程|项目|规划|技术|标准)', '某电网公司', text)
    text = re.sub(r'[一-鿿]{2,4}省电力(?:有限)?(?:公司|设计院|规划院|研究院|经研院)',
                  '某省级电力设计院', text)
    text = re.sub(r'(?:华东|华北|华中|东北|西北|西南|华南)电力(?:有限)?(?:公司|设计院|规划院|研究院|经研院)',
                  '某区域电力设计院', text)
    text = re.sub(r'[一-鿿]{2,4}市电力(?:有限)?公司', '某市电力公司', text)
    text = re.sub(r'[一-鿿]{2,6}(?:供电局|供电公司)', '某供电局', text)
    # Standard publisher/drafter info lines
    text = re.sub(r'(主编|参编|起草|归口)单位[：:]\s*[一-鿿\d、\s]+',
                  r'\1单位: [已脱敏]', text)
    text = re.sub(r'(批准|发布)部门[：:]\s*[一-鿿\d、\s]+',
                  r'\1部门: [已脱敏]', text)

    # Place names (cities, counties, districts, towns, streets)
    text = re.sub(r'[一-鿿]{2,4}(?:市|县|区|镇|乡|村|城|街道|地区)'
                  r'(?!电力|供电|变电站|发电|电网|规划|公司|设计)',
                  '某地', text)
    # Specific location patterns: XX路XX号, XX大道, XX工业园
    text = re.sub(r'[一-鿿]{2,8}(?:路|大道|大街|工业园|开发区|新区|新城|港口|码头)'
                  r'(?!电力|设计|工程)',
                  '某区域', text)

    # Project/case numbers
    text = re.sub(r'(?:编号|案号|文号)[：:]\s*[一-鿿\d〔〕\[\]\(\)\-\s]+',
                  r'编号: [已脱敏]', text)
    text = re.sub(r'[一-鿿\d]+号(?:工程)?\s*可行性研究', '某工程可行性研究', text)

    # Phone/fax numbers
    text = re.sub(r'[\d\-]{8,}(?:[\s]*[转分]\s*\d+)?', '[已脱敏]', text)

    # Dates (keep year for context, remove month-day)
    text = re.sub(r'\b(20\d{2})[年/\-]\d{1,2}[月/\-]\d{1,2}[日号]?\b',
                  r'\1年', text)

    # ── Step 2: Consistent substation/plant name mapping ──
    # Match proper substation names: prefix of 2-4 Chinese chars + facility type.
    # Filters out generic engineering compound words.
    station_pattern = re.compile(
        r'[一-鿿]{2,4}'
        r'(?:变电站|发电厂|换流站|开关站|升压站|降压站)'
        r'(?:[一二三四五六七八九十\d]+期)?'
        r'(?!(?:设计|技术|建设|运行|维护|选址|布置|施工|改造|扩建|工程))'
    )
    filter_words = {'站址','进站','出站','本站','该站','新建','原站','所址',
                    '线路','母线','电缆','间隔','配电','输电','变电','主变',
                    '构架','接地','避雷','围墙','大门','道路','排水','消防',
                    '采用','属于','位于','对于','关于','一个','两个','多个'}
    stations = sorted(set(station_pattern.findall(text)), key=lambda x: -len(x))
    stations_filtered = []
    for s in stations:
        # Remove the suffix to get prefix
        for suffix in ['变电站','发电厂','换流站','开关站','升压站','降压站']:
            if s.endswith(suffix):
                prefix = s[:-len(suffix)]
                break
        else:
            prefix = ''
        if len(prefix) < 2 or prefix in filter_words or re.match(r'^[\d一二三四五六七八九十]+$', prefix):
            continue
        stations_filtered.append(s)

    for i, s in enumerate(stations_filtered):
        label = f'变电站{chr(65+i)}' if i < 26 else f'变电站{i-25}'
        text = text.replace(s, label)

    # ── Step 3: Consistent line number mapping ──
    # Match 3-4 digit line identifiers (specific to power grid)
    line_pattern = re.compile(r'(?:^|\b)(\d{3,4})(?:线|回线|回)(?:\b|$)')
    lines = sorted(set(line_pattern.findall(text)), key=lambda x: int(x))
    line_labels = []
    for i, l in enumerate(lines):
        label = f'线路{chr(65+i)}' if i < 26 else f'线路{i-25}'
        line_labels.append((l, label))
    for original, label in line_labels:
        text = re.sub(rf'\b{original}(?:线|回线|回)?\b',
                      f'{label}路', text)

    # ── Step 4: Person/unit names in standard headers → generic ──
    # Drafting committee member lists
    text = re.sub(r'[一-鿿]{1,3}(?:工|总|师傅|经理|主任|工程师)(?!程|作|艺|具|厂|业|师)',
                  '某专家', text)
    # University/institute names in standard headers (non-power-company entities)
    text = re.sub(r'[一-鿿]{2,6}(?:大学|学院|研究院)(?!设计|规划)',
                  '某研究院', text)

    return text


def parse_sections(text, title):
    """Parse headings into structured sections.

    Supports three heading formats:
    1. Markdown: # Title, ## Subtitle (used in DLT 5429, plans)
    2. Numbered: 3.1 标题, 3.1.1 标题 (used in GB 38755, DLT 5218)
    3. Full-width numbered: １　范围, ３．１　要求 (OCR artifacts)

    Returns list of {section_number, section_title, level, text, char_count, source_file}
    """
    # Compound pattern: Markdown headings OR numbered Chinese-style sections
    # Note: Some files have escaped # (\\#) from prior processing
    heading_pattern = re.compile(
        r"^\\?#{1,4}\s+(.+)$|"                                 # Markdown: # Title (opt. escaped)
        r"^([0-9]+(?:\.[0-9]+){0,3})\s*[\s　]+(.+)$|"     # Numbered: 3.1.1 Title
        r"^（([0-9]+)）\s*(.+)$|"                              # Parenthesized: （1）Title
        r"^[一二三四五六七八九十]、\s*(.+)$",                     # Chinese numbered: 一、Title
        re.MULTILINE,
    )
    matches = list(heading_pattern.finditer(text))

    if not matches:
        # No headings found — treat entire text as one section
        sections = [{
            "section_number": "全文",
            "section_title": title,
            "level": 0,
            "text": text,
            "char_count": len(text),
            "source_file": title,
        }]
        return sections

    sections = []
    for i, match in enumerate(matches):
        groups = match.groups()
        if groups[0]:
            raw_heading = match.group(0).lstrip("\\")
            level = len(raw_heading) - len(raw_heading.lstrip("#"))
            heading_text = groups[0].strip()
        elif groups[1]:
            num_parts = groups[1].split(".")
            level = len(num_parts)
            heading_text = f"{groups[1]} {groups[2].strip()}"
        elif groups[3]:
            level = 2
            heading_text = f"({groups[3]}) {groups[4].strip()}"
        elif groups[5]:
            level = 2
            heading_text = f"{match.group(0).strip()}"
        else:
            continue

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        if len(content) < 50:
            continue

        sections.append({
            "section_number": heading_text[:80],
            "section_title": heading_text,
            "level": min(level, 4),
            "text": content,
            "char_count": len(content),
            "source_file": title,
        })

    if not sections:
        sections = [{
            "section_number": "全文",
            "section_title": title,
            "level": 0,
            "text": text,
            "char_count": len(text),
            "source_file": title,
        }]

    return sections


def load_all_sources():
    """Load all 6 source files, clean, and parse into structured sections.

    Returns:
        sources: list of {title, raw_text, sections, char_count}
        all_sections: flat list of all parsed sections
    """
    sources = []
    all_sections = []

    for filepath in SOURCE_FILES:
        if not os.path.exists(filepath):
            print(f"  [WARN] File not found: {filepath}")
            continue

        title = derive_title(filepath)
        print(f"  Loading: {title} ...")

        raw_text = detect_and_read(filepath)
        cleaned = clean_text(raw_text)
        sections = parse_sections(cleaned, title)

        source = {
            "title": title,
            "filepath": filepath,
            "raw_text": cleaned,
            "sections": sections,
            "char_count": len(cleaned),
        }
        sources.append(source)
        all_sections.extend(sections)

        print(f"    -> {len(sections)} sections, {len(cleaned):,} chars")

    # Assign global indices
    for i, sec in enumerate(all_sections):
        sec["global_index"] = i

    return sources, all_sections


def print_stats(sources, sections):
    """Print loading statistics."""
    print(f"\n{'='*60}")
    print(f"Source Loading Summary")
    print(f"{'='*60}")
    print(f"  Total source files: {len(sources)}")
    print(f"  Total sections:     {len(sections)}")
    total_chars = sum(s["char_count"] for s in sections)
    print(f"  Total characters:   {total_chars:,}")

    print(f"\n  Per-source breakdown:")
    for src in sources:
        print(f"    {src['title']:50s} {len(src['sections']):4d} sections  {src['char_count']:>10,} chars")

    # Section size distribution
    sizes = sorted([s["char_count"] for s in sections])
    if sizes:
        print(f"\n  Section size distribution:")
        print(f"    Min:    {min(sizes):,} chars")
        print(f"    P25:    {sizes[len(sizes)//4]:,} chars")
        print(f"    Median: {sizes[len(sizes)//2]:,} chars")
        print(f"    P75:    {sizes[3*len(sizes)//4]:,} chars")
        print(f"    Max:    {max(sizes):,} chars")


if __name__ == "__main__":
    print("Phase 1: Loading source materials...\n")
    sources, sections = load_all_sources()
    print_stats(sources, sections)

    output = {
        "sources": [
            {"title": s["title"], "char_count": s["char_count"], "section_count": len(s["sections"])}
            for s in sources
        ],
        "total_sections": len(sections),
        "sections": sections,
    }
    with open(CLAUSES_OUTPUT.replace(".json", "_sections.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Sections saved to: {CLAUSES_OUTPUT.replace('.json', '_sections.json')}")
