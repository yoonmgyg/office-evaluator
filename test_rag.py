import sys, logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# [Paste _get_corpus_cache, _find_matching_filenames, query_treasury_corpus]
_corpus_cache = None

def _get_corpus_cache():
    global _corpus_cache
    if _corpus_cache is not None: return _corpus_cache
    import requests, zipfile, io
    zip_url = "https://raw.githubusercontent.com/databricks/officeqa/6aa8c32/treasury_bulletins_parsed/transformed/treasury_bulletins_transformed.zip"
    resp = requests.get(zip_url, timeout=60)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        _corpus_cache = {}
        for fname in z.namelist():
            if fname.endswith(".txt"):
                _corpus_cache[fname] = z.read(fname).decode("utf-8", errors="replace")
    return _corpus_cache

def _find_matching_filenames(year: str, month: str, filenames: list) -> list:
    year = year.strip()
    month = month.strip().lower()
    month_mapping = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12"
    }
    month_num = month_mapping.get(month, month) if month else ""
    matches = []
    for fname in filenames:
        if year in fname:
            if month_num:
                if f"_{month_num}.txt" in fname or f"_{month_num}_" in fname:
                    matches.append(fname)
            else:
                matches.append(fname)
    return sorted(matches)

def query_treasury_corpus(year: str, month: str = "", keyword: str = "") -> str:
    corpus = _get_corpus_cache()
    matching_files = _find_matching_filenames(year, month, list(corpus.keys()))
    if not matching_files and month:
        matching_files = _find_matching_filenames(year, "", list(corpus.keys()))
    if not matching_files: return f"No files for {year}"
    
    results = []
    keyword_lower = keyword.strip().lower() if keyword else ""
    for fname in matching_files[:12]:
        content = corpus[fname]
        if not keyword_lower:
            results.append(f"--- {fname} ---\n{content[:500]}...")
            continue
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        matching_paras = [p for p in paragraphs if keyword_lower in p.lower()]
        if matching_paras:
            results.append(f"--- {fname} (filtered for '{keyword}') ---")
            results.extend(matching_paras[:10])
    return "\n".join(results)

print("Starting RAG test for 1940 January 'national defense'...")
print(query_treasury_corpus("1940", "January", "national defense")[:1500])

