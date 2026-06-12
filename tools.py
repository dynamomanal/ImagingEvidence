import requests
import xml.etree.ElementTree as ET
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════
#  INPUT SCHEMAS
# ══════════════════════════════════════════════════════════════

class PubMedInput(BaseModel):
    query: str = Field(description="Medical findings or diagnosis terms to search")
    max_results: int = Field(default=10, description="Number of articles to return")


class EuropePMCInput(BaseModel):
    query: str = Field(description="Medical terms to search for open access papers")
    max_results: int = Field(default=8, description="Number of articles to return")


# ══════════════════════════════════════════════════════════════
#  RAW FETCHERS — return structured dicts with full abstracts
# ══════════════════════════════════════════════════════════════

def fetch_pubmed_articles(query: str, max_results: int = 15) -> list:
    """
    Search PubMed with MeSH-aware query and return articles with full abstracts.
    Filters for English-language papers that have abstracts.
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    # Build a precise PubMed query: user terms + echocardiography scope + quality filters
    clean = query[:220].strip()
    # Only append echocardiography MeSH if the query doesn't already contain it
    if "echocard" not in clean.lower() and "echo" not in clean.lower():
        mesh_scope = " AND (echocardiography[MeSH Terms] OR echocardiogram[tiab] OR cardiac imaging[tiab])"
    else:
        mesh_scope = ""
    pubmed_term = f"({clean}){mesh_scope} AND hasabstract[text] AND English[Language]"

    try:
        search_resp = requests.get(
            f"{base}esearch.fcgi",
            params={
                "db":      "pubmed",
                "term":    pubmed_term,
                "retmax":  max_results,
                "retmode": "json",
                "sort":    "relevance",
            },
            timeout=15,
        )
        search_resp.raise_for_status()
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []

    if not ids:
        return []

    try:
        fetch_resp = requests.get(
            f"{base}efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "xml"},
            timeout=18,
        )
        root = ET.fromstring(fetch_resp.text)
    except Exception:
        return []

    articles = []
    for article in root.findall(".//PubmedArticle"):
        title   = article.findtext(".//ArticleTitle", "N/A")
        journal = article.findtext(".//Journal/Title", "N/A")
        year    = article.findtext(".//PubDate/Year", "N/A")
        pmid    = article.findtext(".//PMID", "")
        volume  = article.findtext(".//JournalIssue/Volume", "")
        issue   = article.findtext(".//JournalIssue/Issue", "")
        pages   = article.findtext(".//MedlinePgn", "")

        # Authors — up to first 6, then "et al."
        author_nodes = article.findall(".//AuthorList/Author")
        author_parts = []
        for au in author_nodes[:6]:
            last  = au.findtext("LastName", "")
            ini   = au.findtext("Initials", "")
            if last:
                author_parts.append(f"{last} {ini}".strip())
        author_str = ", ".join(author_parts)
        if len(author_nodes) > 6:
            author_str += " et al."

        # DOI
        doi = ""
        for id_node in article.findall(".//ArticleIdList/ArticleId"):
            if id_node.get("IdType") == "doi":
                doi = id_node.text or ""
                break

        # Abstract — multiple labeled sections joined
        abstract_parts = article.findall(".//AbstractText")
        if abstract_parts:
            abstract = " ".join(
                (f"{p.get('Label','')}: " if p.get("Label") else "") + (p.text or "")
                for p in abstract_parts
            ).strip()
        else:
            abstract = ""

        abstract = (abstract[:800] + "…") if len(abstract) > 800 else abstract

        articles.append({
            "source":   "PubMed",
            "title":    title,
            "abstract": abstract,
            "journal":  journal,
            "year":     year,
            "volume":   volume,
            "issue":    issue,
            "pages":    pages,
            "authors":  author_str,
            "doi":      doi,
            "link":     f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })

    return articles


def fetch_pmc_articles(query: str, max_results: int = 8) -> list:
    """
    Search Europe PMC (open-access) and return articles with abstracts + DOIs.
    """
    clean = query[:220].strip()
    if "echocard" not in clean.lower() and "echo" not in clean.lower():
        pmc_query = f"{clean} echocardiography OPEN_ACCESS:y"
    else:
        pmc_query = f"{clean} OPEN_ACCESS:y"

    try:
        resp = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={
                "query":      pmc_query,
                "resultType": "core",
                "pageSize":   max_results,
                "format":     "json",
                "sort":       "CITED desc",
            },
            timeout=12,
        )
        resp.raise_for_status()
        results = resp.json().get("resultList", {}).get("result", [])
    except Exception:
        return []

    articles = []
    for r in results:
        abstract = r.get("abstractText", "")
        abstract = (abstract[:800] + "…") if len(abstract) > 800 else abstract
        if not abstract:
            continue

        doi     = r.get("doi", "")
        authors = r.get("authorString", "")
        year    = str(r.get("pubYear", "N/A"))
        journal = r.get("journalTitle", "N/A")

        articles.append({
            "source":  "PMC",
            "title":   r.get("title", "N/A"),
            "abstract": abstract,
            "journal":  journal,
            "year":     year,
            "volume":   str(r.get("journalVolume", "")),
            "issue":    str(r.get("issue", "")),
            "pages":    str(r.get("pageInfo", "")),
            "authors":  authors,
            "doi":      doi,
            "link":     f"https://doi.org/{doi}" if doi else "N/A",
        })

    return articles


def fetch_semantic_scholar_articles(query: str, max_results: int = 6) -> list:
    """
    Search Semantic Scholar for highly-cited cardiac imaging papers.
    Free API — no key required.
    """
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query":  f"{query[:200]} echocardiography cardiac imaging guideline",
                "limit":  max_results,
                "fields": "title,abstract,authors,year,venue,externalIds,openAccessPdf,citationCount",
            },
            headers={"User-Agent": "ImagingEvidence/1.0 (academic research)"},
            timeout=12,
        )
        papers = resp.json().get("data", [])
    except Exception:
        return []

    articles = []
    for p in papers:
        abstract = p.get("abstract") or ""
        if not abstract:
            continue
        abstract = (abstract[:800] + "…") if len(abstract) > 800 else abstract

        author_names = [a.get("name", "") for a in p.get("authors", [])[:6]]
        author_str   = ", ".join(n for n in author_names if n)
        if len(p.get("authors", [])) > 6:
            author_str += " et al."

        ext   = p.get("externalIds", {}) or {}
        doi   = ext.get("DOI", "")
        pmid  = ext.get("PubMed", "")
        pdf   = (p.get("openAccessPdf") or {}).get("url", "")

        if doi:
            link = f"https://doi.org/{doi}"
        elif pmid:
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        elif pdf:
            link = pdf
        else:
            link = "N/A"

        articles.append({
            "source":   "SemanticScholar",
            "title":    p.get("title", "N/A"),
            "abstract": abstract,
            "journal":  p.get("venue", "N/A"),
            "year":     str(p.get("year", "N/A")),
            "authors":  author_str,
            "doi":      doi,
            "link":     link,
        })

    return articles


# ══════════════════════════════════════════════════════════════
#  LANGCHAIN TOOL WRAPPERS  (kept for backwards compatibility)
# ══════════════════════════════════════════════════════════════

class PubMedSearchTool(BaseTool):
    name: str = "pubmed_search"
    description: str = (
        "Search PubMed for cardiac imaging journal articles. "
        "Use after MedGemma returns findings to find related literature."
    )
    args_schema: type = PubMedInput

    def _run(self, query: str, max_results: int = 10) -> str:
        articles = fetch_pubmed_articles(query, max_results)
        if not articles:
            return "No PubMed articles found for this query."
        rows = []
        for a in articles:
            rows.append(
                f"**{a['title']}**\n"
                f"  📰 {a['journal']} | 📅 {a['year']}\n"
                f"  🔗 {a['link']}"
            )
        return "\n\n".join(rows)

    async def _arun(self, query: str, max_results: int = 10) -> str:
        return self._run(query, max_results)


class EuropePMCTool(BaseTool):
    name: str = "europe_pmc_search"
    description: str = (
        "Search Europe PMC for open-access cardiac imaging papers with abstracts."
    )
    args_schema: type = EuropePMCInput

    def _run(self, query: str, max_results: int = 8) -> str:
        articles = fetch_pmc_articles(query, max_results)
        if not articles:
            return "No open-access articles found on Europe PMC."
        rows = []
        for a in articles:
            rows.append(
                f"**{a['title']}**\n"
                f"  📰 {a['journal']} | 📅 {a['year']}\n"
                f"  📄 {a['abstract']}\n"
                f"  🔗 {a['link']}"
            )
        return "\n\n".join(rows)

    async def _arun(self, query: str, max_results: int = 8) -> str:
        return self._run(query, max_results)
