import os
import asyncio
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from fastmcp import FastMCP
import httpx
from pydantic import BaseModel, Field
from tavily import TavilyClient

# 環境変数読み込み
load_dotenv()

# FastMCP初期化
mcp = FastMCP(name="search")

# Web検索クライアント
class WebSearchClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.client = None

        if not self.api_key:
            print("[search] TAVILY_API_KEY not set; web_* tools will return error.")
            return
        try:
            self.client = TavilyClient(api_key=self.api_key)
            print("[search] Tavily client initialized")
        except Exception as e:
            print(f"[search] Tavily init failed: {e}")
            self.client = None

    def search(self, **kwargs) -> Dict[str, Any]:
        if not self.client:
            raise RuntimeError("Tavily client not initialized")
        return self.client.search(**kwargs)

    def get_search_context(self, **kwargs) -> str:
        if not self.client:
            raise RuntimeError("Tavily client not initialized")
        return self.client.get_search_context(**kwargs)

    def qna_search(self, **kwargs) -> str:
        if not self.client:
            raise RuntimeError("Tavily client not initialized")
        return self.client.qna_search(**kwargs)

# Github検索クライアント
class GitHubClient:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "GitHub-MCP-Server/1.0",
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
            print("[search] GitHub token configured")
        else:
            print("[search] GitHub token NOT configured (rate limit will be low)")

    async def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{self.base_url}{path}", headers=self.headers, params=params)
            if r.status_code == 403:
                remaining = r.headers.get("X-RateLimit-Remaining", "0")
                reset = r.headers.get("X-RateLimit-Reset", "unknown")
                raise RuntimeError(f"GitHub rate limit exceeded (remaining={remaining}, reset={reset})")
            r.raise_for_status()
            return r.json()

    async def search_repositories(self, query: str, sort: str, order: str, per_page: int) -> Dict[str, Any]:
        params = {"q": query, "sort": sort, "order": order, "per_page": per_page}
        return await self._get("/search/repositories", params)

# arXivクライアント
class ArxivClient:
    def __init__(self):
        self.base_url = "https://export.arxiv.org/api/query"
        self.headers = {"User-Agent": "arXiv-MCP-Server/1.0"}
        self.valid_sort_by = {
            "relevance": "relevance",
            "lastUpdatedDate": "lastUpdatedDate",
            "submittedDate": "submittedDate",
            "recent": "submittedDate",
        }
        self.valid_sort_order = {"ascending": "ascending", "descending": "descending", "asc": "ascending", "desc": "descending"}

    async def search_papers(
        self, query: str, max_results: int = 10, start: int = 0, sort_by: str = "relevance", sort_order: str = "descending"
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"search_query": query, "start": start, "max_results": max_results}
        if sort_by in self.valid_sort_by and self.valid_sort_by[sort_by] != "relevance":
            params["sortBy"] = self.valid_sort_by[sort_by]
            params["sortOrder"] = self.valid_sort_order.get(sort_order, "descending")
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            r = await client.get(self.base_url, headers=self.headers, params=params)
            r.raise_for_status()
            return self._parse_arxiv_response(r.content)

    def _parse_arxiv_response(self, xml_content: bytes) -> Dict[str, Any]:
        ns = {"atom": "http://www.w3.org/2005/Atom", "opensearch": "http://a9.com/-/spec/opensearch/1.1/", "arxiv": "http://arxiv.org/schemas/atom"}
        root = ET.fromstring(xml_content)
        total = root.find(".//opensearch:totalResults", ns)
        start = root.find(".//opensearch:startIndex", ns)
        ipp = root.find(".//opensearch:itemsPerPage", ns)
        papers: List[Dict[str, Any]] = []
        for entry in root.findall(".//atom:entry", ns):
            papers.append(self._parse_entry(entry, ns))
        return {
            "total_results": int(total.text) if total is not None else 0,
            "start_index": int(start.text) if start is not None else 0,
            "items_per_page": int(ipp.text) if ipp is not None else 0,
            "papers": papers,
        }

    def _parse_entry(self, entry, ns) -> Dict[str, Any]:
        title = (entry.find("atom:title", ns).text or "").strip()
        id_elem = entry.find("atom:id", ns)
        arxiv_id = (id_elem.text or "").replace("http://arxiv.org/abs/", "") if id_elem is not None else ""
        summary = (entry.find("atom:summary", ns).text or "").strip()
        published = (entry.find("atom:published", ns).text or "")
        updated = (entry.find("atom:updated", ns).text or "")
        authors = []
        for a in entry.findall("atom:author", ns):
            name = (a.find("atom:name", ns).text or "") if a.find("atom:name", ns) is not None else ""
            aff = a.find("arxiv:affiliation", ns)
            authors.append({"name": name, "affiliation": aff.text if aff is not None else ""})
        categories = [c.get("term", "") for c in entry.findall("atom:category", ns)]
        primary_category = entry.find("arxiv:primary_category", ns)
        links: Dict[str, str] = {}
        for link in entry.findall("atom:link", ns):
            rel = link.get("rel", "")
            title_attr = link.get("title", "")
            href = link.get("href", "")
            if rel == "alternate":
                links["abstract"] = href
            elif title_attr == "pdf":
                links["pdf"] = href
        comment = entry.find("arxiv:comment", ns)
        journal_ref = entry.find("arxiv:journal_ref", ns)
        doi = entry.find("arxiv:doi", ns)
        return {
            "id": arxiv_id,
            "title": title,
            "authors": authors,
            "published": published,
            "updated": updated,
            "summary": summary,
            "categories": categories,
            "primary_category": primary_category.get("term", "") if primary_category is not None else "",
            "links": links,
            "comment": comment.text if comment is not None else "",
            "journal_ref": journal_ref.text if journal_ref is not None else "",
            "doi": doi.text if doi is not None else "",
        }

# クライアントのインスタンス化
web_search_client = WebSearchClient()
github_client = GitHubClient()
arxiv_client = ArxivClient()

# Pydanticレスポンス
class WebSearchResult(BaseModel):
    results: List[Dict[str, Any]] = Field(description="Web検索結果")
    total_results: int = Field(description="総検索結果数")
    query: str = Field(description="検索クエリ")

class GithubSearchResult(BaseModel):
    total_count: int = Field(description="総数")
    items: List[Dict[str, Any]] = Field(description="検索結果")

class ArxivSearchResult(BaseModel):
    total_results: int = Field(description="総検索結果数")
    start_index: int = Field(description="開始インデックス")
    items_per_page: int = Field(description="ページあたりの件数")
    papers: List[Dict[str, Any]] = Field(description="論文リスト")

# MCPツール
@mcp.tool(
    name="web_search",
    description="Tavily Search APIでWeb検索",
    tags=["web", "search", "tavily"]
)
async def web_search(
    query: str,
    max_results: int = 3,
    search_depth: str = "basic",
    include_answer: bool = False,
    include_images: bool = False,
    include_raw_content: bool = False,
    country: Optional[str] = None,
    topic: str = "general",
) -> WebSearchResult:
    if not web_search_client.client:
        return WebSearchResult(results=[{"error": "Tavily client not initialized (set TAVILY_API_KEY)"}], total_results=0, query=query)

    try:
        max_results = min(max(1, max_results), 20)
        if search_depth not in {"basic", "advanced"}:
            search_depth = "basic"
        if topic not in {"general", "news"}:
            topic = "general"

        params: Dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_images": include_images,
            "include_raw_content": include_raw_content,
            "topic": topic,
            "country": "Japan",
        }
        try:
            resp = web_search_client.search(**params)
        except Exception as e:
            if "invalid country" in str(e).lower():
                params.pop("country", None)
                resp = web_search_client.search(**params)
            else:
                raise
        results: List[Dict[str, Any]] = []
        for r in resp.get("results", []):
            item = {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
                "published_date": r.get("published_date", ""),
            }
            if include_raw_content and "raw_content" in r:
                item["raw_content"] = (r["raw_content"] or "")[:1000]
            results.append(item)

        if include_answer and "answer" in resp:
            results.insert(0, {"type": "answer", "content": resp["answer"], "title": "AI Generated Answer", "url": "", "score": 1.0})

        if include_images and "images" in resp:
            for img in resp["images"][:3]:
                if isinstance(img, dict):
                    results.append({"type": "image", "url": img.get("url", ""), "description": img.get("description", "")})
                else:
                    results.append({"type": "image", "url": img, "description": ""})

        return WebSearchResult(results=results, total_results=len(results), query=query)
    except Exception as e:
        return WebSearchResult(results=[{"error": f"Web search failed: {e}"}], total_results=0, query=query)

@mcp.tool(
    name="search_github_repositories",
    description="GitHubのリポジトリ検索",
    tags=["github", "search", "repositories", "code"]
)
async def search_github_repositories(query: str, sort: str = "stars", order: str = "desc", per_page: int = 10) -> GithubSearchResult:
    try:
        data = await github_client.search_repositories(query, sort, order, per_page)
        items = [
            {
                "full_name": it.get("full_name"),
                "description": it.get("description"),
                "stars": it.get("stargazers_count"),
                "forks": it.get("forks_count"),
                "language": it.get("language"),
                "topics": it.get("topics"),
                "html_url": it.get("html_url"),
            }
            for it in data.get("items", [])
        ]
        return GithubSearchResult(total_count=data.get("total_count", 0), items=items)
    except Exception as e:
        raise Exception(f"Repository search failed: {e}")

@mcp.tool(
    name="search_arxiv_papers",
    description="arXivで論文検索（タイトル/著者/要約/PDFリンク等）",
    tags=["arxiv", "papers", "search", "research"]
)
async def search_arxiv_papers(
    query: str,
    max_results: int = 10,
    start: int = 0,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    include_urls: bool = True,
    response_format: str = "detailed",
) -> ArxivSearchResult:
    try:
        max_results = min(max_results, 30)
        data = await arxiv_client.search_papers(query=query, max_results=max_results, start=start, sort_by=sort_by, sort_order=sort_order)

        papers_fmt: List[Dict[str, Any]] = []
        for p in data["papers"]:
            if response_format in {"detailed", "urls_first"} and include_urls:
                papers_fmt.append(
                    {
                        "論文ID": p["id"],
                        "PDF": p["links"].get("pdf", "unavailable"),
                        "要約URL": p["links"].get("abstract", "unavailable"),
                        "タイトル": p["title"],
                        "著者": ", ".join([a["name"] for a in p["authors"]]),
                        "公開日": p["published"][:10] if p["published"] else "",
                        "要約": (p["summary"][:200] + "...") if len(p["summary"]) > 200 else p["summary"],
                        "カテゴリ": ", ".join(p["categories"]),
                    }
                )
            else:
                papers_fmt.append(
                    {
                        "id": p["id"],
                        "title": p["title"],
                        "authors": [a["name"] for a in p["authors"]],
                        "summary": (p["summary"][:200] + "...") if len(p["summary"]) > 200 else p["summary"],
                    }
                )
        return ArxivSearchResult(
            total_results=data["total_results"],
            start_index=data["start_index"],
            items_per_page=data["items_per_page"],
            papers=papers_fmt,
        )
    except Exception as e:
        raise Exception(f"arXiv search failed: {e}")

# エントリポイント
if __name__ == "__main__":
    print("[search] Search MCP server starting (stdio)...")
    mcp.run()