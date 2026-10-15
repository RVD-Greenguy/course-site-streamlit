"""Direct port of server/src/search.js — free keyless DuckDuckGo HTML scrape,
plus an optional Google Custom Search JSON API path.
"""
from urllib.parse import urlencode, urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup

from .config import get_secret


def ddgs_search(query: str, limit: int = 5):
    url = f"https://html.duckduckgo.com/html/?{urlencode({'q': query})}"
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; course-site-chat/1.0)"}, timeout=10)
    if not res.ok:
        raise RuntimeError(f"Search request failed: {res.status_code}")

    soup = BeautifulSoup(res.text, "html.parser")
    results = []
    for el in soup.select(".result__body"):
        if len(results) >= limit:
            break
        title_el = el.select_one(".result__title a")
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        href = title_el.get("href")
        snippet_el = el.select_one(".result__snippet")
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        if title and href:
            results.append({"title": title, "url": _resolve_redirect(href), "snippet": snippet})
    return results


def _resolve_redirect(href: str) -> str:
    """DuckDuckGo's HTML result links are redirects
    (//duckduckgo.com/l/?uddg=<encoded-target>) rather than the real
    destination — unwrap them so links in the UI go somewhere useful."""
    try:
        if href.startswith("//"):
            href = "https:" + href
        parsed = urlparse(href)
        target = parse_qs(parsed.query).get("uddg", [None])[0]
        return unquote(target) if target else href
    except Exception:
        return href


def google_search(query: str, limit: int = 5):
    api_key = get_secret("GOOGLE_API_KEY")
    cse_id = get_secret("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        raise RuntimeError("Google search mode needs GOOGLE_API_KEY and GOOGLE_CSE_ID set on the server.")

    url = "https://www.googleapis.com/customsearch/v1"
    res = requests.get(url, params={"key": api_key, "cx": cse_id, "num": limit, "q": query}, timeout=10)
    if not res.ok:
        raise RuntimeError(f"Google search request failed ({res.status_code}): {res.text}")
    data = res.json()
    return [
        {"title": item.get("title"), "url": item.get("link"), "snippet": item.get("snippet", "")}
        for item in data.get("items", [])[:limit]
    ]


def web_search(query: str, engine: str, limit: int = 5):
    if engine == "google":
        return google_search(query, limit)
    return ddgs_search(query, limit)
