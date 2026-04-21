# from ddgs import DDGS

# def web_search(query):
#     with DDGS() as ddgs:
#         results = list(ddgs.text(query, max_results=5))

#     if not results:
#         return "No results"

#     # Combine multiple results → reduces randomness
#     texts = [r.get("body", "") for r in results if r.get("body")]

#     combined = " ".join(texts)

#     # simple compression (first 50 words)
#     return " ".join(combined.split()[:100])

# print(web_search("which team is at the bottom of IPL 2026"))


# import requests
# import json

# url = "https://api.langsearch.com/v1/web-search"

# payload = {
#     "query": "which team won IPL 2025 ?",
#     "freshness": "noLimit",
#     "summary": True,
#     "count": 1
# }

# headers = {
#     "Authorization": "Bearer sk-fe26772647934e8096ebab250584067e",
#     "Content-Type": "application/json"
# }

# response = requests.post(url, headers=headers, json=payload)

# data = response.json()

# # Safe extraction
# try:
#     summary = data["data"]["webPages"]["value"][0].get("summary", "")
# except (KeyError, IndexError):
#     summary = ""

# # Reduce to ~50 words
# short_summary = " ".join(summary.split()[:50])

# print(short_summary)



# tavily_search.py

from tavily import TavilyClient
import os
from dotenv import load_dotenv


class TavilySearch:
    def __init__(self, api_key: str = None, max_results: int = 1):
        load_dotenv()

        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found")

        self.client = TavilyClient(api_key=self.api_key)
        self.max_results = max_results

    def search(self, query: str) -> dict:
        try:
            res = self.client.search(query=query, max_results=self.max_results)

            results = res.get("results", [])
            if not results:
                return {"answer": "No results found", "source": None}

            top = results[0]

            return {
                "answer": top.get("content", ""),
                "source": top.get("url", "")
            }

        except Exception as e:
            return {"answer": f"Error: {str(e)}", "source": None}