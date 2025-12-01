# sanity_groq.py
import os
import requests
import urllib.parse

def sanity_query(project_id, dataset, groq_query, token=""):
    """
    Execute a GROQ query against Sanity HTTP API.
    Returns JSON response or None on fail.
    """
    if not project_id or not dataset or not groq_query:
        return None
    base = f"https://{project_id}.api.sanity.io/v1/data/query/{dataset}"
    q = urllib.parse.quote(groq_query, safe='')
    url = f"{base}?query={q}"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "status_code": getattr(e, 'response', None) and e.response.status_code}
