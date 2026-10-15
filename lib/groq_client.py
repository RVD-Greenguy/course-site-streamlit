"""Direct port of server/src/groq.js — thin wrapper around Groq's OpenAI-
compatible chat completions endpoint."""
import os
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def groq_chat_completion(model: str, messages: list, tools=None, tool_choice=None):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set on the server.")

    body = {"model": model, "messages": messages}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice or "auto"

    res = requests.post(
        GROQ_URL,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=30,
    )
    if not res.ok:
        raise RuntimeError(f"Groq request failed ({res.status_code}): {res.text}")
    return res.json()
