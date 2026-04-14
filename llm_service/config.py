import requests
import json
import re

LLM_PROVIDER = "groq"  # "groq" or "ollama"

# -- Ollama Config --
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:4b"  # User requested lightweight model

# -- Groq Config --
GROQ_API_KEY = ""
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

def _clean_json_response(content: str) -> str:
    """Removes thinking tags and markdown code blocks to extract pure JSON."""
    content = content.strip()
    
    # Remove <think>...</think> blocks common in qwen and deepseek models
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    
    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) > 1:
            # Skip the first line containing ```
            content = "\n".join(lines[1:])
        
        # Remove trailing fence
        if content.endswith("```"):
            content = content[:-3].strip()
    
    return content.strip()

def call_groq_llm(system_prompt: str, user_prompt: str, expect_json: bool = False, retries: int = 2) -> str | dict:
    """Call Groq Cloud API. This API uses OpenAI compatible format."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3
    }
    
    if expect_json:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(retries + 1):
        try:
            response = requests.post(GROQ_BASE_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            content = response.json()["choices"][0]["message"]["content"]
            
            if not expect_json:
                return content.strip()
                
            cleaned_content = _clean_json_response(content)
            
            try:
                return json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                if attempt == retries:
                    raise ValueError(f"Failed to parse JSON from Groq after {retries + 1} attempts. Content: {content}") from e
        except (requests.RequestException, KeyError) as e:
            if attempt == retries:
                raise e

    raise RuntimeError("Unexpected end of retry loop")

def call_ollama(system_prompt: str, user_prompt: str, expect_json: bool = False, retries: int = 2) -> str | dict:
    """
    Call local Ollama model.
    Includes retry logic for JSON parsing robustness.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }
    
    if expect_json:
        payload["format"] = "json"

    for attempt in range(retries + 1):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            content = response.json()["message"]["content"]
            
            if not expect_json:
                # Strip think tags even for text mode to keep output clean
                return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                
            cleaned_content = _clean_json_response(content)
            
            try:
                return json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                if attempt == retries:
                    raise ValueError(f"Failed to parse JSON from LLM after {retries + 1} attempts. Content: {content}") from e
                # Will retry
        except (requests.RequestException, KeyError) as e:
            if attempt == retries:
                raise e
                
    raise RuntimeError("Unexpected end of retry loop")

def call_llm(system_prompt: str, user_prompt: str, expect_json: bool = False, retries: int = 2) -> str | dict:
    """
    Generic wrapper to dynamically call the chosen LLM based on LLM_PROVIDER.
    """
    if LLM_PROVIDER == "groq":
        return call_groq_llm(system_prompt, user_prompt, expect_json, retries)
    elif LLM_PROVIDER == "ollama":
        return call_ollama(system_prompt, user_prompt, expect_json, retries)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
