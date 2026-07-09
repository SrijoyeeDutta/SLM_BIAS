"""
STEP 5 — mitigate(): also just an API call, no models loaded locally.
Sends the biased text + which axes were flagged to one SLM via the
free Hugging Face Inference API, and asks it to rewrite neutrally.
"""
import requests
import os
import re

import streamlit as st
try:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ROUTER_URL = "https://api.groq.com/openai/v1/chat/completions"
MITIGATOR_MODEL_ID = "llama-3.1-8b-instant"


def strip_think_tags(text: str) -> str:
    """Remove hidden reasoning tags some chat models may emit."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def mitigate(
    text: str,
    flagged_axes: list,
    bias_reasons: list | None = None,
    temperature: float = 0.3,
    model_id: str | None = None,
    *_unused_args,
    **_unused_kwargs,
) -> str:
    axes_str = ", ".join(flagged_axes) if flagged_axes else "stereotypes"
    reasons_str = "\n".join(f"- {reason}" for reason in bias_reasons) if bias_reasons else "- No specific example provided."
    prompt = (
        f"Rewrite the following sentence to remove any stereotypes related to "
        f"{axes_str}, while keeping the same meaning and a natural tone. "
        f"Preserve factual demographic terms when the sentence is describing statistics or comparisons, "
        f"such as men, women, male, female, people, or workers. Do not replace them with awkward phrases "
        f"like 'people identifying as...'. Remove unfair generalizations, not the factual group labels. "
        f"Do not introduce new protected-attribute language such as age, religion, caste, or region unless it is already present in the original text. "
        f"Use these bias reasons as guidance:\n{reasons_str}\n\n"
        f"Only output the rewritten sentence, nothing else.\n\n"
        f"Original: {text}"
    )

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model_id or MITIGATOR_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": temperature,
    }

    try:
        response = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        rewritten = strip_think_tags(result["choices"][0]["message"]["content"])
        return rewritten if rewritten else "(mitigation failed — try again)"
    except requests.exceptions.HTTPError as e:
        return f"(mitigation model unavailable: {e.response.status_code})"
    except Exception as e:
        return f"(mitigation error: {e})"


if __name__ == "__main__":
    out = mitigate("The brahmin family is naturally wealthy and educated.", ["Caste"])
    print("Rewritten:", out)
