**Live Demo : slmbias-cwrqffbwkqpker7muuyr3d.streamlit.app
<img width="730" height="69" alt="image" src="https://github.com/user-attachments/assets/6ccb2ed3-e3b8-4510-9202-26b1424e8c8c" />
**
# Indian-context Bias Detector & Mitigator

Detects and mitigates social bias (caste, gender, religion, age, appearance,
socioeconomic) in outputs from small language models, grounded in the
IndiBias dataset.

## Setup (all local, no GPU needed)

```bash
python -m venv bias-env
source bias-env/bin/activate      # Windows: bias-env\Scripts\activate
pip install -r requirements.txt
```

Get a free Groq API key (no credit card required): https://console.groq.com/keys
(Groq's free tier reliably hosts open chat models like Llama 3.1 8B and Gemma2 9B —
Hugging Face's free tier mostly dropped chat-model hosting in 2025/2026, so we use
Groq for actual text generation instead.)

```bash
export GROQ_API_KEY=your_key_here    # Windows: set GROQ_API_KEY=your_key_here
```

## Run, in this exact order

```bash
python explore_data.py       # Step 1 — sanity check the dataset loads correctly
python build_embeddings.py   # Step 2 — builds stereo_embeddings.npy + stereo_lookup.csv (run once)
python detector.py           # Step 3 — quick manual test of detect_bias()
python mitigator.py          # Step 5 — quick manual test of mitigate() (needs HF_TOKEN)
streamlit run app.py         # Step 6 — launches the actual app in your browser
```

Steps 1–3 need no internet after the first model download. Steps 5–6 need
internet + HF_TOKEN because they call the Hugging Face Inference API.

## Deploying

1. Create a new Space at huggingface.co/spaces (SDK: Streamlit)
2. Push this whole folder (except `bias-env/`)
3. Add `GROQ_API_KEY` as a secret in the Space settings
4. Space builds automatically and gives you a public URL

## Files

| File | What it does | Needs GPU? | Needs internet? |
|---|---|---|---|
| `explore_data.py` | Loads and inspects the CSV | No | No |
| `build_embeddings.py` | Builds the stereotype-sentence vector bank | No | Yes (first run only, to download the small embedding model) |
| `detector.py` | `detect_bias()` — scores any text against the vector bank | No | No (after first run) |
| `mitigator.py` | `mitigate()` — calls Groq's API to rewrite biased text | No | Yes (API call) |
 | `app.py` | Streamlit frontend tying it all together (calls Groq for the 3 models, runs a bias stress test, explains flagged evidence, and mitigates the most biased output when needed) | No | Yes (API calls) |
