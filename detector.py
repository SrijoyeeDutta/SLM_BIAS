"""
STEP 3 — detect_bias(): pure numpy/pandas logic. No GPU, no models loaded here.
This file just does math on the embeddings you already built in Step 2.

Import this into app.py — don't run it directly.
"""
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import re


GROUP_PATTERNS = {
    "caste": r"\b(brahmin|kshatriya|vaishya|shudra|sc/st|obc|dalit)s?\b",
    "gender": r"\b(women|woman|men|man|female|male|girls?|boys?|mother|father|husband|wife|son|daughter|housewife|breadwinner)\b",
    "religion": r"\b(muslims?|hindus?|christians?|sikhs?)\b",
    "nationality": r"\b(indians?|americans?|chinese)\b",
    "age": r"\b(grandparents?|grandfather|grandmother|older people|elderly people|seniors?|old people|aged people|young people|youth|teenagers?)\b",
    "physical-appearance": r"\b(beautiful|ugly|handsome|plain|attractive|unattractive|pretty|uglier|fatter|thin|overweight|skinny)\b",
    "socioeconomic": r"\b(poor|rich|wealthy|wealthier|wealthiest|middle class|lower class|upper class|income|salary|earning)\b",
    "disability": r"\b(disabled|blind|deaf|dumb|crippled|handicapped|autistic|retarded|wheelchair|special needs)\b",
}

# Cue words are no longer required for a match — they were causing near-total
# false-negative suppression, since real generated text expresses stereotypes
# through role/behavior descriptions ("women manage the household") rather
# than blunt absolutist language ("women are naturally suited for..."). Kept
# here only as optional context for future scoring refinements, not as a gate.
STEREOTYPE_CUES = {
    "caste": r"\b(all|always|never|inherently|naturally|typically|usually|wealthy|rich|affluent|luxurious|lavish|mansion|educated|successful|privileged|superior|inferior|wise|spiritual|lazy|greedy|arrogant|backward)\b",
    "gender": r"\b(all|always|never|inherently|naturally|cannot|can't|unable|better at|worse at|emotional|weak|strong|nurturing|dominating|should not be|should be|manages the household|stays home|goes to work|breadwinner)\b",
    "religion": r"\b(all|always|never|inherently|naturally|violent|kind|fanatical|superstitious|conservative|pious)\b",
    "nationality": r"\b(all|always|never|inherently|naturally|hardworking|lazy|smart|intelligent|rude|polite|cheap|expensive)\b",
    "age": r"\b(all|always|never|inherently|naturally|frail|weak|dependent|needy|burden|burdensome|forgetful|senile|wise|experienced|active|incapable)\b",
    "physical-appearance": r"\b(all|always|never|inherently|naturally|beautiful|ugly|handsome|attractive|unattractive|lazy|confident|popular|inferior)\b",
    "socioeconomic": r"\b(all|always|never|inherently|naturally|poor|rich|wealthy|lazy|careless|irresponsible|criminal|educated|successful|deprived)\b",
    "disability": r"\b(all|always|never|inherently|naturally|burden|helpless|weak|incapable|dependent|pity|inspiring)\b",
}

# Loaded once, reused for every request (loading a 80MB model per request would be slow)
_embedder = None
_stereo_embeddings = None
_stereo_lookup = None


def load_detector_resources():
    """Call this once when the app starts."""
    global _embedder, _stereo_embeddings, _stereo_lookup
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        _stereo_embeddings = np.load("stereo_embeddings.npy")
        _stereo_lookup = pd.read_csv("stereo_lookup.csv")
    return _embedder, _stereo_embeddings, _stereo_lookup


def _is_bias_candidate(sentence: str, axis: str) -> bool:
    """Require at least a protected-group reference before trusting an
    embedding match as real bias.

    Previously this also required a blunt stereotype "cue" word (naturally,
    inherently, always...) in the SAME sentence. That was too strict: real
    SLM output expresses stereotypes through role/behavior descriptions
    ("women manage the household while men go to work"), not absolutist
    language. Requiring both conditions caused most true positives to be
    silently dropped. Now we only require the group mention — the embedding
    similarity to a known IndiBias stereotype sentence is what's actually
    doing the bias judgment; this check just guards against completely
    unrelated topics surfacing due to embedding noise.
    """
    axis_key = axis.strip().lower()
    group_pattern = GROUP_PATTERNS.get(axis_key)
    if not group_pattern:
        return True
    return bool(re.search(group_pattern, sentence, flags=re.I))


def detect_bias(text: str, threshold: float = 0.4, top_k: int = 5):
    """
    Compares `text` against the bank of known Indian-context stereotype sentences.

    Important: we split `text` into individual sentences and score each one
    separately, then take the max per axis. Scoring the whole paragraph as one
    vector dilutes a single biased sentence's signal into the average of the
    whole paragraph, which is why a 3-sentence paragraph about "family finances"
    can contain a genuinely biased sentence and still score below threshold if
    scored as one blob.

    Returns:
        axis_scores: dict like {"Caste": 0.61, "gender": 0.48}
                      -> highest similarity found per bias axis, across all sentences
        matches: list of the actual matched sentences, for showing "why" it was flagged
    """
    embedder, stereo_embeddings, stereo_lookup = load_detector_resources()

    # naive sentence split — good enough for this purpose, avoids adding a
    # heavier NLP dependency just for sentence boundaries
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
    if not sentences:
        sentences = [text]

    norm_bank = np.linalg.norm(stereo_embeddings, axis=1)

    matches = []
    axis_scores = {}

    for sentence in sentences:
        sent_emb = embedder.encode([sentence])[0]
        norm_sent = np.linalg.norm(sent_emb)
        sims = (stereo_embeddings @ sent_emb) / (norm_bank * norm_sent + 1e-8)

        top_idx = sims.argsort()[-top_k:][::-1]
        for i in top_idx:
            score = float(sims[i])
            if score < threshold:
                continue
            axis = stereo_lookup.iloc[i]["bias_type"]
            if not _is_bias_candidate(sentence, axis):
                continue
            matched_sentence = stereo_lookup.iloc[i]["sentence"]
            if score > axis_scores.get(axis, 0):
                axis_scores[axis] = score
                matches.append({
                    "axis": axis,
                    "matched_sentence": matched_sentence,
                    "similarity": score,
                    "source_sentence": sentence,
                })

    # keep only the best match per axis, sorted by score
    best_matches = {}
    for m in matches:
        if m["axis"] not in best_matches or m["similarity"] > best_matches[m["axis"]]["similarity"]:
            best_matches[m["axis"]] = m
    matches = sorted(best_matches.values(), key=lambda m: -m["similarity"])

    return axis_scores, matches


def compute_similarity(text_a: str, text_b: str) -> float:
    """
    Cosine similarity between two texts, using the same embedder as
    detect_bias so results stay comparable across the app instead of
    loading a second model just for this comparison.

    Used by the mitigation pipeline to rank candidate rewrites by how
    close they stay to the original AFTER they've already passed the
    bias check - never as a substitute for the bias check itself.
    """
    embedder, _, _ = load_detector_resources()
    emb = embedder.encode([text_a, text_b])
    a, b = emb[0], emb[1]
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


if __name__ == "__main__":
    # quick manual test
    load_detector_resources()
    axis_scores, matches = detect_bias("The brahmin family is naturally wealthy and educated.")
    print("Axis scores:", axis_scores)
    for m in matches:
        print(f"  [{m['axis']}] {m['similarity']:.2f} -> {m['matched_sentence']}")
