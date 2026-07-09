"""
Three-stage bias mitigation pipeline.

Stage A - rule-based lexical substitution (fast, cheap, scoped to known
           stereotype phrases per category). If this alone clears the bias
           threshold, we stop here and never call the AI model.

Stage B - AI rewrite via mitigator.mitigate(), called with a few different
           framings/temperatures to produce several candidates instead of
           trusting a single one-shot rewrite. Only runs if Stage A wasn't
           enough.

Stage C - selection: filter candidates down to the ones that ACTUALLY pass
           the bias check, then rank the survivors by embedding similarity
           to the original text. Similarity to the original is never used
           to override safety - a candidate that still fails the bias
           check is disqualified no matter how close it reads to the
           source. Similarity only breaks ties among already-safe
           rewrites, and a minimum floor catches "safe" rewrites that
           drifted too far from the original meaning to trust blindly.

If nothing clears both the bias bar and the similarity floor, the result
is flagged for human review rather than silently returned as-is.
"""
import re
import importlib
import time
from detector import detect_bias, compute_similarity
import mitigator as mitigator_module

SIMILARITY_FLOOR = 0.60  # below this, a "bias-safe" rewrite is treated as unfaithful


# ---------------------------------------------------------------------------
# Stage A: rule-based lexical substitution
# ---------------------------------------------------------------------------
# Deliberately scoped: these catch known stereotype PHRASES per category,
# not general demographic language. Expand this table as your gold set
# surfaces recurring patterns. Anything not covered here safely falls
# through to Stage B rather than being forced into a rigid substitution
# that might mangle a sentence it doesn't understand.
RULE_SUBSTITUTIONS = {
    "Caste": [
        (re.compile(r"\bnaturally (wealthy|educated|successful)\b", re.I), r"often \1"),
        (re.compile(r"\b(brahmin|kshatriya|vaishya|shudra)s? are\b", re.I), r"some \1 families are"),
    ],
    "Gender": [
        (re.compile(r"\bwomen (can'?t|cannot|aren'?t able to)\b", re.I), "some women may find it harder to"),
        (re.compile(r"\bmen are (always|naturally) better at\b", re.I), "men are often assumed to be better at"),
    ],
    "Religion": [
        (re.compile(r"\ball (muslims|hindus|christians|sikhs)\b", re.I), r"some \1"),
    ],
    "Nationality": [
        (re.compile(r"\b(indians|americans|chinese) are all\b", re.I), r"\1 are sometimes stereotyped as"),
    ],
    "Age": [
        (re.compile(r"\b(old|elderly) people can'?t\b", re.I), "some older people may find it harder to"),
    ],
}


def rule_based_rewrite(text: str, flagged_axes: list[str]) -> str:
    """Applies only the substitution rules relevant to the flagged axes.
    Categories with no rule table entries pass the text through unchanged,
    so Stage A never guesses at bias types it wasn't asked to handle."""
    rewritten = text
    for axis in flagged_axes:
        for pattern, replacement in RULE_SUBSTITUTIONS.get(axis, []):
            rewritten = pattern.sub(replacement, rewritten)
    return rewritten


# ---------------------------------------------------------------------------
# Stage B: AI rewrite, multiple candidates
# ---------------------------------------------------------------------------
# (label, temperature, framing hint appended to the mitigation prompt)
CANDIDATE_FRAMINGS = [
    ("minimal edit", 0.2, "Make the smallest possible edit that removes the stereotype - change as few words as you can."),
    ("full rewrite", 0.5, "Rewrite the sentence more freely if needed to fully remove the stereotype, while keeping the same topic."),
    ("balanced", 0.35, "Rewrite to remove the stereotype using natural, everyday phrasing."),
]


def generate_ai_candidates(text: str, flagged_axes: list[str], bias_reasons: list[str] | None = None, model_id: str | None = None) -> list[dict]:
    """Runs mitigator.mitigate() with a few different framings/temperatures
    so Stage C has real alternatives to choose between, instead of relying
    on a single one-shot rewrite that might fail for reasons unrelated to
    the bias itself (an awkward phrasing, an overcorrection, etc.)."""
    importlib.reload(mitigator_module)
    candidates = []
    for label, temperature, framing_hint in CANDIDATE_FRAMINGS:
        extra_reasons = (bias_reasons or []) + [framing_hint]
        text_out = mitigator_module.mitigate(text, flagged_axes, extra_reasons, temperature=temperature, model_id=model_id)
        if text_out and not text_out.startswith("(mitigation") and not text_out.startswith("(error"):
            candidates.append({"label": f"AI rewrite ({label})", "text": text_out})
        time.sleep(1.5)  # Avoid Groq 429 rate limit bursts
    return candidates


# ---------------------------------------------------------------------------
# Stage C: bias-safe filter, then similarity ranking
# ---------------------------------------------------------------------------
def score_axes(axis_scores: dict, axes: list[str]) -> float:
    """Worst remaining flagged axis, not the sum across all flagged axes.

    This threshold (e.g. 0.5) is calibrated for a single cosine-similarity
    value in [0, 1]. Summing scores across 2+ flagged axes could easily
    exceed 1.0 even when EVERY individual axis has been brought down to a
    genuinely safe level — making it nearly impossible for any rewrite to
    ever pass the bias check once more than one axis was originally
    flagged. Taking the max keeps the comparison meaningful: "is the worst
    remaining axis still above the danger threshold?"
    """
    return float(max((axis_scores.get(axis, 0.0) for axis in axes), default=0.0))


def select_best_candidate(original_text: str, candidates: list[dict], flagged_axes: list[str], threshold: float):
    """Filters candidates to the ones that pass the bias threshold, THEN
    ranks the survivors by similarity to the original. Similarity never
    overrides safety - a candidate that still fails the bias check is
    disqualified regardless of how close it reads to the original text."""
    scored = []
    for c in candidates:
        axis_scores, _ = detect_bias(c["text"])
        flagged_score = score_axes(axis_scores, flagged_axes)
        similarity = compute_similarity(original_text, c["text"])
        scored.append({
            **c,
            "axis_scores": axis_scores,
            "flagged_score": flagged_score,
            "similarity": similarity,
            "passes_bias_check": flagged_score < threshold,
        })

    safe_candidates = [c for c in scored if c["passes_bias_check"]]

    if not safe_candidates:
        # Nothing cleared the bias bar. Don't guess - surface the
        # least-biased attempt but mark it clearly as unresolved so the
        # caller doesn't treat it as a successful mitigation.
        best_unsafe = min(scored, key=lambda c: c["flagged_score"]) if scored else None
        return best_unsafe, scored, "no_safe_candidate"

    best_safe = max(safe_candidates, key=lambda c: c["similarity"])

    if best_safe["similarity"] < SIMILARITY_FLOOR:
        # It's bias-safe, but it drifted far enough from the original that
        # it may have distorted the meaning rather than fixed the bias -
        # flag it instead of accepting silently.
        return best_safe, scored, "safe_but_low_fidelity"

    return best_safe, scored, "accepted"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def mitigate_pipeline(
    original_text: str,
    flagged_axes: list[str],
    bias_reasons: list[str] | None = None,
    threshold: float = 0.5,
    model_id: str | None = None,
) -> dict:
    """
    Runs the full 3-stage mitigation pipeline.

    Returns a dict:
        text                - the accepted (or best-effort) rewritten text
        status              - "rule_based_cleared" | "accepted" |
                              "safe_but_low_fidelity" | "no_safe_candidate"
        stage_used          - "A" or "B/C"
        original_scores     - axis scores for the original text
        final_scores        - axis scores for the returned text
        candidates          - full scoring detail for every candidate
                              considered, for transparency in the UI
        needs_human_review  - True if nothing cleared both the bias check
                              and the similarity floor
    """
    original_scores, _ = detect_bias(original_text)

    # --- Stage A: rule-based ---
    rule_based_text = rule_based_rewrite(original_text, flagged_axes)
    rule_axis_scores, _ = detect_bias(rule_based_text)
    rule_flagged_score = score_axes(rule_axis_scores, flagged_axes)

    if rule_based_text != original_text and rule_flagged_score < threshold:
        return {
            "text": rule_based_text,
            "status": "rule_based_cleared",
            "stage_used": "A",
            "original_scores": original_scores,
            "final_scores": rule_axis_scores,
            "candidates": [{
                "label": "Rule-based", "text": rule_based_text,
                "flagged_score": rule_flagged_score, "similarity": 1.0,
                "passes_bias_check": True,
            }],
            "needs_human_review": False,
        }

    # --- Stage B: AI multi-candidate rewrite ---
    # Stage A's output is carried forward as a baseline candidate even if it
    # didn't clear the bar alone - Stage C may still prefer it if it's an
    # improvement and closer to the original than the AI rewrites.
    candidates = [{"label": "Rule-based (partial)", "text": rule_based_text}]
    candidates += generate_ai_candidates(original_text, flagged_axes, bias_reasons, model_id=model_id)

    # --- Stage C: bias-safe filter, then similarity ranking ---
    best, scored, status = select_best_candidate(original_text, candidates, flagged_axes, threshold)

    if status == "no_safe_candidate":
        # Last resort: one stricter AI attempt before giving up.
        stricter_reasons = (bias_reasons or []) + [
            "Do not introduce new protected-attribute language such as age, religion, caste, or region unless it already appears in the original text.",
            "Remove the stereotype completely, even if it requires restructuring the sentence.",
        ]
        importlib.reload(mitigator_module)
        stricter_text = mitigator_module.mitigate(original_text, flagged_axes, stricter_reasons, temperature=0.2, model_id=model_id)
        if stricter_text and not stricter_text.startswith("(mitigation") and not stricter_text.startswith("(error"):
            stricter_candidates = candidates + [{"label": "AI rewrite (stricter, final attempt)", "text": stricter_text}]
        else:
            stricter_candidates = candidates
        best, scored, status = select_best_candidate(original_text, stricter_candidates, flagged_axes, threshold)

    return {
        "text": best["text"] if best else original_text,
        "status": status,
        "stage_used": "B/C",
        "original_scores": original_scores,
        "final_scores": best["axis_scores"] if best else original_scores,
        "candidates": scored,
        "needs_human_review": status in ("no_safe_candidate", "safe_but_low_fidelity"),
    }


if __name__ == "__main__":
    result = mitigate_pipeline(
        "The brahmin family is naturally wealthy and educated.",
        ["Caste"],
    )
    print("Status:", result["status"], "| Stage used:", result["stage_used"])
    print("Text:", result["text"])
    print("Needs human review:", result["needs_human_review"])
