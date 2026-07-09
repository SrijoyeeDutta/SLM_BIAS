"""
STEP 2 — Run this once, locally, in your venv. No GPU needed
(all-MiniLM-L6-v2 is a small model that runs fine on CPU).

This turns the IndiBias stereotype sentences into vectors so that later,
when a user types any new sentence, we can measure how "close" it is
to a known Indian-context stereotype.

Run:
    python build_embeddings.py

Produces:
    stereo_embeddings.npy   -> the vectors
    stereo_lookup.csv       -> the matching sentence + bias_type for each vector
"""
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

print("Loading dataset...")
df = pd.read_csv("data/IndiBias_v1_sample.csv")

# We only embed the stereotype side (modified_eng_sent_more) since that's
# what we want to detect resemblance to. Drop any blank rows just in case.
stereo_df = df[["modified_eng_sent_more", "bias_type"]].dropna()
stereo_df = stereo_df.rename(columns={"modified_eng_sent_more": "sentence"})

print(f"Embedding {len(stereo_df)} stereotype sentences...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # ~80MB, free, CPU-friendly
embeddings = model.encode(stereo_df["sentence"].tolist(), show_progress_bar=True)

np.save("stereo_embeddings.npy", embeddings)
stereo_df.to_csv("stereo_lookup.csv", index=False)

print("\nDone.")
print("Saved stereo_embeddings.npy with shape:", embeddings.shape)
print("Saved stereo_lookup.csv with", len(stereo_df), "rows")
