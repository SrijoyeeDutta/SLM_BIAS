"""
STEP 1 — Run this first, locally, in your venv. No GPU needed.
Just confirms the data loads correctly and shows you what you're working with.

Run:
    python explore_data.py
"""
import pandas as pd

df = pd.read_csv("data/IndiBias_v1_sample.csv")

print("Columns:", df.columns.tolist())
print("\nTotal rows:", len(df))
print("\nBias type breakdown:")
print(df["bias_type"].value_counts())
print("\nStereo vs anti-stereo breakdown:")
print(df["stereo_antistereo"].value_counts())

print("\n--- Example row ---")
row = df.iloc[0]
print("Stereotype sentence :", row["modified_eng_sent_more"])
print("Anti-stereotype     :", row["modified_eng_sent_less"])
print("Bias axis           :", row["bias_type"])
