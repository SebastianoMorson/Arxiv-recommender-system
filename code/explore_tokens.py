# ==============================================================================
# AI-based tools for metrics in Education, STEM, and Social Sciences
# Author: Madalin Mamuleanu
# Contact: madalin.mamuleanu@edu.ucv.ro
# Web: https://www.across-alliance.eu/
# ==============================================================================

import pandas as pd
import argparse
import ast
import random

# === Argument parsing ===
parser = argparse.ArgumentParser(description="Explore tokenized text from the preprocessed dataset.")
parser.add_argument("--input", type=str, default="arxiv_tokenized_balanced.csv", help="Path to tokenized dataset (.csv or .parquet)")
parser.add_argument("--n", type=int, default=5, help="Number of samples to display")
args = parser.parse_args()

# === Load tokenized data ===
print(f"📥 Loading tokenized data from {args.input} ...")
if args.input.endswith(".csv"):
    df = pd.read_csv(args.input)
elif args.input.endswith(".parquet"):
    df = pd.read_parquet(args.input)
else:
    raise ValueError("Input file must be a .csv or .parquet")

print(f"✅ Loaded {len(df):,} rows")

# Parse token lists if stored as strings
def parse_tokens(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x

df["input_ids"] = df["input_ids"].apply(parse_tokens)

# === Show random samples ===
samples = df.sample(n=args.n, random_state=random.randint(0, 10000))

print("\n🔍 Sample tokenized papers:\n")
for i, row in samples.iterrows():
    print("────────────────────────────────────────────")
    print(f"📄 ID:       {row.get('id', 'N/A')}")
    print(f"🏷️  Title:    {row.get('title', 'N/A')}")
    print(f"👥 Authors:  {row.get('authors', 'N/A')}")
    print(f"📝 Abstract (truncated):\n{row.get('abstract', '')[:300]}...\n")
    print(f"🔡 First 20 input tokens: {row['input_ids'][:20]}")
    print()

print("✅ Done.")
