# ==============================================================================
# AI-based tools for metrics in Education, STEM, and Social Sciences
# Author: Madalin Mamuleanu
# Contact: madalin.mamuleanu@edu.ucv.ro
# Web: https://www.across-alliance.eu/
# ==============================================================================

import pandas as pd
import argparse
import random

# === Argument parsing ===
parser = argparse.ArgumentParser(description="Explore random papers from the arXiv dataset.")
parser.add_argument("--input", type=str, default="data/arxiv_dataset.csv", help="Path to raw dataset CSV")
parser.add_argument("--n", type=int, default=5, help="Number of random papers to display")
args = parser.parse_args()

# === Load dataset ===
print(f"📥 Loading dataset from {args.input} ...")
df = pd.read_csv(args.input)
print(f"✅ Loaded {len(df):,} rows")

# === Drop rows with missing essentials ===
df.dropna(subset=["title", "abstract"], inplace=True)

# === Show random samples ===
samples = df.sample(n=args.n, random_state=random.randint(0, 10000))

print("\n🔍 Random paper samples:\n")
for i, row in samples.iterrows():
    print("────────────────────────────────────────────")
    print(f"📄 ID:       {row.get('id', 'N/A')}")
    print(f"🏷️  Title:    {row.get('title', 'N/A')}")
    print(f"👥 Authors:  {row.get('authors', 'N/A')}")
    print(f"🏷️  Category: {row.get('categories', 'N/A')}")
    print(f"📝 Abstract:\n{row.get('abstract', 'N/A')[:1000]}...\n")

print("✅ Done.")