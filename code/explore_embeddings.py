# ==============================================================================
# AI-based tools for metrics in Education, STEM, and Social Sciences
# Author: Madalin Mamuleanu
# Contact: madalin.mamuleanu@edu.ucv.ro
# Web: https://www.across-alliance.eu/
# ==============================================================================

import numpy as np
import pandas as pd
import argparse
import random
import os

# === Argument parsing ===
parser = argparse.ArgumentParser(description="Explore precomputed paper embeddings.")
parser.add_argument("--model", type=str, default="bert", choices=["bert", "minilm", "scibert", "specter"],
                    help="Model used for embeddings")
parser.add_argument("--base_path", type=str, default="data/", help="Base directory for embedding files")
parser.add_argument("--n", type=int, default=5, help="Number of random samples to inspect")
args = parser.parse_args()

# === Resolve filenames ===
embedding_file = os.path.join(args.base_path, f"arxiv_{args.model}_embeddings.npy")
metadata_file = os.path.join(args.base_path, f"arxiv_{args.model}_embeddings.csv")

# === Load embeddings and metadata ===
print(f"📥 Loading embeddings from {embedding_file}")
embeddings = np.load(embedding_file)
print(f"✅ Embeddings loaded: shape = {embeddings.shape}")

print(f"📥 Loading metadata from {metadata_file}")
df = pd.read_csv(metadata_file)
print(f"✅ Metadata loaded: {len(df)} rows")

# === Show random samples ===
print(f"\n🔍 Showing {args.n} random samples from '{args.model}':")
indices = random.sample(range(len(df)), args.n)

for idx in indices:
    print("────────────────────────────────────────────")
    print(f"📄 ID:       {df.iloc[idx].get('id', 'N/A')}")
    print(f"🏷️  Title:    {df.iloc[idx].get('title', 'N/A')[:100]}")
    print(f"📊 Embedding preview: {embeddings[idx][:8]}...")  # show first 8 dims
    print()

print("✅ Done.")