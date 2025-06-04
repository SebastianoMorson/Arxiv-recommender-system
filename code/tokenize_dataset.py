# ==============================================================================
# AI-based tools for metrics in Education, STEM, and Social Sciences
# Author: Madalin Mamuleanu
# Contact: madalin.mamuleanu@edu.ucv.ro
# Web: https://www.across-alliance.eu/
# ==============================================================================

import pandas as pd
import time
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import os

# === Argument parsing ===
parser = argparse.ArgumentParser(description="Tokenize a balanced subset of the arXiv dataset.")
parser.add_argument("--input", type=str, default="data/arxiv_dataset.csv", help="Path to full raw arXiv CSV")
parser.add_argument("--output_prefix", type=str, default="data/arxiv_tokenized_balanced", help="Output file prefix (without extension)")
parser.add_argument("--target_size", type=int, default=200_000, help="Total number of papers to sample")
parser.add_argument("--tokenizer_model", type=str, default="bert-base-uncased", help="Tokenizer model name")
args = parser.parse_args()

# === Step 1: Load and Balance the Dataset ===
print("📥 Loading dataset...")
df = pd.read_csv(args.input)
print("📦 Loaded:", len(df), "rows")

df.dropna(subset=["title", "authors", "abstract", "categories"], inplace=True)
df["main_category"] = df["categories"].apply(lambda cat: cat.split()[0].split('.')[0])
domain_counts = df["main_category"].value_counts()
print("📊 Categories:\n", domain_counts)

n_per_cat = args.target_size // len(domain_counts)
balanced_df = pd.concat([
    df[df["main_category"] == cat].sample(min(n_per_cat, len(df[df["main_category"] == cat])), random_state=42)
    for cat in domain_counts.index
], ignore_index=True)

print(f"✅ Balanced dataset with {len(balanced_df)} papers across {len(domain_counts)} categories")

# Save raw balanced subset
subset_csv = args.output_prefix.replace("tokenized", "subset") + ".csv"
os.makedirs(os.path.dirname(subset_csv), exist_ok=True)
balanced_df.to_csv(subset_csv, index=False)
print(f"💾 Raw balanced subset saved: {subset_csv}")

# === Step 2: Tokenization ===
balanced_df.fillna("", inplace=True)
balanced_df["combined_text"] = balanced_df.apply(
    lambda row: f"{row['title']} {row['authors']} {row['abstract']}", axis=1
)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)

def tokenize_text(text):
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None
    )
    return encoded["input_ids"], encoded["attention_mask"]

tqdm.pandas()
start_time = time.time()

tokenized = balanced_df["combined_text"].progress_apply(tokenize_text)
balanced_df["input_ids"] = tokenized.apply(lambda x: x[0])
balanced_df["attention_mask"] = tokenized.apply(lambda x: x[1])

elapsed = time.time() - start_time
print(f"⏱️ Tokenization completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")

# === Step 3: Save Output ===
balanced_df["id"] = balanced_df["id"].astype(str)

csv_path = f"{args.output_prefix}.csv"
parquet_path = f"{args.output_prefix}.parquet"

balanced_df.to_csv(csv_path, index=False)
balanced_df.to_parquet(parquet_path, index=False)

print(f"✅ Tokenized CSV saved: {csv_path}")
print(f"✅ Tokenized Parquet saved: {parquet_path}")