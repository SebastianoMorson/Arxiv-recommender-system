import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import random
import numpy as np

macro_categories = {
    "cs": "Computer Science",
    "econ": "Economics",
    "eess": "Electrical Engineering and Systems Science",
    "math": "Mathematics",
    "astro-ph": "Astrophysics",
    "cond-mat": "Condensed Matter",
    "gr-qc": "General Relativity and Quantum Cosmology",
    "hep-ex": "High Energy Physics - Experiment",
    "hep-lat": "High Energy Physics - Lattice",
    "hep-ph": "High Energy Physics - Phenomenology",
    "hep-th": "High Energy Physics - Theory",
    "math-ph": "Mathematical Physics",
    "nlin": "Nonlinear Sciences",
    "nucl-ex": "Nuclear Experiment",
    "nucl-th": "Nuclear Theory",
    "physics": "Physics",
    "quant-ph": "Quantum Physics",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "stat": "Statistics"
}

def load_data(ds_file_path, embeddings_file_path):
    # load the dataset from a CSV file
    print('Loading dataset...')
    df_dataset = pd.read_csv(ds_file_path, usecols=['id', 'categories']) 
    df_emb = pd.read_csv(embeddings_file_path)
    df_emb = df_emb.merge(df_dataset[['id', 'categories']], on='id', how='left')
    print('Dataset loaded successfully.')
    return df_emb

df = load_data('/Users/enricotazzer/Desktop/hackathon/data/arxiv_dataset.csv', '/Users/enricotazzer/Desktop/hackathon/data/arxiv_specter_embeddings.csv') # Replace with actual file paths

def user_interest_filtering(user_interests):
    df_user = df[df['categories'].apply(lambda x: isinstance(x, str) and any(cat in x.split() for cat in user_interests))]
    return df_user


def extract_samples(user_interests, n_samples=2):
    samples = []
    for cat in user_interests:
        cat_samples = df[df['categories'].str.contains(cat, na=False)].sample(n=min(n_samples, len(df)), random_state=42)
        samples.append(cat_samples)
    return pd.concat(samples)


def knn_recommender(user_interests):
    df_user = user_interest_filtering(user_interests)
    guess_samples = extract_samples(user_interests, n_samples=2)
    guess_samples = guess_samples[['id', 'categories']].drop_duplicates()
    embedding_cols = [col for col in df_user.columns if col not in ['id', 'categories', 'title']]
    embeddings = df_user[embedding_cols].values
    y = df_user.index

    nn = KNeighborsClassifier(n_neighbors=20, metric='cosine', n_jobs=-1)
    nn.fit(embeddings, y)

    for _ in range(5):
        
        idx = random.randint(0, len(y)-1)
        sample_title = df_user.iloc[idx]["id"] if "title" not in df_user.columns else df_user.iloc[idx]["title"]
        dist, index = nn.kneighbors(X=embeddings[idx, :].reshape(1, -1))
        print(f"Sample:\n{sample_title}\n")
        for i in range(1, 20):
            rec_idx = index[0][i]
            rec_title = df_user.iloc[rec_idx]["id"] if "title" not in df_user.columns else df_user.iloc[rec_idx]["title"]
            print(f"Recommendation {i}:\n{rec_title}\n")
        print("===============\n")


