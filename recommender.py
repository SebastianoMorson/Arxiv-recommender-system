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

#df = load_data('/Users/enricotazzer/Desktop/hackathon/data/arxiv_dataset.csv', '/Users/enricotazzer/Desktop/hackathon/data/arxiv_specter_embeddings.csv') # Replace with actual file paths

def user_interest_filtering(user_interests):
    df_user = df[df['categories'].apply(lambda x: isinstance(x, str) and any(cat in x.split() for cat in user_interests))]
    return df_user

def random_not_user_interests(df, user_interests, n=20):
    mask = df['categories'].apply(
        lambda x: isinstance(x, str) and not any(cat in x.split() for cat in user_interests)
    )
    df_not_interests = df[mask]
    return df_not_interests.sample(n=min(n, len(df_not_interests)), random_state=42)

def extract_samples(user_interests, n_samples=2):
    samples = []
    for cat in user_interests:
        cat_samples = df[df['categories'].str.contains(cat, na=False)].sample(n=min(n_samples, len(df)), random_state=42)
        samples.append(cat_samples)
    return pd.concat(samples)


def knn_recommender(user_interests):
    if not isinstance(user_interests, list): ## concorrenza attendere file registration.json
        raise ValueError("user_interests must be a list of categories.")
    
    df_user = user_interest_filtering(user_interests)
    embedding_cols = [col for col in df_user.columns if col not in ['id', 'categories', 'title']]
    embeddings = df_user[embedding_cols].values
    y = df_user.index
    nn = KNeighborsClassifier(n_neighbors=100, metric='cosine', n_jobs=-1)
    nn.fit(embeddings, y)
    idxs = [np.random.randint(0, len(y)-1) for _ in range(10)]  # Randomly select 10 indices from the user dataset
    mean_embedding = np.mean(embeddings[idxs, :], axis=0)
    idx = random.randint(0, len(y)-1) # Randomly select an index from the user dataset
    dist, index = nn.kneighbors(X=mean_embedding.reshape(1, -1))
    best_reccomendation = index[:70][0]
    worst_reccomendation = index[-10:][0]
    random_recommendation = random_not_user_interests(df_emb, user_interests)
    recommendations = pd.concat([df_user.iloc[best_reccomendation], df_user.iloc[worst_reccomendation], random_recommendation])

