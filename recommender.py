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
sub_categories = {
    "cs": [
        ("cs.AI", "Artificial Intelligence"),
        ("cs.AR", "Hardware Architecture"),
        ("cs.CC", "Computational Complexity"),
        ("cs.CE", "Computational Engineering, Finance, and Science"),
        ("cs.CG", "Computational Geometry"),
        ("cs.CL", "Computation and Language"),
        ("cs.CR", "Cryptography and Security"),
        ("cs.CV", "Computer Vision and Pattern Recognition"),
        ("cs.CY", "Computers and Society"),
        ("cs.DB", "Databases"),
        ("cs.DC", "Distributed, Parallel, and Cluster Computing"),
        ("cs.DL", "Digital Libraries"),
        ("cs.DM", "Discrete Mathematics"),
        ("cs.DS", "Data Structures and Algorithms"),
        ("cs.ET", "Emerging Technologies"),
        ("cs.FL", "Formal Languages and Automata Theory"),
        ("cs.GL", "General Literature"),
        ("cs.GR", "Graphics"),
        ("cs.GT", "Computer Science and Game Theory"),
        ("cs.HC", "Human-Computer Interaction"),
        ("cs.IR", "Information Retrieval"),
        ("cs.IT", "Information Theory"),
        ("cs.LG", "Machine Learning"),
        ("cs.LO", "Logic in Computer Science"),
        ("cs.MA", "Multiagent Systems"),
        ("cs.MM", "Multimedia"),
        ("cs.MS", "Mathematical Software"),
        ("cs.NA", "Numerical Analysis"),
        ("cs.NE", "Neural and Evolutionary Computing"),
        ("cs.NI", "Networking and Internet Architecture"),
        ("cs.OH", "Other Computer Science"),
        ("cs.OS", "Operating Systems"),
        ("cs.PF", "Performance"),
        ("cs.PL", "Programming Languages"),
        ("cs.RO", "Robotics"),
        ("cs.SC", "Symbolic Computation"),
        ("cs.SE", "Software Engineering"),
        ("cs.SI", "Social and Information Networks"),
        ("cs.SY", "Systems and Control"),
    ],
    "econ": [
        ("econ.EM", "Econometrics"),
        ("econ.GN", "General Economics"),
        ("econ.TH", "Theoretical Economics"),
    ],
    "eess": [
        ("eess.AS", "Audio and Speech Processing"),
        ("eess.IV", "Image and Video Processing"),
        ("eess.SP", "Signal Processing"),
        ("eess.SY", "Systems and Control"),
    ],
    "math": [
        ("math.AC", "Commutative Algebra"),
        ("math.AG", "Algebraic Geometry"),
        ("math.AP", "Analysis of PDEs"),
        ("math.AT", "Algebraic Topology"),
        ("math.CA", "Classical Analysis and ODEs"),
        ("math.CO", "Combinatorics"),
        ("math.CT", "Category Theory"),
        ("math.CV", "Complex Variables"),
        ("math.DG", "Differential Geometry"),
        ("math.DS", "Dynamical Systems"),
        ("math.FA", "Functional Analysis"),
        ("math.GM", "General Mathematics"),
        ("math.GN", "General Topology"),
        ("math.GR", "Group Theory"),
        ("math.GT", "Geometric Topology"),
        ("math.HO", "History and Overview"),
        ("math.IT", "Information Theory"),
        ("math.KT", "K-Theory and Homology"),
        ("math.LO", "Logic"),
        ("math.MG", "Metric Geometry"),
        ("math.MP", "Mathematical Physics"),
        ("math.NA", "Numerical Analysis"),
        ("math.NT", "Number Theory"),
        ("math.OA", "Operator Algebras"),
        ("math.OC", "Optimization and Control"),
        ("math.PR", "Probability"),
        ("math.QA", "Quantum Algebra"),
        ("math.RA", "Rings and Algebras"),
        ("math.RT", "Representation Theory"),
        ("math.SG", "Symplectic Geometry"),
        ("math.SP", "Spectral Theory"),
        ("math.ST", "Statistics Theory"),
    ],
    "astro-ph": [
        ("astro-ph.CO", "Cosmology and Nongalactic Astrophysics"),
        ("astro-ph.EP", "Earth and Planetary Astrophysics"),
        ("astro-ph.GA", "Astrophysics of Galaxies"),
        ("astro-ph.HE", "High Energy Astrophysical Phenomena"),
        ("astro-ph.IM", "Instrumentation and Methods for Astrophysics"),
        ("astro-ph.SR", "Solar and Stellar Astrophysics"),
    ],
    "cond-mat": [
        ("cond-mat.dis-nn", "Disordered Systems and Neural Networks"),
        ("cond-mat.mes-hall", "Mesoscale and Nanoscale Physics"),
        ("cond-mat.mtrl-sci", "Materials Science"),
        ("cond-mat.other", "Other Condensed Matter"),
        ("cond-mat.quant-gas", "Quantum Gases"),
        ("cond-mat.soft", "Soft Condensed Matter"),
        ("cond-mat.stat-mech", "Statistical Mechanics"),
        ("cond-mat.str-el", "Strongly Correlated Electrons"),
        ("cond-mat.supr-con", "Superconductivity"),
    ],
    "gr-qc": [
        ("gr-qc", "General Relativity and Quantum Cosmology"),
    ],
    "hep-ex": [
        ("hep-ex", "High Energy Physics - Experiment"),
    ],
    "hep-lat": [
        ("hep-lat", "High Energy Physics - Lattice"),
    ],
    "hep-ph": [
        ("hep-ph", "High Energy Physics - Phenomenology"),
    ],
    "hep-th": [
        ("hep-th", "High Energy Physics - Theory"),
    ],
    "math-ph": [
        ("math-ph", "Mathematical Physics"),
    ],
    "nlin": [
        ("nlin.AO", "Adaptation and Self-Organizing Systems"),
        ("nlin.CD", "Chaotic Dynamics"),
        ("nlin.CG", "Cellular Automata and Lattice Gases"),
        ("nlin.PS", "Pattern Formation and Solitons"),
        ("nlin.SI", "Exactly Solvable and Integrable Systems"),
    ],
    "nucl-ex": [
        ("nucl-ex", "Nuclear Experiment"),
    ],
    "nucl-th": [
        ("nucl-th", "Nuclear Theory"),
    ],
    "physics": [
        ("physics.acc-ph", "Accelerator Physics"),
        ("physics.app-ph", "Applied Physics"),
        ("physics.atm-clus", "Atomic and Molecular Clusters"),
        ("physics.atom-ph", "Atomic Physics"),
        ("physics.bio-ph", "Biological Physics"),
        ("physics.chem-ph", "Chemical Physics"),
        ("physics.class-ph", "Classical Physics"),
        ("physics.comp-ph", "Computational Physics"),
        ("physics.data-an", "Data Analysis, Statistics and Probability"),
        ("physics.ed-ph", "Physics Education"),
        ("physics.flu-dyn", "Fluid Dynamics"),
        ("physics.gen-ph", "General Physics"),
        ("physics.geo-ph", "Geophysics"),
        ("physics.hist-ph", "History and Philosophy of Physics"),
        ("physics.ins-det", "Instrumentation and Detectors"),
        ("physics.med-ph", "Medical Physics"),
        ("physics.optics", "Optics"),
        ("physics.plasm-ph", "Plasma Physics"),
        ("physics.pop-ph", "Popular Physics"),
        ("physics.soc-ph", "Physics and Society"),
        ("physics.space-ph", "Space Physics"),
    ],
    "quant-ph": [
        ("quant-ph", "Quantum Physics"),
    ],
    "q-bio": [
        ("q-bio.BM", "Biomolecules"),
        ("q-bio.CB", "Cell Behavior"),
        ("q-bio.GN", "Genomics"),
        ("q-bio.MN", "Molecular Networks"),
        ("q-bio.NC", "Neurons and Cognition"),
        ("q-bio.OT", "Other Quantitative Biology"),
        ("q-bio.PE", "Populations and Evolution"),
        ("q-bio.QM", "Quantitative Methods"),
        ("q-bio.SC", "Subcellular Processes"),
        ("q-bio.TO", "Tissues and Organs"),
    ],
    "q-fin": [
        ("q-fin.CP", "Computational Finance"),
        ("q-fin.EC", "Economics"),
        ("q-fin.GN", "General Finance"),
        ("q-fin.MF", "Mathematical Finance"),
        ("q-fin.PM", "Portfolio Management"),
        ("q-fin.PR", "Pricing of Securities"),
        ("q-fin.RM", "Risk Management"),
        ("q-fin.ST", "Statistical Finance"),
        ("q-fin.TR", "Trading and Market Microstructure"),
    ],
    "stat": [
        ("stat.AP", "Applications"),
        ("stat.CO", "Computation"),
        ("stat.ME", "Methodology"),
        ("stat.ML", "Machine Learning"),
        ("stat.OT", "Other Statistics"),
        ("stat.TH", "Statistics Theory"),
    ]
}


def load_data(ds_file_path, embeddings_file_path):
    # load the dataset from a CSV file
    print('Loading dataset...')
    df_complete = pd.read_csv(ds_file_path, nrows=500000, low_memory=False)
    
    df_emb = pd.read_csv(embeddings_file_path, low_memory=False)
    # Assicurati che la colonna 'id' sia presente in entrambi i DataFrame e sia dello stesso tipo
    df_complete['id'] = df_complete['id'].astype(str)
    df_emb['id'] = df_emb['id'].astype(str)
    
    # genero un df più piccolo a partire dal df_complete
    df_ids_cats = df_complete[['id', 'categories']]
    
    # Unisci solo le colonne necessarie da df_dataset
    df_emb = df_emb.merge(df_ids_cats[['id', 'categories']], on='id', how='left')
    print('Dataset loaded successfully.')
    
    #ritorno un dizionario con i DataFrame e gli ID del dataset 
    return {'df_emb': df_emb, 'df_complete': df_complete, 'dataset_ids': df_ids_cats}


def load_data_with_filtering(ds_file_path, embeddings_file_path):
    """
    Si comporta come la funzione load_data, ma carica i dati dal dataset embedding_file_path, seleziona gli indici disponibili e 
    carica i dati dal dataset ds_file_path, selezionando solo gli indici disponibili.
    """
    global df, df_ids_cats, df_emb
    data = load_data(ds_file_path, embeddings_file_path)
    df_emb = data['df_emb']
    df = data['df_complete']
    df_ids_cats = data['dataset_ids']
    
    # filtro il dataset in base agli ID disponibili nel dataset di embedding
    df_ids_cats = df_ids_cats[df_ids_cats['id'].isin(df_emb['id'])]
    
    # filtro il dataset in base agli ID disponibili nel dataset di embedding
    df_emb = df_emb[df_emb['id'].isin(df_ids_cats['id'])]
    
    return {'df_emb': df_emb, 'df_complete': df, 'dataset_ids': df_ids_cats}


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


def check_macro(df, macro_category):
    """
    Controlla se le categorie contengono elementi della classe macro_categories
    I codici delle categorie sono stringhe del tipo 'cs.AI', 'econ.EM', etc. in cui la parte prima del punto è la macro categoria e la parte dopo il punto è la sotto categoria.
    """
    if not isinstance(macro_category, str):
        raise ValueError("macro_category deve essere una stringa.")
    if macro_category not in macro_categories:
        raise ValueError(f"{macro_category} non è una macro categoria valida.")
    
    # estraggo le categorie che contengono la macro categoria
    mask = df['categories'].apply(lambda x: isinstance(x, str) and any(cat.startswith(macro_category + '.') for cat in x.split()))
    return df[mask]

    
def extract_samples(macro_categories=None, sub_categories=None,nsamples=50, p_subcat=60, p_randcat=40):
    """
    extract_samples è il metodo che fornisce gli articoli da valutare durante la fase
    di registrazione.
    - sub_categories è la lista dei codici delle sottocategorie preferite dall'utente
    - macro_categories è la lista dei codici delle categorie preferite dall'utente
    - p_subcat è la percentuale di articoli da selezionare e presi dalle sottocategorie
    - p_randcat è la percentuale di articoli da selezionare e presi random dalle macro categorie
    - nsamples è il numero di articoli che verranno ritornati
    Il metodo restituisce una lista di nsamples articoli così composta:
    - p_subcat% di articoli presi dalle sottocategorie
    - p_randcat% di articoli presi dalle macro categorie
    - il resto degli articoli è preso random dal dataset

    Il metodo restituisce un DataFrame con tutte le informazioni degli articoli selezionati contenute nel dataset df 
    df_ids_cat è un DataFrame che contiene gli ID degli articoli e la lista delle categorie associate a ciascun articolo
    df è il DataFrame che contiene tutte le informazioni degli articoli, tra cui l'ID, il titolo, le categorie, l'abstract e le informazioni di pubblicazione.
    """
    global df, df_ids_cats

    print("Inizio estrazione campioni...")
    print("Macro categorie selezionate:", macro_categories)
    print("Sottocategorie selezionate:", sub_categories)


    # estraggo gli indici degli articoli appartenenti a una sottocategoria
    if sub_categories:
        mask_sub = df_ids_cats['categories'].apply(
            lambda x: isinstance(x, str) and any(cat in x.split() for cat in sub_categories)
        )
        df_sub = df_ids_cats[mask_sub]
    else:
        print("cazzo")
        #df_sub = pd.DataFrame(columns=df_ids_cats.columns)
    # estraggo gli indici degli articoli appartenenti a una macro categoria
    if macro_categories:
        mask_macro = df_ids_cats['categories'].apply(
            lambda x: isinstance(x, str) and any(cat in x.split() for cat in macro_categories)
        )
        df_macro = df_ids_cats[mask_macro]


    print("DF_MACRO: ", df_macro)
    # calcolo il numero di articoli da selezionare dalle sottocategorie e dalle macro categorie
    n_sub = int(nsamples * p_subcat / 100)
    n_rand = int(nsamples * p_randcat / 100)
    # seleziono gli articoli dalle sottocategorie
    if not df_sub.empty:
        df_sub_sample = df_sub.sample(n=min(n_sub, len(df_sub)), random_state=42)
    else:
        print("cazzo cazzo cazzo cazzo")
        #df_sub_sample = pd.DataFrame(columns=df_ids_cats.columns)
    
    # seleziono gli articoli dalle macro categorie
    if not df_macro.empty:
        df_macro_sample = df_macro.sample(n=min(n_rand, len(df_macro)), random_state=42)
    else:
        print("cazzo cazzo cazzo cazzo cazzo")
        #df_macro_sample = pd.DataFrame(columns=df_ids_cats.columns)
    
    
    # unisco i due DataFrame
    df_samples = pd.concat([df_sub_sample, df_macro_sample], ignore_index=True)
    # se il numero di articoli selezionati è inferiore a nsamples, aggiungo articoli random presi dalle macro categorie
    if df_ids_cats is None or df_ids_cats.empty:
        raise ValueError("df_ids_cats non può essere vuoto. Assicurati di aver caricato il dataset correttamente.")
    if len(df_samples) < nsamples:
        # calcolo il numero di articoli da aggiungere
        n_add = nsamples - len(df_samples)
        # seleziono articoli random dalle macro categorie
        df_rand_sample = df_ids_cats.sample(n=n_add, random_state=42)
        # unisco i due DataFrame
        df_samples = pd.concat([df_samples, df_rand_sample], ignore_index=True)
    
    # se il numero di articoli selezionati è superiore a nsamples, riduco il numero di articoli
    if len(df_samples) > nsamples:
        df_samples = df_samples.sample(n=nsamples, random_state=42)
    # unisco i campi del DataFrame df con quelli di df_ids_cats
    df_samples = df.merge(df_samples, on='id', how='inner')
    # seleziono le colonne che mi interessano
    #df_samples = df_samples[['id', 'title', 'categories', 'abstract', 'authors', 'published']]
    # resetto l'indice
    df_samples.reset_index(drop=True, inplace=True)
    # ritorno il DataFrame con gli articoli selezionati
    return df_samples

"""
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
    dist, index = nn.kneighbors(X=mean_embedding.reshape(1, -1))
    best_reccomendation = index[:70][0]
    worst_reccomendation = index[-10:][0]
    random_recommendation = random_not_user_interests(df, user_interests)
    recommendations = pd.concat([df_user.iloc[best_reccomendation], df_user.iloc[worst_reccomendation], random_recommendation])
    return recommendations.loc[:, ['id', 'title']]

"""

def code_to_description(code):
    # Macro category
    if code in macro_categories:
        return macro_categories[code]
    # Subcategory
    for macro, subs in sub_categories.items():
        for sub_code, desc in subs:
            if code == sub_code:
                return desc
            
def raw2codes(raw_topics=None, raw_subtopics=None):
    # estraggo la lista di codici che sono contenute nella macro categoria
    # i codici sono nella forma "macro.sub"
    if raw_topics is None or raw_subtopics is None:
        raise ValueError("raw_topics e raw_subtopics devono essere specificati.")
    topics_codes = []
    # estraggo i codici delle macro categorie
    for topic in raw_topics:
        found = False
        for macro, desc in macro_categories.items():
            if topic == desc or topic == macro:
                topics_codes.append(macro)
                found = True
                break
        if not found:
            raise ValueError(f"{topic} non è una macro categoria valida.")
    print(f"Macro categories: {topics_codes}")
    
    # per ogni macro categoria contenuta in topics_codes, estraggo la lista delle sottocategorie
    macro_areas_codes = []
    for macro in topics_codes:
        if macro not in sub_categories:
            raise ValueError(f"{macro} non ha sotto categorie valide.")
        macro_areas_codes.extend([code for code, desc in sub_categories[macro]])
    print(f"Sub categories: {macro_areas_codes}")

    # estraggo da sub_categories i codici che hanno come prefisso una macro categoria contenuta in topics_codes
    subtopics_codes = []
    for subtopic in raw_subtopics:
        found = False
        for macro in topics_codes:
            for code, desc in sub_categories.get(macro, []):
                if subtopic == desc or subtopic == code:
                    subtopics_codes.append(code)
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError(f"{subtopic} non è una sotto categoria valida.")
    
    return topics_codes, macro_areas_codes, subtopics_codes


from sklearn.metrics.pairwise import cosine_similarity

def recommendation_weighted(user_feedback):
    global df_ids_cats, df_emb, df
    """
    user_feedback = [
        # Esempio:
        # {'id': '2505.10224', 'embedding': np.array([...]), 'like': 1, 'clicks': 3},
        # {'id': '2505.10225', 'embedding': np.array([...]), 'like': -1, 'clicks': 1},
        # ...
    ]
    """

    # Costruisci array di embeddings e pesi
    embeddings_list = []
    weights = []
    for item in user_feedback:
        emb = item['embedding']
        # Salta se l'embedding non è valido
        if emb is None or not hasattr(emb, "shape"):
            continue
        # Flatten se necessario
        if len(emb.shape) == 2 and emb.shape[0] == 1:
            emb = emb.flatten()
        embeddings_list.append(emb)
        # Peso: like/dislike ha molto peso, i click aumentano il peso
        like = item['like'][0] if isinstance(item['like'], list) else item['like']
        try:
            like_weight = 10 * int(like)  # like=1 -> +10, dislike=-1 -> -10, 0 -> 0
        except Exception:
            like_weight = 0
        click_weight = int(item['clicks'][0]) if isinstance(item['clicks'], list) else int(item['clicks'])
        total_weight = like_weight + click_weight
        weights.append(total_weight)
    
    if not embeddings_list or not weights:
        print("Nessun embedding valido trovato in user_feedback.")
        return [], []

    embeddings_arr = np.vstack(embeddings_list)
    weights_arr = np.array(weights)

    # Assicura che weights_arr abbia la stessa lunghezza di embeddings_arr lungo axis=0
    if embeddings_arr.shape[0] != weights_arr.shape[0]:
        print("Mismatch tra numero di embeddings e pesi, controllo dati.")
        return [], []

    # Normalizza i pesi (opzionale, solo se vuoi che la somma sia 1)
    if np.sum(np.abs(weights_arr)) > 0:
        weights_arr = weights_arr / np.sum(np.abs(weights_arr))

    # Calcola l'embedding pesato del gruppo
    group_embedding_weighted = np.average(embeddings_arr, axis=0, weights=weights_arr).reshape(1, -1)

    # Prendi solo le colonne embedding da df_emb (solo float, escludi tutte le colonne non numeriche)
    exclude_cols = ['id', 'categories', 'title', 'abstract', 'authors', 'published']
    embedding_cols = [col for col in df_emb.columns if col not in exclude_cols and np.issubdtype(df_emb[col].dtype, np.number)]
    all_embeddings = df_emb[embedding_cols].values

    # uso kNN per trovare i documenti più simili
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=100, metric='cosine', n_jobs=-1)
    nn.fit(all_embeddings)
    distances, indices = nn.kneighbors(group_embedding_weighted)
    similar_indices = indices[0]
    # Prendi i documenti validi
    similar_docs = df_emb.iloc[similar_indices].copy()
    # Calcola la similarità coseno tra l'embedding del gruppo e gli embeddings dei documenti simili
    similarities = cosine_similarity(group_embedding_weighted, similar_docs[embedding_cols].values).flatten()
    # Aggiungi le similarità come colonna al DataFrame dei documenti simili
    similar_docs['similarity'] = similarities
    # Ordina i documenti simili per similarità in ordine decrescente
    similar_docs = similar_docs.sort_values(by='similarity', ascending=False)
    # Prendi i primi 50 documenti come raccomandazioni
    recommendations = similar_docs.head(50)
    # Aggiungi le categorie e i titoli al DataFrame delle raccomandazioni
    recommendations = recommendations.merge(df_ids_cats[['id', 'categories']], on='id', how='left')
    recommendations = recommendations.merge(df[['id', 'title']], on='id', how='left')
    # Ritorna le raccomandazioni come lista di dizionari
    recommendations_list = recommendations.to_dict(orient='records')
    #for rec in recommendations_list:
    #    rec['categories'] = rec['categories'].split() if isinstance(rec['categories'], str) else []
    #    rec['title'] = rec.get('title', 'No title available')
    return recommendations_list, similar_docs




def actions_parsed(user_actions):
    global df, df_emb, df_ids_cats
    embedding_cols = [col for col in df_emb.columns if col not in ['id', 'categories', 'title']]
    data = []
    for action in user_actions:
        actions = user_actions[action]
        dic = {}
        dic["id"] = action
        dic["clicks"] = actions["clicks"]
        dic["like"] = actions["likes"]
        dic["time_spent"] = actions["time_spent"]
        dic["embedding"] = df_emb[df_emb["id"] == action][embedding_cols].values[0] if not df_emb[df_emb["id"] == action].empty else None
        data.append(dic)
    return data



def wait_user_interests():
    # check if there exist user_registration_info.json
    while not os.path.exists('user_registration_info.json'):
        pass
    with open('user_registration_info.json', 'r') as f:
        user_registration_info = json.load(f)
        raw_topics = user_registration_info[0]['topics']
        raw_subtopics = user_registration_info[0]['subtopics']

    topics_codes, macro_codes, subtopics_codes  = raw2codes(raw_topics, raw_subtopics)

    return {'topics_codes': topics_codes, 'subtopics_codes': subtopics_codes, 'macro_codes': macro_codes}




def recommendation_loop():
    """
    Questa funzione legge continuamente il file user_actions.json e calcola le raccomandazioni
    basate sulle azioni degli utenti. Le raccomandazioni vengono poi scritte in user_rec.json.
    La funzione si interrompe solo quando viene terminato il processo o quando si verifica un errore.
    La funzione attende 5000 secondi tra un ciclo e l'altro per evitare di sovraccaricare il sistema.

    Il formato atteso per user_actions.json è:
    {
        "paper_id_1": {
            "clicks": ["1"],
            "likes": ["1"],
            "favorites": ["1"],
            "time_spent": [],
            "searches": [],
            "last_active": "2023-10-01T12:00:00"
        },
        "paper_id_2": {
            "clicks": ["0"],
            "likes": ["-1"],
            "favorites": ["0"],
            "time_spent": [],
            "searches": [],
            "last_active": "2023-10-01T12:05:00"
        },
        ...
    }
    """
    while True:
        # leggi il file user_actions.json
        with open('user_actions.json', 'r') as f:
            content = f.read().strip()
            if not content:
                print("Il file user_actions.json è vuoto. Attendo 3 secondi e riprovo...")
                time.sleep(3)
                continue
            user_actions = json.loads(content)
        # se il file non è vuoto, procedo con il calcolo delle raccomandazioni
        print("User actions read successfully. Processing recommendations...")
        # controllo che user_actions sia un dizionario
        if not isinstance(user_actions, dict):
            print("Il file user_actions.json non è un dizionario valido. Attendo 5000 secondi e riprovo...")
            time.sleep(5000)
            continue

        # ristrutturo user_actions in un formato utilizzabile
        user_actions_dict = actions_parsed(user_actions)
        # calcolo le raccomandazioni sulla base del nuovo dizionario user_actions_dict
        recommendations, similar_docs = recommendation_weighted(user_actions_dict)        

        # scrivo le raccomandazioni in un file seguendo il formato di registration_samples.json
        # ossia una lista di dizionari con le chiavi 'id', 'title', 'categories', 'abstract', 'authors', 'published'
        # estratti da df
        recommendations = pd.DataFrame(recommendations)

        # considero gli indici di reccommendations come gli ID degli articoli
        recommendations_ids = recommendations['id'].astype(str)
        # estraggo da df le entry che hanno gli ID presenti in recommendations_ids
        recommendations = df[df['id'].isin(recommendations_ids)]
        
        with open('user_rec.json', 'w') as f:
            json.dump(recommendations.to_dict(orient='records'), f, indent=4)

        time.sleep(30) # Attendo 30 secondi prima di ripetere il ciclo





import os, json
import time 
if __name__ == "__main__":
    global df, df_emb, dataset_ids
    try:
        # Load the dataset and embeddings
        dict_dfs= load_data_with_filtering('/home/justamonkey/Documenti/HACKATHON2025/data/arxiv_dataset.csv', '/home/justamonkey/Documenti/HACKATHON2025/data/arxiv_specter_embeddings.csv')
        df = dict_dfs['df_complete']
        df_emb = dict_dfs['df_emb']
        df_ids_cats = dict_dfs['dataset_ids']
        ##########################################
        print("Dataset and embeddings loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset or embeddings: {e}")
        exit(1)

    # wait user interests from user_registration_info.json
    topics_dict = wait_user_interests()

    macro_codes = topics_dict['macro_codes']
    subtopics_codes = topics_dict['subtopics_codes']

    registration_samples = extract_samples(macro_categories=macro_codes, nsamples=50, sub_categories=subtopics_codes, p_subcat=60, p_randcat=40)

    # write the user interests to a file
    with open('registration_samples.json', 'w') as f:
        json.dump(registration_samples.to_dict(orient='records'), f, indent=4)
    

    print("Waiting for user actions...")
    while not os.path.exists('user_actions.json'):
        pass
    print("User actions file found. Starting recommendation process...")

    # finchè non viene richiamato un segnale di stop, continuo a leggere il file user_actions.json
    recommendation_loop()



    # scrivo delle raccomandazioni basate sulle informazioni contenute dentro user_actions.json
    # aspetto che il file user_actions.json venga modificato