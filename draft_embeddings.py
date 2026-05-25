import sqlite3
import json
import numpy as np
from scipy.sparse.linalg import svds

def get_svd_hero_embeddings(db_path="./mlbb_data.db", K=16):
    """
    Loads historical drafts from the SQLite database, computes the Positive Pointwise
    Mutual Information (PPMI) co-occurrence matrix, and performs Singular Value Decomposition
    (SVD) to return a 16-dimensional dense embedding vector for each hero.
    
    Returns:
        embeddings (dict): Map of HeroName -> list of K floats.
        fallback_vector (list): Average vector to use for new/unseen heroes.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT picks FROM games WHERE picks IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    # Extract all team compositions
    sentences = []
    all_heroes = set()

    for (picks_json,) in rows:
        try:
            picks = json.loads(picks_json)
        except Exception:
            continue
        blue_picks = picks.get('blue', [])
        red_picks = picks.get('red', [])
        if len(blue_picks) == 5:
            sentences.append(blue_picks)
            all_heroes.update(blue_picks)
        if len(red_picks) == 5:
            sentences.append(red_picks)
            all_heroes.update(red_picks)

    hero_list = sorted(list(all_heroes))
    hero_to_idx = {h: i for i, h in enumerate(hero_list)}
    idx_to_hero = {i: h for i, h in enumerate(hero_list)}
    num_heroes = len(hero_list)

    if num_heroes == 0:
        # Return dummy mappings if db is empty (should not happen)
        return {}, [0.0] * K

    # 1. Build Co-occurrence Matrix
    C = np.zeros((num_heroes, num_heroes))
    for sentence in sentences:
        for i in range(len(sentence)):
            for j in range(i + 1, len(sentence)):
                h1, h2 = sentence[i], sentence[j]
                if h1 in hero_to_idx and h2 in hero_to_idx:
                    idx1, idx2 = hero_to_idx[h1], hero_to_idx[h2]
                    C[idx1, idx2] += 1.0
                    C[idx2, idx1] += 1.0

    # 2. Compute PPMI Matrix
    T = np.sum(C)
    D = np.sum(C, axis=1)

    PPMI = np.zeros((num_heroes, num_heroes))
    for i in range(num_heroes):
        for j in range(num_heroes):
            if C[i, j] > 0 and D[i] > 0 and D[j] > 0:
                pmi = np.log((C[i, j] * T) / (D[i] * D[j]))
                PPMI[i, j] = max(0, pmi)

    # 3. SVD Decomposition
    # In case K is greater than or equal to num_heroes, cap it
    k_dim = min(K, num_heroes - 1)
    if k_dim < 1:
        k_dim = 1

    try:
        U, S, Vt = svds(PPMI, k=k_dim)
        # Hero representations: W = U * sqrt(S)
        W = U * np.sqrt(S)
    except Exception as e:
        print(f"SVD warning (using fallback random projection): {e}")
        # Robust fallback using standard numpy SVD or random projection if svds fails
        U, S, Vt = np.linalg.svd(PPMI)
        W = U[:, :k_dim] * np.sqrt(S[:k_dim])

    # Pad vectors to exactly K dimensions if needed
    if W.shape[1] < K:
        padding = np.zeros((W.shape[0], K - W.shape[1]))
        W = np.hstack((W, padding))

    # 4. Map to dictionary
    embeddings = {}
    for h, idx in hero_to_idx.items():
        embeddings[h] = W[idx].tolist()

    # Precompute fallback vector (mean vector)
    fallback_vector = np.mean(W, axis=0).tolist()

    return embeddings, fallback_vector

if __name__ == "__main__":
    emb, fallback = get_svd_hero_embeddings()
    print(f"Generated embeddings for {len(emb)} heroes.")
    print(f"Fallback vector: {fallback}")
