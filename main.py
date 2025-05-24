import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.utils.extmath import randomized_svd
import kagglehub


def load_data(directory_path):
    ratings_path = os.path.join(directory_path, 'ratings.dat')
    df = pd.read_csv(
        ratings_path, 
        sep='::', 
        engine='python',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        dtype={'user_id': int, 'item_id': int}
    )
    
    # create categorical mappings
    user_ids = df['user_id'].unique()
    item_ids = df['item_id'].unique()
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}
    
    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)
    
    return df, len(user_ids), len(item_ids)

def temporal_split(df):
    # temporal split using timestamps
    df = df.sort_values(['user_id', 'timestamp'])
    train, test = [], []
    
    for uid in df['user_id'].unique():
        user_data = df[df['user_id'] == uid]
        split_idx = int(0.8 * len(user_data))
        train.append(user_data.iloc[:split_idx])
        test.append(user_data.iloc[split_idx:])
        
    return pd.concat(train), pd.concat(test)

def create_train_mask(train_data, num_users, num_items):
    """Create boolean mask for training interactions"""
    mask = np.zeros((num_users, num_items), dtype=bool)
    for u, i in zip(train_data['user_idx'], train_data['item_idx']):
        mask[u, i] = True
    return mask
    
def preprocess(R):
    # normalized Laplacian with epsilon handling
    user_degrees = np.array(R.sum(axis=1)).ravel()
    item_degrees = np.array(R.sum(axis=0)).ravel()

    user_degrees[user_degrees == 0] = 1e-6
    item_degrees[item_degrees == 0] = 1e-6
    
    D_u = diags(1 / np.sqrt(user_degrees))
    D_i = diags(1 / np.sqrt(item_degrees))

    # normalize R
    return D_u @ R @ D_i

def svd_ae(R_hat, R_train, gamma):
    # closed form soln from paper
    m = int(gamma * min(R_hat.shape))
    U, S, Vt = randomized_svd(R_hat, n_components=m, random_state=42)
    
    # Paper's exact formulation
    Sigma_inv = np.diag(1 / S)
    Q_m = U
    V_m = Vt.T
    
    B = V_m @ Sigma_inv @ Q_m.T @ R_train
    return R_hat @ B
    
def evaluate(test_users, test_items, scores, train_mask, top_k=10):
    hr_sum = 0
    ndcg_sum = 0
    user_counts = 0

    test_df = pd.DataFrame({'user': test_users, 'item': test_items})
    user_groups = test_df.groupby('user')['item'].apply(list)
    
    for user_idx, test_items in user_groups.items():
        user_scores = scores[user_idx].copy()
        
        # mask training interactions
        user_scores[train_mask[user_idx]] = -np.inf
        
        # top k prediction
        top_items = np.argpartition(user_scores, -top_k)[-top_k:]
        top_items_set = set(top_items)
        
        hits = sum(1 for item in test_items if item in top_items_set)
        if hits > 0:
            hr_sum += 1

            ranks = []
            for item in test_items:
                if item in top_items:
                    ranks.append(np.where(top_items == item)[0][0] + 1)  # 1-based index
            
            dcg = sum(1 / np.log2(r + 1) for r in ranks)
            idcg = sum(1 / np.log2(i + 1) for i in range(1, len(ranks)+1))
            ndcg_sum += dcg / idcg
            
        user_counts += 1
    
    return hr_sum/user_counts, ndcg_sum/user_counts

def main():
    path = kagglehub.dataset_download("odedgolden/movielens-1m-dataset")
    df, num_users, num_items = load_data(path)
    train_data, test_data = temporal_split(df)
    
    # create sparse matrices
    R_train = csr_matrix(
        (np.ones(len(train_data)), 
         (train_data['user_idx'], train_data['item_idx'])),
        shape=(num_users, num_items)
    )
    
    # preprocessing and training
    R_hat = preprocess(R_train)
    reconstructed = svd_ae(R_hat, R_train, gamma=0.3)
    
    train_mask = np.zeros((num_users, num_items), dtype=bool)
    for u, i in zip(train_data['user_idx'], train_data['item_idx']):
        train_mask[u, i] = True
    
    # test data preparation
    test_users = test_data['user_idx'].values
    test_items = test_data['item_idx'].values
    
    # evaluate
    hr, ndcg = evaluate(test_users, test_items, reconstructed, train_mask)
    print(f"HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}")

if __name__ == "__main__":
    main()
