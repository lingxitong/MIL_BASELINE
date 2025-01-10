import argparse
import glob
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.mixture import BayesianGaussianMixture


def get_bayesian_gaussian_pt(pt_file_path,save_path):
    if os.path.exists(save_path):
        return
    bag_feat = torch.load(pt_file_path,weights_only = True).unsqueeze(0).cpu().numpy()
    if len(bag_feat.shape) == 3:
        bag_feat = bag_feat.squeeze(0)
    dp_cluster = BayesianGaussianMixture(n_components=10, random_state=0, max_iter=30, weight_concentration_prior=0.1)
    dp_cluster.fit(bag_feat)
    assignments = dp_cluster.predict(bag_feat)
    print(assignments.shape)
    centroids = np.array([np.mean(bag_feat[assignments == i], axis=0) for i in np.unique(assignments)])
    centroids = torch.tensor(centroids).squeeze(0)
    torch.save(centroids, save_path)
    
    
if __name__ == '__main__':
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    parser = argparse.ArgumentParser(description='base dictionary construction')
    parser.add_argument('--dataset_csv_path',default = '/Data/lxt166/MB工程化/cdp_extra_dir/Camelyon-16.csv', type=str)
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--concentration', type=float, default=0.1)
    parser.add_argument('--save_dir', type=str, default='/Data/lxt166/MB工程化/cdp_extra_dir/cdp_bayesianGaussion_dir2')
    args = parser.parse_args()
    
    '''
    --dataset_csv_path: format like /datasets/example_Dataset.csv
    '''
    os.makedirs(args.save_dir, exist_ok=True)
    dataset = pd.read_csv(args.dataset_csv_path)
    pt_file_paths = dataset['slide_path'].dropna().tolist()
    for pt_file_path in tqdm(pt_file_paths):
        save_path = os.path.join(args.save_dir, os.path.basename(pt_file_path).replace('.pt', '_bayesian_gaussian.pt'))
        get_bayesian_gaussian_pt(pt_file_path,save_path)
    print('Done!')
