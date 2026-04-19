import pandas as pd
import numpy as np
import torch_geometric
import torch
from torch_geometric.data import Data, Batch
import pickle
import math
import os
from recursive_decomposition import recursive_generation
from Data.make_mp_data import peak_detection

def convert_data(angle, intensity, path_save=None):
    intensity = intensity / (np.max(intensity) + 1e-5)

    angle = angle[intensity > 0.01]
    intensity = intensity[intensity > 0.01]

    intensity = intensity[angle < 60]
    angle = angle[angle < 60]

    data = Data(
        x=torch.tensor(angle), intensity=torch.tensor(intensity),
        attr=torch.ones(angle.shape[0], 2), components_num=2,
        x_detected=torch.tensor(angle),
        intensity_detected=torch.tensor(intensity),
        attr_detected=torch.ones(angle.shape[0], 2),
        intensity_detected_raw=torch.tensor(intensity),
        source_index=torch.tensor([1, 2]).unsqueeze(0))

    if path_save is not None:
        with open(path_save, mode='wb') as fo:
            pickle.dump([data], fo)

    return data

if __name__ == "__main__":
    path = "ExperimentalData/YI00_rutile+anatase_10deg_per_min.csv"
    df = pd.read_csv(path, skiprows=591, header=0)  # for miniflex

    pattern_angle = df.values[:, 0]
    pattern_intensity = df.values[:, 1]

    angle, intensity = peak_detection(pattern_angle, pattern_intensity)

    path_model = "result/phasedifformer/confidence_estimator/version_0/model_checkpoint/best.ckpt"
    path_config = "result/phasedifformer/confidence_estimator/config_use.yaml"
    path_database = "Data/MP_database/Only90deg/test.pkl"

    device = "cuda:0"
    max_iter = 5
    batch_size = 100
    path_save_dir = "test/inference/cubic_2phase"

    os.makedirs(path_save_dir, exist_ok=True)

    convert_data(angle, intensity, path_save=path_save_dir + "/input_processed.pkl")

    recursive_generation(
        path_save_dir + "/input_processed.pkl",
        path_config, path_model,
        device, max_iter, batch_size, path_save_dir,
        max_batch_num=3, path_database=path_database,
        candidates_num=15, exclude_index="no"
    )
