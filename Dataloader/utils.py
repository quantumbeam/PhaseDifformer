import pickle
import torch_geometric

def load_pkl(path_pkl):
    with open(path_pkl, mode='br') as fi:
        data = pickle.load(fi)
    return data
