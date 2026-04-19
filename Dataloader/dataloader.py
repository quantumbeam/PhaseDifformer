from torch_geometric.data import Data, DataLoader
from Dataloader import utils
import copy

def convert_to_mydata(data_list):
    new_list = []
    for data in data_list:
        if not isinstance(data, dict):
            raise NotImplementedError()
        else:
            data_dict = {}
            for key in data.keys():
                if key in ['source_index', 'search_result']:
                    data_dict[key] = copy.deepcopy(data[key][0].unsqueeze(0))
                else:
                    data_dict[key] = copy.deepcopy(data[key][0])
        new_data = MyData(**data_dict)
        new_list.append(new_data)
    return new_list

class MyData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ['source_index', 'search_result']:
            return 0
        return super().__inc__(key, value, *args, **kwargs)

class DataModule():
    def __init__(self, path_train, path_val, path_test, batch_size):
        self.batch_size = batch_size

        self.train_graph_list = convert_to_mydata(utils.load_pkl(path_train))
        self.val_graph_list = convert_to_mydata(utils.load_pkl(path_val))
        self.test_graph_list = convert_to_mydata(utils.load_pkl(path_test))

    def train_dataloader(self, train_non_shuffle=False):
        if train_non_shuffle:
            return DataLoader(self.train_graph_list, batch_size=self.batch_size, shuffle=False, exclude_keys=["mixed_pattern", "angle_pattern"])
        else:
            return DataLoader(self.train_graph_list, batch_size=self.batch_size, shuffle=True, exclude_keys=["mixed_pattern", "angle_pattern"])

    def val_dataloader(self):
        return DataLoader(self.val_graph_list, batch_size=self.batch_size, shuffle=False, exclude_keys=["mixed_pattern", "angle_pattern"])

    def test_dataloader(self):
        return DataLoader(self.test_graph_list, batch_size=self.batch_size, shuffle=False, exclude_keys=["mixed_pattern", "angle_pattern"])