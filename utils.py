from torch.utils.data import DataLoader, Dataset
from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset
import copy


class CustomDataLoader:
    def __init__(self, tfrecord_file_path, index_file_path, mini_batch_size, num_workers=8):
        self.tfrecord_file_path = tfrecord_file_path
        self.index_file_path = index_file_path
        self.mini_batch_size = mini_batch_size
        self.dataset = TFRecordDataset(data_path=self.tfrecord_file_path,  
                                        index_path=self.index_file_path)
        self.loader = DataLoader(self.dataset, batch_size=self.mini_batch_size, 
                                num_workers=num_workers)

class EarlyStopper(object):
    def __init__(self, patience, mode="max"):
        self.patience = patience
        self.trial_counter = 0
        self.best_value = None
        self.best_weights = None
        self.mode = mode

    def stop_training(self, value, model):
        if (self.best_value is None) or\
            (self.mode == "max" and value > self.best_value) or\
            (self.mode == "min" and value < self.best_value):
            self.best_value = value
            self.trial_counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
            return False
        elif self.trial_counter + 1 < self.patience:
            self.trial_counter += 1
            return False
        else:
            return True