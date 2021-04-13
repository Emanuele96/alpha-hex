import numpy as np
from torch.utils.data import Dataset, TensorDataset
import torch
class Replay_buffer():
    def __init__(self):
        #self.x_train_episode = list()
        #self.y_train_episode = list()
        #self.x_train = list()
        #self.y_train = list()
        self.dataset = dict()
        self.long_time_dataset = dict()

    def get_training_episode(self):
        x_train = []
        y_train = []
        for hashed_state in self.dataset.keys():
            state = np.expand_dims(np.asarray(hashed_state), axis=0)
            x_train.append(state)
            y_train.append(self.dataset[hashed_state])
        return x_train, y_train
    
    def get_training_dataset(self):
        x_train = []
        y_train = []
        for hashed_state in self.long_time_dataset.keys():
            state = np.expand_dims(np.asarray(hashed_state), axis=0)
            x_train.append(state)
            y_train.append(self.long_time_dataset[hashed_state])
        
        x_train = np.stack(x_train)
        y_train = np.stack(y_train)

        x_train_tensor = torch.from_numpy(x_train).float()
        y_train_tensor = torch.from_numpy(y_train).float()
        #print(x_train_tensor[0], y_train_tensor[0])
        return CustomDataset(x_train_tensor, y_train_tensor)
        

    
    def add_train_case(self, train_case):
        #self.x_train.append(train_case[0])
        #self.y_train.append(train_case[1])
        hashable_action = tuple(train_case[0][0])
        self.dataset[hashable_action] = train_case[1]

    
    def flush_episode(self):
        #self.x_train_episode = list()
        #self.y_train_episode = list()
        self.long_time_dataset.update(self.dataset)
        self.dataset = dict()

class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)