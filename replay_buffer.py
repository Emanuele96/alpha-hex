import numpy as np

class Replay_buffer():
    def __init__(self):
        self.x_train_episode = list()
        self.y_train_episode = list()
        self.x_train = list()
        self.y_train = list()
        self.dataset = dict()

    def get_training_dataset(self):
        x_train = []
        y_train = []
        for hashed_state in self.dataset.keys():
            state = np.expand_dims(np.asarray(hashed_state), axis=0)
            x_train.append(state)
            y_train.append(self.dataset[hashed_state])
        return x_train, y_train
    
    def add_train_case(self, train_case):
        #self.x_train.append(train_case[0])
        #self.y_train.append(train_case[1])
        hashable_action = tuple(train_case[0][0])
        self.dataset[hashable_action] = train_case[1]

    def get_training_episode(self):
        self.x_train.extend(self.x_train_episode)
        self.y_train.extend(self.y_train_episode)
        return self.x_train_episode, self.y_train_episode
    
    def flush_episode(self):
        self.x_train_episode = list()
        self.y_train_episode = list()
        self.dataset = dict()