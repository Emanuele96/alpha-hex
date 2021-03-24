class Replay_buffer():
    def __init__(self):
        self.x_train_episode = list()
        self.y_train_episode = list()
        self.x_train = list()
        self.y_train = list()


    def get_training_dataset(self):
        return self.x_train, self.y_train
    
    def add_train_case(self, train_case):
        self.x_train_episode.append(train_case[0])
        self.y_train_episode.append(train_case[1])

    def get_training_episode(self):
        self.x_train.extend(self.x_train_episode)
        self.y_train.extend(self.y_train_episode)
        return self.x_train_episode, self.y_train_episode