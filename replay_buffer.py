
class Replay_buffer():
    def __init__(self):
        self.x_train = list()
        self.y_train = list()

    def get_training_dataset(self):
        return self.x_train, self.y_train
    
    def add_train_case(self, train_case):
        self.x_train.append(train_case[0])
        self.y_train.append(train_case[1])
