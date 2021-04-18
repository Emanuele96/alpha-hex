import random
import numpy as np
import ann_model
import torch.optim as optim
import torch.nn as nn
import torch
import math
from tqdm import tqdm


'''def cross_entropy_loss(prediction, label):
    loss = 0
    for i in range(len(prediction[0])):
        loss += label[0][i] * math.log(prediction[0][i])
    loss = loss * (-1)
    return loss'''

def cross_entropy(pred, soft_targets, dim):
    #logsoftmax = nn.LogSoftmax(dim=dim)
    return torch.mean(torch.sum(- soft_targets * torch.log(pred), 1))

class Actor:

    def __init__(self, cfg):
            self.lr = cfg["anet_lr"]
            self.use_cuda = cfg["use_cuda"]
            if self.use_cuda and torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
            self.nn_layers = cfg["anet_layers"]
            self.input_size = cfg["board_size"] ** 2 + 1
            self.model = ann_model.Net(self.input_size,self.nn_layers, self.use_cuda)
            self.optimizer =  self.initiate_optim(cfg["anet_optim"])
            self.loss_name = cfg["loss"]
            self.loss_fn = self.initiate_loss() #nn.MSELoss()#reduction="mean") #nn.KLDivLoss(reduction = 'batchmean') # nn.CrossEntropyLoss()  nn.BCELoss()
            self.trained_episodes = 0
            self.minibatch_size = cfg["minibatch_size"]

            if cfg["training_type"] == "full_minibatch":
                self.dim = 2
            elif cfg["training_type"] == "episode":
                self.dim = 1
            self.softmax = nn.Softmax(dim=self.dim)
            self.cfg = cfg
            self.board_size = cfg["board_size"]

    def initiate_loss(self):
        if self.loss_name == "mse":
            return nn.MSELoss(reduction="mean")
        elif self.loss_name == "kld":
            return nn.KLDivLoss(reduction = 'batchmean')
        elif self.loss_name =="nl":
            return nn.NLLLoss()
        elif self.loss_name =="ce":
            return None #cross_entropy()
        elif self.loss_name =="l1":
            return nn.L1Loss()


    def initiate_optim(self, optim_name):
        if optim_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=self.lr)
        elif optim_name == "adam":
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif optim_name == "rms":
            return optim.RMSprop(self.model.parameters(), lr=self.lr)

    def get_action(self, state, possible_actions):
        #if playing as p2, toggle the player2 routine
        p2_routine = False
        #state = np.array(state_t, copy=True)
        if state[0][0] == 2:
            p2_routine = True
            #print("state before ", state)
            #state[0][0] = 1
            state[0] = self.toggle_players(state[0])
            state[0][1:] = self.flip_array(state[0][1:])
            #print("state after ", state)

        #Forward the state in the model and get a action distribution
        state_tensor = torch.from_numpy(state)
        action_distribution = self.model(state_tensor.float()).detach().numpy()
        if p2_routine:
            action_distribution = self.flip_array(action_distribution)
            state[0] = self.toggle_players(state[0])
            state[0][1:] = self.flip_array(state[0][1:])
        #Filter away the illegal actions and normalize
        filtered_action_distribution = self.filter_action_distribution(action_distribution, possible_actions, state[0][0])
        #Find the index corrisponding the action with the most visits. Gets the first one if multiple actions has the same visit value
        choosen_action = self.get_max_action_from_distribution(filtered_action_distribution, state[0][0])
        return choosen_action

    def flip_array(self, array):
        array = np.reshape(array, (self.board_size, self.board_size))
        array = np.fliplr(array)
        array = np.flipud(array)
        array = np.reshape(array, (1, -1))
        return array

    def toggle_players(self, array):
        #print("array before toogle ", array)
        for i in range(len(array)):
            if array[i] == 1:
                array[i] = 2
            elif array[i] == 2:
                array[i] = 1
        #print("array after toogle ", array)
        return array

    def filter_action_distribution(self, action_distribution, possible_actions, active_player):
        #Remove the illegal action from the distribution and normalize the vector
        # Create an empty mask
        mask = np.zeros(possible_actions[0].shape)
        # Fill up all the positions of legals actions
        for action in possible_actions:
            mask += action
        # If player 2, remove 2 with 1 (np.where)
        if active_player == 2:
            mask[0] = mask[0] /2
        # Apply the mask to the distribution, removing the non legal actions. Normalize the result
        action_distribution[0] = np.multiply(action_distribution[0], mask[0])
        norm_factor = sum(action_distribution[0])
        if norm_factor == 0:
            #When RELU activation used, the probability distribution can be zeroed after the mask has been applied. Take a random choice
            random_action_index = random.randint(0, action_distribution.shape[1] - 1)
            action_distribution[0][random_action_index] = 1
        else:
            action_distribution[0] = action_distribution[0] / sum(action_distribution[0])
        return action_distribution

    def get_max_action_from_distribution(self, distribution, player_id):
        #Find the index corrisponding the action with the most visits. Gets the first one if multiple actions has the same visit value
        choosen_action_index = np.where(distribution[0] == np.amax(distribution[0]))[0][0]
        #Create a new action and popolate it with the active player code in the choosen_action_index position
        choosen_action = np.zeros(distribution.shape)
        choosen_action[0][choosen_action_index] = player_id
        return choosen_action

    def train_step(self, input_data, label):
        self.model.train()
        prediction = self.model(input_data)
        

        
        print("input ", input_data[0])
        print("prediction ", prediction[0])
        print("label ", label[0])
        print("input size ", input_data.size())
        print("prediction size ", prediction.size())
        
        #prediction = self.softmax(prediction)
        #print("softmaxa prediction ", prediction[0] )
        #print("softmaxa prediction ", torch.sum(prediction[0]) )
        #print("dims ",len(list(prediction.size())))

        if self.loss_name == "ce":
                loss = cross_entropy(prediction, label, self.dim)
        else:
                loss = self.loss_fn(prediction, label)
        print("loss ", loss)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model.train(mode=False)
        return loss
    
    def episode_train(self, x_train, y_train):
        self.optimizer.zero_grad()
        for i in range(len(x_train)):
            x_sample = torch.from_numpy(x_train[i]).float()
            y_label = torch.from_numpy(y_train[i]).float()
            self.train_step(x_sample, y_label)
        self.trained_episodes += 1
        return self.train_step(x_sample, y_label)

    def full_train(self, train_loader, n_epochs):
        losses = list()
        i = 0
        #for i in tqdm(range(n_epochs), "Training ",position = 2, leave = False):
        for x_batch, y_batch in train_loader:
            #print(i)
            #send minibatch to device from cpu
            #x_batch = x_batch.to(device)
            #y_batch = y_batch.to(device)
            losses.append(self.train_step(x_batch, y_batch))
            if i == n_epochs:
                #break after 1 minibatch
                break
            i += 1

        self.trained_episodes += 1
        return losses
