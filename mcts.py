import math
import simworld
import copy
import numpy as np
from game_visualize import pil_image_to_pygame
import pygame
import random
import operator
from tqdm import tqdm

def softmax(x):
    e_x = np.exp(x - np.max(x)) 
    return e_x / np.sum(e_x)

class MTCS_node():
    
    def __init__(self, state, action, parent):
        self.state = state
        self.action = action
        self.parent = parent
        self.is_leaf = True
        self.is_root = False
        #N(s)
        self.total_visits = 0
        #action as key
        self.childrens = dict()
        #Q(s,a)
        self.q_values = dict()
        #Et(s,a)
        self.total_value = dict()
        #N(s,a)
        self.stats = dict()
        ##self.q_value = 0

    def increase_total_visits(self, amount=1):
        self.total_visits += amount
    
    def increase_branch_visit(self, action, amount=1):
        self.stats[action] += amount

    def increase_total_amount(self,action, amount):
        self.total_value[action] += amount
    
    def update_q_value(self, action):
        #for action in self.q_values.keys:
        self.q_values[action] = self.total_value[action]/ self.stats[action]


class MTCS():

    def __init__(self, init_state, player_1, player_2, replay_buffer,  cfg):
        self.player_1 = player_1
        self.player_2 = player_2
        self.init_state = init_state
        self.root = self.import_state(init_state)
        self.number_of_simulations = cfg["number_of_simulations"]
        self.visualize = cfg["rollout_visualize"]
        self.board = simworld.Board(cfg["board_size"],"Rollout", self.visualize, cfg["verbose"])
        self.initialize_board()
        self.replay_buffer = replay_buffer
        self.verbose = cfg["verbose"]
        self.frame_latency = cfg["frame_latency_rollout"]

    
    def initialize_board(self):
        self.board.set_state(self.init_state, True)
        
    def reset(self):
        self.initialize_board()
        self.root = self.import_state(self.init_state)


    def run_simulation(self):
        simulation = 1
        #Cache the board. This will be the same for each simulation
        cached_simulation_board = copy.deepcopy(self.board)
        for simulation in tqdm(range(self.number_of_simulations), "Simulation ", position = 1, leave = False):
            if self.verbose == 1:
                print("### Begin simulation nr ", simulation, ", starting at root: ", self.root.state)
            pointer = self.root
            self.board.set_state(self.root.state, True)
            #Check wether pointer points to a leaf node
            #While the node is not a leaf node, point to the next one using the active player and tree policy
            while not pointer.is_leaf:
                if self.verbose == 1:
                    print("Not leaf node, select next node with tree policy")
                pointer = self.choose_next_node(pointer)
            #If the node has been sampled before or it is the root, expand it, select the first of the childrens and run a rollout and backpropagation
            if pointer.total_visits > 0 or pointer.is_root:
                if self.verbose == 1:
                    print("### Leaf node sampled before or is root. Expanding Node and selecting a child.")
                self.expand_leaf(pointer)
                #Bruk aktoren?
                if len(pointer.childrens)== 0:
                    #What to do if reached a goal state before rollout?Get reward and backpropagate. 
                    if self.verbose == 1:
                        print("##### Reached goal node before rollout")
                else:
                    #Do a random choice of next node
                    pointer = random.choice(list(pointer.childrens.values()))
            elif self.verbose == 1:
                print("### Leaf node never sampled before.")
            #Update the board to the leaf node state
            self.board.set_state(pointer.state, True)
            if self.verbose == 1:
                print("### Starting rollout from ", pointer)
            reward = self.rollout()
            if self.verbose == 1:
                print("### Rollout result ", reward, ". Backpropagating till root")
            self.backpropagate(reward, pointer)
            #simulation += 1
            self.board = copy.deepcopy(cached_simulation_board)
        action_distribution = self.get_actions_distribution()
        #del cached_simulation_board
        #Add training case to the replay buffer
        self.replay_buffer.add_train_case((self.root.state, action_distribution))#((cached_simulation_board.get_state(), action_distribution))
        if self.verbose == 2:
            print("Simulation distribution\n", self.root.stats.values())
            for key in self.root.stats.keys():
                print(key)
            print(action_distribution)
        return action_distribution
    def get_actions_distribution(self):
        #get a normalized distribution of all actions from root
        hashed_action = next(iter(self.root.childrens))
        action = np.expand_dims(np.asarray(hashed_action), axis=0)
        distribution = np.zeros(action.shape)
        #print("branches\n", self.root.stats)
        for branch in self.root.stats:
            action = np.expand_dims(np.asarray(branch), axis=0)
            distribution += action * self.root.stats[branch]
        distribution = distribution / self.root.total_visits
        #print("distribution \n", distribution)
        #print("distribution sum \n", np.sum(distribution))
        #print("softmaxed \n", softmax(distribution))
        #print("softmax sum \n", np.sum( softmax(distribution)))
        #print("before norm ", softmax(distribution))
        #print("after norm ", softmax(distribution / self.root.total_visits))
        #distribution = softmax(distribution)
        return distribution

    def get_suggested_action(self, board = None):
        #Use actor policy to select the next action
        if board is None:
            board = self.board
        active_player = board.state_t[0][0]
        if active_player == 1:
            actor = self.player_1
        elif active_player == 2:
            actor = self.player_2
        possible_actions = board.get_all_possible_actions()
        return actor.get_action(board.get_state(), possible_actions)

    def is_goal_state(self, state):
        return self.board.is_goal_state()

    def rollout(self):
        #Check wether the node to start the rollout is not a goal state for either players. 
        # This rule out some edge cases.
        if not self.board.is_goal_state(active_player=1) and not self.board.is_goal_state(active_player=2):
            #From the leaf node, let the actor take some actions until reached goal node
            rollout_board = copy.deepcopy(self.board)
            i = 1
            if  self.visualize:
                    pygame.init()
                    #Show start board, generate an img, get the size and initializate a pygame display
                    img = rollout_board.update(None)
                    X, Y = img.size
                    display_surface = pygame.display.set_mode((X,Y)) 
                    frame = pil_image_to_pygame(img)
                    pygame.display.set_caption('Alpha Hex - Emanuele Caprioli')
                    display_surface.blit(frame, (0, 0)) 
                    pygame.display.update() 
                    pygame.time.delay(self.frame_latency)
                    last_pil_frame = None

            end_visualization = False

            is_rollout_goal_state = rollout_board.is_goal_state()
            while (not is_rollout_goal_state or self.visualize) and not end_visualization:
                if not is_rollout_goal_state:
                    if i > 1:
                        rollout_board.change_player()
                    action = self.get_suggested_action(board=rollout_board)
                    if self.verbose == 2:
                        print("#### Roll nr ", i)
                        print("##### Actual State: ", rollout_board.get_state())
                        print("##### Active Player: ", rollout_board.active_player)
                        print("##### Choosen Action: ", action)
                    new_pil_frame = rollout_board.update(action)
                    i += 1

                if self.visualize:
                        #Performe the routine for visualization
                        if new_pil_frame != last_pil_frame:
                            new_frame = pil_image_to_pygame(new_pil_frame)
                            last_pil_frame = new_pil_frame
                            # Update the pygame display with the new frames a delay between each frames
                            display_surface.blit(new_frame, (0, 0)) 
                            pygame.display.update() 
                            pygame.time.wait(self.frame_latency)
                        for event in pygame.event.get() :
                            if event.type == pygame.QUIT :
                                pygame.quit()
                                end_visualization = True
                is_rollout_goal_state = rollout_board.is_goal_state()
                new_pil_frame = rollout_board.update(None)#rollout_board.show_board()
            reward =  rollout_board.get_reward()
            del rollout_board
        else: 
            reward = self.board.get_reward()
        return reward
    
    def backpropagate(self, reward, node):
        #Backpropagate reward
        #First, the leaf node: No branches, update only visits count
        backpropagate_path = list()
        pointer = node
        backpropagate_path.append(pointer)
        pointer.increase_total_visits()
        while not pointer.is_root:
            #While pointing a non root node, cache the action used to get to the node, go to his parent and update values
            action_used = pointer.action
            pointer = pointer.parent
            pointer.increase_total_visits()
            pointer.increase_branch_visit(action_used)
            pointer.increase_total_amount(action_used, reward)
            pointer.update_q_value(action_used)
            if self.verbose == 1:
                backpropagate_path.append(pointer)
        if self.verbose == 1:
            print("####Backpropagation successfull. Backpropagate Path:\n")
            for n in backpropagate_path:
                print(n.state)

    def expand_leaf(self, node):
        self.board.set_state(node.state, True)
        if not self.board.is_goal_state(active_player=1) and not self.board.is_goal_state(active_player=2):
            if self.verbose == 1:
                print("##### Expanding Node ", node)
            possible_actions = self.board.get_all_possible_actions(node.state)
            for action in possible_actions:
                #Action is a np array and as list are unhashable. A key for a dict must be hashable, so convert to bytes (action.tobytes()) or tuple(action))
                hashable_action = tuple(action[0])
                tmp_state = self.board.get_next_state(action=action, state_t=node.state, change_player=True)
                tmp_node = MTCS_node(tmp_state, hashable_action, node)
                node.childrens[hashable_action]= tmp_node
                node.q_values[hashable_action] = 0
                node.stats[hashable_action] = 0
                node.total_value[hashable_action] = 0
            node.is_leaf = False
    
    def choose_next_node(self, node):
        #Calculate usa values and do the best greedy choice relate to the player playing
        if self.verbose == 1:
            print("##### Choosing next Node")
            print("From ", node.state)
        tmp = dict()
        if self.board.active_player == 1:
            for action in node.q_values:
                tmp[action] = node.q_values[action] + self.calculate_usa(node, action) 
            #choosen_action = max(tmp, key= tmp.get)
            max_val = max(tmp.items(), key=operator.itemgetter(1))
            max_keys = [k for k, v in tmp.items() if v == max_val[1]]
            choosen_action = random.choice(max_keys)
            #print("max_val", max_val)
            #print("max_keys", max_keys)
            #print("tmp", tmp.values())
            #print("random choice", choosen_action)
        elif self.board.active_player == 2:
            for action in node.q_values:
                tmp[action] = node.q_values[action] - self.calculate_usa(node, action) 
            #choosen_action = min(tmp, key= tmp.get)
            min_val = min(tmp.items(), key=operator.itemgetter(1))
            min_keys = [k for k, v in tmp.items() if v == min_val[1]]
            choosen_action = random.choice(min_keys)
        next_node = node.childrens[choosen_action]
        next_node.parent = node
        return next_node
            

    def calculate_usa(self, node, action):
        if node.stats[action] == 0:
            return math.inf
        return math.sqrt((math.log(node.total_visits)/(1 + node.stats[action])))

    def import_state(self, state):
        #Take the state of the board and return a MCTS root node
        # (self, state, action, parent)
        node = MTCS_node(state, None,  None)
        node.is_root = True
        return node

    def prune_tree(self, action):
        #Prune the tree: remove the root status to the actual root, point to the children of old root via the action and make it the new root
        self.root.is_root = False
        hashed_action = tuple(action[0])
        new_root = self.root.childrens[hashed_action]
        new_root.is_root = True
        self.root = new_root
        self.board.set_state(new_root.state, True)
