import math
import simworld
import copy
import numpy as np
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

    def __init__(self, init_state, actor,  cfg):
        self.actor = actor
        self.init_state = init_state
        self.root = self.import_state(init_state)
        self.epsilon = cfg["epsilon"]
        self.number_of_simulations = cfg["number_of_simulations"]
        self.board = simworld.Board(cfg["board_size"], False, cfg["verbose"])
        #self.initialize_board()
        self.verbose = cfg["verbose"]
    
    def initialize_board(self):
        self.board.set_state(self.init_state, True)

    def run_simulation(self):
        simulation = 1
        #pointer = self.root
        #Cache the board. This will be the same for each simulation
        cached_simulation_board = copy.deepcopy(self.board)
        while simulation < self.number_of_simulations:
            if self.verbose:
                print("### Begin simulation nr ", simulation, ", starting at root")
            pointer = self.root
            self.board.set_state(self.root.state, True)
            #Check wether pointer points to a leaf node
            #While the node is not a leaf node, point to the next one using the active player and tree policy
            while not pointer.is_leaf:
                if self.verbose:
                    print("Not leaf node, select next node with tree policy")
                hashed_action = self.choose_next_node(pointer)
                action = np.expand_dims(np.asarray(hashed_action), axis=0)
                self.board.update(action)
                self.board.change_player()
                next_node = pointer.childrens[hashed_action]
                next_node.parent = pointer
                pointer = next_node
            #If the node has been sampled before, expand it, select the first of the childrens and run a rollout and backpropagation
            if pointer.total_visits > 0 or pointer.is_root:
                if self.verbose:
                    print("### Leaf node sampled before or is root. Expanding Node and selecting a child.")
                self.expand_leaf(pointer)
                #Bruk aktoren?
                if len(pointer.childrens)== 0:
                    #What to do if reached a goal state before rollout?
                    if self.verbose:
                        print("##### Reached goal node before rollout")
                    #break
                hashed_action = next(iter(pointer.childrens))
                action = np.expand_dims(np.asarray(hashed_action), axis=0)
                self.board.update(action)
                self.board.change_player()
                pointer = pointer.childrens[hashed_action]
            elif self.verbose:
                print("### Leaf node never sampled before.")

            if self.verbose:
                print("### Starting rollout from ", pointer)
            reward = self.rollout(pointer)
            if self.verbose:
                print("### Rollout result ", reward, ". Backpropagating till root")
            self.backpropagate(reward, pointer)
            simulation += 1
            self.board = copy.deepcopy(cached_simulation_board)
        suggested_action = self.get_actions_distribution()
        del cached_simulation_board
        return suggested_action
    def get_actions_distribution(self):
        #get a normalized distribution of all actions from root
        distribution = dict()
        for branch in self.root.stats:
            distribution[branch] = self.root.stats[branch] / self.root.total_visits
        return distribution

    def get_suggested_action(self, board = None):
        #Use actor policy to select the next action
        if board is None:
            board = self.board
        possible_actions = board.get_all_possible_actions()
        return self.actor.get_action(board.get_state(), possible_actions)

    def is_goal_state(self, state):
        return self.board.is_goal_state()

    def rollout(self, node):
        #From the leaf node, let the actor take some actions until reached goal node
        rollout_board = copy.deepcopy(self.board)
        i = 1
        while not rollout_board.is_goal_state():
            if i > 1:
                rollout_board.change_player()
            action = self.get_suggested_action(rollout_board)
            if self.verbose:
                print("#### Roll nr ", i)
                i += 1
                print("##### Actual State: ", rollout_board.get_state())
                print("##### Active Player: ", rollout_board.active_player)
                print("##### Choosen Action: ", action)
            rollout_board.update(action)
        reward =  rollout_board.get_reward()
        del rollout_board
        return reward
    
    def backpropagate(self, reward, node):
        #Backpropagate reward
        #First, the leaf node: No branches, update only visits count
        pointer = node
        pointer.increase_total_visits()
        while not pointer.is_root:
            #While pointing a non root node, cache the action used to get to the node, go to his parent and update values
            action_used = pointer.action
            pointer = pointer.parent
            pointer.increase_total_visits()
            pointer.increase_branch_visit(action_used)
            pointer.increase_total_amount(action_used, reward)
            pointer.update_q_value(action_used)
            

    def expand_leaf(self, node):
        if self.verbose:
            print("##### Expanding Node ", node)
        possible_actions = self.board.get_all_possible_actions()
        for action in possible_actions:
            #Action is a np array and as list are unhashable. A key for a dict must be hashable, so convert to bytes (action.tobytes()) or tuple(action))
            hashable_action = tuple(action[0])
            tmp_state = self.board.get_next_state(action=action, change_player=True)
            tmp_node = MTCS_node(tmp_state, hashable_action, node)
            node.childrens[hashable_action]= tmp_node
            node.q_values[hashable_action] = 0
            node.stats[hashable_action] = 0
            node.total_value[hashable_action] = 0
        node.is_leaf = False
    
    def choose_next_node(self, node):
        #Calculate usa values and do the best greedy choice relate to the player playing
        if self.verbose:
            print("##### Choosing next Node")
        tmp = dict()
        if self.board.active_player == 1:
            for action in node.q_values:
                tmp[action] = node.q_values[action] + self.calculate_usa(node, action) 
            choosen_action = max(tmp, key= tmp.get)
        elif self.board.active_player == 2:
            for action in node.q_values:
                tmp[action] = node.q_values[action] - self.calculate_usa(node, action) 
            choosen_action = min(tmp, key= tmp.get)
        return choosen_action


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
        self.board.set_state(self.board.get_next_state(action=action), True)
        self.board.change_player()