import math
import simworld

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
    
    def increase_branch_visit(self, action, amount=1)
        self.stats[action] += amount

    def increase_total_amount(self,action, amount):
        self.total_value[action] += amount
    
    def update_q_value(self, action):
        #for action in self.q_values.keys:
        self.q_values[action] = self.total_value[action]/ self.stats[action]


class MTCS():

    def __init__(self, init_state, simworld, player, actor,  cfg):
        self.actor = actor
        self.init_state = init_state
        self.root = self.import_node(init_state)
        self.epsilon = cfg["epsilon"]
        self.number_of_simulations = cfg["number_of_simulations"]
        self.simworld = simworld.board(cfg["board_size"], False)
        self.active_player = player
        #TODO: Update the board with the root state

    def run_simulation(self):
        simulation = 1
        pointer = self.root
        self.board.set_state(pointer.state)
        while simulation < self.number_of_simulations:
            #Check wether pointer points to a leaf node
            #While the node is not a leaf node, point to the next one using the active player and tree policy
            while not pointer.isLeaf:
                pointer = self.choose_next_node(pointer)
                self.board.set_state(pointer.state)
            #If the node has been sampled before, expand it, select the first of the childrens and run a rollout and backpropagation
            if pointer.total_visits > 0:
                self.expand_leaf(pointer)
                pointer = pointer.childrens[pointer.childrens.keys[0]]
                self.board.set_state(pointer.state)
            reward = self.rollout(pointer)
            self.backpropagate(reward, pointer)
            simulation += 1
        suggested_action = self.get_suggested_action()
        self.prune_tree(suggested_action)
        return suggested_action

    def get_suggested_action(self, state, possible_actions):
        return self.actor.get_action(state, possible_actions)

    def is_goal_state(self, state):
        if self.board.is_goal_state():
            return True
        return False

    def rollout(self, node):
        #From the leaf node, let the actor take some actions until reached goal node
        state = node.state
        self.board.set_state(state)
        while not is_goal_state():
            possible_actions = self.board.find_all_legal_actions()
            action = self.get_suggested_action(state, possible_actions)
            self.board.update(action)
        return self.board.get_reward()
    
    def backpropagate(self, reward, node)
        #Backpropagate reward
        #First, the leaf node: No branches, update only visits count
        pointer = node
        pointer.increase_total_visits()
        while not pointer.is_root:
            #While pointing a non root node, cache the action used to get to the node, go to his parent and update values
            action_used = node.action
            pointer = node.parent
            pointer.increase_total_visits()
            pointer.increase_branch_visit(action_used)
            pointer.increase_total_amount(action_used, reward)
            pointer.update_q_value(action_used)

    def expand_leaf(self, node):
        possible_actions = self.board.find_all_legal_actions()
        for action in possible_actions:
            tmp_state = self.board.get_next_state(node.state, action)
            tmp_node = MTCS_node(tmp_state, action, node)
    
    def choose_next_node(self, node):
        #Calculate usa values and do the best greedy choice relate to the player playing
        node.isLeaf = False
        tmp = dict()
        if self.active_player == 1:
            for action in node.q_values.keys:
                tmp[action] = node.q_values[action] + self.calculate_usa(node, action) 
            choosen_action = max(tmp, key= tmp.get)
        elif self.active_player == 2:
            for action in node.q_values.keys:
                tmp[action] = node.q_values[action] - self.calculate_usa(node, action) 
            choosen_action = min(tmp, key= tmp.get)
        self.change_player()
        return choosen_action


    def calculate_usa(self, node, action):
        if node.stats[action] = 0:
            return math.inf
        return math.sqrt((math.log(node.total_visits)/(1 + node.stats[action])))

    def import_state(self, state):
        #Take the state of the board and return a MCTS root node
        # (self, state, action, parent)
        node = MTCS_node(state, None,  None)
        return node

    def change_player(self):
        self.active_player = (self.active_player + 1) % 3 + 1

    def prune_tree(self, action):
        #Prune the tree: remove the root status to the actual root, point to the children of old root via the action and make it the new root
        self.root.is_root = False
        new_root = self.root.childrens[action]
        new_root.is_root = True
        self.root = new_root