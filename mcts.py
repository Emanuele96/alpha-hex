import math
import simworld

class MTCS_node():
    
    def __init__(self, parent, rollout):
        self.parent = parent
        self.original_node = None
        self.isRollout = rollout
        self.is_leaf = True
        #N(s)
        self.total_visits = 0
        #action as key
        #Q(s,a)
        self.childrens = dict()
        #Et(s,a)
        self.total_value = dict()
        #N(s,a)
        self.stats = dict()
        self.q_value = 0


class MTCS():

    def __init__(self, init_state, simworld, player, cfg):
        self.init_state = init_state
        self.root = self.import_node(root)
        self.epsilon = cfg["epsilon"]
        self.simworld = simworld.board(cfg["board_size"], False)
        self.number_of_simulations = cfg["number_of_simulations"]
        self.active_player = player

    def run_simulation(self):
        simulation = 1
        pointer = self.root
        while simulation < self.number_of_simulations:
            #Traversate the tree while pointing to a non goal state
            while is_goal_state(pointer):
                #Check wether pointer points to a leaf node
                #If the node is not a leaf node, point to the next one using the active player and tree policy
                if pointer.isLeaf:
                    pointer = self.choose_next_node(pointer)
                #Else, check wether the node has been sampled before
                else:
                    #If the node has been sampled before, expand it, select the first of the childrens and run a rollout and backpropagation
                    if pointer.total_visits > 0:
                        self.expand_leaf(pointer)
                        pointer = pointer.childrens[pointer.childrens.keys[0]]
                    reward = self.rollout(pointer)
                    self.backpropagate(reward, pointer)
            

        return -1
    def is_goal_state(self, node):
        return False

    def rollout(self, node):
        return -1
    
    def backpropagate(self, reward, node)
    return -1

    def expand_leaf(self, node):
        return -1
    
    def choose_next_node(self, node):
        #Calculate usa values and do the best greedy choice relate to the player playing
        node.isLeaf = False
        tmp = dict()
        if self.active_player == 1:
            for action in node.childrens.keys:
                tmp[action] = node.childrens[action] + self.calculate_usa(node, action) 
            choosen_action = max(tmp, key= tmp.get)
        elif self.active_player == 2:
            for action in node.childrens.keys:
                tmp[action] = node.childrens[action] - self.calculate_usa(node, action) 
            choosen_action = min(tmp, key= tmp.get)
        self.change_player()
        return choosen_action


    def calculate_usa(self, node, action):
        if node.stats[action] = 0:
            return math.inf
        return math.sqrt((math.log(node.total_visits)/(1 + node.stats[action])))

    def calculate_q_value(self, node):

        return -1

    def import_node(self, node):
        return node
    
    def change_player(self):
        self.active_player = (self.active_player + 1) % 3 + 1