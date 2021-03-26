import math
import simworld
import copy
import numpy as np
from game_visualize import pil_image_to_pygame
import pygame

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

    def __init__(self, init_state, actor, replay_buffer,  cfg):
        self.actor = actor
        self.init_state = init_state
        self.root = self.import_state(init_state)
        self.epsilon = cfg["epsilon"]
        self.number_of_simulations = cfg["number_of_simulations"]
        self.visualize = cfg["rollout_visualize"]
        self.board = simworld.Board(cfg["board_size"],"Rollout1", self.visualize, cfg["verbose"])
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
        print("ROOT STATE", self.root.state)
        simulation = 1
        #pointer = self.root
        #Cache the board. This will be the same for each simulation
        cached_simulation_board = copy.deepcopy(self.board)
        while simulation < self.number_of_simulations:
            if self.verbose:
                print("### Begin simulation nr ", simulation, ", starting at root: ", self.root.state)
            pointer = self.root
            self.board.set_state(self.root.state, True)
            #Check wether pointer points to a leaf node
            #While the node is not a leaf node, point to the next one using the active player and tree policy
            while not pointer.is_leaf:
                if self.verbose:
                    print("Not leaf node, select next node with tree policy")
                pointer = self.choose_next_node(pointer)
                #action = np.expand_dims(np.asarray(hashed_action), axis=0)
                #self.board.update(action)
                #self.board.change_player()
                #next_node = pointer.childrens[hashed_action]
                #next_node.parent = pointer
                #pointer = next_node
            #If the node has been sampled before or it is the root, expand it, select the first of the childrens and run a rollout and backpropagation
            if pointer.total_visits > 0 or pointer.is_root:
                if self.verbose:
                    print("### Leaf node sampled before or is root. Expanding Node and selecting a child.")
                self.expand_leaf(pointer)



                
                print("###########################################")
                print("Node expanded. State ", pointer.state)
                print("childrens\n")
                for key in pointer.childrens.keys():
                    print("action", key)
                    print("state", pointer.childrens[key].state)
                print("###########################################")





                #Bruk aktoren?
                if len(pointer.childrens)== 0:
                    #What to do if reached a goal state before rollout?
                    if self.verbose:
                        print("##### Reached goal node before rollout")
                    print("##### Reached goal node before rollout!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    #break
                hashed_action = next(iter(pointer.childrens))
                action = np.expand_dims(np.asarray(hashed_action), axis=0)
                #self.board.update(action)
                #self.board.change_player()
                pointer = pointer.childrens[hashed_action]
            elif self.verbose:
                print("### Leaf node never sampled before.")

            #Update the board to the leaf node state
            self.board.set_state(pointer.state, True)
            print("board state before rollout", self.board.get_state())
            if self.verbose:
                print("### Starting rollout from ", pointer)
            reward = self.rollout()
            if self.verbose:
                print("### Rollout result ", reward, ". Backpropagating till root")
            self.backpropagate(reward, pointer)
            simulation += 1
            self.board = copy.deepcopy(cached_simulation_board)
        action_distribution = self.get_actions_distribution()
        #del cached_simulation_board
        #Add training case to the replay buffer
        self.replay_buffer.add_train_case((cached_simulation_board.get_state(), action_distribution))
        return action_distribution
    def get_actions_distribution(self):
        #get a normalized distribution of all actions from root
        hashed_action = next(iter(self.root.childrens))
        action = np.expand_dims(np.asarray(hashed_action), axis=0)
        distribution = np.zeros(action.shape)
        for branch in self.root.stats:
            action = np.expand_dims(np.asarray(branch), axis=0)
            distribution += action * self.root.stats[branch]
        distribution = distribution / self.root.total_visits
        return distribution

    def get_suggested_action(self, board = None):
        #Use actor policy to select the next action
        if board is None:
            board = self.board
        possible_actions = board.get_all_possible_actions()
        return self.actor.get_action(board.get_state(), possible_actions)

    def is_goal_state(self, state):
        return self.board.is_goal_state()

    def rollout(self):

        #From the leaf node, let the actor take some actions until reached goal node
        rollout_board = copy.deepcopy(self.board)
        i = 1

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Rollout start state", rollout_board.get_state())
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
                #print("act p ",rollout_board.active_player)
                '''try:
                    action = self.get_suggested_action(board=rollout_board)
                except:
                    print("**********************************************")
                    print("state ", rollout_board.get_state())
                    a = np.zeros((1, 5, 5))
                    for n in rollout_board.pawns.keys():
                        a[0][n] = rollout_board.pawns[n].populated_by
                    print("before a\n", a)
                    print("active player", rollout_board.active_player)
                    print("is goal for p",rollout_board.active_player, " : ", rollout_board.is_goal_state(verbose= True))
                    rollout_board.change_player()
                    print("is goal for p",rollout_board.get_next_player(), " : ", rollout_board.is_goal_state(verbose=True))'''
                action = self.get_suggested_action(board=rollout_board)

                if self.verbose:
                    print("#### Roll nr ", i)
                    print("##### Actual State: ", rollout_board.get_state())
                    print("##### Active Player: ", rollout_board.active_player)
                    print("##### Choosen Action: ", action)
                new_pil_frame = rollout_board.update(action)
                a = np.zeros((1, 5, 5))
                for n in rollout_board.pawns.keys():
                    a[0][n] = rollout_board.pawns[n].populated_by
                if False:
                    print("after a\n", a)
                    print("state", rollout_board.get_state())
                    print("is goal for p",rollout_board.active_player, " : ", rollout_board.is_goal_state())
                    print("**********************************************")
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
        reward =  rollout_board.get_reward()
        del rollout_board
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
            if self.verbose:
                backpropagate_path.append(pointer)
        if self.verbose:
            print("####Backpropagation successfull. Backpropagate Path:\n")
            for n in backpropagate_path:
                print(n.state)

    def expand_leaf(self, node):
        if self.verbose:
            print("##### Expanding Node ", node)
        possible_actions = self.board.get_all_possible_actions(node.state)




        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("Expanding NODe", node.state)
        print("ALll possible moves")
        for m in possible_actions:
            print(m)



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
        print("acp", self.board.state.shape)
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
        #action = np.expand_dims(np.asarray(choosen_action), axis=0)
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


        print("pruninggggggggg")
        print("ROOT", self.root.state)
        print("action", action)
        
        print("Childrens\n")
        for key in self.root.childrens.keys():
            print(key)
        print("is action in roots children ", hashed_action in self.root.childrens.keys())


        new_root = self.root.childrens[hashed_action]
        new_root.is_root = True
        self.root = new_root
        self.board.set_state(new_root.state, True)
        #self.board.change_player()