import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import io
import variables
class Node:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.neighbours = [] #{}
        # 0 for empty, 1 for player 1 and 2 for player 2
        self.populated_by = 0
    
    def empty_the_node(self):
        self.populated_by = 0

class Board:
    def __init__(self, size, visualize, verbose):
        self.verbose = verbose
        self.initial_state = np.zeros((1, size**2 + 1))
        self.initial_state[0] = 1
        self.size = size
        self.active_player = 1
        self.pawns = {}
        self.state_t = self.initial_state
        self.move_counter = 0
        self.populate_board()
        self.possible_actions = None
        self.graph = None
        self.visualize = visualize
        if self.visualize:
            self.graph = self.generate_graph()

    def find_valid_neighbours(self,node):
        #find all possible neighbours using defined direction rules. Save thoose neighbours in the neighboard-list of the node
        if self.verbose:
            print ("node: " + str(node.coordinates))

        possible_neighbors = ((0,-1),(-1,0),(-1,1),(0,1),(1,0),(1,-1))
        for possible_neighbor in possible_neighbors:
            tmp_coordinate = (node.coordinates[0] + possible_neighbor[0], node.coordinates[1] + possible_neighbor[1])
            if self.verbose:
                print("tmp_coordinates: " + str(tmp_coordinate))
            if tmp_coordinate != node.coordinates and  tmp_coordinate[0] >=0 and tmp_coordinate[0] < self.size and tmp_coordinate[1] >= 0 and tmp_coordinate[1] < self.size:
                if self.pawns[tmp_coordinate] not in node.neighbours:
                    node.neighbours[possible_neighbor] = self.pawns[tmp_coordinate]

    def reset(self, visualize):
        #Reset the board
        self.visualize = visualize
        self.move_counter = 0
        self.remove_all_pawns()
        self.set_state(self.initial_state)
        if self.visualize:
            self.graph = self.generate_graph()

    def remove_all_pawns(self):
        for node in self.pawns.values():
            node.empty_the_node()
    
    def get_state(self):
        return self.state_t

    def set_state(self, state, recompute_population = False):
        self.state_t = state
        self.set_active_player(state[0])
        if recompute_population:
            for i in range(state.shape[1]-1):
                coordinates_1d = i + 1
                coordinates_2d_y = move_coordinates_1d // self.size
                coordinates_2d_x = move_coordinates_1d % self.size
                self.pawns[coordinates_2d_y, coordinates_2d_x].populated_by = state[coordinates_1d]
        self.compute_all_possible_actions()

    
    def set_active_player(self, player_id):
        self.active_player = player_id
    
    def get_next_player(self, active_player = self.active_player):
        return (active_player + 1) % 3 + 1

    def get_next_state(self, state_t = self.state, action):
        #return the state t1 from state t taken action t. NB: This will not update the state of the board
        if action not in self.possible_actions:
            return None
            #TODO implement raise exception
        next_state = self.state_t[0, 1:] + action
        next_state[0] = self.get_next_player()
        return next_state

    def populate_board(self):
        #Generate all the board nodes
        self.state = np.zeros((1, self.size**2 + 1))
        for i in range(self.size):
            for j in range(self.size):
                node = Node((i,j))   
                self.pawns[(i,j)] = node
        for coordinate in self.pawns:
            self.find_valid_neighbours(self.pawns[coordinate])
      
    def to_numpy_array(self):
        #Convert the board to a numpy array, so that can be visualized.
        #Generate a list of all nodes, this will be rows and columns for the adjacent matrix
        all_nodes = list(())
        for node in self.pawns.keys():
            all_nodes.append(node)
        if variables.debug:
            print(all_nodes)
        adj_matrix = np.full((len(all_nodes), len(all_nodes)), 0, dtype=int)
        #Then iterate through every node (row) and for each neighbour, find the corrispondent index(column) and fill that box with 1.
        for i in range(len(all_nodes)):
            node = self.pawns[all_nodes[i]]
            all_neigbours_keyes = list(())
            for neighbour_key in node.neighbours.keys():
                all_neigbours_keyes.append(neighbour_key)
            for key in all_neigbours_keyes:
                j = all_nodes.index(node.neighbours[key].coordinates)
                adj_matrix[i][j] = 1
        return adj_matrix

    def generate_graph(self):
        adj_matrix = self.to_numpy_array()
        G = nx.Graph(adj_matrix)
        #Add node attributes for "coordinate" and "is_empty", taken from board "pawns" dictionary
        all_nodes = list(())
        for node in self.pawns.keys():
            all_nodes.append(node)
        for i in range(adj_matrix.shape[0]):
            G.add_node(i, coordinates = self.pawns[all_nodes[i]].coordinates, is_selected = False, is_being_eaten = False, is_empty = self.pawns[all_nodes[i]].is_empty, diamond_plan = all_nodes[i][0] + all_nodes[i][1], triangle_plan =  all_nodes[i][0] )
        if variables.debug:
            print(adj_matrix)
            print(G.nodes.data())
        return G

    def update_graph(self):
        #Iterate foreach node in the graph and syncronize attributes with the new values
        nodes = list(self.graph.nodes(data=True))
        for node in nodes:
            if variables.debug:
                print(node)
            coordinates = node[1]["coordinates"]
            node[1]["is_empty"] = self.pawns[coordinates].is_empty
            node[1]["is_selected"] = self.pawns[coordinates].is_selected
            node[1]["is_being_eaten"] = self.pawns[coordinates].is_being_eaten

    def show_board(self):
        #Plot the graph nodes and edges  with regards of plans (1 different plan per row), then do the necessary flipping and rotation for matching the board to the assigment
        if self.form == "diamond":
            rotation = 180
            plan_key = "diamond_plan"
        elif self.form == "triangle":
            rotation = 180
            plan_key = "triangle_plan"
        position = nx.multipartite_layout(self.graph, subset_key=plan_key, align="horizontal", center=[0,5])
        #Decide the color of each node, based on if it is empty or not
        nodes_color = []
        nodes = list(self.graph.nodes(data=True))
        for node in nodes:
            color = ""
            if node[1]["is_empty"]:                
                color = "#ffffff"
            elif node[1]["is_selected"]:                
                color = "#17e310"
            elif node[1]["is_being_eaten"]:                
                color = "#fa0000"
            else:
                color = "#000000"
            nodes_color.append(color)
        options = {'node_size': 400,'width': 1, 'pos' : position, 'with_labels':False, 'font_weight':'bold', 'node_color': nodes_color, 'linewidths':5}
        nx.draw(self.graph, **options )
        #outline each node and specify linewidt to show an empty node. ax.collection is the path collection of matplotlib.
        # documentation :  https://matplotlib.org/3.3.3/api/collections_api.html
        ax = plt.gca()
        ax.collections[0].set_edgecolor("#000000") 
        ax.collections[0].set_linewidth(2)
        fig = plt.gcf()
        img = self.matplotlib_to_pil(fig)
        img = img.rotate(rotation)
        if self.form == "diamond":
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        plt.clf()
        #Print the move 
        font = ImageFont.truetype('arial.ttf', 30)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0),"Move nr. " + str(self.move_counter), (0,0,0), font=font)
        return img

    def matplotlib_to_pil(self, fig):
        #Convert a Matplotlib object to an PIL Image   
        bufffer = io.BytesIO()
        fig.savefig(bufffer)
        bufffer.seek(0)
        return Image.open(bufffer)
    
    def compute_all_possible_actions(self):
        #Analize the board and check all possible actions. 
        all_actions = list(())
        board_state = self.state_t.shape[0, 1:]
        empty_action = np.zeros(board_state.shape)
        for i in range(board_state):
            if self.state_t[i] == 0:
                action = empty_action
                action[i] = self.active_player
                all_actions.append(action)                         
        if self.verbose:
            print("all legal actions: " + str(len(all_actions)))
            for action in all_actions:
                print(action)
        self.possible_actions =  all_actions
                    
    def get_all_possible_actions(self):
        return self.possible_actions

    def is_goal_state(self):
        #Start a DFS from each node on the active player side and check if there is a path to the other side
        visited_nodes = list()
        i=0
        if self.active_player == 1:
            start_coordinate = (0, i)
        else:
            start_coordinate = (i, 0)
        for i in range(self.size):
            node = self.pawns[start_coordinate]
            if (node.populated_by == self.active_player) and (node not in visited_nodes):
                is_win = DFS_path_check(node, visited_nodes)
        return is_win

    def DFS_path_check(node, visited_nodes):
        #Perform recursive DFS with a list of visited nodes and domain specific terminal path settings.
        visited_nodes.append(node)
        if node.coordinate[0] == self.size - 1 and self.active_player == 1:
            return True
        elif node.coordinate[1] == self.size - 1 and self.active_player == 2:
            return True    
        for adj_node in node.neighboards:
            if (adj_node.populated_by == self.active_player) and (adj_node not in visited_nodes):
                is_terminal_path = DFS_path_check(adj_node, visited_nodes)
            if is_terminal_path:
                return True
        return False

    def get_reward(self):
        #return reward for being in state self.state_t at time t for player_id
        if self.is_goal_state():
            if self.active_player == 1:
                return 1
            return -1
        return 0        

    def update(self, action):
        #Apply the action to the board and change interested nodes propriety such that it can be visualized 
        #return a img frame of the new board stat  
        if action is not None:
            self.set_state(self.get_next_state(action))
            for i in range(action.shape[1]):
                if action[0, i] == self.active_player:
                    move_coordinates_1d = i
                    break
            move_coordinates_2d_y = move_coordinates_1d // self.size
            move_coordinates_2d_x = move_coordinates_1d % self.size
            self.pawns[move_coordinates_2d_y, move_coordinates_2d_x].populated_by = self.active_player
            self.set_active_player(self.get_next_player())
        self.compute_all_possible_actions()
        if self.visualize:
            self.update_graph()
            frame = self.show_board()
            return frame
        return None
