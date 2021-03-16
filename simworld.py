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
        self.neighbours = {}
        # 0 for empty, 1 for player 1 and 2 for player 2
        self.populated_by = 0
        self.parent = None
        #self.is_empty = is_empty
        #self.is_selected = False
        #self.is_being_eaten = False
    
    def empty_the_node(self):
        self.is_empty = 0

class Board:
    def __init__(self, size, visualize, verbose):
        self.verbose = verbose
        self.size = size
        self.active_player = 1
        self.pawns = {}
        self.state_t = None
        self.move_counter = 0
        self.populate_board()
        self.graph = None
        self.visualize = visualize
        if self.visualize:
            self.graph = self.generate_graph()

    def reset(self, visualize):
        #Reset the board
        self.visualize = visualize
        self.move_counter = 0
        for node in self.pawns.values():
            node.empty_the_node()
        self.update_state()
        if self.visualize:
            self.graph = self.generate_graph()
    
    def get_state(self):
        return self.state_t

    def set_state(self, state):
        self.state_t = state
    
    def get_next_state(self, state_t, action):
        next_state = self.state_t + action
        return next_state

    def populate_board(self):
        #Generate all the board nodes
        state = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                is_empty = False
                node = Node((i,j), is_empty)   
                self.pawns[(i,j)] = node
        self.state_t = state 
      
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
    
    def update_state(self):
        #Iterate through all the pawns (nodes) and fill the string representing the state with 0 if that peg is empty or 1 if is not
        state =''
        for coordinate in self.pawns:
            node = self.pawns[coordinate]
            if node.is_empty:
                state = state + '0'
            else:
                state = state + '1'
        self.state_t = state


    def find_all_legal_actions(self):
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
        return all_actions
                    
    def is_goal_state(self):
        return False
    
    def get_reward(self, player_id):
        #return reward for being in state self.state_t at time t for player_id
        if self.is_goal_state():
            if self.active_player == player_id:
                return 1
            return -1
        return 0
        

    def update(self, action):
        #Apply the action to the board and change interested nodes propriety such that it can be visualized 
        #return a tuple of img frames with the first being the selected action and second the new board state
        selected_node = action[0]
        offer = selected_node.neighbours[action[1]]
        empty_node = offer.neighbours[action[1]]
        if self.visualize:
            self.move_counter = self.move_counter + 1
            selected_node.is_selected = True
            offer.is_being_eaten = True
            self.update_graph()
            frame_1 = self.show_board()
            selected_node.is_selected = False
            offer.is_being_eaten = False
        selected_node.is_empty = True
        offer.is_empty= True
        empty_node.is_empty = False
        self.update_state()
        if self.visualize:
            self.update_graph()
            frame_2 = self.show_board()
            return (frame_1, frame_2)
        return None
