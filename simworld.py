import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import io
import variables
class Node:
    def __init__(self, coordinates, is_empty):
        self.coordinates = coordinates
        self.neighbours = {}
        self.is_empty = is_empty
        self.is_selected = False
        self.is_being_eaten = False
    
    def empty_the_node(self):
        self.is_empty = True
class Board:
    def __init__(self, form, size, empty_nodes, visualize):
        self.form = form
        self.size = size
        self.empty_nodes = empty_nodes
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
            if node.coordinates in self.empty_nodes:
                node.is_empty = True
            else:
                node.is_empty = False
            node.is_being_eaten = False
            node.is_selected = False
        self.update_state()
        if self.visualize:
            self.graph = self.generate_graph()
    
    def get_state(self):
        return self.state_t
    
    def find_valid_neighbours(self,node):
        #find all possible neighbours using defined direction rules. Save thoose neighbours in the neighboard-list of the node as a tuple (direction, node)  
        if variables.debug:
            print ("node: " + str(node.coordinates))
        if self.form == "diamond":
            possible_neighbors = ((0,-1),(-1,0),(-1,1),(0,1),(1,0),(1,-1))
            for possible_neighbor in possible_neighbors:
                tmp_coordinate = (node.coordinates[0] + possible_neighbor[0], node.coordinates[1] + possible_neighbor[1])
                if variables.debug:
                    print("tmp_coordinates: " + str(tmp_coordinate))
                if tmp_coordinate != node.coordinates and  tmp_coordinate[0] >=0 and tmp_coordinate[0] < self.size and tmp_coordinate[1] >= 0 and tmp_coordinate[1] < self.size:
                    if self.pawns[tmp_coordinate] not in node.neighbours:
                        node.neighbours[possible_neighbor] = self.pawns[tmp_coordinate]
        elif self.form == "triangle":
            possible_neighbors = ((-1,-1),(-1,0),(0,1),(1,1),(1,0),(0,-1))
            for possible_neighbor in possible_neighbors:
                tmp_coordinate = (node.coordinates[0] + possible_neighbor[0], node.coordinates[1] + possible_neighbor[1])
                if tmp_coordinate != node.coordinates and tmp_coordinate[0] >=0 and tmp_coordinate[0] < self.size and tmp_coordinate[1] >= 0 and tmp_coordinate[1] <= tmp_coordinate[0]:
                    if self.pawns[tmp_coordinate] not in node.neighbours:
                        node.neighbours[possible_neighbor] = self.pawns[tmp_coordinate]

    def populate_board(self):
        #Generate all the pegs (nodes) and find all they legal neighbours
        state = ''
        for i in range(self.size):
            for j in range(i+1 if self.form == "triangle" else self.size):
                if (i,j) in self.empty_nodes:
                    is_empty = True
                    state = state + '0'
                else:
                    is_empty = False
                    state = state + '1'
                node = Node((i,j), is_empty)   
                self.pawns[(i,j)] = node
        self.state_t = state 
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
        #Analize the board and check all possible actions. Iterate trough all the nodes and for each neighboard, 
        # check if with the move that needs to take from node-->neighboard, it comes to neighboard-->adj_to_neighboard
        # and this is an empty node. 
        all_actions = list(())
        for node in self.pawns.values():
            if node.is_empty:
                continue
            for neighbour_object in node.neighbours.items():
                move = neighbour_object[0]
                neighbour = neighbour_object[1]
                if neighbour.is_empty:
                    continue
                if move in neighbour.neighbours:
                    if neighbour.neighbours[move].is_empty:
                        all_actions.append((node, move))

                                
        if variables.debug:
            print("all legal actions: " + str(len(all_actions)))
            for action in all_actions:
                print(str(action[0].coordinates) + "   " + str(action[1]) )
        return all_actions
                    
    def get_reward(self, game_over):
        #return reward for being in state self.state_t at time t
        if game_over and int(self.state_t.replace('0','')) == 1:
            return variables.terminal_goal_state_reward 
        elif game_over:
            return variables.terminal_state_penalty
        return variables.non_terminal_state_reward

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
