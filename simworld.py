import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import io
class Node:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.neighbours = [] #{}
        # 0 for empty, 1 for player 1 and 2 for player 2
        self.populated_by = 0
    
    def empty_the_node(self):
        self.populated_by = 0

class Board:
    def __init__(self, size, board_name, visualize, verbose):
        self.verbose = verbose
        self.initial_state = np.zeros((1, size**2 + 1))
        self.initial_state[0,0] = 1
        self.size = size
        self.active_player = 1
        self.pawns = {}
        self.state_t = self.initial_state
        self.move_counter = 1
        self.populate_board()
        self.board_name = board_name
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
            if tmp_coordinate != node.coordinates and  tmp_coordinate[0] >=0 and tmp_coordinate[0] < self.size and tmp_coordinate[1] >= 0 and tmp_coordinate[1] < self.size:
                if self.pawns[tmp_coordinate] not in node.neighbours:
                    if self.verbose:
                        print("Neighbours coordinates: " + str(tmp_coordinate))
                    node.neighbours.append(self.pawns[tmp_coordinate])

    def reset(self, visualize):
        #Reset the board
        self.visualize = visualize
        #self.move_counter = 0
        #self.remove_all_pawns()
        self.set_state(self.initial_state, True)
        #self.compute_all_possible_actions()
        if self.visualize:
            self.graph = self.generate_graph()

    def get_state(self):
        return self.state_t

    def set_state(self, state, recompute_population = False):
        self.state_t = state
        self.set_active_player(state[0,0])
        if recompute_population:
            for i in range(state.shape[1]-1):
                coordinates_1d = i
                coordinates_2d_y = coordinates_1d // self.size
                coordinates_2d_x = coordinates_1d % self.size
                self.pawns[coordinates_2d_y, coordinates_2d_x].populated_by = state[0, coordinates_1d + 1]

    def set_active_player(self, player_id):
        self.active_player = player_id
        self.state_t[0,0] = player_id
    
    def get_next_player(self, active_player = None):
        if active_player is None:
            active_player = self.active_player
        if active_player == 1:
            return 2
        else:
            return 1
    def change_player(self):
        #Use this to change player outside this class
        self.set_active_player(self.get_next_player()) 
        self.move_counter += 1

    def get_next_state(self, action,  state_t = None, change_player = False):
        #return the state t1 from state t taken action t. NB: This will not update the state of the board
        if state_t is None:
            state_t = self.state_t
        next_state = np.zeros(self.state_t.shape) + self.state_t
        next_state[:,1:] += action
        if change_player:
            next_player = self.get_next_player()
            next_state[0,0] = next_player
        if self.verbose:
            print("Get next state ", next_state)
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
        adj_matrix = np.zeros((self.size**2, self.size**2), dtype=int)
        for node in self.pawns.values():
            row = node.coordinates[1] + node.coordinates[0] * self.size
            for adj_node in node.neighbours:
                column = adj_node.coordinates[1] + adj_node.coordinates[0] * self.size
                adj_matrix[row][column] = 1
        
        return adj_matrix

    def generate_graph(self):
        adj_matrix = self.to_numpy_array()
        G = nx.Graph(adj_matrix)
        #Add node attributes for "coordinate" and "is_empty", taken from board "pawns" dictionary
        all_nodes = list(())
        for node in self.pawns.keys():
            all_nodes.append(node)
        for i in range(adj_matrix.shape[0]):
            G.add_node(i, coordinates = self.pawns[all_nodes[i]].coordinates,  populated_by = self.pawns[all_nodes[i]].populated_by, diamond_plan = all_nodes[i][0] + all_nodes[i][1])
        if self.verbose:
            print(adj_matrix)
            print(G.nodes.data())
        return G

    def update_graph(self):
        #Iterate foreach node in the graph and syncronize attributes with the new values
        nodes = list(self.graph.nodes(data=True))
        for node in nodes:
            if self.verbose:
                print(node)
            coordinates = node[1]["coordinates"]
            node[1]["populated_by"] = self.pawns[coordinates].populated_by

    def show_board(self):
        #Plot the graph nodes and edges  with regards of plans (1 different plan per row), then do the necessary flipping and rotation for matching the board to the assigment
        rotation = 180
        position = nx.multipartite_layout(self.graph, subset_key= "diamond_plan", align="horizontal", center=[0,5])
        #Decide the color of each node, based on if it is empty or not
        nodes_color = []
        nodes = list(self.graph.nodes(data=True))
        for node in nodes:
            color = ""
            if node[1]["populated_by"] == 0:                
                color = "#ffffff"
            elif node[1]["populated_by"] == 1:                
                color = "#ff0000"
            elif node[1]["populated_by"] == 2:                
                color = "#000000"
            else:
                color = "#ffff00"
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
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        plt.clf()
        #Print the move 
        font = ImageFont.truetype('arial.ttf', 30)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), self.board_name, (0,0,0), font=font)
        draw.text((0, 60),"Move nr. " + str(self.move_counter), (0,0,0), font=font)
        return img

    def matplotlib_to_pil(self, fig):
        #Convert a Matplotlib object to an PIL Image   
        bufffer = io.BytesIO()
        fig.savefig(bufffer)
        bufffer.seek(0)
        return Image.open(bufffer) 
                    
    def get_all_possible_actions(self, state = None):
        if state is None:
            state = self.state_t
        #Analize the board and check all possible actions. 
        all_actions = list(())
        tot_possible_actions = self.size**2
        for i in range(tot_possible_actions):
            if state[0, i+1] == 0:
                action = np.zeros((1, tot_possible_actions))
                action[0, i] = state[0][0]
                all_actions.append(action)                         
        return all_actions

    def is_goal_state(self, verbose=False):
        if self.board_name== "Rollout":
            print("**************Start goal check *************")
            a = np.zeros((1, 5, 5))
            for n in self.pawns.keys():
                a[0][n] = self.pawns[n].populated_by
            print("check actual board\n", a)
            print("state is ", self.state_t)
            print("active player", self.active_player)
        #Start a DFS from each node on the active player side and check if there is a path to the other side
        visited_nodes = list()
        win_path = list()
        is_win = False
        if self.active_player == 1:
            start_coordinate = lambda i : (0, i)
        else:
            start_coordinate = lambda i : (i, 0)
        for i in range(self.size):
            #print("start coordinates", start_coordinate(i))
            node = self.pawns[start_coordinate(i)]
            if self.verbose:
                print("valuating node ",node.coordinates )
                print("node populated by p", self.active_player, " ", node.populated_by == self.active_player)
            if (node.populated_by == self.active_player) and (node not in visited_nodes):
                is_win = self.DFS_path_check(node, visited_nodes, win_path, verbose=verbose)
            if is_win:
                break
        if self.verbose:
            print(self.get_state()[0,1:].reshape(1,self.size, self.size))
            print("#####Is Goal State for p", self.active_player, "? ", is_win)
            if is_win:
                print("##### Win Path: \n")
                for node in reversed(win_path):
                    print(node.coordinates, "\n")
        return is_win

    def DFS_path_check(self, node, visited_nodes,win_path, verbose = False):
        if self.board_name== "Rollout":
            print("Visited node", node.coordinates)

        if self.verbose:
            print("DFS visited node ", node.coordinates)
        #Perform recursive DFS with a list of visited nodes and domain specific terminal path settings.
        visited_nodes.append(node)
        if node.coordinates[0] == self.size - 1 and self.active_player == 1:
            win_path.append(node)
            return True
        elif node.coordinates[1] == self.size - 1 and self.active_player == 2:
            win_path.append(node)
            return True    
        is_terminal_path = False
        for adj_node in node.neighbours:
            if verbose:
                print("Adj Node ", adj_node.coordinates, " is populated by ", self.active_player, "  ", adj_node.populated_by == self.active_player, "   and is visited before  ", adj_node in visited_nodes)
            if (adj_node.populated_by == self.active_player) and (adj_node not in visited_nodes):
                is_terminal_path = self.DFS_path_check(adj_node, visited_nodes,win_path)
            if is_terminal_path:
                win_path.append(node)
                return True
        return False

    def get_reward(self):
        #return reward for being in state self.state_t at time t for player_id
        if self.is_goal_state():
            if self.active_player == 1:
                return 1
            return -1
        return 0        
    def populate_pawn(self,action, player_id):
        #Populate pawn with player_id dictated by action
        for i in range(action.shape[1]):
            if action[0, i] == player_id:
                move_coordinates_1d = i
                break
        move_coordinates_2d_y = move_coordinates_1d // self.size
        move_coordinates_2d_x = move_coordinates_1d % self.size
        self.pawns[move_coordinates_2d_y, move_coordinates_2d_x].populated_by = player_id

    def update(self, action):
        #Apply the action to the board and change interested nodes propriety such that it can be visualized 
        #return a img frame of the new board state
        if action is not None:
            self.populate_pawn(action, self.active_player)
            self.set_state(self.get_next_state(action), True)    
            #self.set_active_player(self.get_next_player())        
        if self.visualize:
            self.update_graph()
            frame = self.show_board()
            return frame
        return None
