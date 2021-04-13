import math
import numpy as np
import actor
from pathlib import Path
import pickle
from BasicClientActorAbs import BasicClientActorAbs

class BasicClientActor(BasicClientActorAbs):

    def __init__(self, player, IP_address=None, verbose=True):
        self.series_id = -1
        self.player = player
        BasicClientActorAbs.__init__(self, IP_address, verbose=verbose)

    def handle_get_action(self, state):
        """
        Here you will use the neural net that you trained using MCTS to select a move for your actor on the current board.
        Remember to use the correct player_number for YOUR actor! The default action is to select a random empty cell
        on the board. This should be modified.
        :param state: The current board in the form (1 or 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), where
        1 or 2 indicates the number of the current player.  If you are player 2 in the current series, for example,
        then you will see a 2 here throughout the entire series, whereas player 1 will see a 1.
        :return: Your actor's selected action as a tuple (row, column)
        """

        # This is an example player who picks random moves. REMOVE THIS WHEN YOU ADD YOUR OWN CODE !!
        #next_move = tuple(self.pick_random_free_cell(
        #    state, size=int(math.sqrt(len(state)-1))))
        #############################
        #
        #
        # YOUR CODE HERE
        #
        state_array = np.asarray(state).reshape(1,-1)
        all_possible_actions = self.get_all_possible_actions(state_array)
        next_action = player.get_action(state_array, all_possible_actions)

        '''print("*************************")
        print("state ", state_array)
        print("all possib. actions ", all_possible_actions)
        print("choosen action ", next_action)'''

        #reshape action from (1,36) to (6,6)
        next_action = next_action.reshape((6,6))
        row = None
        column = None
        for i in range(6):
            for j in range(6):
                if next_action[i][j] != 0:
                    row = i
                    column = j
                    break
        next_move = (row, column)
        #print("reshaped action ", next_action)
        #print("result ", next_move)
        print("I'm player ", state[0])
        ##############################
        return next_move

    def get_all_possible_actions(self, state):
        #Analize the board and check all possible actions. 
        all_actions = list(())
        tot_possible_actions = 36
        for i in range(tot_possible_actions):
            if state[0, i+1] == 0:
                action = np.zeros((1, tot_possible_actions))
                action[0, i] = state[0][0]
                all_actions.append(action)   
        return all_actions

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """
        Set the player_number of our actor, so that we can tell our MCTS which actor we are.
        :param unique_id - integer identifier for the player within the whole tournament database
        :param series_id - (1 or 2) indicating which player this will be for the ENTIRE series
        :param player_map - a list of tuples: (unique-id series-id) for all players in a series
        :param num_games - number of games to be played in the series
        :param game_params - important game parameters.  For Hex = list with one item = board size (e.g. 5)
        :return

        """
        self.series_id = series_id
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_start(self, start_player):
        """
        :param start_player: The starting player number (1 or 2) for this particular game.
        :return
        """
        self.starting_player = start_player
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_over(self, winner, end_state):
        """
        Here you can decide how to handle what happens when a game finishes. The default action is to print the winner and
        the end state.
        :param winner: Winner ID (1 or 2)
        :param end_state: Final state of the board.
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        ##############################
        print("Game over, these are the stats:")
        print('Winner: ' + str(winner))
        print('End state: ' + str(end_state))

    def handle_series_over(self, stats):
        """
        Here you can handle the series end in any way you want; the initial handling just prints the stats.
        :param stats: The actor statistics for a series = list of tuples [(unique_id, series_id, wins, losses)...]
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Series ended, these are the stats:")
        print(str(stats))

    def handle_tournament_over(self, score):
        """
        Here you can decide to do something when a tournament ends. The default action is to print the received score.
        :param score: The actor score for the tournament
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Tournament over. Your score was: " + str(score))

    def handle_illegal_action(self, state, illegal_action):
        """
        Here you can handle what happens if you get an illegal action message. The default is to print the state and the
        illegal action.
        :param state: The state
        :param action: The illegal action
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("An illegal action was attempted:")
        print('State: ' + str(state))
        print('Action: ' + str(illegal_action))

def unpickle_file(path, filename):
    path = Path(path)
    filepath = path / filename
    if filepath.is_file():
        f = open(filepath, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj
    return None

if __name__ == '__main__':

    p_name = "actor_b6_ep990"
    player = unpickle_file("data/actor", p_name + ".pkl" )
    bsa = BasicClientActor(player, verbose=True)
    bsa.connect_to_server()
