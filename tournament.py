import simworld
import actor
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
class Tournament():
    
    def __init__(self, cfg, players):
        self.players = players
        # A list of touples of already played games [(p1, p2)...]
        self.played_games = list()
        self.wins = [0] * (len(players))
        self.loss = [0] * (len(players))
        self.board_size = cfg["board_size"]
    def run(self, games):
        for i in range (len(self.players)):
            p1 = i
            for j in range(len(self.players)):
                if i == j:
                    break
                p2 = j
                for k in range(games):
                    reward = self.play_game(False, self.players[p1], self.players[p2])
                    if reward == 1:
                        self.wins[i] += 1
                        self.loss[j] += 1
                    elif reward == -1:
                        self.wins[j] += 1
                        self.loss[i] += 1              
        self.plot_results()

    def play_game(self, visualize, p1, p2):
        board = simworld.Board(self.board_size, visualize, False)
        move = 1
        print("*****************************************************")
        print("Move nr. ", move, " - Player ", int(board.active_player))
        print("Before\n", board.get_state()[0,1:].reshape(1, self.board_size, self.board_size))
        while not board.is_goal_state():
            if move > 1:
                board.change_player()
                print("*****************************************************")
                print("Move nr. ", move, " - Player ", int(board.active_player))
                print("Before\n", board.get_state()[0,1:].reshape(1, self.board_size, self.board_size))
            possible_actions =  board.get_all_possible_actions()
            state = board.get_state()
            if board.active_player == 1:
                choosen_action = p1.get_action(state, possible_actions)
            elif board.active_player == 2:
                choosen_action = p2.get_action(state, possible_actions)
            board.update(choosen_action)
            move += 1
            print("After\n", board.get_state()[0,1:].reshape(1, self.board_size, self.board_size))
        reward = board.get_reward()
        print("*****************************************************")
        print("Game finished. Reward: ", reward )
        return reward
    
    def plot_results(self):
        labels = []
        for player in self.players:
            labels.append("a_" + str(player.trained_episodes))
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, self.wins, width, label='Wins')
        rects2 = ax.bar(x - width/2, self.loss, width, label='Losses', bottom=self.wins)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Games')
        ax.set_title('Tournament Results')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        #autolabel(rects1, ax)
        #autolabel(rects2, ax)

        fig.tight_layout()

        plt.show()

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

