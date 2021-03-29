import simworld
import actor
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygame
class Tournament():
    
    def __init__(self, cfg, players):
        self.players = players
        # A list of touples of already played games [(p1, p2)...]
        self.played_games = list()
        self.wins = [0] * (len(players))
        self.loss = [0] * (len(players))
        self.board_size = cfg["board_size"]
        self.visualize = cfg["tournament_visualize"]
        self.frame_latency = cfg["frame_latency_tournament"]
        self.games = cfg["number_tournament_games"]
    def run(self):
        print(self.players[0].trained_episodes)
        print(self.players[1].trained_episodes)
        game_nr =0
        for i in range (len(self.players)):
            p1 = i
            for j in range(len(self.players)):
                if i == j or (i, j) in self.played_games:
                    continue
                self.played_games.append((i, j))
                self.played_games.append((j, i))
                p2 = j
                player_1 = self.players[p1]
                player_2 = self.players[p2]
                flip_reward = False
                for k in range(self.games):
                    game_nr += 1
                    print("p1 ", p1)
                    print("p2 ", p2)
                    print("Player 1: ", player_1.trained_episodes)
                    print("Player 2: ", player_2.trained_episodes)
                    reward = self.play_game(game_nr, player_1, player_2)

                    if (reward == 1 and not flip_reward) or (reward == -1 and flip_reward):
                        self.wins[i] += 1 
                        self.loss[j] += 1 
                    elif (reward == -1 and not flip_reward) or (reward == 1 and flip_reward):
                        self.wins[j] += 1 
                        self.loss[i] += 1 

                    tmp = player_1
                    player_1 = player_2
                    player_2 = tmp
                    flip_reward = not flip_reward     
                    print("so lenge\n")
                    print("wins ", self.wins)
                    print("loss ", self.loss)
                    print("###############")
        self.plot_results()

    def play_game(self, game_nr, p1, p2):
        board = simworld.Board(self.board_size,"Tournament", self.visualize, False)
        move = 1
        print("******************** , Game ", str(game_nr), " ********************")
        #print("Move nr. ", move, " - Player ", int(board.active_player))
        #print("Before\n", board.get_state()[0,1:].reshape(1, self.board_size, self.board_size))
        
        
        if  self.visualize:
                pygame.init()
                #Show start board, generate an img, get the size and initializate a pygame display
                img = board.update(None)
                X, Y = img.size
                display_surface = pygame.display.set_mode((X,Y)) 
                frame = pil_image_to_pygame(img)
                pygame.display.set_caption('Alpha Hex - Emanuele Caprioli')
                display_surface.blit(frame, (0, 0)) 
                pygame.display.update() 
                pygame.time.delay(self.frame_latency)
                last_pil_frame = None

        is_main_game_goal = board.is_goal_state()
        end_visualization = False
        while (not is_main_game_goal or self.visualize) and not end_visualization:
            if not is_main_game_goal:
                if move > 1:
                    board.change_player()
                    #print("*****************************************************")
                    #print("Move nr. ", move, " - Player ", int(board.active_player))
                    #print("Before\n", board.get_state()[0,1:].reshape(1, self.board_size, self.board_size))
                possible_actions =  board.get_all_possible_actions()
                state = board.get_state()
                if board.active_player == 1:
                    #print(p1.trained_episodes)
                    choosen_action = p1.get_action(state, possible_actions)
                elif board.active_player == 2:
                    #print(p2.trained_episodes)
                    choosen_action = p2.get_action(state, possible_actions)
                board.update(choosen_action)
                move += 1
                #print("After\n", board.get_state()[0,1:].reshape(1, self.board_size, self.board_size))
                is_main_game_goal = board.is_goal_state()
                new_pil_frame = board.update(None) #board.show_board()
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
        print(np.reshape(board.get_state()[:,1:], (1, board.size, board.size)))
        #print("is goal state start ", is_main_game_goal)
        reward = board.get_reward()
        #print("*****************************************************")
        print("Game finished. Reward: ", reward )
        return reward
    
    def plot_results(self):
        labels = []
        for player in self.players:
            labels.append("b"+ str(self.board_size) + "_a" + str(player.trained_episodes))
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, self.wins, width, label='Wins', color = "seagreen")
        rects2 = ax.bar(x - width/2, self.loss, width, label='Losses', bottom=self.wins, color = "salmon")

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Games')
        ax.set_title('Tournament Results')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        autolabel(rects1, ax)
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

# Convert a PIL object to a Pygame image object
def pil_image_to_pygame(pilImage):
    return pygame.image.fromstring(
        pilImage.tobytes(), pilImage.size, pilImage.mode).convert()