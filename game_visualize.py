import argparse
import json
import actor
import mcts as mc
import simworld
import tournament
import replay_buffer
import pickle
import numpy as np
from pathlib import Path
import pygame
from tqdm import tqdm
import matplotlib.pyplot as plt


import torch


# Tools 
def read_config_from_json(filename):
    with open('configs/' + filename, 'r') as fp:
        cfg = json.load(fp)
    return cfg

def dump_config_to_json(filename):
    with open('configs/' + filename, 'w') as fp:
        json.dump(cfg, fp)

def pickle_file(path, filename, obj):
    path = Path(path)
    filepath = path / filename
    f = open(filepath, 'wb')
    pickle.dump(obj, f, -1)
    f.close()

def unpickle_file(path, filename):
    path = Path(path)
    filepath = path / filename
    if filepath.is_file():
        f = open(filepath, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj
    return None

# Convert a PIL object to a Pygame image object
def pil_image_to_pygame(pilImage):
    return pygame.image.fromstring(
        pilImage.tobytes(), pilImage.size, pilImage.mode).convert()

if __name__ == "__main__":

    # Parse config file of choice
    parser = argparse.ArgumentParser("AlphaHex - a MCTS RL solver for HEX")
    parser.add_argument('--config', default="config.json", type=str, help="Select configuration file to load")
    parser.add_argument('--train', default=False, type=bool, help="Choose whether to train actors or not")
    parser.add_argument('--tournament', default=True, type=bool, help="Choose whether to run a tournaments or not")
    parser.add_argument('--actor', default="none", type=str, help="Select actor file to load")

    args = parser.parse_args()
    cfg = read_config_from_json(args.config)
    print(cfg)

    board = simworld.Board(cfg["board_size"], "Main Game", cfg["board_visualize"], cfg["verbose"])
    if args.actor == "none":
        actor = actor.Actor(cfg)
    else: 
        actor = unpickle_file("data/actor", args.actor + ".pkl" )
    buffer = unpickle_file("data/dataset", "buffer_b" + str(board.size) + ".pkl" )
    if buffer is None:
        buffer = replay_buffer.Replay_buffer()
    mcts = mc.MTCS(board.get_state(), actor, buffer, cfg)

    losses = list()
    p1 = 0
    p2 = 0
    if args.train:
        for i in tqdm(range(cfg["episodes"]+1), "Episode ", position = 0):
            if i % (cfg["episodes"] / cfg["actors_to_save"]+1)== 0:
                filename = "actor_b" + str(board.size) + "_ep" + str(actor.trained_episodes) +".pkl"  
                pickle_file("data/actor", filename, actor)
            move = 1


            #print("*****************************************************")
            #print("Move nr. ", move, " - Player ", int(board.active_player))
            #print("Before\n", board.get_state()[0,1:].reshape(1, cfg["board_size"], cfg["board_size"]))
            
            if  cfg["board_visualize"]:
                pygame.init()
                #Show start board, generate an img, get the size and initializate a pygame display
                img = board.update(None)
                X, Y = img.size
                display_surface = pygame.display.set_mode((X,Y)) 
                frame = pil_image_to_pygame(img)
                pygame.display.set_caption('Alpha Hex - Emanuele Caprioli')
                display_surface.blit(frame, (0, 0)) 
                pygame.display.update() 
                pygame.time.delay(cfg["frame_latency"])
                last_pil_frame = None
            is_main_game_goal = board.is_goal_state()
            end_visualization = False
            while (not is_main_game_goal or cfg["board_visualize"]) and not end_visualization :
                if not is_main_game_goal:
                    if move > 1:
                        board.change_player()
                        mcts.prune_tree(choosen_action)
                        #print("*****************************************************")
                        #print("Move nr. ", move, " - Player ", int(board.active_player))
                        #print("Before\n", board.get_state()[0,1:].reshape(1, cfg["board_size"], cfg["board_size"]))
                    action_distribution = mcts.run_simulation()
                    #print("ACTion distrisbution", action_distribution)
                    #hashed_action = max(action_distribution, key= action_distribution.get)
                    #choosen_action = np.expand_dims(np.asarray(hashed_action), axis=0)
                    choosen_action = actor.get_max_action_from_distribution(action_distribution, board.active_player)
                    
                    new_pil_frame = board.update(choosen_action)
                    move += 1
                    #print("After\n", board.get_state()[0,1:].reshape(1, cfg["board_size"], cfg["board_size"]))
                    is_main_game_goal = board.is_goal_state()
                    new_pil_frame = board.update(None) #board.show_board()

                if cfg["board_visualize"]:
                    #Performe the routine for visualization
                    if new_pil_frame != last_pil_frame:
                        new_frame = pil_image_to_pygame(new_pil_frame)
                        last_pil_frame = new_pil_frame
                        # Update the pygame display with the new frames a delay between each frames
                        display_surface.blit(new_frame, (0, 0)) 
                        pygame.display.update() 
                        pygame.time.wait(cfg["frame_latency"])
                    for event in pygame.event.get() :
                        if event.type == pygame.QUIT :
                            pygame.quit()
                            end_visualization = True
                
                
            reward = board.get_reward()
            #print("*****************************************************")
            #print("Episode ", i, " Finished. Reward: ", reward )
            if reward == 1:
                p1 += 1
            elif reward == -1:
                p2 += 1

            #board.reset(False)
            #mcts.reset()
            board = simworld.Board(cfg["board_size"], "Main Game", cfg["board_visualize"], cfg["verbose"])
            mcts = mc.MTCS(board.get_state(), actor, buffer, cfg)
            x_train, y_train = buffer.get_training_dataset()
            buffer.flush_episode()
            loss = actor.train(x_train, y_train)
            losses.append(loss)
            if i % 5 == 0:
                pickle_file("data/dataset", "buffer_b" + str(board.size) + ".pkl", buffer)
            
            #Alternate episode, change p1 or p2 starting
            if i % 2 == 0:
               board.change_player() 

        print("All episodes run. The stats are:")
        print("P1 won : ", p1, " P2 won : ", p2)
        print("Saving Replay Buffer to disc....")
        pickle_file("data/dataset", "buffer_b" + str(board.size) + ".pkl", buffer)
        print("Replay Buffer saved.")

        time = np.linspace(0, len(losses), num=len(losses))
        plt.plot(time, losses)
        plt.show()

    if args.tournament:
        players = []
        for actor_name in cfg["tournament_players"]:
            filename ="actor_b"+str(board.size)+"_ep" + str(actor_name) + ".pkl"
            actor = unpickle_file("data/actor", filename)
            players.append(actor)
        
        '''for i in range(len(players)):
            for j in range(len(players)):
                if i == j:
                    continue
                print("is actor ",players[i].trained_episodes, " the same as player ", players[j].trained_episodes)
                
                params1  = list()
                params2 = list()
                for param in players[i].model.parameters():
                    #print(param)
                    params1.append(param)
                print("##########")
                for param in players[j].model.parameters():
                    #print(param)
                    params2.append(param)
                for k in range(len(params2)):
                    print("¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤")
                    print(params1[k])
                    print(params2[k])

                    print(torch.all(torch.eq(params1[k], params2[k])))'''
        tournament = tournament.Tournament(cfg, players)
        tournament.run()