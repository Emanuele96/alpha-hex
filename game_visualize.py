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
from torch.utils.data import DataLoader
import torch
import random
import copy
import math

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
    parser.add_argument('--tournament', default=False, type=bool, help="Choose whether to run a tournaments or not")
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
        buffer = replay_buffer.Replay_buffer(cfg["buffer_type"])

    all_agents = unpickle_file("data/actor", "all_agents_b" + str(board.size) + ".pkl" )
    if all_agents is None:
        all_agents = list()
        all_agents.append(copy.deepcopy(actor))

    print(all_agents)
    #Set up players
    p1 = actor
    if cfg["random_adversary_training"] and random.random() < cfg["random_adversary_probability"]:
        p2 = random.choice(all_agents[math.floor(cfg["only_last_adversary"]*len(all_agents)):])
    else:
        p2 = actor
    print("\n", p1.trained_episodes, " playing against ", p2.trained_episodes)

    mcts = mc.MTCS(board.get_state(), p1, p2, buffer, cfg)


    losses = list()
    p1_wins = 0
    p2_wins = 0
    if args.train:
        for i in tqdm(range(cfg["episodes"]+1), "Episode ", position = 0):
            if i % (cfg["episodes"] / cfg["actors_to_save"]+1)== 0 and i!=0:
                filename = "actor_b" + str(board.size) + "_ep" + str(actor.trained_episodes) +".pkl"  
                copied_actor = copy.deepcopy(actor)
                pickle_file("data/actor", filename, copied_actor)
                all_agents.append(copied_actor)
                pickle_file("data/actor", "all_agents_b" + str(board.size) + ".pkl", all_agents)
            move = 1

            
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

                    action_distribution = mcts.run_simulation()

                    choosen_action = actor.get_max_action_from_distribution(action_distribution, board.active_player)
                    
                    new_pil_frame = board.update(choosen_action)
                    move += 1
                    is_main_game_goal = board.is_goal_state()
                    new_pil_frame = board.update(None)

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

            if reward == 1:
                p1_wins += 1
            elif reward == -1:
                p2_wins += 1

            # Start new board and new players
            board = simworld.Board(cfg["board_size"], "Main Game", cfg["board_visualize"], cfg["verbose"])
            if cfg["random_adversary_training"] and random.random() < cfg["random_adversary_probability"]:
                if random.random() <0.3:
                    unpickle_file("data/actor", "actor_b6_ep99999.pkl" )
                else:
                    p2 = random.choice(all_agents[math.floor(cfg["only_last_adversary"]*len(all_agents)):])
            else:
                p2 = actor
            print("\n", p1.trained_episodes, " playing against ", p2.trained_episodes)

            #Alternate episode, change p1 or p2 starting
            if random.random() < 0.5: #i % 2 == 0:
               board.change_player()
                
            mcts = mc.MTCS(board.get_state(), p1, p2, buffer, cfg)
            if cfg["training_type"] == "episode":
                x_train, y_train = buffer.get_training_episode()
                buffer.flush_episode()
                loss = actor.episode_train(x_train, y_train)
                losses.append(loss)
            elif cfg["training_type"] == "full_minibatch":
                buffer.flush_episode()
                train_data = buffer.get_training_dataset()
                train_loader = DataLoader(dataset=train_data, batch_size=cfg["minibatch_size"], shuffle=True)
                losses.extend(actor.full_train(train_loader, cfg["n_epochs"]))
            
            if i % math.floor(cfg["clear_buffer_after_episode"] * cfg["episodes"]) == 0:
                buffer.clear(cfg["clear_buffer_amount"])

            if i % 5 == 0:
                pickle_file("data/dataset", "buffer_b" + str(board.size) + ".pkl", buffer)
            print("buffer size: ", len(buffer.long_time_dataset))

        print("All episodes run. The stats are:")
        print("Actor won : ", p1_wins, " Contester won : ", p2_wins)
        print("Saving Replay Buffer to disc....")
        pickle_file("data/dataset", "buffer_b" + str(board.size) + ".pkl", buffer)
        print("Replay Buffer saved.")

        time = np.linspace(0, len(losses), num=len(losses))
        plt.plot(time, losses)
        plt.show()

    if args.tournament:
        if cfg["tournament_few_players"]:
            all_agents = []
            for actor_name in cfg["tournament_players"]:
                filename ="actor_b"+str(board.size)+"_ep" + str(actor_name) + ".pkl"
                actor = unpickle_file("data/actor", filename)
                all_agents.append(actor)
        else:
            all_agents = unpickle_file("data/actor", "all_agents_b" + str(board.size) + ".pkl" )

        tournament = tournament.Tournament(cfg, all_agents)
        tournament.run()