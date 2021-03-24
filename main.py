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

if __name__ == "__main__":

    # Parse config file of choice
    parser = argparse.ArgumentParser("AlphaHex - a MCTS RL solver for HEX")
    parser.add_argument('--config', default="config.json", type=str, help="Select configuration file to load")
    parser.add_argument('--train', default=False, type=bool, help="Choose whether to train actors or not")
    parser.add_argument('--tournament', default=True, type=bool, help="Choose whether to run a tournaments or not")

    args = parser.parse_args()
    cfg = read_config_from_json(args.config)
    print(cfg)

    buffer = unpickle_file("data/dataset", "buffer.pkl" )
    if buffer is None:
        buffer = replay_buffer.Replay_buffer()
    board = simworld.Board(cfg["board_size"], cfg["board_visualize"], cfg["verbose"])
    actor = actor.Actor(cfg)
    mcts = mc.MTCS(board.get_state(), actor, buffer, cfg)

    p1 = 0
    p2 = 0
    if args.train:
        for i in range(cfg["episodes"]):
            if i % (cfg["episodes"] / cfg["actors_to_save"]) == 0:
                filename = "actor_" + str(actor.trained_episodes) +"_ep.pkl"  
                pickle_file("data/actor", filename, actor)
            move = 1
            print("*****************************************************")
            print("Move nr. ", move, " - Player ", int(board.active_player))
            print("Before\n", board.get_state()[0,1:].reshape(1, cfg["board_size"], cfg["board_size"]))
            while not board.is_goal_state():
                if move > 1:
                    board.change_player()
                    mcts.prune_tree(choosen_action)
                    print("*****************************************************")
                    print("Move nr. ", move, " - Player ", int(board.active_player))
                    print("Before\n", board.get_state()[0,1:].reshape(1, cfg["board_size"], cfg["board_size"]))
                action_distribution = mcts.run_simulation()
                #hashed_action = max(action_distribution, key= action_distribution.get)
                #choosen_action = np.expand_dims(np.asarray(hashed_action), axis=0)
                choosen_action = actor.get_max_action_from_distribution(action_distribution, board.active_player)
                
                board.update(choosen_action)
                move += 1
                print("After\n", board.get_state()[0,1:].reshape(1, cfg["board_size"], cfg["board_size"]))
            reward = board.get_reward()
            print("*****************************************************")
            print("Episode ", i, " Finished. Reward: ", reward )
            if reward == 1:
                p1 += 1
            elif reward == -1:
                p2 += 1

            #board.reset(False)
            #mcts.reset()
            board = simworld.Board(cfg["board_size"], cfg["board_visualize"], cfg["verbose"])
            mcts = mc.MTCS(board.get_state(), actor, buffer, cfg)
            x_train, y_train = buffer.get_training_episode()
            actor.train(x_train, y_train)
        print("All episodes run. The stats are:")
        print("P1 won : ", p1, " P2 won : ", p2)
        print("Saving Replay Buffer to disc....")
        pickle_file("data/dataset", "buffer.pkl", buffer)
        print("Replay Buffer saved.")
    

    if args.tournament:
        players = []
        for actor_name in cfg["tournament_players"]:
            filename ="actor_" + actor_name + "_ep.pkl"
            actor = unpickle_file("data/actor", filename)
            players.append(actor)
        tournament = tournament.Tournament(cfg, players)
        tournament.run(cfg["number_tournament_games"])