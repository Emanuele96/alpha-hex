import argparse
import json
import actor
import mcts
import simworld
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
    args = parser.parse_args()
    cfg = read_config_from_json(args.config)
    print(cfg)

    buffer = unpickle_file("dataset", "buffer.pkl" )
    if buffer is None:
        buffer = replay_buffer.Replay_buffer()
    board = simworld.Board(cfg["board_size"], cfg["board_visualize"], cfg["verbose"])
    actor = actor.Actor(cfg)
    mcts = mcts.MTCS(board.get_state(), actor, buffer, cfg)
    for i in range(cfg["episodes"]):
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
            hashed_action = max(action_distribution, key= action_distribution.get)
            choosen_action = np.expand_dims(np.asarray(hashed_action), axis=0)
            board.update(choosen_action)
            move += 1
            print("After\n", board.get_state()[0,1:].reshape(1, cfg["board_size"], cfg["board_size"]))
        print("*****************************************************")
        print("Episode ", i, " Finished. Reward: ", board.get_reward())
        board.reset(False)
        mcts.reset()
        x_train, y_train = buffer.get_training_dataset()
        #actor.train(x_train, y_train)
    print("Saving Replay Buffer to disc....")
    pickle_file("dataset", "buffer.pkl", buffer)
    print("Replay Buffer saved.")
    