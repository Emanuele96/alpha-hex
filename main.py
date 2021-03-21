import argparse
import json
import actor
import mcts
import simworld

# Tools 
def read_config_from_json(filename):
    with open('configs/' + filename, 'r') as fp:
        cfg = json.load(fp)
    return cfg

def dump_config_to_json(filename):
    with open('configs/' + filename, 'w') as fp:
        json.dump(cfg, fp)



if __name__ == "__main__":

    # Parse config file of choice
    parser = argparse.ArgumentParser("AlphaHex - a MCTS RL solver for HEX")
    parser.add_argument('--config', default="config.json", type=str, help="Select configuration file to load")
    args = parser.parse_args()
    cfg = read_config_from_json(args.config)
    print(cfg)


    board = simworld.Board(cfg["board_size"], cfg["board_visualize"], cfg["verbose"])
    actor = actor.Actor()
    mcts = mcts.MTCS(board.get_state(), actor, cfg)
    while not board.is_goal_state():
        #possible_actions = board.get_all_possible_actions()
        choosen_action = mcts.run_simulation()
        board.update(choosen_action)
    print(board.get_reward())
