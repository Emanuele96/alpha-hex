# Alpha HEX 

## Project Description
This project is a implementation of AlphaGo algorithm for solving the game of HEX.

The reinforcement learner will play against himself and previous versions of himself in order to generate training case to train on. No prior domain knowledge has been given to the model, except the rule of HEX.
Each move in the main game will be treated as a root node of a tree: Monte Carlo Tree Search and Rollouts will be used to expand the tree and a specific tree policy will be used to traversate the tree. 
When reached a leaf node, the program will run rollout, playing the game from that state, choosing the next move using the learner policy.

After the MCTS simulation has been completed, a distribution of visits from the root node will be generated and saved in a buffer as a pair (root state, distribution), and used later under training.
After a game/episode is completed, the learner retrieves a batch from the buffer and performes batch learning to improve his policy. The first elements of the buffer will removed with time.

### Models
Some pretrained model that gave good results can be found under the "models" folder. 

## Installation Guide
The project is developed used Python version 3.7, but newer version should be compatible too. Compatibility with Python 2.7 is not assured.
### CUDA Toolkit installation
CUDA Toolkit is necessary if you want to utilize your NVIDIA GPU for tensors operations.
Check if you have a compatibel GPU here: https://developer.nvidia.com/cuda-gpus
CUDA Toolkit used when development on this project: CUDA 10.2
Other versions should work too but not assured.
Installation guide for Windows and Linux: https://docs.nvidia.com/cuda/index.html#installation-guides

### Pythorch installation
If pythorch is not installed, follow the guide found on the official Pythorch website for installing the correct version for your machine.
Different versions are available, both with support of different CUDA versions and only CPU support.
Pythorch packages version used when developing this project: "torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0"
Official starting guide: https://pytorch.org/get-started/locally/

### Install required packages
Run "pip install -r requirements.txt" for installing the lasts of the required packages for this project.

If you use a Python version older then  3.2, run this program for installing ulterior needed packages:
"pip install argparse pathlib".

## User Guide
Run "main.py" for starting the program. The "main.py" script accepts the following parameters:
* **--actor** (""filename without .pkl") The name of the model to continue training from. Default is None.
* **--train** (True or false) Wether to train or not. Default is False.
* **--tournament** (True or false) Wether to run a tournament or not. Default is False.

### Some examples of use:

* "**python main.py --train True**". This will run the program in "train" mode without loading a previous model. 
* "**python main.py --train True --actor actor_b6_ep3560**". The program will load a previously trained model continue training.
* "**python main.py --tournament True**". With this command, the program will then conduct a tournament.

### Config
A config file that gave good results
```
config.json
...
{
    "verbose" : 0,
    "board_visualize": false, 
    "rollout_visualize": false,
    "tournament_visualize":false,
    "frame_latency": 0,
    "frame_latency_rollout":100,
    "frame_latency_tournament":100,
    "board_size" : 6,
    "episodes": 1000,
    "anet_lr": 5e-4,
    "buffer_type": "list",
    "clear_buffer_after_episode":0.1,
    "clear_buffer_amount":0.10,
    "training_type" : "full_minibatch",
    "random_adversary_training": true,
    "random_adversary_probability":0.2,
    "only_last_adversary": 0.8,
    "minibatch_size": 16,
    "n_epochs":5,
    "anet_layers": [

        {
            "neurons": 50,
            "activation": "selu"
        },
        {
            "neurons": 50,
            "activation": "selu"
        },
        {
            "neurons": 36,
            "activation": "linear"
        }
    ],
    "loss":"bce",
    "anet_optim" : "adam",
    "number_of_simulations": 1000,
    "number_tournament_games": 2,
    "actors_to_save":500,
    "tournament_few_players": false,
    "tournament_players":[2774,99999],
    "use_cuda":false
}
```
### Pictures
![MCTS1](https://github.com/Emanuele96/alpha-hex/blob/main/images/b2_mcts_0.png)
MCTS with a board of 2x2 for simplicity. First step

![MCTS2](https://github.com/Emanuele96/alpha-hex/blob/main/images/b2_mcts_1.png)
MCTS with a board of 2x2 for simplicity. Second step

![MCTS3](https://github.com/Emanuele96/alpha-hex/blob/main/images/b2_mcts_1.png)
MCTS with a board of 2x2 for simplicity. Third step

![MCTS4](https://github.com/Emanuele96/alpha-hex/blob/main/images/b2_mcts_3.png)
MCTS with a board of 2x2 for simplicity. Fourth step
