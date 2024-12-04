import numpy as np
from copy import deepcopy
from agents.agent import Agent
from store import register_agent

# Important: you should register your agent with a name.
@register_agent("approach_agent")
class ApproachAgent(Agent):
    """
    An agent which approaches adversary as close as possible without ending the game unfavourably.
    """

    def __init__(self):
        super(ApproachAgent, self).__init__()
        self.name = "ApproachAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.move_map = {
            "u": (-1,0),
            "r": (0,1),
            "d": (1,0),
            "l": (0,-1),
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        pass

    # MCST Class
    #   Holds a hash tree of nodes
    #   Selection(): Chooses a new next move at random, puts it in the root container.
    #   Update(): Input is the actual move that was made.
    #             Prune root array, get rid of all the moves except for the one that was made.
    #               Take the children of pruned nodes and put them into the root array.
    #             
    #    i need to figure out how to hold the nodes: (1) root container and each node has its descendants or (2) all nodes container
    #       ok here's an implementation:
    #           root container. sure.
    #           hash table. Each time a node is created, it is hashed. If it exists in the table 
    #               then reject the hashing and reject the creation of this node.
    #           Now we have a root container of all the next moves we care about and we can easily access any node.
    #           So we'll need a hashing formula for each board. Gotta be unique. Based on position of players and barriers.


    # Node Class
    #   Contains a board representing a game state.
    #   Is_end_game() function
    #   Propogate(): Chooses move(s) at random
    #   Whose turn is it is kept track of 

    # Selection
    #   Choose at random
    #   Simulate random games (random no endgame games? yeah)
    # Backpropogation and score tallying for moves
    #   this is all done in the tree automatically
    #   i dont know exactly how it works so itll be fun to imagine
    #   i imagine i have to make a way of creating games as nodes on the tree
    #       so i think i should make a hashcode for each game, not sure for what yet
    #   and the choice of each new node/game step is random
    #   