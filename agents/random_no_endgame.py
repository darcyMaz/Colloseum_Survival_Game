import numpy as np
from copy import deepcopy
from agents.agent import Agent
from store import register_agent

# Important: you should register your agent with a name.
@register_agent("random_no_endgame")
class RandomNoEndgame(Agent):
    """
    An agent which moves randomly but tries to avoid choosing a game that will cause an endgame.
    It randomly chooses a move, sees if its an endgame, and then avoids it if it is.
    If 8 moves in a row are chosen which lead to endgames then it gives up and ends the game on another random choice.

    """

    def __init__(self):
        super(RandomNoEndgame, self).__init__()
        self.name = "RandomNoEndgame"
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

        value = 0
        while value < 9:
            new_pos, new_dir = self.random_step(chess_board, my_pos, adv_pos, max_step)

            temp_board = deepcopy(chess_board)
            temp_board[new_pos[0], new_pos[1], new_dir] = True

            if self.is_end_game(temp_board, new_pos, adv_pos):
                value = value + 1
                continue

            return new_pos, new_dir
        
        return self.random_step(chess_board, my_pos, adv_pos, max_step)



    def random_step(self, chess_board, my_pos, adv_pos, max_step):
        # Moves (Up, Right, Down, Left)
        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_step + 1)

        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir

    # This function checks to see if a chess_board is in a finished state.
    # I.e. whether the board has been partitioned such that player A can never reach player B.
    # This will be done via a Depth First Search starting from this player.
    # The search looks for the other player.
    # If the search ends without finding them return true. If the search finds them return false.
    def is_end_game(self, chess_board, my_pos, adv_pos):
        # The visited dictionary maps the whole chess_board as not having been visited by the search.
        # As the search progresses, we can use the dict to mark down what has been visited and avoid repeats.
        visited = {}
        for r in range( len(chess_board[0]) ):
            for c in range( len(chess_board[0]) ):
                visited[(r,c)] = False


        def dfs(new_pos):
            # Right away, we mark this spot as visited.
            visited[new_pos] = True

            # If we've reached the position of the adversary than it is not an endgame.
            if new_pos == adv_pos:
                return False

            toReturn = True

            # for each direction possible where _ is up, right, left, or down and d is 0,1,2,3.
            for _ in self.dir_map:
                # If the barrier at the new position in the direction d does not exist.    
                # print("current position", new_pos, _, "if statement: ", chess_board[new_pos[0], new_pos[1], self.dir_map[_]])
                if chess_board[new_pos[0], new_pos[1], self.dir_map[_]] == False:
                    next_dir = self.move_map[_]
                    # Next position is the current position plus the direction we're moving.
                    next_pos = (new_pos[0] + next_dir[0], new_pos[1] + next_dir[1])
                    # If this spot is already visited skip over it.
                    if visited[next_pos] == False:
                        # Having a variable called to_return allows us to check all directions.
                        # If we just did return bfs(next_pos) then the search may end prematurely
                        #   if the recursive loop returns to the first iteration without having
                        #   done a full search.
                        if not dfs(next_pos):
                            toReturn = False
                            # Break is important. The search must end once we find the adversary.
                            # If we don't break out of the search once its found, we will do a full
                            # search of the partition when it is not necessary.
                            break
            
            # If the search arrives at a spot on the board where all neighbors have a barrier or have been visited, then return True.
            # If the adversary is never found, then the first call of this recursive function returns True.
            return toReturn

        return dfs(my_pos)