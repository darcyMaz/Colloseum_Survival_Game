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
        return self.approach(chess_board, my_pos, adv_pos, max_step)
    
    def approach(self, chess_board, my_pos, adv_pos, max_step):
        
        # Initialize the visited dictionary to minus 1.
        # -1 indicates that a tile hasn't been visited.
        # 0 and positive numbers indicate distance from the start.
        board_size = len(chess_board[0])
        visited = self.init_visited(board_size, -1)
        search_queue = []

        # Define a recursive function for bfs.
        def bfs(chess_board, new_pos, adversary_pos, prev_dist): 
            visited[(new_pos[0], new_pos[1])] = prev_dist + 1

            if new_pos == adversary_pos:
                return
            
            for _ in self.dir_map:
                if chess_board[new_pos[0], new_pos[1], self.dir_map[_]] == False:
                    next_dir = self.move_map[_]
                    next_pos = (new_pos[0] + next_dir[0], new_pos[1] + next_dir[1])
                    # If this tile has never been searched before.
                    if visited[next_pos] == -1:
                        search_queue.append((next_pos, visited[new_pos]))
                    # If this tile has been searched before (>-1) and it's not the start (not 0) and 
                    #   this tile's previous search was from a longer path (new_pos + 1 < next_pos's current distance).
                    # Then change it to reflect the shorter path.
                    elif visited[new_pos] + 1 < visited[next_pos] > 0:
                        search_queue.append((next_pos, visited[new_pos]))
            
        # Add the position of our player to the queue.
        # While the queue is not empty, pop a position to search off the top and call bfs.
        search_queue.append((my_pos, -1))
        while not len(search_queue) == 0:
            next_search = search_queue.pop(0)
            bfs(chess_board, next_search[0], adv_pos, next_search[1])

        # The function call below allows you to see the 'visited' array.
        # Try playing a game with it uncommented to see!
        # self.print_visited(board_size, visited)

        # At this point, the best path is in the visited dict.
        # We just need to find it.
        # Start from the adversary position and move towards the player's position.
        # Always choose the smallest number which represents the shortest distance.
        shortest_path = []
        curr_pos = adv_pos

        while curr_pos != my_pos:
            
            shortest_pos = None
            for _ in self.dir_map:
                search_dir = self.move_map[_]
                search_pos = (curr_pos[0] + search_dir[0], curr_pos[1] + search_dir[1])

                # If this position does not exist in the visited dict then skip.
                # Means its out of bounds, not part of the board.
                if not search_pos in visited:
                    continue

                # If this value on the board was never part of the search.
                # Means it's unreachable.
                if visited[search_pos] == -1:
                    continue

                # If there is a barrier in the way of this search.
                # Continue
                if chess_board[curr_pos[0], curr_pos[1], self.dir_map[_]] == True:
                    continue

                # If the best position to search for hasn't been selected yet (i.e. first of four iterations in the loop)
                #   then select it.
                if shortest_pos == None:
                    shortest_pos = search_pos
                    continue

                # If the value at the shortest position is bigger than the one we're searching over,
                #  that means there is a shorter path along the search_pos. Select it.
                if visited[shortest_pos] > visited[search_pos]:
                    shortest_pos = search_pos
                    # I can make it >= and then when equal flip a coin
                    #   to see if it should be chosen or not.
            
            # The shortest position here is an adjacent position to curr_pos, the closest one to the player.
            
            # Choose it for the next iteration.
            curr_pos = shortest_pos
            # Add to the front of the shortest path array.
            shortest_path.insert(0, curr_pos)

        # This pops the first value in the shortest path which is always the position of
        #   this player.
        shortest_path.pop(0)

        # Choosing the next move based on maxstep.
        # The case where the shortest path is longer than the distance the player can travel.
        if len( shortest_path ) > max_step:
            # Choose the spot on the path that is maxsteps away.
            return_pos = shortest_path[max_step - 1]
            
            # Choosing the direction.
            # This will be based on where the next spot in the shortest_path is.
            next_pos = shortest_path[max_step]
        # The case where the length of the shortest path is equal to or less than the maximum number of steps.
        else:
            if len(shortest_path) == 0:
                return_pos = my_pos
            else:
                return_pos = shortest_path[-1]
            next_pos = adv_pos

        # Return_pos is adjacent to next_pos as they are next to each other on the path.
        # This operation will find the direction of next_pos relative to return_pos, which
        #   will be the position to put the barrier.
        next_dir = (next_pos[0] - return_pos[0], next_pos[1] - return_pos[1])
        next_dir_char = None
        for _ in self.move_map:
            if self.move_map[_] == next_dir:
                next_dir_char = _
                break
        if next_dir_char == None:
            print("There was a problem choosing the barrier for this move. Up is chosen and if it is invalid, a different move is automatically choosen a random move.")

        return return_pos, self.dir_map[next_dir_char]

    # This function returns a dictionary that keeps track of the spots visited in our search.
    # The array can take as input any initializing value. The default value is false.
    # We could also set the default value to 0 and count the steps away from the start.
    def init_visited(self, board_size, value="False"):
        visited = {}
        for r in range( board_size ):
            for c in range( board_size ):
                visited[(r,c)] = value
        return visited
    
    # This is a debugging function which prints the dictionary visited.
    # This dictionary has board_size x board_size tuples, one for each spot on the board.
    def print_visited(self, board_size, visited):
        for r in range( board_size ):
            for c in range( board_size ):
                print( visited[(r,c)], end=" ")
            print()


# Possible Imporvements:
#   It can choose a closest approach which loses the game for itself.
#       I can gather all the possible best approaches and only choose one which doesn't end the game as a loss.
#       I can choose one which, if it results in a losing endgame, uses the random_no_endgame AI instead.
#           This one could run approach 1-3 times then do random no endgame.