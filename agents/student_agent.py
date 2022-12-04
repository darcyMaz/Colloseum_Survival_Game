# Student agent: Add your own agent here
import math
import random
import time
from copy import deepcopy
from collections import defaultdict

from agents.agent import Agent
from store import register_agent

import sys


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.tree = MCTree()
        self.first_iteration = True
        self.node = None

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        if self.first_iteration:
            new_set = set()
            new_set.add(my_pos)
            self.node = Node(chess_board, my_pos, adv_pos, max_step, new_set)
            t_end = time.time() + 28
            while time.time() < t_end:
                self.tree.do_rollout(self.node)
            self.first_iteration = False
        else:
            self.node.board = chess_board
            self.node.adv_pos = adv_pos
            t_end = time.time() + 1.5
            while time.time() < t_end:
                self.tree.do_rollout(self.node)

        move = self.tree.choose(self.node)
        self.node = move
        return move.cur_pos, move.d


class MCTree:
    # Monte Carlo tree searcher. First rollout the tree then choose a move.

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        # Choose the best successor of node.

        if node not in self.children:
            if node.terminal:
                return None

            copy = deepcopy(node)
            is_end, x = copy.is_end_game()
            if is_end:
                node.terminal = True
                return None

            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        # Train for one iteration.
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        # i = 0
        # while i < 5:
        reward = self._simulate(leaf)
        # i = i + 1
        self._backpropagate(path, reward)

    def _select(self, node):
        # Find an unexplored descendent of `node`
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                while unexplored:
                    n = unexplored.pop()
                    nr, nc = n.cur_pos
                    nd = n.d
                    if not node.board[nr, nc, nd]:
                        break
                    self.children[node].remove(n)
                path.append(n)
                return path
            while True:
                n = self._uct_select(node)
                nr, nc = n.cur_pos
                nd = n.d
                if not node.board[nr, nc, nd]:
                    node = n  # descend a layer deeper
                    break
                if n in self.children.keys():
                    self.children.pop(n)
                if n in self.children[node]:
                    self.children[node].remove(n)

    def _expand(self, node):
        # Update the `children` dict with the children of `node`
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        # Returns the reward for a random simulation (to completion) of `node`
        test_node = deepcopy(node)

        # Opponent makes next move
        test_node = Node(test_node.board, test_node.adv_pos, test_node.cur_pos, test_node.max_step,
                         test_node.visited_pos)

        invert_reward = True
        while True:
            term, rew = test_node.is_end_game()
            if term:
                test_node.terminal = True
                reward = rew
                return 1 - reward if invert_reward else reward
            test_node = test_node.find_random_child()
            # Not 100% sure this makes sense...
            test_node = Node(test_node.board, test_node.adv_pos, test_node.cur_pos, test_node.max_step,
                             test_node.visited_pos)
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        # Send the reward back up to the ancestors of the leaf
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        # Select a child of node, balancing exploration & exploitation

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_n_vertex = math.log(self.N[node])

        def uct(n):
            # Upper confidence bound for trees
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_n_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


class Node:
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    def __init__(self, board, cur_pos, adv_pos, max_step, sibling_pos, d=-1, terminal=False):
        self.board = board
        self.terminal = terminal
        self.cur_pos = cur_pos
        self.adv_pos = adv_pos
        self.max_step = max_step
        self.board_size = len(board[0])
        self.d = d
        self.visited_pos = sibling_pos

    def set_curr_pos(self, curr_pos):
        self.cur_pos = curr_pos

    def set_adv_pos(self, adv_pos):
        self.adv_pos = adv_pos

    def find_children(self):
        self.visited_pos = set()
        self.visited_pos.add(self.cur_pos)

        if self.terminal:
            return set()

        copy = deepcopy(self)

        is_end, x = copy.is_end_game()
        if is_end:
            self.terminal = True
            return set()

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}

        # set containing children nodes
        children: set = set()
        children_pos = set()
        children_pos.add(self.cur_pos)

        k = 0
        r, c = self.cur_pos
        while k < 4:
            if not self.board[r, c, k]:
                # if the guard can be placed in the location
                # add step and barrier placement move to set of possible moves
                # next_move.add(((x, y), i))
                new_board = deepcopy(self.board)
                # Set the barrier to True
                new_board[r, c, k] = True
                # Set the opposite barrier to True
                move = moves[k]
                new_board[r + move[0], c + move[1], opposites[k]] = True
                pos2 = (r, c)
                children.add(Node(new_board, pos2, self.adv_pos, self.max_step, children_pos, k))

            k = k + 1

        first_step = True
        # set containing all possible moves
        # possible_moves = set()
        # set containing the possible moves from some step
        curr_step_pos = set()

        i = 0
        # get all possible moves of all possible lengths
        while i < self.max_step:
            if first_step:
                one_step_pos, r_children = self.next_move()
                # possible_moves.update(one_step_moves)
                curr_step_pos.update(one_step_pos)
                children.update(r_children)
                children_pos.update(one_step_pos)
                first_step = False
            else:
                next_step_pos = set()
                for pos in curr_step_pos:
                    new_board = deepcopy(self.board)
                    new_node = Node(new_board, pos, self.adv_pos, self.max_step, children_pos)
                    step_pos, r_children = new_node.next_move()
                    next_step_pos.update(step_pos)
                    children.update(r_children)
                    children_pos.update(next_step_pos)

                curr_step_pos = set()
                curr_step_pos.update(next_step_pos)

            i = i + 1

        # All possible successors of this board state
        return children

    def next_move(self):
        # possible one-step moves
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}

        # set of possible next positions
        next_pos = set()
        # set of possible next positions and guard placement
        # next_move = set()
        # set containing possible children nodes
        next_node = set()

        r, c = self.cur_pos

        # move 1 step up
        m_r, m_c = moves[0]
        new_pos = (r + m_r, c + m_c)
        # check that there isn't a barrier blocking movement
        # and check adversary is not in that new position
        if not self.board[r, c, 0] and not new_pos == self.adv_pos and new_pos not in self.visited_pos:
            # add the possible position to the set
            next_pos.add(new_pos)

        # move 1 step to the right
        m_r, m_c = moves[1]
        new_pos = (r + m_r, c + m_c)
        if not self.board[r, c, 1] and not new_pos == self.adv_pos and new_pos not in self.visited_pos:
            next_pos.add(new_pos)

        # move 1 step down
        m_r, m_c = moves[2]
        new_pos = (r + m_r, c + m_c)
        if not self.board[r, c, 2] and not new_pos == self.adv_pos and new_pos not in self.visited_pos:
            next_pos.add(new_pos)

        # move one step to the left
        m_r, m_c = moves[3]
        new_pos = (r + m_r, c + m_c)
        if not self.board[r, c, 3] and not new_pos == self.adv_pos and new_pos not in self.visited_pos:
            next_pos.add(new_pos)

        # get all possible guard placements for each new position
        for pos in next_pos:
            i = 0
            x, y = pos
            while i < 4:
                if not self.board[x, y, i]:
                    # if the guard can be placed in the location
                    # add step and barrier placement move to set of possible moves
                    # next_move.add(((x, y), i))
                    new_board = deepcopy(self.board)
                    # Set the barrier to True
                    new_board[x, y, i] = True
                    # Set the opposite barrier to True
                    move = moves[i]
                    new_board[x + move[0], y + move[1], opposites[i]] = True
                    pos2 = (x, y)
                    next_node.add(Node(new_board, pos2, self.adv_pos, self.max_step, self.visited_pos, i))

                i = i + 1

        return next_pos, next_node

    def find_random_child(self):

        # Moves (Up, Right, Down, Left)
        ori_pos = deepcopy(self.cur_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        steps = random.randint(0, self.max_step)
        my_pos = self.cur_pos

        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            d = random.randint(0, 3)
            m_r, m_c = moves[d]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while self.board[r, c, d] or my_pos == self.adv_pos:
                k += 1
                if k > 50:
                    break
                d = random.randint(0, 3)
                m_r, m_c = moves[d]
                my_pos = (r + m_r, c + m_c)

            if k > 50:
                my_pos = ori_pos
                break

        # Put Barrier
        d = random.randint(0, 3)
        r, c = my_pos
        while self.board[r, c, d]:
            d = random.randint(0, 3)

        new_board = deepcopy(self.board)
        new_board[r, c, d] = True
        # Set the opposite barrier to True
        move = moves[d]
        new_board[r + move[0], c + move[1], opposites[d]] = True

        visited = set()
        visited.add(self.cur_pos)

        return Node(new_board, my_pos, self.adv_pos, self.max_step, visited, d)

    def is_end_game(self):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Union-Find
        father = dict()
        for r in range(self.board_size):
            for c in range(self.board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self.board_size):
            for c in range(self.board_size):
                for d, move in enumerate(
                        moves[1:3]
                ):  # Only check down and right
                    if self.board[r, c, d + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))
        p0_r = find(tuple(self.cur_pos))
        p1_r = find(tuple(self.adv_pos))

        if p0_r == p1_r:
            return False, 0.0

        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)

        if p0_score > p1_score:
            return True, 1.0
        elif p0_score < p1_score:
            return True, 0.0
        else:
            return True, 0.5

    def is_terminal(self):
        # Returns True if the node has no children
        return self.terminal
