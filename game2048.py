import numpy as np
import random
from tkinter import *
from enum import IntEnum


class Direction(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

class Game:
    """
    Handles game logic and objects.

    The active board is stored in an NP matrix. Instead of storing tile values directly, only the exponent is stored
    """

    def __init__(self):
        self.action_space = Direction
        self.chance_codes = list(range(32))
        self.matrix = np.zeros((4, 4), dtype=int)
        # place first tile
        sel = np.random.randint(4, size=2)
        self.matrix[sel[0], sel[1]] = 1
        self.matrix_rotations = None
        self.update_rotations()

    def update_rotations(self):
        # used for swiping different directions
        # must be called every time self.matrix is re-assigned
        self.matrix_rotations = [self.matrix, np.fliplr(self.matrix), np.rot90(self.matrix, axes=(0, 1)), np.rot90(self.matrix, axes=(1, 0))]

    def get_legal_moves(self):
        is_legal = [True, True, True, True]
        for direction in Direction:
            old_matrix = self.matrix.copy()
            is_legal[direction] = self.move(direction)
            self.set_matrix(old_matrix)
        return is_legal

    def new_game(self):
        self.matrix = np.zeros((4, 4), dtype=int)
        sel = np.random.randint(4, size=2)
        self.matrix[sel[0], sel[1]] = 1  # place first tile
        self.update_rotations()

    def set_matrix(self, matrix):
        self.matrix = matrix.copy()
        self.update_rotations()

    def new_tile(self):
        """
        stick a new tile on the board. 9:1 odds of 2 vs 4 (2^1 or 2^2)
        :return: none
        """
        outcome = self.generate_chance_outcome()
        self.apply_chance_outcome(outcome)

    def is_game_over(self):

        if np.any(self.matrix == 0): # if board has open positions then game is still on
            return False

        pairs = [
            ((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0)),  # updown pairs, col 0
            ((0, 1), (1, 1)), ((1, 1), (2, 1)), ((2, 1), (3, 1)),  # updown pairs, col 1
            ((0, 2), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (3, 2)),  # updown pairs, col 2
            ((0, 3), (1, 3)), ((1, 3), (2, 3)), ((2, 3), (3, 3)),  # updown pairs, col 3
            ((0, 0), (0, 1)), ((1, 0), (1, 1)), ((2, 0), (2, 1)), ((3, 0), (3, 1)),  # leftright pairs, col 0/1
            ((0, 1), (0, 2)), ((1, 1), (1, 2)), ((2, 1), (2, 2)), ((3, 1), (3, 2)),  # leftright pairs, col 1/2
            ((0, 2), (0, 3)), ((1, 2), (1, 3)), ((2, 2), (2, 3)), ((3, 2), (3, 3))  # leftright pairs, col 2/3
        ]

        for p in pairs:  # since board is full, any potential swipes are from adjacent pairs
            if self.matrix[p[0][0]][p[0][1]] == self.matrix[p[1][0]][p[1][1]]:
                return False

        # only reached when board has no empty spaces AND all adjacent cells have different numbers
        return True

    def move(self, direction):
        """
        play the move specified by direction, and add a tile if the move did something
        :param direction: direction to swipe the board
        :return: True/False if the move did something
        """
        old_matrix = self.matrix.copy()  # compare with matrix after move to determine if board has changed

        reward = self.apply_action(direction)
        is_legal = not np.all(np.equal(old_matrix, self.matrix))
        if is_legal:
            self.new_tile()

        return is_legal

    def generate_chance_outcome(self):
        """
        Select whether new tile is a 2 or 4, then select an empty position to place the tile.
        Encode the selections as an integer x from 0-31, where (x<16) indicates (new tile==2),
        and x % 16 gives the position of the tile.
        :return: outcome
        """

        zero_indices = np.flatnonzero(self.matrix.flat == 0)

        if zero_indices.size == 0:
            # if board is full, it doesn't matter what chance code we generate.
            # this happens if we reach a terminal state in mcts
            return 0

        if random.random() > 0.9:  # 90% chance of 1, 10% chance of 2
            base = 16
        else:
            base = 0


        offset = np.random.choice(zero_indices)

        return base + offset

    def apply_chance_outcome(self, outcome: int):
        """
        Add a tile to the board, as specified by the outcome code.
            0-15 indicates a 2 was generated at position (outcome)
            16-31 indicates a 4 was generated at position (outcome-16)
        If the position is already occupied, do nothing.
        :param outcome: integer from 0-31 indicating outcome
        :return: None
        """
        if outcome < 16:
            if self.matrix.flat[outcome] == 0:  # only fails during MCTS where illegal chance outcomes are possible
                self.matrix.flat[outcome] = 1
        else:
            if self.matrix.flat[outcome-16] == 0:
                self.matrix.flat[outcome - 16] = 2

    def apply_action(self, direction):
        """
        flip the board around, swipe left, flip around again. does not generate a new tile
        :param direction: direction to be swiped. string. LEFT RIGHT UP DOWN
        :return: Gain in score resulting from the action
        """

        def lSwipeRow(row):
            """
            Take the input row and push all the tiles to the left. Modifies the row in place

            We make use of the fact that the finalized output will be "full" from the left side. This perspective allows us
            to easily avoid accidentally merging the same tile twice (eg. [4,2,2,0] could mistakenly become [8,0,0,0])

            :param row: ndarray. tiles are pushed left
            """
            reward = 0
            preVal = -1  # value of merge candidate. -1 if no merge candidate
            j = 0  # current write index
            for i in range(0, 4):  # i is current read index
                val = row[i]
                if val != 0:
                    if preVal == val:  # if we have a match then merge
                        row[j - 1] = val + 1  # j-1 is the last thing we wrote to (i.e. merge candidate)
                        preVal = -1
                        reward += val+1
                    else:  # otherwise just copy the current cell over
                        preVal = val
                        row[j] = val
                        j += 1
            row[j:] = 0  # remainder of array is zeros

            return reward

        def lswipe(matrix):
            """
            swipe all rows left. modifies in place
            :param: matrix: board to be swept
            """
            reward = 0
            for row in matrix:
                reward += lSwipeRow(row)
            return reward

        # Move shit around
        view = self.matrix_rotations[direction]
        return lswipe(view)  # return reward

    def __eq__(self, other):
        return self.matrix == other.matrix


class GameWindow:
    fcolours = {
        0: "#aaa69d",
        1: "#2f3542",
        2: "#2f3542",
        3: "white",
        4: "white",
        5: "white",
        6: "white",
        7: "#ffffff",
        8: "#ffffff",
        9: "#ffffff",
        10: "#ffffff",
        11: "#ffffff",
        12: "#ffffff",
        13: "#ffffff",
        14: "#ffffff",
        15: "#ffffff",
        16: "#ffffff",
        17: "#ffffff"
    }
    bcolours = {
        0: "#aaa69d",
        1: "#f7f1e3",
        2: "#f8c291",
        3: "#ffbe76",
        4: "#fa983a",
        5: "#ff6348",
        6: "#e55039",
        7: "#f6e58d",
        8: "#f7d794",
        9: "#ffc048",
        10: "#feca57",
        11: "#fed330",
        12: "#ff9f43",
        13: "#25CCF7",
        14: "#1B9CFC",
        15: "#5352ed",
        16: "#3B3B98",
        17: "#182C61"
    }

    def __init__(self):
        self.root = Tk()
        self.gridFrame = Frame(self.root)
        self.gridFrame.pack()

    def paint(self, game):
        for child in self.gridFrame.winfo_children():
            child.grid_forget()
            child.destroy()

        for i in range(4):
            for j in range(4):
                Label(
                    self.gridFrame,
                    text=1 << game.matrix[i][j],
                    font=("", 30),
                    fg=self.fcolours[game.matrix[i][j]],
                    bg=self.bcolours[game.matrix[i][j]], width=6, height=3
                ).grid(row=i + 1, column=j, padx=2, pady=2)
        self.root.update()
