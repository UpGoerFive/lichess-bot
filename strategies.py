"""
Some example strategies for people who want to create a custom, homemade bot.
And some handy classes to extend
"""

import chess
from chess.engine import PlayResult
import random
from engine_wrapper import EngineWrapper
from tensorflow.keras import models
import numpy as np

class FillerEngine:
    """
    Not meant to be an actual engine.

    This is only used to provide the property "self.engine"
    in "MinimalEngine" which extends "EngineWrapper"
    """
    def __init__(self, main_engine, name=None):
        self.id = {
            "name": name
        }
        self.name = name
        self.main_engine = main_engine

    def __getattr__(self, method_name):
        main_engine = self.main_engine

        def method(*args, **kwargs):
            nonlocal main_engine
            nonlocal method_name
            return main_engine.notify(method_name, *args, **kwargs)

        return method


class MinimalEngine(EngineWrapper):
    """
    Subclass this to prevent a few random errors

    Even though MinimalEngine extends EngineWrapper,
    you don't have to actually wrap an engine.

    At minimum, just implement `search`,
    however you can also change other methods like
    `notify`, `first_search`, `get_time_control`, etc.
    """
    def __init__(self, commands, options, stderr, draw_or_resign, name=None, **popen_args):
        super().__init__(options, draw_or_resign)

        self.engine_name = self.__class__.__name__ if name is None else name

        self.engine = FillerEngine(self, name=self.name)
        self.engine.id = {
            "name": self.engine_name
        }

    def search(self, board, time_limit, ponder, draw_offered):
        """
        The method to be implemented in your homemade engine

        NOTE: This method must return an instance of "chess.engine.PlayResult"
        """
        raise NotImplementedError("The search method is not implemented")

    def notify(self, method_name, *args, **kwargs):
        """
        The EngineWrapper class sometimes calls methods on "self.engine".
        "self.engine" is a filler property that notifies <self>
        whenever an attribute is called.

        Nothing happens unless the main engine does something.

        Simply put, the following code is equivalent
        self.engine.<method_name>(<*args>, <**kwargs>)
        self.notify(<method_name>, <*args>, <**kwargs>)
        """
        pass


class ExampleEngine(MinimalEngine):
    pass


# Strategy names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    def search(self, board, *args):
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    def search(self, board, *args):
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Gets the first move when sorted by uci representation"""
    def search(self, board, *args):
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)


def fen_to_array(fen):
    """
    Takes a FEN board position, and converts it to a 64 x 13 array. The array is structured similarly to the FEN with array[0] representing a8 on the chess board,
    with the rest of the rank following and then each successive rank in descending order, array[-1] is h1.
    """
    board_array = np.zeros(832)

    board_string = ''
    for row in fen.split()[0].split('/'):
        for char in row:
            addon = char if char not in '12345678' else '0' * int(char)
            board_string += addon

    if len(board_string) != 64:
        raise ValueError(f"Board string has incorrect string length of {len(board_string)}")

    # Maps a FEN character to an index
    piece_dict = {'0': 0,
                  'P': 1,
                  'N': 2,
                  'B': 3,
                  'R': 4,
                  'Q': 5,
                  'K': 6,
                  'p': 7,
                  'n': 8,
                  'b': 9,
                  'r': 10,
                  'q': 11,
                  'k': 12}

    for _ in range(64):
        board_array[13 * _ + piece_dict[board_string[_]]] = 1

    return board_array


class PuzzledBot(MinimalEngine):
    def search(self, board, *args):
        """
        Plays best predicted move and displays board.
        """
        model = models.load_model('tuned_model.h5')
        options = list(board.legal_moves)
        fens = []

        for move in options:
            board.push(move)
            fens.append(board.fen())
            board.pop()

        fens = np.array([fen_to_array(fen).reshape(8,8,13) for fen in fens]) # This is slow, but I just need something working first
        move_vals = model.predict(fens)
        best_move = options[np.argmax(move_vals)]

        board.push(best_move)
        return PlayResult(best_move, None)


