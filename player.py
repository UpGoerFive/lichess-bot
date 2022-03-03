import numpy as np
import chess
from chess.engine import PlayResult
from strategies import MinimalEngine
from tensorflow.keras import models

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

class Player:
    def __init__(self, model, start_fen=chess.STARTING_FEN):
        """
        Basic player that uses python chess to make moves.
        """
        self.model = models.load_model(model)
        self.set_position(fen=start_fen)
        self.show_board()

    def set_position(self, fen):
        """
        Sets the board position for the Player, useful for having the player play a new game.
        """
        self.board = chess.Board(fen=fen)

    def show_board(self):
        """
        Displays board for play.
        """
        return self.board

    def play_move(self):
        """
        Plays best predicted move and displays board.
        """
        options = list(self.board.legal_moves)
        fens = []

        for move in options:
            self.board.push(move)
            fens.append(self.board.fen())
            self.board.pop()

        fens = np.array([fen_to_array(fen).reshape(8,8,13) for fen in fens]) # This is slow, but I just need something working first
        move_vals = self.model.predict(fens)
        best_move = options[np.argmax(move_vals)]

        self.board.push(best_move)
        self.show_board()
        return best_move

    def op_move(self, move):
        """
        Receives opponent move.
        """
        self.board.push(move) if type(move) == chess.Move else self.board.push_san(move)
        self.show_board()


class Single_Game:
    """
    Plays one game between two Player objects. player_names should be a dictionary with 'white' and 'black'
    as the keys, and the desired names as values.
    """
    def __init__(self, white: Player, black: Player, player_names=None):
        self.players = {'white': white, 'black': black}
        self.active_player = 'white'
        self.passive_player = 'black'

        self.game_board = chess.Board()

        self.players['white'].set_position(self.game_board.fen())
        self.players['black'].set_position(self.game_board.fen())

        if player_names:
            self.names = player_names

    def play(self):
        """
        Plays out the game. Returns basic information.
        """
        ply_num = 0
        switcher = ('white', 'black')
        while not self.game_board.outcome():
            next_passive = ply_num % 2
            move = self.players[self.active_player].play_move()
            self.game_board.push(move)
            self.players[self.passive_player].op_move(move)

            # Advance Ply by one and switch players
            ply_num += 1
            self.active_player = switcher[ply_num % 2]
            self.passive_player = switcher[next_passive]

        return self.game_board.outcome(), self.game_board.move_stack


class PuzzledBot(MinimalEngine):
    def search(self, board, *args):
        """
        Plays best predicted move and displays board.
        """
        model = models.load_model('LongModel.h5')
        options = list(board.legal_moves)
        fens = []

        for move in options:
            board.push(move)
            fens.append(self.board.fen())
            board.pop()

        fens = np.array([fen_to_array(fen).reshape(8,8,13) for fen in fens]) # This is slow, but I just need something working first
        move_vals = self.model.predict(fens)
        best_move = options[np.argmax(move_vals)]

        board.push(best_move)
        return PlayResult(best_move, None)


