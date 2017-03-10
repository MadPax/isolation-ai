"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game._ _player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return max_dist_from_center(game, player)

def min_dist_from_center(game, player):

    center = (4,4)
    position = game.get_player_location(player)

    return float(-((position[0]-center[0]) + (position[1]-center[1])))


def max_dist_from_center(game, player):
    center = (4, 4)
    position = game.get_player_location(player)

    return float((position[0] - center[0]) + (position[1] - center[1]))



def closest_to_opponent(game, player):
    if player == game.__player_1__:
        opponent = game.__player_2__
    else:
        opponent = game.__player_1__

    player_position = game.get_player_location(player)
    opponent_position = game.get_player_location(opponent)

    return float(-((player_position[0] - opponent_position[0]) + (player_position[1] - opponent_position[1])))


def furthest_to_opponent(game, player):
    if player == game.__player_1__:
        opponent = game.__player_2__
    else:
        opponent = game.__player_1__

    player_position = game.get_player_location(player)
    opponent_position = game.get_player_location(opponent)

    return float((player_position[0] - opponent_position[0]) + (player_position[1] - opponent_position[1]))


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=25.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.MIN_TIME_LEFT = 20

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        center_position = (4, 4)
        corner_position = (0, 0)
        no_legal_moves = (-1, -1)

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        if len(legal_moves) == 0:
            return no_legal_moves

        if len(legal_moves) == 49:
            return center_position

        if len(legal_moves) == 48:
            if game.move_is_legal(center_position):
                return center_position
            else:
                return corner_position

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.method == "minimax":
                score, move = self.minimax(game, self.search_depth, True)
                return move

            elif self.method == "alphabeta":
                score, move = self.alphabeta(game, self.search_depth, True)
                return move

        except Timeout:
            raise TimeoutError

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        if not self.iterative:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()
            return minimax(self, game, depth, maximizing_player)

        elif self.iterative:
            best_score = None
            best_move = None

            iterative_depth = 1
            while True:
                if self.time_left() < self.TIMER_THRESHOLD:
                    return best_score, best_move
                best_score, best_move = minimax(self, game, iterative_depth, maximizing_player)
                iterative_depth += 1

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if not self.iterative:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()
            return alphabeta(self, game, depth, alpha, beta, maximizing_player)

        elif self.iterative:
            best_score = None
            best_move = None

            iterative_depth = 1
            while True:
                if self.time_left() < self.TIMER_THRESHOLD:
                    return best_score, best_move
                best_score, best_move = alphabeta(self, game, iterative_depth, alpha, beta, maximizing_player)
                iterative_depth += 1

def minimax(player, game, depth, maximizing_player):
    legal_moves = game.get_legal_moves()

    if len(legal_moves) == 0:
        if maximizing_player:
            return float("-inf"), (-1, -1)
        if not maximizing_player:
            return float("inf"), (-1, -1)

    if depth == 1 or player.time_left() < player.MIN_TIME_LEFT:
        if maximizing_player:
            return get_minimax_max_move(player, game, legal_moves)
        else:
            return get_minimax_min_move(player, game, legal_moves)

    else:
        if maximizing_player:
            max_score = float("-inf")
            max_move = None
            for move in legal_moves:
                if player.time_left() < player.MIN_TIME_LEFT:
                    return max_score, max_move
                move_score, placeholder = minimax(player, game.forecast_move(move), depth - 1, False)
                if move_score > max_score:
                    max_score = move_score
                    max_move = move
            return max_score, max_move

        else:
            min_score = float("inf")
            min_move = None
            for move in legal_moves:
                if player.time_left() < player.MIN_TIME_LEFT:
                    return min_score, min_move
                move_score, placeholder = minimax(player, game.forecast_move(move), depth - 1, True)
                if move_score < min_score:
                    min_score = move_score
                    min_move = move
            return min_score, min_move


def alphabeta(player, game, depth, alpha, beta, maximizing_player):

    legal_moves = game.get_legal_moves()

    if len(legal_moves) == 0:
        if maximizing_player:
            return float("-inf"), (-1, -1)
        if not maximizing_player:
            return float("inf"), (-1, -1)

    if depth == 1 or player.time_left() < player.MIN_TIME_LEFT:
        if maximizing_player:
            return get_alphabeta_max_move(player, game, legal_moves, alpha, beta)

        elif not maximizing_player:
            return get_alphabeta_min_move(player, game, legal_moves, alpha, beta)

    else:
        if maximizing_player:
            best_value = alpha
            best_move = None
            for move in legal_moves:
                value, placeholder = alphabeta(player, game.forecast_move(move), depth - 1, alpha, beta, False)
                if value > best_value:
                    best_value = value
                    best_move = move
                    alpha = max(alpha, best_value)
                    if beta <= alpha or player.time_left() < player.MIN_TIME_LEFT:
                        break
            return best_value, best_move
        elif not maximizing_player:
            best_value = beta
            best_move = None
            for move in legal_moves:
                value, placeholder = alphabeta(player, game.forecast_move(move), depth - 1, alpha, beta, True)
                if value < best_value:
                    best_value = value
                    best_move = move
                    beta = min(beta, best_value)
                    if beta <= alpha or player.time_left() < player.MIN_TIME_LEFT:
                        break
            return best_value, best_move



def get_minimax_max_move(player, game, legal_moves):

    max_score = float("-inf")
    max_score_move = None

    for move in legal_moves:
        if player.time_left() < player.MIN_TIME_LEFT:
            return max_score, max_score_move
        score = player.score(game.forecast_move(move), player)
        if score > max_score:
            max_score = score
            max_score_move = move
    return max_score, max_score_move


def get_minimax_min_move(player, game, legal_moves):

    min_score = float("inf")
    min_score_move = None

    for move in legal_moves:
        if player.time_left() < player.MIN_TIME_LEFT:
            return min_score, min_score_move
        score = player.score(game.forecast_move(move), player)
        if score < min_score:
            min_score = score
            min_score_move = move
    return min_score, min_score_move


def get_alphabeta_max_move(player, game, legal_moves, alpha, beta):

    max_score = alpha
    max_score_move = None

    for move in legal_moves:
        if player.time_left() < player.MIN_TIME_LEFT:
            return max_score, max_score_move
        score = player.score(game.forecast_move(move), player)
        if score > max_score:
            max_score = score
            max_score_move = move
            alpha = max(alpha, max_score)
            if beta <= alpha:
                break
    return max_score, max_score_move


def get_alphabeta_min_move(player, game, legal_moves, alpha, beta):

    min_score = beta
    min_score_move = None

    for move in legal_moves:
        if player.time_left() < player.MIN_TIME_LEFT:
            return min_score, min_score_move
        score = player.score(game.forecast_move(move), player)
        if score < min_score:
            min_score = score
            min_score_move = move
            beta = min(beta, min_score)
            if beta <= alpha:
                break
    return min_score, min_score_move
