"""
Monte Carlo Tic-Tac-Toe Player
"""

import random
import ttt_provided as provided

# Constants for Monte Carlo simulator
# Change as desired
NTRIALS = 500 # Number of trials to run
MCMATCH = 1.0  # Score for squares played by the machine player
MCOTHER = 1.0  # Score for squares played by the other player

def mc_trial(board, player):
	"""
	Play a game with the given Player and modify the board input.
	"""
	while not board.check_win() == 'None':
		# Get all empty squares
		empty_squares = board.get_empty_squares()

		# If there is no empty square, stop the game
		if len(empty_squares) == 0:
			break

		# Randomly choose a random square to try
		random_empty_square = empty_squares[random.randrange(len(empty_squares))]
		row = random_empty_square[0]
		col = random_empty_square[1]

		# Decide the move
		board.move(row, col, player)

		# Switch the player
		player = provided.switch_player(player)

def mc_update_scores(scores, board, player):
	"""
	Score the completed board and update the scores grid.
	"""
	# Keep current scores if the board is a draw
	if board.check_win() == provided.DRAW:
		return None
	
	dummy_row = 0
	dummy_col = 0
	machine_score = MCMATCH
	other_score = MCOTHER
	if player == board.check_win():
		other_score = -MCOTHER
	else:
		machine_score = -MCMATCH
	other_player = provided.switch_player(player)

	# Iterate over board and update scores
	while dummy_row < board.get_dim():
		if board.square(dummy_row, dummy_col) == player:
			scores[dummy_row][dummy_col] += machine_score
		elif board.square(dummy_row, dummy_col) == other_player:
			scores[dummy_row][dummy_col] += other_score
		else:
			scores[dummy_row][dummy_col] += 0
		if dummy_col < (board.get_dim() - 1):
			dummy_col += 1
		else:
			dummy_col = 0
			dummy_row += 1

def get_best_move(board, scores):
	"""
	Find all of the empty squares with the maximum score and
	randomly return one as a (row, column) tuple.
	"""
	# Return error message when board is full
	if not board.get_empty_squares():
		return None

	# Get the best move
	best_move = None

	# Get an empty squares
	empty_squares = board.get_empty_squares()

	# Update scores for empty squares
	highest_score = -1000
	for dummy_square in empty_squares:
		row = dummy_square[0]
		col = dummy_square[1]
		if scores[row][col] >= highest_score:
			highest_score = scores[row][col]
			best_move = (row, col)
	return best_move

def mc_move(board, player, trials):
	"""
	Use Monte Carlo simulation to return a move for the machine player
	in the form of a (row, column) tuple.
	"""
	move = None

	# scores is a board like list of rewards for each cell.
	scores = [[0 for dummy_col in range(board.get_dim())] for 
		dummy_row in range(board.get_dim())]

	for dummy_index in range(trials):
		temp_board = board.clone()

		# Get rewards from trials.
		mc_trial(temp_board, player)
		
		# Update rewards for each cell.
		mc_update_scores(scores, temp_board, player)
	move = get_best_move(board, scores)
	return move

def ttt_play():
	return provided.play_game(mc_move, NTRIALS, False)

# for testing
# if __name__ == '__main__':
# 	ttt_play()
