import random, time
import pygame
import math
from connect4 import connect4
import sys
from copy import deepcopy


class connect4Player(object):
	def __init__(self, position, seed=0, CVDMode=False):
		self.position = position
		self.opponent = None
		self.seed = seed
		random.seed(seed)
		if CVDMode:
			global P1COLOR
			global P2COLOR
			P1COLOR = (227, 60, 239)
			P2COLOR = (0, 255, 0)

	def play(self, env: connect4, move_dict: dict) -> None:
		move_dict["move"] = -1

class humanConsole(connect4Player):
	'''
	Human player where input is collected from the console
	'''
	def play(self, env: connect4, move_dict: dict) -> None:
		move_dict['move'] = int(input('Select next move: '))
		while True:
			if int(move_dict['move']) >= 0 and int(move_dict['move']) <= 6 and env.topPosition[int(move_dict['move'])] >= 0:
				break
			move_dict['move'] = int(input('Index invalid. Select next move: '))

class humanGUI(connect4Player):
	'''
	Human player where input is collected from the GUI
	'''

	def play(self, env: connect4, move_dict: dict) -> None:
		done = False
		while(not done):
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					sys.exit()

				if event.type == pygame.MOUSEMOTION:
					pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
					posx = event.pos[0]
					if self.position == 1:
						pygame.draw.circle(screen, P1COLOR, (posx, int(SQUARESIZE/2)), RADIUS)
					else: 
						pygame.draw.circle(screen, P2COLOR, (posx, int(SQUARESIZE/2)), RADIUS)
				pygame.display.update()

				if event.type == pygame.MOUSEBUTTONDOWN:
					posx = event.pos[0]
					col = int(math.floor(posx/SQUARESIZE))
					move_dict['move'] = col
					done = True

class randomAI(connect4Player):
	'''
	connect4Player that elects a random playable column as its move
	'''

	def play(self, env: connect4, move_dict: dict) -> None:
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		move_dict['move'] = random.choice(indices)

class stupidAI(connect4Player):
	'''
	connect4Player that will play the same strategy every time
	Tries to fill specific columns in a specific order 
	'''
	def play(self, env: connect4, move_dict: dict) -> None:
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		if 3 in indices:
			move_dict['move'] = 3
		elif 2 in indices:
			move_dict['move'] = 2
		elif 1 in indices:
			move_dict['move'] = 1
		elif 5 in indices:
			move_dict['move'] = 5
		elif 6 in indices:
			move_dict['move'] = 6
		else:
			move_dict['move'] = 0

class minimaxAI(connect4Player):
    def play(self, env: connect4, move_dict: dict) -> None:
        random.seed(self.seed)
        env = deepcopy(env)
        env.visualize = False

        # Get legal moves (columns with available space)
        indices = [i for i, p in enumerate(env.topPosition) if p >= 0]
        best_value = float('-inf')
        best_move = indices[0]

        # Try each legal move
        for move in indices:
            env_copy = deepcopy(env)
            self.simulateMove(env_copy, move, self.position)
            value = self.minimax(env_copy, depth=4, maximizing=False, last_move=move, player=self.position)
            if value > best_value:
                best_value = value
                best_move = move

        move_dict['move'] = best_move

    def minimax(self, env, depth: int, maximizing: bool, last_move: int, player: int) -> float:
        # Terminal condition: win, loss, or full board
        if env.gameOver(last_move, player) or depth == 0:
            return self.evaluate(env)

        indices = [i for i, p in enumerate(env.topPosition) if p >= 0]

        if maximizing:
            best = float('-inf')
            for move in indices:
                env_copy = deepcopy(env)
                self.simulateMove(env_copy, move, self.position)
                score = self.minimax(env_copy, depth-1, False, move, self.position)
                best = max(best, score)
            return best
        else:
            opp = 3 - self.position
            best = float('inf')
            for move in indices:
                env_copy = deepcopy(env)
                self.simulateMove(env_copy, move, opp)
                score = self.minimax(env_copy, depth-1, True, move, opp)
                best = min(best, score)
            return best

    def evaluate(self, env) -> float:
        score = 0
        board = env.board
        ROW_COUNT, COLUMN_COUNT = board.shape

        # Score center column: pieces in the center count more.
        center_col = COLUMN_COUNT // 2
        center_array = list(board[:, center_col])
        center_count = center_array.count(self.position)
        score += center_count * 1.1

        # Horizontal score
        for r in range(ROW_COUNT):
            row_array = list(board[r, :])
            for c in range(COLUMN_COUNT - 3):
                window = row_array[c:c+4]
                score += self.evaluate_window(window)

        # Vertical score
        for c in range(COLUMN_COUNT):
            col_array = list(board[:, c])
            for r in range(ROW_COUNT - 3):
                window = col_array[r:r+4]
                score += self.evaluate_window(window)

        # Positive slope diagonal
        for r in range(ROW_COUNT - 3):
            for c in range(COLUMN_COUNT - 3):
                window = [board[r+i, c+i] for i in range(4)]
                score += self.evaluate_window(window)

        # Negative slope diagonal
        for r in range(ROW_COUNT - 3):
            for c in range(COLUMN_COUNT - 3):
                window = [board[r+3-i, c+i] for i in range(4)]
                score += self.evaluate_window(window)

        return score

    def evaluate_window(self, window: list) -> float:
        score = 0
        opp = 1 if self.position == 2 else 2

        player_count = window.count(self.position)
        empty_count = window.count(0)
        opp_count = window.count(opp)

        if player_count == 4:
            score += 100
        elif player_count == 3 and empty_count == 1:
            score += 15  # Increase reward for three in a row
        elif player_count == 2 and empty_count == 2:
            score += 5

        if opp_count == 3 and empty_count == 1:
            score -= 95  # Heavy penalty if the opponent is close to winning
        elif opp_count == 2 and empty_count == 2:
            score -= 5

        return score

    def simulateMove(self, env: connect4, move: int, player: int):
        # Play the move like monteCarloAI does
        env.board[env.topPosition[move]][move] = player
        env.topPosition[move] -= 1
        env.history[player-1].append(move)	

class alphaBetaAI(connect4Player):
    def play(self, env: connect4, move_dict: dict) -> None:
        random.seed(self.seed)
        self.start_time = time.time()
        time_threshold = 2.8  

        env = deepcopy(env)
        env.visualize = False

        legal_moves = [i for i, pos in enumerate(env.topPosition) if pos >= 0]
        best_value = float('-inf')
        best_move = legal_moves[0]
        alpha = float('-inf')
        beta = float('inf')

        for move in legal_moves:
            if time.time() - self.start_time > time_threshold:
                # Out of time – use the best move found so far.
                break
            env_copy = deepcopy(env)
            self.simulateMove(env_copy, move, self.position)
            value = self.alphabeta(
                env_copy,
                depth=5,
                maximizing=False,
                alpha=alpha,
                beta=beta,
                last_move=move,
                player=self.position,
                time_threshold=time_threshold
            )
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)
        
        move_dict['move'] = best_move

    def alphabeta(self, env, depth: int, maximizing: bool, alpha: float, beta: float,
                  last_move: int, player: int, time_threshold: float) -> float:
        # If we're nearly out of time or reached a terminal condition, return the evaluation.
        if time.time() - self.start_time > time_threshold or depth == 0 or env.gameOver(last_move, player):
            return self.evaluate(env)
        
        legal_moves = [i for i, pos in enumerate(env.topPosition) if pos >= 0]
        if maximizing:
            value = float('-inf')
            for move in legal_moves:
                if time.time() - self.start_time > time_threshold:
                    break
                env_copy = deepcopy(env)
                self.simulateMove(env_copy, move, self.position)
                value = max(value, self.alphabeta(env_copy, depth-1, False, alpha, beta, move, self.position, time_threshold))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            opp = 3 - self.position
            value = float('inf')
            for move in legal_moves:
                if time.time() - self.start_time > time_threshold:
                    break
                env_copy = deepcopy(env)
                self.simulateMove(env_copy, move, opp)
                value = min(value, self.alphabeta(env_copy, depth-1, True, alpha, beta, move, opp, time_threshold))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def evaluate(self, env) -> float:
        score = 0
        board = env.board
        ROW_COUNT, COLUMN_COUNT = board.shape

        # Score center column: pieces in the center count more.
        center_col = COLUMN_COUNT // 2
        center_array = list(board[:, center_col])
        center_count = center_array.count(self.position)
        score += center_count * 5

        # Horizontal score
        for r in range(ROW_COUNT):
            row_array = list(board[r, :])
            for c in range(COLUMN_COUNT - 3):
                window = row_array[c:c+4]
                score += self.evaluate_window(window)

        # Vertical score
        for c in range(COLUMN_COUNT):
            col_array = list(board[:, c])
            for r in range(ROW_COUNT - 3):
                window = col_array[r:r+4]
                score += self.evaluate_window(window)

        # Positive slope diagonal
        for r in range(ROW_COUNT - 3):
            for c in range(COLUMN_COUNT - 3):
                window = [board[r+i, c+i] for i in range(4)]
                score += self.evaluate_window(window)

        # Negative slope diagonal
        for r in range(ROW_COUNT - 3):
            for c in range(COLUMN_COUNT - 3):
                window = [board[r+3-i, c+i] for i in range(4)]
                score += self.evaluate_window(window)

        return score

    def evaluate_window(self, window: list) -> float:
        score = 0
        opp = 1 if self.position == 2 else 2

        player_count = window.count(self.position)
        empty_count = window.count(0)
        opp_count = window.count(opp)

        if player_count == 4:
            score += 100
        elif player_count == 3 and empty_count == 1:
            score += 75  # Increase reward for three in a row
        elif player_count == 2 and empty_count == 2:
            score += 5

        if opp_count == 3 and empty_count == 1:
            score -= 80  # Heavy penalty if the opponent is close to winning
        elif opp_count == 2 and empty_count == 2:
            score -= 5

        return score

    def simulateMove(self, env: connect4, move: int, player: int):
        # Update the board, top positions, and move history as in monteCarloAI.
        env.board[env.topPosition[move]][move] = player
        env.topPosition[move] -= 1
        env.history[player-1].append(move)

# Defining Constants
SQUARESIZE = 100
BLUE = (0,0,255)
BLACK = (0,0,0)
P1COLOR = (255,0,0)
P2COLOR = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size)




