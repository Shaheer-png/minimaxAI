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
    def __init__(self, position, seed=0, CVDMode=False):
        super().__init__(position, seed, CVDMode)
        self.first_move = True

    def evaluate_window(self, window, player):
        """Evaluate a window of 4 positions"""
        opponent = self.opponent.position if player == self.position else self.position
        
        player_count = sum(1 for piece in window if piece == player)
        empty_count = sum(1 for piece in window if piece == 0)
        opponent_count = sum(1 for piece in window if piece == opponent)
        
        # Can't make 4 in this window if both players have pieces
        if opponent_count > 0 and player_count > 0:
            return 0
            
        score = 0
        if player_count == 4:
            score += 1000  # Win
        elif player_count == 3 and empty_count == 1:
            score += 50    # One move from win
        elif player_count == 2 and empty_count == 2:
            score += 10    # Two connected
        
        if opponent_count == 3 and empty_count == 1:
            score -= 40    # Block opponent near-win
            
        return score

    def evaluate_position(self, env, player):
        """Evaluate entire board position"""
        score = 0
        board = env.board
        rows, cols = board.shape

        # Check horizontal windows
        for row in range(rows):
            for col in range(cols - 3):
                window = [board[row][col+i] for i in range(4)]
                score += self.evaluate_window(window, player)

        # Check vertical windows
        for row in range(rows - 3):
            for col in range(cols):
                window = [board[row+i][col] for i in range(4)]
                score += self.evaluate_window(window, player)

        # Check diagonal windows (positive slope)
        for row in range(rows - 3):
            for col in range(cols - 3):
                window = [board[row+i][col+i] for i in range(4)]
                score += self.evaluate_window(window, player)

        # Check diagonal windows (negative slope)
        for row in range(3, rows):
            for col in range(cols - 3):
                window = [board[row-i][col+i] for i in range(4)]
                score += self.evaluate_window(window, player)

        # Prefer center control
        center_col = cols // 2
        center_count = sum(1 for row in range(rows) if board[row][center_col] == player)
        score += center_count * 3

        return score

    def get_valid_moves(self, env):
        """Get list of valid moves"""
        return [col for col in range(env.board.shape[1]) if env.topPosition[col] >= 0]

    def make_move(self, env, col, player):
        """Make a move in the given environment"""
        row = env.topPosition[col]
        env.board[row][col] = player
        env.topPosition[col] -= 1
        return row

    def minimax(self, env, depth, maximizing_player, start_time):
        """
        Basic minimax implementation without alpha-beta pruning
        """
        import time
        from copy import deepcopy

        # Check if we're running out of time
        if time.time() - start_time > 2.9:
            return self.evaluate_position(env, self.position)

        valid_moves = self.get_valid_moves(env)
        
        # Check for immediate wins/losses
        for col in valid_moves:
            test_env = deepcopy(env)
            row = self.make_move(test_env, col, test_env.turnPlayer.position)
            if test_env.gameOver(col, test_env.turnPlayer.position):
                return 1000 if maximizing_player else -1000

        # Reached max depth, evaluate position
        if depth == 0:
            return self.evaluate_position(env, self.position)

        if maximizing_player:
            value = float('-inf')
            for col in valid_moves:
                test_env = deepcopy(env)
                self.make_move(test_env, col, self.position)
                value = max(value, self.minimax(test_env, depth - 1, False, start_time))
            return value
        else:
            value = float('inf')
            for col in valid_moves:
                test_env = deepcopy(env)
                self.make_move(test_env, col, self.opponent.position)
                value = min(value, self.minimax(test_env, depth - 1, True, start_time))
            return value

    def play(self, env: connect4, move_dict: dict) -> None:
          pass
          '''
        # First move - play center
        if self.first_move:
            self.first_move = False
            move_dict['move'] = 3
            return

        start_time = time.time()
        valid_moves = self.get_valid_moves(env)
        best_move = valid_moves[0]  # Default to first valid move

        # Iterative deepening - start shallow and go deeper if time permits
        depth = 1
        while time.time() - start_time < 2.5:  # Leave 0.5s buffer
            current_best_move = best_move
            best_score = float('-inf')

            # Try each possible move
            for col in valid_moves:
                test_env = deepcopy(env)
                row = self.make_move(test_env, col, self.position)
                
                # If this move wins, take it immediately
                if test_env.gameOver(col, self.position):
                    move_dict['move'] = col
                    return
                    
                score = self.minimax(test_env, depth, False, start_time)
                
                if score > best_score:
                    best_score = score
                    current_best_move = col

            # Only update best_move if we completed the search at this depth
            if time.time() - start_time < 2.8:
                best_move = current_best_move
                depth += 1
            else:
                break

        move_dict['move'] = best_move
		'''

class alphaBetaAI(connect4Player):
    def __init__(self, position, seed=0, CVDMode=False):
        super().__init__(position, seed, CVDMode)
        self.first_move = True

    def order_moves(self, valid_moves, env):
        """
        Order moves based only on current board state:
        1. Winning moves
        2. Blocking opponent wins
        3. Center and near-center columns
        """
        from copy import deepcopy
        
        move_rankings = []
        for col in valid_moves:
            score = 0
            test_env = deepcopy(env)
            row = self.make_test_move(test_env, col, test_env.turnPlayer.position)
            
            # Prioritize winning moves
            if test_env.gameOver(col, test_env.turnPlayer.position):
                score += 10000
            
            # Check if opponent would win in this column
            if row + 1 < test_env.board.shape[0]:  # If not at top
                test_env_2 = deepcopy(env)
                test_env_2.board[row][col] = self.opponent.position
                if test_env_2.gameOver(col, self.opponent.position):
                    score += 5000
            
            # Prefer center columns - the closer to center the better
            score += 50 * (4 - abs(col - 3))
            
            move_rankings.append((score, col))
            
        # Sort moves by score
        move_rankings.sort(reverse=True)
        return [move for score, move in move_rankings]

    def evaluate_position(self, env, player):
        """Same evaluation as before"""
        score = 0
        board = env.board
        rows, cols = board.shape

        # Evaluate horizontal windows
        for row in range(rows):
            for col in range(cols - 3):
                window = [board[row][col+i] for i in range(4)]
                score += self.evaluate_window(window, player)

        # Evaluate vertical windows
        for row in range(rows - 3):
            for col in range(cols):
                window = [board[row+i][col] for i in range(4)]
                score += self.evaluate_window(window, player)

        # Evaluate diagonal windows (positive slope)
        for row in range(rows - 3):
            for col in range(cols - 3):
                window = [board[row+i][col+i] for i in range(4)]
                score += self.evaluate_window(window, player)

        # Evaluate diagonal windows (negative slope)
        for row in range(3, rows):
            for col in range(cols - 3):
                window = [board[row-i][col+i] for i in range(4)]
                score += self.evaluate_window(window, player)

        return score

    def evaluate_window(self, window, player):
        opponent = self.opponent.position if player == self.position else self.position
        
        player_count = sum(1 for piece in window if piece == player)
        empty_count = sum(1 for piece in window if piece == 0)
        opponent_count = sum(1 for piece in window if piece == opponent)
        
        if opponent_count > 0 and player_count > 0:
            return 0
            
        score = 0
        if player_count == 4:
            score += 1000
        elif player_count == 3 and empty_count == 1:
            score += 70
        elif player_count == 2 and empty_count == 2:
            score += 10
        
        if opponent_count == 3 and empty_count == 1:
            score -= 60
            
        return score

    def alpha_beta(self, env, depth, alpha, beta, maximizing_player, start_time):
        import time
        from copy import deepcopy
        
        if time.time() - start_time > 2.8:  # Time check
            return self.evaluate_position(env, self.position)
            
        valid_moves = self.get_valid_columns(env)
        if not valid_moves:
            return 0
            
        # Check immediate wins/losses
        for col in valid_moves:
            test_env = deepcopy(env)
            row = self.make_test_move(test_env, col, test_env.turnPlayer.position)
            if test_env.gameOver(col, test_env.turnPlayer.position):
                return 1000 if maximizing_player else -1000

        if depth == 0:
            return self.evaluate_position(env, self.position)

        ordered_moves = self.order_moves(valid_moves, env)
        
        if maximizing_player:
            value = float('-inf')
            for col in ordered_moves:
                test_env = deepcopy(env)
                self.make_test_move(test_env, col, self.position)
                value = max(value, self.alpha_beta(test_env, depth - 1, alpha, beta, False, start_time))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float('inf')
            for col in ordered_moves:
                test_env = deepcopy(env)
                self.make_test_move(test_env, col, self.opponent.position)
                value = min(value, self.alpha_beta(test_env, depth - 1, alpha, beta, True, start_time))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def get_valid_columns(self, env):
        return [col for col in range(env.board.shape[1]) if env.topPosition[col] >= 0]

    def make_test_move(self, env, col, player):
        row = env.topPosition[col]
        env.board[row][col] = player
        env.topPosition[col] -= 1
        return row

    def play(self, env: connect4, move_dict: dict) -> None:
		
        if self.first_move:
            self.first_move = False
            move_dict['move'] = 3
            return

        start_time = time.time()
        valid_moves = self.get_valid_columns(env)
        ordered_moves = self.order_moves(valid_moves, env)
        best_move = ordered_moves[0]
        
        # Iterative deepening
        depth = 2
        while time.time() - start_time < 2.5:
            current_best_move = best_move
            best_score = float('-inf')
            alpha = float('-inf')
            beta = float('inf')

            for col in ordered_moves:
                test_env = deepcopy(env)
                row = self.make_test_move(test_env, col, self.position)
                
                if test_env.gameOver(col, self.position):
                    move_dict['move'] = col
                    return
                    
                score = self.alpha_beta(test_env, depth, alpha, beta, False, start_time)
                
                if score > best_score:
                    best_score = score
                    current_best_move = col
                
                alpha = max(alpha, best_score)
                
            if time.time() - start_time < 2.8:
                best_move = current_best_move
                depth += 1
            else:
                break

        move_dict['move'] = best_move

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




