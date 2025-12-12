import random
import time
from extension.board_utils import list_legal_moves_for, copy_piece_move

# global dict to cache board states we've already seen
# saves us from recalc-ing the same positions over and over. huge speedup.
TRANSPOSITION_TABLE = {}

# values for pieces. king is massive obvs.
# tweaked these a bit based on testing.
PIECE_VALUES = {
    "pawn": 100, "knight": 320, "bishop": 330, "right": 500, "queen": 900, "king": 20000
}

# position tables (heatmaps) to guide pieces where to go
# white pawns wanna go UP (index 0)
PAWN_TABLE_WHITE = [
    [50, 50, 50, 50, 50], # promo rank!
    [30, 30, 30, 30, 30],
    [10, 15, 20, 15, 10],
    [ 5,  5,  5,  5,  5],
    [ 0,  0,  0,  0,  0]
]

# black pawns wanna go DOWN (index 4)
PAWN_TABLE_BLACK = [
    [ 0,  0,  0,  0,  0],
    [ 5,  5,  5,  5,  5],
    [10, 15, 20, 15, 10],
    [30, 30, 30, 30, 30],
    [50, 50, 50, 50, 50] # promo rank!
]

# keep king safe in corners, away from center "death zone"
KING_TABLE = [
    [ 20, 30, 10, 30, 20],
    [  5,  0, -10, 0,  5],
    [-10, -20, -40, -20, -10], # stay away from here
    [  5,  0, -10, 0,  5],
    [ 20, 30, 10, 30, 20]
]

def get_board_key(board):
    """
    makes a unique key for the board so we can save it in the dict.
    tuple is immutable so it works as a dict key.
    """
    pieces = []
    for p in board.get_pieces():
        # store type, player, x, y
        pieces.append((p.name, p.player.name, p.position.x, p.position.y))
    pieces.sort()
    # also need to track whose turn it is
    return (tuple(pieces), board.current_player.name)

def evaluate_board(board, player_name):
    """
    calculates how good the board is for us.
    positive score = winning, negative = losing.
    """
    score = 0
    for piece in board.get_pieces():
        # basic material value
        val = PIECE_VALUES.get(piece.name.lower(), 0)
        x, y = piece.position.x, piece.position.y
        bonus = 0
        
        # add positional bonuses
        if piece.name.lower() == 'pawn':
            # check which table to use based on color
            bonus = PAWN_TABLE_WHITE[y][x] if piece.player.name == 'white' else PAWN_TABLE_BLACK[y][x]
        elif piece.name.lower() == 'king':
            bonus = KING_TABLE[y][x]
        elif 1 <= x <= 3 and 1 <= y <= 3:
            # knights/bishops etc should control center
            bonus = 10
            
        if piece.player.name == player_name:
            score += (val + bonus)
        else:
            score -= (val + bonus)
    return score

def quiescence(board, alpha, beta, maximizing_player, player_name, start_time, time_limit):
    """
    extra search at the end of depth to prevent 'horizon effect'.
    basically keeps looking if captures are happening so we dont stop mid-fight.
    """
    # check if we're outta time
    if time.perf_counter() - start_time > time_limit:
        return evaluate_board(board, player_name)

    # stand pat score (eval without moving)
    stand_pat = evaluate_board(board, player_name)
    
    if maximizing_player:
        if stand_pat >= beta: return beta
        if stand_pat > alpha: alpha = stand_pat
    else:
        if stand_pat <= alpha: return alpha
        if stand_pat < beta: beta = stand_pat

    # figure out whose turn it is to see valid moves
    current_player_obj = None
    for p in board.players:
        if maximizing_player:
            if p.name == player_name: current_player_obj = p
        else:
            if p.name != player_name: current_player_obj = p
    
    if not current_player_obj: return stand_pat

    legal_moves = list_legal_moves_for(board, current_player_obj)
    # ONLY check captures here. ignore quiet moves.
    capture_moves = [m for m in legal_moves if getattr(m[1], "captures", [])]
    
    if not capture_moves: return stand_pat

    if maximizing_player:
        for piece, move in capture_moves:
            temp_board = board.clone()
            _, t_piece, t_move = copy_piece_move(temp_board, piece, move)
            if t_piece and t_move:
                t_piece.move(t_move)
                score = quiescence(temp_board, alpha, beta, False, player_name, start_time, time_limit)
                if score > alpha: alpha = score
                if beta <= alpha: break
        return alpha
    else:
        for piece, move in capture_moves:
            temp_board = board.clone()
            _, t_piece, t_move = copy_piece_move(temp_board, piece, move)
            if t_piece and t_move:
                t_piece.move(t_move)
                score = quiescence(temp_board, alpha, beta, True, player_name, start_time, time_limit)
                if score < beta: beta = score
                if beta <= alpha: break
        return beta

def minimax(board, depth, alpha, beta, maximizing_player, player_name, start_time, time_limit):
    """
    standard minimax algo with alpha-beta pruning.
    uses global memory (transposition table) to speed stuff up.
    """
    # timeout check
    if time.perf_counter() - start_time > time_limit:
        return None, evaluate_board(board, player_name)

    # 1. check memory (TT) to see if we've seen this pos before
    board_key = get_board_key(board)
    if board_key in TRANSPOSITION_TABLE:
        entry = TRANSPOSITION_TABLE[board_key]
        # only use cached result if it was deep enough
        if entry['depth'] >= depth:
            if entry['flag'] == 'EXACT':
                return None, entry['score']
            elif entry['flag'] == 'LOWERBOUND':
                alpha = max(alpha, entry['score'])
            elif entry['flag'] == 'UPPERBOUND':
                beta = min(beta, entry['score'])
            
            if alpha >= beta:
                return None, entry['score']

    # grab current player obj
    current_player_obj = None
    for p in board.players:
        if maximizing_player:
            if p.name == player_name: current_player_obj = p
        else:
            if p.name != player_name: current_player_obj = p
            
    legal_moves = list_legal_moves_for(board, current_player_obj)
    
    # check for checkmate/stalemate
    # if no moves, its game over man.
    if not legal_moves:
        if maximizing_player: return None, -99999 # we lose
        else: return None, 99999 # we win

    # base case: switch to quiescence search
    if depth == 0:
        val = quiescence(board, alpha, beta, maximizing_player, player_name, start_time, time_limit)
        # save leaf node to memory
        TRANSPOSITION_TABLE[board_key] = {'depth': 0, 'score': val, 'flag': 'EXACT'}
        return None, val

    # move ordering: try captures first! huge optimization for pruning.
    legal_moves.sort(key=lambda m: 1 if getattr(m[1], "captures", []) else 0, reverse=True)

    best_move = None
    original_alpha = alpha 

    if maximizing_player:
        max_eval = -float('inf')
        for piece, move in legal_moves:
            temp_board = board.clone()
            _, t_piece, t_move = copy_piece_move(temp_board, piece, move)
            if t_piece and t_move:
                t_piece.move(t_move)
                
                _, eval_score = minimax(temp_board, depth - 1, alpha, beta, False, player_name, start_time, time_limit)
                
                # bubble up timeout
                if eval_score is None: return None, None
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = (piece, move)
                alpha = max(alpha, eval_score)
                if beta <= alpha: break 
        
        # save result to memory
        flag = 'EXACT'
        if max_eval <= original_alpha: flag = 'UPPERBOUND'
        elif max_eval >= beta: flag = 'LOWERBOUND'
        
        TRANSPOSITION_TABLE[board_key] = {'depth': depth, 'score': max_eval, 'flag': flag}
        return best_move, max_eval

    else:
        min_eval = float('inf')
        for piece, move in legal_moves:
            temp_board = board.clone()
            _, t_piece, t_move = copy_piece_move(temp_board, piece, move)
            if t_piece and t_move:
                t_piece.move(t_move)
                
                _, eval_score = minimax(temp_board, depth - 1, alpha, beta, True, player_name, start_time, time_limit)
                
                if eval_score is None: return None, None
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = (piece, move)
                beta = min(beta, eval_score)
                if beta <= alpha: break
        
        # save min result to memory
        flag = 'EXACT'
        if min_eval <= original_alpha: flag = 'UPPERBOUND' 
        elif min_eval >= beta: flag = 'LOWERBOUND'

        TRANSPOSITION_TABLE[board_key] = {'depth': depth, 'score': min_eval, 'flag': flag}
        return best_move, min_eval

def agent(board, player, var):
    """
    Main agent function required for submission.
    Uses iterative deepening to play as best as possible within time limit.
    """
    # extract time budget from var
    # var[0] is ply, var[1] is budget
    time_budget = var[1]
    start_time = time.perf_counter()
    
    # buffer of 0.1s to make sure we return in time
    hard_limit = start_time + time_budget - 0.1
    
    legal = list_legal_moves_for(board, player)
    # if trapped, just return None (let game engine handle loss)
    if not legal: return None, None
    
    # default random-ish move just in case
    legal.sort(key=lambda m: 1 if getattr(m[1], "captures", []) else 0, reverse=True)
    best_piece, best_move = legal[0]
    
    # iterative deepening loop
    # start depth 1, go deeper until we run outta time
    max_depth = 1 
    # print(f"--- Move for {player.name} ---") # debug log
    
    while True:
        # hard stop if time is up
        if time.perf_counter() > hard_limit: break
        
        # dont start a new depth if we have < 0.2s left
        remaining = hard_limit - time.perf_counter()
        if remaining < 0.2: break 
        
        try:
            # call minimax with remaining time
            result_move, result_score = minimax(board, max_depth, -float('inf'), float('inf'), True, player.name, start_time, time_budget - 0.1)
            
            if result_move:
                best_piece, best_move = result_move
                # print(f"Depth {max_depth} | Score: {result_score} | Time: {remaining:.2f}s")
                max_depth += 1
                # if found a forced mate, stop searching and execute order 66
                if abs(result_score) > 10000: break
            else:
                # timed out mid-search, break loop
                break
        except Exception:
            break
            
    return best_piece, best_move
