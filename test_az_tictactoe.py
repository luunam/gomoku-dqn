import time
import sys
import numpy as np
from game.state import State
from agents.alphazero.az_agent import AlphaZeroAgent
from agents.shared_net import SharedNet

SIZE = 3
WIN_LENGTH = 3
NUM_ENVS = 16
BATCH_SIZE = 64
EPISODES = 5000

def evaluate_against_random(agent, num_games=100):
    wins = 0
    draws = 0
    losses = 0
    for _ in range(num_games):
        state = State(SIZE, win_length=WIN_LENGTH)
        while not state.done():
            if state.turn == 2: # Agent's turn
                # Use greedy move
                original_epsilon = getattr(agent, 'epsilon', 0.0)
                if hasattr(agent, 'epsilon'):
                    agent.epsilon = 0.0
                result = agent.find_best_move(state)
                action = result[1] if isinstance(result, tuple) else result
                if hasattr(agent, 'epsilon'):
                    agent.epsilon = original_epsilon
            else: # Random agent's turn
                possible_moves = []
                for i in range(SIZE * SIZE):
                    if state.valid_move(i):
                        possible_moves.append(i)
                action = np.random.choice(possible_moves)
            
            state, reward, done = state.step(action)
        
        # turn 2 is the agent. if winner == 2, agent won.
        if state.winner == 2:
            wins += 1
        elif state.winner == 1:
            losses += 1
        else:
            draws += 1
            
    win_rate = wins / num_games * 100
    draw_rate = draws / num_games * 100
    loss_rate = losses / num_games * 100
    print(f"AlphaZero TicTacToe Results: {win_rate:.1f}% Wins, {draw_rate:.1f}% Draws, {loss_rate:.1f}% Losses")
    return win_rate

def main():
    print(f"Train AlphaZero on TicTacToe ({SIZE}x{SIZE}, win_length={WIN_LENGTH})")
    print(f"Vectorized Environments: {NUM_ENVS} parallel games. Batch size: {BATCH_SIZE}")
    
    agent = AlphaZeroAgent(board_size=SIZE)
    agent.batch_size = BATCH_SIZE
    
    states = [State(SIZE, win_length=WIN_LENGTH) for _ in range(NUM_ENVS)]
    history = [[] for _ in range(NUM_ENVS)]
    
    episodes_completed = 0
    next_print = 10
    next_eval = 50
    start = time.time()
    
    while episodes_completed < EPISODES:
        actions, all_probs = agent.find_best_moves_batch(states)
        
        for i in range(NUM_ENVS):
            state = states[i]
            action = actions[i]
            probs = all_probs[i]
            
            next_state, reward, done = state.step(action)
            
            history[i].append({
                'state': SharedNet.state_to_tensor(state),
                'probs': probs
            })
            
            states[i] = next_state
            
            if done:
                is_win = (next_state.winner != 0)
                
                for step_idx, step_data in enumerate(history[i]):
                    if is_win:
                        if (len(history[i]) - 1 - step_idx) % 2 == 0:
                            r = 1.0
                        else:
                            r = -1.0
                    else:
                        r = 0.0 # Draw
                    
                    agent.memory.states.append(step_data['state'])
                    agent.memory.mcts_probs.append(step_data['probs'])
                    agent.memory.winners.append(r)
                
                states[i] = State(SIZE, win_length=WIN_LENGTH)
                history[i] = []
                
                episodes_completed += 1
                
                if episodes_completed >= next_print:
                    print(f'Episodes: {episodes_completed}, Time: {time.time() - start:.2f}s')
                    start = time.time()
                    next_print += 10
                    
                if episodes_completed >= next_eval:
                    evaluate_against_random(agent, 10)
                    next_eval += 50
                    
        agent.replay()

if __name__ == '__main__':
    main()
