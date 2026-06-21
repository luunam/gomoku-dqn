import time
import sys
import numpy as np
from game.state import State
from agents.dqn.dqn_agent import DQNAgent

SIZE = 3
WIN_LENGTH = 3
NUM_ENVS = 128
BATCH_SIZE = 256
EPISODES = 50000

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
    print(f"DQN TicTacToe Results: {win_rate:.1f}% Wins, {draw_rate:.1f}% Draws, {loss_rate:.1f}% Losses")
    return win_rate

def main():
    print(f"Train DQN on TicTacToe ({SIZE}x{SIZE}, win_length={WIN_LENGTH})")
    print(f"Vectorized Environments: {NUM_ENVS} parallel games. Batch size: {BATCH_SIZE}")
    
    agent = DQNAgent(board_size=SIZE)
    # Accelerate epsilon decay specifically for this quick TicTacToe test
    agent.epsilon_decay = 0.99
    
    states = [State(SIZE, win_length=WIN_LENGTH) for _ in range(NUM_ENVS)]
    history = [[] for _ in range(NUM_ENVS)]
    
    episodes_completed = 0
    start = time.time()
    
    while episodes_completed < EPISODES:
        actions = agent.find_best_moves_batch(states)
        
        for i in range(NUM_ENVS):
            state = states[i]
            action = actions[i]
            next_state, reward, done = state.step(action)
            
            history[i].append((state, action))
            
            if len(history[i]) >= 2 and not done:
                prev_state, prev_action = history[i][-2]
                agent.augment_and_queue(prev_state, prev_action, 0.0, next_state, False)
                
            states[i] = next_state
            
            if done:
                winner_state, winner_action = history[i][-1]
                if next_state.winner != 0:
                    agent.augment_and_queue(winner_state, winner_action, 1.0, next_state, True)
                    if len(history[i]) >= 2:
                        loser_state, loser_action = history[i][-2]
                        agent.augment_and_queue(loser_state, loser_action, -1.0, next_state, True)
                else: # Draw
                    agent.augment_and_queue(winner_state, winner_action, 0.0, next_state, True)
                    if len(history[i]) >= 2:
                        loser_state, loser_action = history[i][-2]
                        agent.augment_and_queue(loser_state, loser_action, 0.0, next_state, True)
                
                states[i] = State(SIZE, win_length=WIN_LENGTH)
                history[i] = []
                
                episodes_completed += 1
                
                if episodes_completed % 100 == 0:
                    print(f'Episodes: {episodes_completed}, Time: {time.time() - start:.2f}s')
                    start = time.time()
                    agent.duplicate_model.load_state_dict(agent.model.state_dict())
                    
                if episodes_completed % 500 == 0:
                    evaluate_against_random(agent, 100)
                    
        agent.replay(BATCH_SIZE)

if __name__ == '__main__':
    main()
