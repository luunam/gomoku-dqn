from __future__ import print_function
from game.game import Game
from game.state import State
from agents import HumanAgent, DQNAgent, MinimaxAgent, PPOAgent, AlphaZeroAgent
from agents.shared_net import SharedNet
from benchmark import Benchmarker
import argparse
import logging
import time
import os
import glob
import re
import multiprocessing

EPISODES = 150000
SIZE = 15
BATCH_SIZE = 256

def evaluate_against_random(agent1, num_games=100):
    orig_eps1 = getattr(agent1, 'epsilon', 0.0)
    if hasattr(agent1, 'epsilon'):
        agent1.epsilon = 0.0

    random_agent = DQNAgent(agent1.board_size)
    random_agent.epsilon = 1.0

    wins = 0

    for i in range(num_games):
        if i % 2 == 0:
            game = Game(agent1, random_agent, test=True)
            my_piece = 1
        else:
            game = Game(random_agent, agent1, test=True)
            my_piece = 2
            
        game.run()
        winner_piece = 3 - game.state.turn
        
        if game.state.occupied < 225:
            if winner_piece == my_piece:
                wins += 1

    if hasattr(agent1, 'epsilon'):
        agent1.epsilon = orig_eps1
    return wins / num_games


def train_dqn():
    agent1 = DQNAgent(SIZE)
    episodes_completed = 0
    
    if os.path.exists('./trained'):
        files = glob.glob('./trained/agent_ep*.h5')
        if files:
            def extract_ep(name):
                m = re.search(r'ep(\d+)', name)
                return int(m.group(1)) if m else -1
            latest = max(files, key=extract_ep)
            print(f"Resuming DQN from {latest}")
            agent1.load(latest)
            episodes_completed = extract_ep(latest)
            print(f"DQN Current Episode: {episodes_completed}")

    NUM_ENVS = 128
    print(f'Train DQN with Vectorized Environments: {NUM_ENVS} parallel games. Batch size: {BATCH_SIZE}')

    if not os.path.exists('./trained'):
        os.makedirs('./trained')

    states = [State(SIZE) for _ in range(NUM_ENVS)]
    history = [[] for _ in range(NUM_ENVS)]
    
    start = time.time()

    try:
        while episodes_completed < EPISODES:
            actions = agent1.find_best_moves_batch(states)
            
            for i in range(NUM_ENVS):
                state = states[i]
                action = actions[i]
                next_state, reward, done = state.step(action)
                
                history[i].append((state, action))
                
                if len(history[i]) >= 2 and not done:
                    prev_state, prev_action = history[i][-2]
                    agent1.augment_and_queue(prev_state, prev_action, 0.0, state, False)
                    
                states[i] = next_state
                
                if done:
                    winner_state, winner_action = history[i][-1]
                    agent1.augment_and_queue(winner_state, winner_action, 1.0, next_state, True)
                    
                    if len(history[i]) >= 2:
                        loser_state, loser_action = history[i][-2]
                        agent1.augment_and_queue(loser_state, loser_action, -1.0, next_state, True)
                    
                    states[i] = State(SIZE)
                    history[i] = []
                    episodes_completed += 1
                    
                    if episodes_completed % 50 == 0:
                        print(f'DQN Episodes: {episodes_completed}, Time: {time.time() - start:.2f}s')
                        start = time.time()

                    if episodes_completed > 0 and episodes_completed % 1000 == 0:
                        print(f'Saving DQN snapshot at episode {episodes_completed}')
                        agent1.save(f'./trained/agent_ep{episodes_completed}.h5')
                        
                    if episodes_completed > 0 and episodes_completed % 5000 == 0:
                        win_rate = evaluate_against_random(agent1, 100)
                        print(f'DQN Win Rate against Random Agent: {win_rate * 100:.1f}%')

            agent1.replay(BATCH_SIZE)

        agent1.save('./trained/agent_final.h5')

    except KeyboardInterrupt:
        print('Key board interrupt, emergency save')
        agent1.save('./trained/agent1_interrupt.h5')

def train_ppo():
    agent = PPOAgent(SIZE)
    episodes_completed = 0
    if os.path.exists('./trained'):
        files = glob.glob('./trained/ppo_ep*.h5')
        if files:
            def extract_ep(name):
                m = re.search(r'ep(\d+)', name)
                return int(m.group(1)) if m else -1
            latest = max(files, key=extract_ep)
            print(f"Resuming PPO from {latest}")
            agent.load(latest)
            episodes_completed = extract_ep(latest)
            print(f"PPO Current Episode: {episodes_completed}")

    NUM_ENVS = 64
    print(f'Train PPO with Vectorized Environments: {NUM_ENVS} parallel games.')

    if not os.path.exists('./trained'):
        os.makedirs('./trained')

    states = [State(SIZE) for _ in range(NUM_ENVS)]
    history = [[] for _ in range(NUM_ENVS)]
    start = time.time()

    try:
        while episodes_completed < EPISODES:
            actions, logprobs, values = agent.find_best_moves_batch(states)
            
            for i in range(NUM_ENVS):
                state = states[i]
                action = actions[i]
                logprob = logprobs[i]
                value = values[i]
                next_state, reward, done = state.step(action)
                
                history[i].append({
                    'state': SharedNet.state_to_tensor(state),
                    'action': action,
                    'logprob': logprob,
                    'value': value
                })
                
                states[i] = next_state
                
                if done:
                    for step_idx, step_data in enumerate(history[i]):
                        if (len(history[i]) - 1 - step_idx) % 2 == 0:
                            r = 1.0
                        else:
                            r = -1.0
                        
                        agent.memory.states.append(step_data['state'])
                        agent.memory.actions.append(step_data['action'])
                        agent.memory.logprobs.append(step_data['logprob'])
                        agent.memory.values.append(step_data['value'])
                        agent.memory.rewards.append(r)
                        agent.memory.is_terminals.append(True if step_idx == len(history[i])-1 else False)
                        
                    states[i] = State(SIZE)
                    history[i] = []
                    episodes_completed += 1
                    
                    if episodes_completed % 50 == 0:
                        print(f'PPO Episodes: {episodes_completed}, Time: {time.time() - start:.2f}s')
                        start = time.time()

                    if episodes_completed > 0 and episodes_completed % 1000 == 0:
                        agent.save(f'./trained/ppo_ep{episodes_completed}.h5')
            
            agent.replay(BATCH_SIZE)

        agent.save('./trained/ppo_final.h5')

    except KeyboardInterrupt:
        agent.save('./trained/ppo_interrupt.h5')

def train_az():
    agent = AlphaZeroAgent(SIZE)
    episodes_completed = 0
    if os.path.exists('./trained'):
        files = glob.glob('./trained/az_ep*.h5')
        if files:
            def extract_ep(name):
                m = re.search(r'ep(\d+)', name)
                return int(m.group(1)) if m else -1
            latest = max(files, key=extract_ep)
            print(f"Resuming AlphaZero from {latest}")
            agent.load(latest)
            episodes_completed = extract_ep(latest)
            print(f"AlphaZero Current Episode: {episodes_completed}")
    
    NUM_ENVS = 8
    print(f'Train AlphaZero with Sequential MCTS: {NUM_ENVS} parallel games.')

    if not os.path.exists('./trained'):
        os.makedirs('./trained')

    states = [State(SIZE) for _ in range(NUM_ENVS)]
    history = [[] for _ in range(NUM_ENVS)]
    start = time.time()

    try:
        while episodes_completed < EPISODES:
            actions, mcts_probs = agent.find_best_moves_batch(states)
            
            for i in range(NUM_ENVS):
                state = states[i]
                action = actions[i]
                probs = mcts_probs[i]
                next_state, reward, done = state.step(action)
                
                history[i].append({
                    'state': SharedNet.state_to_tensor(state),
                    'probs': probs
                })
                
                states[i] = next_state
                
                if done:
                    for step_idx, step_data in enumerate(history[i]):
                        if (len(history[i]) - 1 - step_idx) % 2 == 0:
                            winner = 1.0
                        else:
                            winner = -1.0
                        
                        agent.memory.states.append(step_data['state'])
                        agent.memory.mcts_probs.append(step_data['probs'])
                        agent.memory.winners.append(winner)
                        
                    states[i] = State(SIZE)
                    history[i] = []
                    episodes_completed += 1
                    
                    if episodes_completed % 10 == 0:
                        print(f'AZ Episodes: {episodes_completed}, Time: {time.time() - start:.2f}s')
                        start = time.time()

                    if episodes_completed > 0 and episodes_completed % 100 == 0:
                        agent.save(f'./trained/az_ep{episodes_completed}.h5')
            
            agent.replay(BATCH_SIZE)

        agent.save('./trained/az_final.h5')

    except KeyboardInterrupt:
        agent.save('./trained/az_interrupt.h5')

def train_all():
    print("Launching all 3 agents in parallel processes...")
    processes = []
    
    p1 = multiprocessing.Process(target=train_dqn)
    processes.append(p1)
    
    p2 = multiprocessing.Process(target=train_ppo)
    processes.append(p2)
    
    p3 = multiprocessing.Process(target=train_az)
    processes.append(p3)
    
    for p in processes:
        p.start()
        
    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command, either train or test')
    parser.add_argument('command', type=str)
    parser.add_argument('-l', '--log', type=str)
    parser.add_argument('-e', '--episodes', type=int)
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('--agent', type=str, default='all', choices=['dqn', 'ppo', 'az', 'all'])

    args = parser.parse_args()

    command = args.command

    if args.log:
        print('Saving log to: ' + args.log)
        logging.basicConfig(filename=args.log,
                            level=logging.DEBUG,
                            format='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    if args.episodes:
        EPISODES = args.episodes

    if args.batch_size:
        BATCH_SIZE = args.batch_size

    if command == 'train':
        if args.agent == 'dqn':
            train_dqn()
        elif args.agent == 'ppo':
            train_ppo()
        elif args.agent == 'az':
            train_az()
        else:
            train_all()
