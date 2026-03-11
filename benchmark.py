import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os
from env import ReversiEnv
from train import DualHeadResNet

# --- 1. Baseline Agents ---

def random_agent(action_mask):
    """Picks a completely random valid move."""
    valid_actions = np.where(action_mask == 1)[0]
    return np.random.choice(valid_actions)

def greedy_agent(env, action_mask):
    """Simulates valid moves and picks the one that flips the most pieces."""
    valid_actions = np.where(action_mask == 1)[0]
    
    # If the only move is to pass
    if len(valid_actions) == 1 and valid_actions[0] == 64:
        return 64
        
    best_action = -1
    max_flips = -1
    
    for action in valid_actions:
        action = int(action) 
        if action == 64: continue
        
        # Simulate the move using your environment's bitboard logic
        new_player_bb, _ = env._apply_move(action, env.current_player_bb, env.opp_bb)
        
        # Calculate how many pieces were gained
        flips = new_player_bb.bit_count() - env.current_player_bb.bit_count()
        
        if flips > max_flips:
            max_flips = flips
            best_action = action
            
    return best_action

# --- 2. Neural Network Agent ---

def neural_net_agent(obs, action_mask, model, device):
    """Uses the raw Policy Head of the ResNet to pick the highest probability move."""
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        policy, _ = model(obs_tensor)
        policy = policy.squeeze(0).cpu().numpy()
        
        # Mask out illegal moves
        masked_policy = policy * action_mask
        
        if masked_policy.sum() == 0: # Fallback just in case
            return random_agent(action_mask)
            
        return np.argmax(masked_policy)

# --- 3. Match Engine ---

def play_match(model, device, opponent_type, ai_is_black):
    """Plays one full game between the ResNet and a chosen baseline."""
    env = ReversiEnv()
    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        is_ai_turn = (env.is_black_turn == ai_is_black)
        action_mask = info["action_mask"]
        
        if is_ai_turn:
            action = neural_net_agent(obs, action_mask, model, device)
        else:
            if opponent_type == "random":
                action = random_agent(action_mask)
            elif opponent_type == "greedy":
                action = greedy_agent(env, action_mask)
                
        obs, reward, terminated, truncated, info = env.step(action)
        
    # Determine winner
    black_score = env.current_player_bb.bit_count() if env.is_black_turn else env.opp_bb.bit_count()
    white_score = env.opp_bb.bit_count() if env.is_black_turn else env.current_player_bb.bit_count()
    
    if black_score > white_score:
        return 1 if ai_is_black else -1
    elif white_score > black_score:
        return -1 if ai_is_black else 1
    return 0 # Draw

# --- 4. Tournament Logic ---

def run_benchmark(checkpoint_step=500, games_per_match=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadResNet().to(device)
    
    # Find all checkpoint files and sort them numerically by game number
    files = glob.glob(r"checkpoints\reversi_model_game_*.pth")
    checkpoints = []
    for f in files:
        match = re.search(r"game_(\d+)", f)
        if match:
            checkpoints.append((int(match.group(1)), f))
            
    checkpoints.sort(key=lambda x: x[0]) # Sort by game number
    
    # Filter checkpoints based on the step size to save time (e.g., every 500 games)
    checkpoints_to_test = [cp for cp in checkpoints if cp[0] % checkpoint_step == 0 or cp[0] == checkpoints[-1][0]]
    
    game_numbers = []
    random_win_rates = []
    greedy_win_rates = []

    print(f"Found {len(checkpoints)} checkpoints. Testing {len(checkpoints_to_test)} of them...")

    for game_num, path in checkpoints_to_test:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        
        print(f"\n--- Benchmarking Game {game_num} Checkpoint ---")
        
        # Test vs Random
        ai_wins = 0
        for i in range(games_per_match):
            ai_is_black = (i % 2 == 0) # Swap colors every game
            result = play_match(model, device, "random", ai_is_black)
            if result == 1: ai_wins += 1
            elif result == 0: ai_wins += 0.5 # Count draw as half-win
        
        random_wr = (ai_wins / games_per_match) * 100
        random_win_rates.append(random_wr)
        
        # Test vs Greedy
        ai_wins = 0
        for i in range(games_per_match):
            ai_is_black = (i % 2 == 0)
            result = play_match(model, device, "greedy", ai_is_black)
            if result == 1: ai_wins += 1
            elif result == 0: ai_wins += 0.5
            
        greedy_wr = (ai_wins / games_per_match) * 100
        greedy_win_rates.append(greedy_wr)
        
        game_numbers.append(game_num)
        print(f"Win Rate vs Random: {random_wr}% | Win Rate vs Greedy: {greedy_wr}%")

    # --- 5. Graphing ---
    plt.figure(figsize=(10, 6))
    plt.plot(game_numbers, random_win_rates, label='Win % vs Random', marker='o', color='blue')
    plt.plot(game_numbers, greedy_win_rates, label='Win % vs Greedy', marker='s', color='red')
    
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7) # 50% Win Rate Line
    
    plt.title('AlphaZero Progression Benchmarks (Raw Policy Intuition)')
    plt.xlabel('Self-Play Games Trained')
    plt.ylabel('Win Rate (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig('benchmark_results.png')
    print("\nBenchmark complete! Graph saved as 'benchmark_results.png'.")
    plt.show()

if __name__ == "__main__":
    # Test every 500th game checkpoint, playing 20 matches per opponent
    run_benchmark(checkpoint_step=500, games_per_match=200)