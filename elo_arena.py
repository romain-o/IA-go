import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from collections import defaultdict
import math
from mcts import MCTS

from env import ReversiEnv
from train import DualHeadResNet

# --- 1. Agents ---

class LocalEvaluator:
    """A fast, synchronous bridge to evaluate boards without multiprocessing."""
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            policy, value = self.model(state_tensor)
            policy = policy.squeeze(0).cpu().numpy()
            return {a: p for a, p in enumerate(policy)}, value.item()

def random_agent(action_mask):
    """Picks a completely random valid move."""
    valid_actions = np.where(action_mask == 1)[0]
    return int(np.random.choice(valid_actions))

# --- 2. Competitive Match Engine ---

def play_match(agent_black, agent_white, models_dict, device, mcts_sims=500):
    """
    Plays one game using competitive MCTS lookahead. 
    Agents are either the string "random" or an integer representing a checkpoint.
    """
    env = ReversiEnv()
    obs, info = env.reset()
    terminated = False
    step_count = 0
    
    # Initialize separate MCTS trees for each player (if they aren't the random agent)
    mcts_black = MCTS(num_simulations=mcts_sims) if agent_black != "random" else None
    mcts_white = MCTS(num_simulations=mcts_sims) if agent_white != "random" else None
    
    # Initialize the evaluators
    eval_black = LocalEvaluator(models_dict[agent_black], device) if agent_black != "random" else None
    eval_white = LocalEvaluator(models_dict[agent_white], device) if agent_white != "random" else None
    
    while not terminated:
        current_agent = agent_black if env.is_black_turn else agent_white
        action_mask = info["action_mask"]
        
        if current_agent == "random":
            action = random_agent(action_mask)
        else:
            # Grab the correct tree and evaluator for the current turn
            current_mcts = mcts_black if env.is_black_turn else mcts_white
            current_eval = eval_black if env.is_black_turn else eval_white
            
            # 1. Run the search with Dirichlet noise DISABLED
            _, mcts_policy = current_mcts.search(env, current_eval, add_noise=False)
            
            # 2. Temperature Scheduling
            if step_count < 2:
                # First two moves: Add slight randomness to prevent identical games
                action = int(np.random.choice(65, p=mcts_policy))
            else:
                # Rest of the game: Absolute deterministic, ruthless exploitation
                action = int(np.argmax(mcts_policy))
                
            # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
    # Determine the true winner
    black_score = env.current_player_bb.bit_count() if env.is_black_turn else env.opp_bb.bit_count()
    white_score = env.opp_bb.bit_count() if env.is_black_turn else env.current_player_bb.bit_count()
    
    print(f"Final Score -> Black: {black_score} | White: {white_score}")
    if black_score > white_score: return 1.0  # Black wins
    elif white_score > black_score: return 0.0 # White wins
    return 0.5 # Draw

# --- 3. Elo Math ---

def get_expected_score(rating_a, rating_b):
    return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, score_a, k=32):
    expected_a = get_expected_score(rating_a, rating_b)
    new_rating_a = rating_a + k * (score_a - expected_a)
    
    # Score B is inverted (1 - score_a)
    expected_b = get_expected_score(rating_b, rating_a)
    new_rating_b = rating_b + k * ((1.0 - score_a) - expected_b)
    
    return new_rating_a, new_rating_b

# --- 4. The Arena Tournament ---

def run_elo_tournament(checkpoint_step=5000, games_per_matchup=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Discover Checkpoints
    files = glob.glob("checkpoints/reversi_model_game_*.pth")
    checkpoints = []
    for f in files:
        match = re.search(r"game_(\d+)", f)
        if match:
            checkpoints.append((int(match.group(1)), f))
            
    checkpoints.sort(key=lambda x: x[0])
    checkpoints_to_test = [cp[0] for cp in checkpoints if cp[0] % checkpoint_step == 0 or cp[0] == checkpoints[-1][0]]
    
    print(f"Loading {len(checkpoints_to_test)} models into memory...")
    
    # 2. Load Models and Initialize Elos
    models_dict = {}
    elos = {"random": 1200.0}
    
    for game_num in checkpoints_to_test:
        path = f"checkpoints/reversi_model_game_{game_num}.pth"
        model = DualHeadResNet().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        models_dict[game_num] = model
        elos[game_num] = 1200.0 # Everyone starts at 1200

    print("Models loaded. Let the tournament begin!\n")

    # 3. Play the Matches
    for i, current_cp in enumerate(checkpoints_to_test):
        print(f"--- Processing Checkpoint: Game {current_cp} ---")
        
        # A. Play against Random
        for game in range(games_per_matchup):
            # Alternate colors
            if game % 2 == 0:
                result = play_match(current_cp, "random", models_dict, device)
                score_cp = result
            else:
                result = play_match("random", current_cp, models_dict, device)
                score_cp = 1.0 - result
                
            elos[current_cp], elos["random"] = update_elo(elos[current_cp], elos["random"], score_cp)

        # B. Play against the previous checkpoint (to establish relative AI-vs-AI strength)
        if i > 0:
            prev_cp = checkpoints_to_test[i-1]
            for game in range(games_per_matchup):
                if game % 2 == 0:
                    result = play_match(current_cp, prev_cp, models_dict, device)
                    score_cp = result
                else:
                    result = play_match(prev_cp, current_cp, models_dict, device)
                    score_cp = 1.0 - result
                    
                elos[current_cp], elos[prev_cp] = update_elo(elos[current_cp], elos[prev_cp], score_cp)
                
        print(f"Current Elo -> CP {current_cp}: {elos[current_cp]:.0f} | Random: {elos['random']:.0f}")

    # --- 5. Graphing ---
    plot_games = [cp for cp in checkpoints_to_test]
    plot_elos = [elos[cp] for cp in plot_games]
    
    plt.figure(figsize=(10, 6))
    plt.plot(plot_games, plot_elos, marker='o', color='purple', label='AlphaZero Checkpoint Elo')
    plt.axhline(y=elos["random"], color='gray', linestyle='--', label=f'Random Agent Elo ({elos["random"]:.0f})')
    
    plt.title('AlphaZero Progression: True Elo Rating')
    plt.xlabel('Self-Play Games Trained')
    plt.ylabel('Elo Rating')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig('elo_progression.png')
    print("\nTournament complete! Graph saved as 'elo_progression.png'.")
    plt.show()

if __name__ == "__main__":
    run_elo_tournament(checkpoint_step=5000, games_per_matchup=20)