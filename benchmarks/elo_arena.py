import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import time

from mcts import MCTS
from env import ReversiEnv
from train import DualHeadResNet

class LocalEvaluator:
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
    valid_actions = np.where(action_mask == 1)[0]
    return int(np.random.choice(valid_actions))

def play_match(agent_black, agent_white, models_dict, device, mcts_sims=40):
    env = ReversiEnv()
    obs, info = env.reset()
    terminated = False
    step_count = 0
    
    mcts_black = MCTS(num_simulations=mcts_sims) if agent_black != "random" else None
    mcts_white = MCTS(num_simulations=mcts_sims) if agent_white != "random" else None
    
    eval_black = LocalEvaluator(models_dict[agent_black], device) if agent_black != "random" else None
    eval_white = LocalEvaluator(models_dict[agent_white], device) if agent_white != "random" else None
    
    while not terminated:
        current_agent = agent_black if env.is_black_turn else agent_white
        action_mask = info["action_mask"]
        
        if current_agent == "random":
            action = random_agent(action_mask)
        else:
            current_mcts = mcts_black if env.is_black_turn else mcts_white
            current_eval = eval_black if env.is_black_turn else eval_white
            
            _, mcts_policy = current_mcts.search(env, current_eval, add_noise=False)
            
            if step_count < 6:
                action = int(np.random.choice(65, p=mcts_policy))
            else:
                action = int(np.argmax(mcts_policy))
                
        obs, _, terminated, _, info = env.step(action)
        step_count += 1
        
    black_score = env.current_player_bb.bit_count() if env.is_black_turn else env.opp_bb.bit_count()
    white_score = env.opp_bb.bit_count() if env.is_black_turn else env.current_player_bb.bit_count()
    
    if black_score > white_score: return 1.0  
    elif white_score > black_score: return 0.0 
    return 0.5 

def get_expected_score(rating_a, rating_b):
    return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, score_a, k=32):
    expected_a = get_expected_score(rating_a, rating_b)
    new_rating_a = rating_a + k * (score_a - expected_a)
    
    expected_b = get_expected_score(rating_b, rating_a)
    new_rating_b = rating_b + k * ((1.0 - score_a) - expected_b)
    
    return new_rating_a, new_rating_b

def run_elo_tournament(games_per_matchup=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoints_to_test = [
        r'checkpoints\reversi_model_game_5000.pth',
        r'checkpoints\reversi_model_game_20000.pth',
        r'checkpoints\reversi_model_game_31700.pth',
        r'checkpoints\reversi_bundle_game_43000.pth',
        r'checkpoints/reversi_bundle_game_65000.pth'
    ]
    
    print(f"Loading {len(checkpoints_to_test)} models into memory...")
    
    models_dict = {}
    model_names = ["random"]
    elos = {"random": 1000.0}
    
    for path in checkpoints_to_test:
        game_num = f"{int(path.split('game_')[-1].split('.pth')[0])//1000}k"
        model = DualHeadResNet().to(device)
        
        data = torch.load(path, map_location=device, weights_only=False)
        if 'model_state_dict' in data:
            model.load_state_dict(data['model_state_dict'])
        else:
            model.load_state_dict(data)
            
        model.eval()
        models_dict[game_num] = model
        model_names.append(game_num)
        elos[game_num] = 1000.0 

    win_matrix = pd.DataFrame(index=model_names[1:], columns=model_names)
    win_matrix = win_matrix.fillna(0.0)

    print("Models loaded.\n")
    start_time = time.time()

    for i, current_cp in enumerate(model_names[1:]):
        print(f"\n--- Testing {current_cp} ---")

        for opponent in model_names[:i+1]:
            wins = 0.0
            print(f"  vs {opponent:8s} | ", end="", flush=True)
            
            for game in range(games_per_matchup):
                if game % 2 == 0:
                    score = play_match(current_cp, opponent, models_dict, device)
                else:
                    score = 1.0 - play_match(opponent, current_cp, models_dict, device)
                
                wins += score
                elos[current_cp], elos[opponent] = update_elo(elos[current_cp], elos[opponent], score)
                print(".", end="", flush=True)
                
            win_rate = wins / games_per_matchup
            win_matrix.loc[current_cp, opponent] = win_rate * 100 
            print(f" | WR: {win_rate*100:.1f}%")

    print(f"\nTournament complete in {(time.time() - start_time)/60:.1f} minutes!")
    
    plot_games = model_names[1:] 
    plot_elos = [elos[cp] for cp in plot_games]
    
    plt.figure(figsize=(10, 6))
    plt.plot(plot_games, plot_elos, marker='o', linewidth=3, color='#2c3e50')
    plt.axhline(y=elos["random"], color='#e74c3c', linestyle='--', linewidth=2, label=f'Random Agent Baseline')
    
    plt.title('AlphaZero Learning Curve (Elo Rating)', fontsize=16, fontweight='bold')
    plt.xlabel('Self-Play Training Games', fontsize=12)
    plt.ylabel('Calculated Elo', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('report_elo_curve.png', dpi=300)

    plt.figure(figsize=(10, 8))
    mask = win_matrix.isnull() | (win_matrix == 0.0).applymap(lambda x: False) 
    
    sns.heatmap(win_matrix.astype(float), annot=True, fmt=".1f", cmap="YlGnBu", 
                cbar_kws={'label': 'Win Rate (%)'}, linewidths=1,
                vmin=0, vmax=100)
    
    plt.title('Cross-Evaluation Matrix (Win Rates)', fontsize=16, fontweight='bold')
    plt.ylabel('Challenger Model', fontsize=12)
    plt.xlabel('Opponent Model', fontsize=12)
    plt.tight_layout()
    plt.savefig('report_heatmap.png', dpi=300)

    print("Saved 'report_elo_curve.png' and 'report_heatmap.png'")

if __name__ == "__main__":
    run_elo_tournament(games_per_matchup=20)