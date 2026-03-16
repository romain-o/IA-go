import struct
import torch
import numpy as np

from env import ReversiEnv
from train import DualHeadResNet

def parse_wthor_file(filepath):
    """
    Parses a standard .wtb file and returns a list of games.
    Each game is a list of sequential actions (integers 0-63).
    """
    games = []
    
    with open(filepath, "rb") as f:
        header = f.read(16)

        num_games = struct.unpack('<I', header[4:8])[0]
        print(f"Detected {num_games} games in WTHOR file.")
        
        for _ in range(num_games):
            game_data = f.read(68)
            if len(game_data) < 68:
                break
                
            moves_bytes = game_data[8:68]
            moves = []
            
            for b in moves_bytes:
                if b == 0:
                    break 
                    
                col = (b // 10) - 1
                row = (b % 10) - 1
                action = row * 8 + col
                moves.append(action)
                
            games.append(moves)
            
    return games

def evaluate_checkpoint(model, device, games):
    correct_predictions = 0
    total_predictions = 0
    
    phase_stats = {
        "Opening (1-20)":   {"correct": 0, "total": 0},
        "Midgame (21-40)":  {"correct": 0, "total": 0},
        "Endgame (41-60)":  {"correct": 0, "total": 0}
    }

    env = ReversiEnv()
    
    for game_idx, human_moves in enumerate(games):
        obs, info = env.reset()
        move_number = 0
        
        for human_action in human_moves:
            action_mask = info["action_mask"]
            
            while np.array_equal(np.where(action_mask == 1)[0], [64]):
                obs, reward, terminated, truncated, info = env.step(64)
                action_mask = info["action_mask"]
                if terminated: break

            if env.pass_count >= 2:
                break

            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                policy, _ = model(obs_tensor)
                policy = policy.squeeze(0).cpu().numpy()
                
                masked_policy = policy * action_mask
                ai_predicted_action = int(np.argmax(masked_policy))

            move_number += 1
            is_correct = (ai_predicted_action == human_action)
            
            correct_predictions += is_correct
            total_predictions += 1
            
            if move_number <= 20: phase = "Opening (1-20)"
            elif move_number <= 40: phase = "Midgame (21-40)"
            else: phase = "Endgame (41-60)"
            
            phase_stats[phase]["total"] += 1
            phase_stats[phase]["correct"] += is_correct

            obs, reward, terminated, truncated, info = env.step(human_action)

        if (game_idx + 1) % 100 == 0:
            print(f"Processed {game_idx + 1}/{len(games)} games...")

    # --- Print Final Results ---
    overall_accuracy = (correct_predictions / total_predictions) * 100
    print("\n" + "="*40)
    print(f"WTHOR STATIC EVALUATION RESULTS")
    print("="*40)
    print(f"Total Positions Evaluated: {total_predictions}")
    print(f"Overall AI Accuracy:       {overall_accuracy:.2f}%\n")
    
    for phase, stats in phase_stats.items():
        if stats["total"] > 0:
            acc = (stats["correct"] / stats["total"]) * 100
            print(f"{phase} Accuracy: {acc:.2f}%")
    print("="*40)


if __name__ == "__main__":
    # 1. Load the WTHOR file
    WTHOR_FILE = r"WTHOR/WTH_2025.wtb" 
    
    try:
        wthor_games = parse_wthor_file(WTHOR_FILE)
    except FileNotFoundError:
        print(f"Could not find '{WTHOR_FILE}'. Please check the path.")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadResNet().to(device)
    

    latest_checkpoint_bundle_path = r'model.pth'
    print(f"\nLoading Model: {latest_checkpoint_bundle_path}")
    
    checkpoint_bundle = torch.load(latest_checkpoint_bundle_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint_bundle['model_state_dict'])
    model.eval()

    print("\nStarting evaluation")
    evaluate_checkpoint(model, device, wthor_games)