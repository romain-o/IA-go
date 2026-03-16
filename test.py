import numpy as np
import random

from env import ReversiEnv

def test_random_agent():
    env = ReversiEnv()
    obs, info = env.reset()
    
    print("=== Initial Board State ===")
    env.render()

    terminated = False
    step_count = 0

    while not terminated:
        action_mask = info["action_mask"]
        
        valid_actions = np.where(action_mask == 1)[0]
        
        action = random.choice(valid_actions)
        
        player = "Black" if env.is_black_turn else "White"
        if action == 64:
            move_str = "PASS"
        else:
            row, col = divmod(action, 8)
            move_str = f"Row {row}, Col {col}"
            
        print(f"\n--- Step {step_count + 1}: {player} plays {move_str} (Action {action}) ---")

        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        
        if terminated:
            print("=== Game Over ===")
            if reward == 1.0:
                print(f"Result: {player} won the game!")
            elif reward == -1.0:
                print(f"Result: {player} lost the game!")
            else:
                print("Result: It's a Draw!")

        step_count += 1

if __name__ == "__main__":
    test_random_agent()