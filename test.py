import numpy as np
import random

from env import ReversiEnv

def test_random_agent():
    # 1. Initialize the environment
    env = ReversiEnv()
    obs, info = env.reset()
    
    print("=== Initial Board State ===")
    env.render()

    terminated = False
    step_count = 0

    # 2. Run the game loop
    while not terminated:
        # Extract the action mask to see which moves are legal
        action_mask = info["action_mask"]
        
        # Find the indices of all valid actions (where mask is 1)
        valid_actions = np.where(action_mask == 1)[0]
        
        # Choose a random valid action
        action = random.choice(valid_actions)
        
        # Format the move for printing
        player = "Black" if env.is_black_turn else "White"
        if action == 64:
            move_str = "PASS"
        else:
            row, col = divmod(action, 8)
            move_str = f"Row {row}, Col {col}"
            
        print(f"\n--- Step {step_count + 1}: {player} plays {move_str} (Action {action}) ---")

        # 3. Step the environment forward
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 4. Render the board
        env.render()
        
        # Print reward if the game ended
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