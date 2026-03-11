import pygame
import sys
import numpy as np
import time
from env import ReversiEnv  # Assumes your environment class is in env.py
import torch
from train import DualHeadResNet # Assuming you saved the class in train.py
from benchmark import greedy_agent

# --- Configuration ---
SQUARE_SIZE = 80
BOARD_SIZE = SQUARE_SIZE * 8
WINDOW_SIZE = (BOARD_SIZE, BOARD_SIZE)

# Colors
GREEN = (34, 139, 34)
DARK_GREEN = (24, 100, 24)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
YELLOW = (255, 255, 0)

def trained_ai_policy(obs, action_mask, model, device):
    """
    Passes the board to the ResNet and picks the highest-probability valid move.
    """
    with torch.no_grad(): # Disable gradient tracking for fast inference
        # Convert the (3, 8, 8) numpy array into a (1, 3, 8, 8) PyTorch batch
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Ask the network for the policy (we ignore the value head here)
        policy, _ = model(obs_tensor)
        
        # Move the policy back to the CPU and convert to a 1D numpy array
        policy = policy.squeeze(0).cpu().numpy()
        
        # Apply the Action Mask: Multiply probabilities by 0 for illegal moves
        masked_policy = policy * action_mask
        
        # Fallback: If the network somehow gave 0 probability to all legal moves
        if masked_policy.sum() == 0:
            valid_actions = np.where(action_mask == 1)[0]
            return np.random.choice(valid_actions)
            
        # Pick the legal action with the highest network probability
        best_action = np.argmax(masked_policy)
        
        return best_action

def draw_board(screen, env, action_mask):
    screen.fill(GREEN)
    
    # Dynamically check which bitboard belongs to which color
    black_bb = env.current_player_bb if env.is_black_turn else env.opp_bb
    white_bb = env.opp_bb if env.is_black_turn else env.current_player_bb

    # Draw Grid
    for i in range(9):
        pygame.draw.line(screen, BLACK, (0, i * SQUARE_SIZE), (BOARD_SIZE, i * SQUARE_SIZE), 2)
        pygame.draw.line(screen, BLACK, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE, BOARD_SIZE), 2)

    # Draw Pieces
    for i in range(64):
        r, c = divmod(i, 8)
        center = (c * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE // 2)
        
        # Use the dynamic bitboards here
        if black_bb & (1 << i):
            pygame.draw.circle(screen, BLACK, center, SQUARE_SIZE // 2 - 4)
        elif white_bb & (1 << i):
            pygame.draw.circle(screen, WHITE, center, SQUARE_SIZE // 2 - 4)
            
        # Highlight valid moves for the current player
        if action_mask[i] == 1:
            pygame.draw.circle(screen, GRAY, center, SQUARE_SIZE // 6)

    pygame.display.flip()

def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Reversi AI - Human vs AlphaZero")

    env = ReversiEnv()
    obs, info = env.reset()
    
    # --- NEW: Load the Trained Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ai_model = DualHeadResNet().to(device)
    
    
    # Replace 'reversi_model_game_100.pth' with your actual latest save file
    checkpoint_path = r"checkpoints\reversi_model_game_7300.pth" 
    ai_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    ai_model.eval() # CRITICAL: Set the network to evaluation mode
    print("AI Model loaded and ready!")
    # -----------------------------------

    human_is_black = True
    game_over = False
    clock = pygame.time.Clock()

    while True:
        action_mask = info["action_mask"]
        is_human_turn = (env.is_black_turn == human_is_black)

        # 1. Handle Automatic Passing
        # If the only valid action is 64 (Pass), execute it automatically
        if np.array_equal(np.where(action_mask == 1)[0], [64]) and not game_over:
            print("No valid moves. Auto-passing.")
            time.sleep(0.5) # Slight delay so the user sees what happened
            obs, reward, terminated, truncated, info = env.step(64)
            game_over = terminated
            continue

        # 2. Draw the Current State
        draw_board(screen, env, action_mask if (is_human_turn and not game_over) else np.zeros(65))

        # 3. Handle Events (Human Input & Window Closing)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Human Turn (Mouse Click)
            if event.type == pygame.MOUSEBUTTONDOWN and is_human_turn and not game_over:
                x, y = event.pos
                col = x // SQUARE_SIZE
                row = y // SQUARE_SIZE
                action = row * 8 + col
                
                if action_mask[action] == 1:
                    obs, reward, terminated, truncated, info = env.step(action)
                    game_over = terminated

        # 4. Handle AI Turn
        if not is_human_turn and not game_over:
            pygame.event.pump() 
            time.sleep(0.5) # Slight delay so the AI doesn't move instantly
            
            # --- Pass the board, the mask, the model, and the device ---
            action = trained_ai_policy(obs, action_mask, ai_model, device)
            # -----------------------------------------------------------
            
            obs, reward, terminated, truncated, info = env.step(action)
            game_over = terminated

        # 5. Handle Game Over
        if game_over:
            screen.fill(DARK_GREEN)
            font = pygame.font.Font(None, 64)
            
            # Use dynamic bitboards for the final score
            final_black_bb = env.current_player_bb if env.is_black_turn else env.opp_bb
            final_white_bb = env.opp_bb if env.is_black_turn else env.current_player_bb
            
            black_score = final_black_bb.bit_count()
            white_score = final_white_bb.bit_count()
            
            if black_score > white_score:
                text = "Black Wins!"
            elif white_score > black_score:
                text = "White Wins!"
            else:
                text = "It's a Tie!"
                
            text_surf = font.render(f"{text} ({black_score} - {white_score})", True, YELLOW)
            text_rect = text_surf.get_rect(center=(BOARD_SIZE//2, BOARD_SIZE//2))
            screen.blit(text_surf, text_rect)
            pygame.display.flip()
            
            # Wait for user to close window
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

        clock.tick(30) # Limit to 30 FPS

if __name__ == "__main__":
    main()