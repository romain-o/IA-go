import pygame
import sys
import numpy as np
import torch
from env import ReversiEnv  
from train import DualHeadResNet 
from mcts import MCTS 

SQUARE_SIZE = 80
BOARD_SIZE = SQUARE_SIZE * 8
PANEL_WIDTH = 300  
WINDOW_SIZE = (BOARD_SIZE + PANEL_WIDTH, BOARD_SIZE)

BOARD_GREEN = (39, 119, 73)      
LINE_COLOR = (25, 80, 48)        
PANEL_BG = (33, 37, 43)          
TEXT_LIGHT = (220, 224, 232)     
TEXT_MUTED = (130, 137, 151)     
PIECE_BLACK = (30, 32, 34)       
PIECE_WHITE = (240, 242, 245)    
SHADOW_COLOR = (20, 60, 35, 120) 
LAST_MOVE_COLOR = (255, 215, 0) 
OVERLAY = (0, 0, 0, 180)         

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

def draw_board(screen, env, font, large_font, is_game_over, is_ai_thinking=False, last_action=None):
    screen.fill(BOARD_GREEN, (0, 0, BOARD_SIZE, BOARD_SIZE))
    screen.fill(PANEL_BG, (BOARD_SIZE, 0, PANEL_WIDTH, BOARD_SIZE))
    
    black_bb = env.current_player_bb if env.is_black_turn else env.opp_bb
    white_bb = env.opp_bb if env.is_black_turn else env.current_player_bb
    black_score = black_bb.bit_count()
    white_score = white_bb.bit_count()

    for i in range(1, 8):
        pygame.draw.line(screen, LINE_COLOR, (0, i * SQUARE_SIZE), (BOARD_SIZE, i * SQUARE_SIZE), 2)
        pygame.draw.line(screen, LINE_COLOR, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE, BOARD_SIZE), 2)

    transparent_surface = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)

    for i in range(64):
        r, c = divmod(i, 8)
        center = (c * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE // 2)
        shadow_offset = (center[0] + 3, center[1] + 4)
        radius = SQUARE_SIZE // 2 - 8
        
        if black_bb & (1 << i):
            pygame.draw.circle(transparent_surface, SHADOW_COLOR, shadow_offset, radius)
            pygame.draw.circle(screen, PIECE_BLACK, center, radius)
        elif white_bb & (1 << i):
            pygame.draw.circle(transparent_surface, SHADOW_COLOR, shadow_offset, radius)
            pygame.draw.circle(screen, PIECE_WHITE, center, radius)
            
        if last_action == i:
            pygame.draw.circle(screen, LAST_MOVE_COLOR, center, radius + 2, 3)

    screen.blit(transparent_surface, (0, 0))

    panel_center_x = BOARD_SIZE + (PANEL_WIDTH // 2)
    
    title_text = large_font.render("REVERSAI", True, TEXT_LIGHT)
    screen.blit(title_text, title_text.get_rect(center=(panel_center_x, 40)))

    pygame.draw.rect(screen, PIECE_BLACK, (BOARD_SIZE + 25, 100, 250, 60), border_radius=10)
    b_score_text = font.render(f"Black: {black_score}", True, PIECE_WHITE)
    screen.blit(b_score_text, b_score_text.get_rect(center=(panel_center_x, 130)))

    pygame.draw.rect(screen, PIECE_WHITE, (BOARD_SIZE + 25, 180, 250, 60), border_radius=10)
    w_score_text = font.render(f"White: {white_score}", True, PIECE_BLACK)
    screen.blit(w_score_text, w_score_text.get_rect(center=(panel_center_x, 210)))

    if is_game_over:
        status = "Game Over"
        color = LAST_MOVE_COLOR
    elif is_ai_thinking:
        status = "AI is thinking..."
        color = TEXT_MUTED
    else:
        status = "Guess the move!"
        color = TEXT_LIGHT

    turn_text = font.render(status, True, color)
    screen.blit(turn_text, turn_text.get_rect(center=(panel_center_x, 290)))
    
    if not is_game_over and not is_ai_thinking:
        instruction = font.render("Press [ -> ]", True, LAST_MOVE_COLOR)
        screen.blit(instruction, instruction.get_rect(center=(panel_center_x, 330)))

    if not is_game_over:
        turn_color = PIECE_BLACK if env.is_black_turn else PIECE_WHITE
        pygame.draw.circle(screen, turn_color, (panel_center_x, 400), 20)
        pygame.draw.circle(screen, TEXT_MUTED, (panel_center_x, 400), 21, 2) 

    pygame.display.flip()

def main():
    pygame.init()
    
    try:
        main_font = pygame.font.SysFont("Segoe UI, Helvetica, Arial", 28, bold=True)
        title_font = pygame.font.SysFont("Segoe UI, Helvetica, Arial", 42, bold=True)
        giant_font = pygame.font.SysFont("Segoe UI, Helvetica, Arial", 72, bold=True)
    except:
        main_font = pygame.font.Font(None, 36)
        title_font = pygame.font.Font(None, 48)
        giant_font = pygame.font.Font(None, 80)

    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("ReversAI - Self-Play Analysis")

    env = ReversiEnv()
    obs, info = env.reset()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ai_model = DualHeadResNet().to(device)

    checkpoint_path = r"model.pth" 
    try:
        checkpoint_bundle = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint_bundle:
            ai_model.load_state_dict(checkpoint_bundle['model_state_dict'])
        else:
            ai_model.load_state_dict(checkpoint_bundle)
    except Exception as e:
        print(f"Warning: Could not load model. Error: {e}")
        
    ai_model.eval() 
    evaluator = LocalEvaluator(ai_model, device)
   
    mcts_black = MCTS(num_simulations=200)
    mcts_white = MCTS(num_simulations=200)

    game_over = False
    last_action = None
    clock = pygame.time.Clock()

    draw_board(screen, env, main_font, title_font, game_over)

    while True:
        action_mask = info["action_mask"]
        terminated = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT and not game_over:
                
                if np.array_equal(np.where(action_mask == 1)[0], [64]):
                    last_action = 64
                    obs, reward, terminated, truncated, info = env.step(64)
                else:
                    draw_board(screen, env, main_font, title_font, game_over, is_ai_thinking=True, last_action=last_action)
                    current_mcts = mcts_black if env.is_black_turn else mcts_white

                    _, mcts_policy = current_mcts.search(env, evaluator, add_noise=False)

                    last_action = int(np.argmax(mcts_policy))

                    obs, reward, terminated, truncated, info = env.step(last_action)

                game_over = terminated
                draw_board(screen, env, main_font, title_font, game_over, is_ai_thinking=False, last_action=last_action)

        if game_over:
            overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
            overlay.fill(OVERLAY)
            screen.blit(overlay, (0, 0))
            
            black_score = env.current_player_bb.bit_count() if env.is_black_turn else env.opp_bb.bit_count()
            white_score = env.opp_bb.bit_count() if env.is_black_turn else env.current_player_bb.bit_count()
            
            if black_score > white_score:
                text = "Black Wins!"
            elif white_score > black_score:
                text = "White Wins!"
            else:
                text = "It's a Tie!"
                
            text_surf = giant_font.render(text, True, TEXT_LIGHT)
            text_rect = text_surf.get_rect(center=(WINDOW_SIZE[0]//2, WINDOW_SIZE[1]//2 - 40))
            
            score_surf = title_font.render(f"{black_score} - {white_score}", True, TEXT_LIGHT)
            score_rect = score_surf.get_rect(center=(WINDOW_SIZE[0]//2, WINDOW_SIZE[1]//2 + 30))
            
            screen.blit(text_surf, text_rect)
            screen.blit(score_surf, score_rect)
            pygame.display.flip()
            
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

        clock.tick(30) 

if __name__ == "__main__":
    main()