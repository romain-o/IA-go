import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import time
import queue
from env import ReversiEnv
from mcts import MCTS
import torch.optim as optim
from collections import deque
import random
import torch.nn.functional as F
import pickle

class ResBlock(nn.Module):
    """
    A standard Residual Block.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out)
        
        return out

class DualHeadResNet(nn.Module):
    """
    The AlphaZero-style architecture optimized for an 8x8 board.
    """
    def __init__(self, num_blocks=5, channels=128):
        super().__init__()
        self.conv_input = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)

        self.policy_fc = nn.Linear(2 * 8 * 8, 65) 

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = F.relu(x)

        for block in self.res_blocks:
            x = block(x)

        # Policy Head
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1) 
        policy = F.softmax(self.policy_fc(p), dim=1)

        # Value Head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value
    
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def save_game(self, game_history):
        """game_history is a list of (state, mcts_policy, player_who_moved, result)"""
        for state, policy, player, result in game_history:
            value = 1.0 if player == result else -1.0
            if result == 0: value = 0.0 # Draw
            self.buffer.append((state, policy, value))
            
    def sample_batch(self, batch_size=128):
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        return (np.array(states), np.array(policies), np.array(values, dtype=np.float32))
    
    def save_buffer(self, filepath):
        """Serializes the deque to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)
            
    def load_buffer(self, filepath):
        """Loads a saved buffer from a file."""
        try:
            with open(filepath, 'rb') as f:
                loaded_list = pickle.load(f)
                self.buffer = deque(loaded_list, maxlen=self.buffer.maxlen)
            print(f"Successfully loaded {len(self.buffer)} positions into the Replay Buffer.")
        except FileNotFoundError:
            print("No existing buffer found. Starting fresh.")


# --- 2. The GPU Batching Engine ---
def gpu_batch_evaluator(input_queue, output_pipes, weight_sync_queue, batch_size=16):
    """
    Dedicated process that sits on the GPU.
    It waits for board states from the CPU workers, batches them, 
    and sends the predictions back.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadResNet().to(device)
    model.eval() # Set to evaluation mode for MCTS
    
    print(f"GPU Evaluator online using: {device}")

    with torch.no_grad():
        while True:
            try:
                new_state_dict = weight_sync_queue.get_nowait()
                model.load_state_dict(new_state_dict)
                print("GPU Evaluator: Model updated with new weights!")
            except queue.Empty:
                pass
            batch_states = []
            worker_ids = []
            
            while len(batch_states) < batch_size:
                try:
                 
                    worker_id, state = input_queue.get_nowait()
                    worker_ids.append(worker_id)
                    batch_states.append(state)
                except queue.Empty:
                    if len(batch_states) > 0:
                        break
                    else:
                        time.sleep(0.001)

            if not batch_states:
                continue
     
            batch_tensor = torch.tensor(np.array(batch_states), dtype=torch.float32).to(device)
            
            policies, values = model(batch_tensor)
            
            policies = policies.cpu().numpy()
            values = values.cpu().numpy()

            for i, w_id in enumerate(worker_ids):
                output_pipes[w_id].send((policies[i], values[i].item()))


class RemoteEvaluator:
    """
    Acts as a bridge between the MCTS search tree and the GPU process.
    To MCTS, this looks like a normal neural network.
    """
    def __init__(self, worker_id, input_queue, pipe_conn):
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.pipe_conn = pipe_conn

    def predict(self, state):
        self.input_queue.put((self.worker_id, state))
        policy, value = self.pipe_conn.recv()
        action_probs = {action: prob for action, prob in enumerate(policy)}
        
        return action_probs, value
    
def get_symmetries(board, policy):
    """
    Takes a 3x8x8 board and a 65-length policy.
    Returns 8 mathematically equivalent (board, policy) pairs.
    """
    symmetries = []
    policy_board = policy[:64].reshape(8, 8)
    pass_prob = policy[64]
    
    for i in range(4):
        for flip in [False, True]:
            b = np.rot90(board, i, axes=(1, 2))
            p = np.rot90(policy_board, i, axes=(0, 1))
            
            if flip:
                b = np.flip(b, axis=2)
                p = np.flip(p, axis=1)
           
            p_flat = np.append(p.flatten(), pass_prob)
            
            symmetries.append((b.copy(), p_flat.copy()))
            
    return symmetries

def self_play_worker(worker_id, input_queue, pipe_conn, experience_queue, num_games=10000):
    print(f"Worker {worker_id} started.")
    env = ReversiEnv()
    mcts = MCTS(num_simulations=200)
    evaluator = RemoteEvaluator(worker_id, input_queue, pipe_conn)
    
    for game in range(num_games):
        obs, info = env.reset()
        terminated = False
        game_history = []
        
        while not terminated:
            best_action, mcts_policy = mcts.search(env, evaluator)
            if env.pass_count == 0 and len(game_history) < 8: # Dirichlet noise to encourage exploration
                best_action = np.random.choice(65, p=mcts_policy)
                
            current_player = 1 if env.is_black_turn else -1
            
            for sym_obs, sym_policy in get_symmetries(obs.copy(), mcts_policy):
                game_history.append((sym_obs, sym_policy, current_player))
            
            obs, reward, terminated, truncated, info = env.step(best_action)

        final_black_bb = env.current_player_bb if env.is_black_turn else env.opp_bb
        final_white_bb = env.opp_bb if env.is_black_turn else env.current_player_bb
        
        p1_score = final_black_bb.bit_count()
        p2_score = final_white_bb.bit_count()
        
        if p1_score > p2_score: true_winner = 1
        elif p2_score > p1_score: true_winner = -1
        else: true_winner = 0

        history_with_result = [(s, p, curr_p, true_winner) for s, p, curr_p in game_history]
        experience_queue.put(history_with_result)
        
    print(f"Worker {worker_id} finished all games.")
    
def train_network(model, optimizer, replay_buffer, batch_size=128, device="cuda"):
    """Pulls a batch of MCTS data and updates the ResNet weights."""
    if len(replay_buffer.buffer) < batch_size:
        return 0.0, 0.0
        
    model.train() 
    states, target_policies, target_values = replay_buffer.sample_batch(batch_size)
    
    states = torch.tensor(states, dtype=torch.float32).to(device)
    target_policies = torch.tensor(target_policies, dtype=torch.float32).to(device)
    target_values = torch.tensor(target_values).unsqueeze(1).to(device)
    
    optimizer.zero_grad()
    
    predicted_policies, predicted_values = model(states)
    
    value_loss = torch.nn.functional.mse_loss(predicted_values, target_values)
    policy_loss = -torch.sum(target_policies * torch.log(predicted_policies + 1e-8)) / batch_size
    
    total_loss = value_loss + policy_loss
    
    total_loss.backward()
    optimizer.step()
    
    return value_loss.item(), policy_loss.item()

if __name__ == "__main__":
    mp.set_start_method('spawn')

    NUM_WORKERS = 14 
    BATCH_SIZE = 16 

    input_queue = mp.Queue()
    parent_pipes = []
    child_pipes = []
    
    for _ in range(NUM_WORKERS):
        parent_conn, child_conn = mp.Pipe()
        parent_pipes.append(parent_conn)
        child_pipes.append(child_conn)

    weight_sync_queue = mp.Queue()
    gpu_process = mp.Process(
        target=gpu_batch_evaluator, 
        args=(input_queue, parent_pipes, weight_sync_queue, BATCH_SIZE)
    )
    gpu_process.start()


    experience_queue = mp.Queue()

    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(
            target=self_play_worker, 
            args=(i, input_queue, child_pipes[i], experience_queue)
        )
        p.start()
        workers.append(p)

  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    master_model = DualHeadResNet().to(device)
    optimizer = optim.Adam(master_model.parameters(), lr=0.0005, weight_decay=1e-4) 
    checkpoint_path = r"model.pth"
    checkpoint_bundle = torch.load(checkpoint_path, map_location=device, weights_only=False)

    master_model.load_state_dict(checkpoint_bundle['model_state_dict'])
    optimizer.load_state_dict(checkpoint_bundle['optimizer_state_dict'])
    games_played = checkpoint_bundle['games_played']
    replay_buffer = ReplayBuffer(capacity=100000)

    buffer_path = checkpoint_path.replace("reversi_bundle_", "reversi_buffer_").replace(".pth", ".pkl").replace("checkpoints", "buffer_checkpoints")
    replay_buffer.load_buffer(buffer_path)

        
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    print(f"Starting training loop. Games already played: {games_played}")
    
    try:
        while True:
            try:
                game_history = experience_queue.get(timeout=10)
                replay_buffer.save_game(game_history)
                games_played += 1
                
                if games_played % 20 == 0 and len(replay_buffer.buffer) > 10000:
                    v_loss, p_loss = train_network(master_model, optimizer, replay_buffer, batch_size=512, device=device)
                    
                    scheduler.step()
                    print(f"Game {games_played} | Value Loss: {v_loss:.4f} | Policy Loss: {p_loss:.4f}")
                    cpu_state_dict = {k: v.cpu() for k, v in master_model.state_dict().items()}
                    weight_sync_queue.put(cpu_state_dict)

                    if games_played % 500 == 0:
                        checkpoint_bundle = {
                            'games_played': games_played,
                            'model_state_dict': master_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                        }
                        torch.save(checkpoint_bundle, f"checkpoints/reversi_bundle_game_{games_played}.pth")
                        replay_buffer.save_buffer(f"buffer_checkpoints/reversi_buffer_game_{games_played}.pkl")
                        print("--> Model and Replay Buffer Checkpoints Saved!")

                    
            except queue.Empty:
                if not any(p.is_alive() for p in workers):
                    break

        for p in workers:
            p.join()

        gpu_process.terminate()
        print("Self-play data generation complete.")
        
    except KeyboardInterrupt:
        print(f"\n[!] Halting training at Game {games_played}...")
        
        print("Saving emergency checkpoints...")
        torch.save(master_model.state_dict(), f"reversi_model_EMERGENCY_game_{games_played}.pth")
        replay_buffer.save_buffer(f"reversi_buffer_EMERGENCY_game_{games_played}.pkl")
        print("--> Model and Replay Buffer safely saved to disk.")
        
        print("Terminating background worker processes (this may take a second)...")
        for p in workers:
            p.terminate()
            p.join()

        gpu_process.terminate()
        gpu_process.join()
        
        print("All processes cleanly shut down. Exiting.")