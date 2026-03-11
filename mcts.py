import math
import numpy as np

class Node:
    """
    A lightweight MCTS Node. 
    Optimized to minimize memory overhead for the 7800X3D's L3 Cache.
    """
    __slots__ = ['state', 'parent', 'action_taken', 'children', 
                 'visit_count', 'value_sum', 'prior_prob']

    def __init__(self, state, parent=None, action_taken=None, prior_prob=0.0):
        self.state = state                  # The environment bitboards/state
        self.parent = parent                # Parent Node
        self.action_taken = action_taken    # Action that led to this node
        self.prior_prob = prior_prob        # P(s, a) from Neural Net
        
        self.children = {}                  # Action -> Node
        self.visit_count = 0                # N(s, a)
        self.value_sum = 0.0                # Total value from backpropagation

    @property
    def q_value(self):
        """Q(s, a): Expected value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, action_probs):
        """
        Expands the node by creating children for all legal actions.
        action_probs: A dictionary of {action: probability} from the Neural Net.
        """
        for action, prob in action_probs.items():
            if action not in self.children:
                # In a full implementation, you'd apply the move to get the next state
                # next_state = apply_move(self.state, action)
                self.children[action] = Node(state=None, parent=self, action_taken=action, prior_prob=prob)

    def select_child(self, c_puct=1.5):
        """Selects the child with the highest PUCT score."""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children.values():
            # PUCT Formula
            u_score = -child.q_value + c_puct * child.prior_prob * (
                math.sqrt(self.visit_count) / (1 + child.visit_count)
            )
            
            if u_score > best_score:
                best_score = u_score
                best_child = child
                
        return best_child

    def backpropagate(self, value):
        """Passes the evaluation up the tree to the root."""
        self.visit_count += 1
        self.value_sum += value
        if self.parent is not None:
            # Reversi is zero-sum, so the value flips perspective for the opponent
            self.parent.backpropagate(-value)


class MCTS:
    def __init__(self, num_simulations=800):
        self.num_simulations = num_simulations
        
    def search(self, env, neural_net_evaluator, add_noise=True):
        root = Node(state=env._get_obs())
        
        action_probs, _ = neural_net_evaluator.predict(root.state)
        valid_mask = env._get_info()["action_mask"]
        
        legal_actions = [a for a in range(65) if valid_mask[a] == 1]
        action_probs = {a: action_probs[a] for a in legal_actions}
        
        # --- NEW: Only add Dirichlet noise if we are training ---
        if add_noise and legal_actions:
            noise = np.random.dirichlet([1.0] * len(legal_actions))
            for i, action in enumerate(legal_actions):
                action_probs[action] = 0.75 * action_probs[action] + 0.25 * noise[i]
                
        root.expand(action_probs)

        for _ in range(self.num_simulations):
            node = root
            
            # --- 1. Snapshot the actual environment state ---
            saved_state = env.get_state()
            
            # --- 2. SELECTION: Traverse tree and advance environment ---
            while node.children:
                node = node.select_child()
                env.step(node.action_taken) # Physically play the move in the env

            # If this is a new node, grab the real observation from the env
            if node.state is None:
                node.state = env._get_obs()

            # --- 3. EXPANSION & EVALUATION ---
            # Check if the simulated game is over
            if env.pass_count >= 2 or (env.current_player_bb | env.opp_bb) == 0xFFFFFFFFFFFFFFFF:
                # Terminal state: Calculate actual win/loss value
                p1_score = env.current_player_bb.bit_count()
                p2_score = env.opp_bb.bit_count()
                if p2_score > p1_score: value = -1.0
                elif p2_score < p1_score: value = +1.0
                else: value = 0.0
                action_probs = {} # No moves to expand
            else:
                # Ask GPU for predictions
                action_probs, value = neural_net_evaluator.predict(node.state)
                
                # Filter out illegal moves so the AI doesn't explore them
                valid_mask = env._get_info()["action_mask"]
                action_probs = {a: p for a, p in action_probs.items() if valid_mask[a] == 1}
                
                node.expand(action_probs)

            # --- 4. BACKPROPAGATION ---
            node.backpropagate(value)
            
            # --- 5. Restore the environment for the next simulation ---
            env.set_state(saved_state)

        # Return the most robustly explored action
        total_visits = sum(child.visit_count for child in root.children.values())
        mcts_policy = np.zeros(65, dtype=np.float32)
        for action, child in root.children.items():
            mcts_policy[action] = child.visit_count / total_visits
            
        best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        
        return best_action, mcts_policy