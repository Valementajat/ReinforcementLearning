import numpy as np
from env import _DELTA
from helper import EpisodeLogger



import numpy as np

# Must match env.py constants
EMPTY, TOXIC, HEALTHY, START, GOAL, WALL, FOG = 0, 1, 2, 3, 4, 5, 6

FOG_RADIUS = 3
WINDOW_SIDE = 2 * FOG_RADIUS + 1   # 7
CENTER = FOG_RADIUS         # index 3 — agent's own position in the window



def featurize(state, action, max_energy_bins=10, lookahead=3):
    obs = np.asarray(state)

    # Split window (49) from energy_bin (1)
    window_flat = obs[:-1]
    energy_bin = obs[-1]

    window_side = int(np.sqrt(len(window_flat)))   # 7
    window = window_flat.reshape(window_side, window_side)
    center = window_side // 2

    
    dr, dc = _DELTA[action]
    target_r, target_c = center + dr, center + dc
    target_cell = window[target_r, target_c]

    n_cells = window_side * window_side

    energy_norm = float(energy_bin) / max_energy_bins
    fog_fraction = float((window == FOG).sum()) / n_cells


    # Completely made with AI, as a last hail mary to get this model to work
    # ── Directional ray-cast lookahead ──────────────────────────────────
    # Walk `lookahead` cells out from the agent in the action's own
    # direction, weighting closer cells more heavily (1/i). This is what
    # lets up/right/down/left look different to the linear model even when
    # the immediate next cell is the same type for two directions.
    k = min(lookahead, center)
    ray_toxic = ray_healthy = ray_goal = ray_fog = 0.0
    for i in range(1, k + 1):
        rr, cc = center + i * dr, center + i * dc
        if 0 <= rr < window_side and 0 <= cc < window_side:
            cell = window[rr, cc]
        else:
            cell = FOG  # past the lookahead array: treat as unknown
        weight = 1.0 / i
        ray_toxic   += weight * float(cell == TOXIC)
        ray_healthy += weight * float(cell == HEALTHY)
        ray_goal    += weight * float(cell == GOAL)
        ray_fog     += weight * float(cell == FOG)

  
    features = [
        1.0,
        float(target_cell == TOXIC),
        float(target_cell == HEALTHY),
        float(target_cell == WALL),
        float(target_cell == GOAL),

        # Try to learn a penalty we hang around the start pos
        float((window == START).sum() > 0) * (1.0 - energy_norm),
        # desperate toxic 
        float((target_cell==TOXIC) * (1-energy_norm) ),
        energy_norm,      
        float((window == TOXIC).sum()) / n_cells,
        float((window == HEALTHY).sum()) / n_cells,
        #Encourage healthy pickup
        float((target_cell == HEALTHY)) * (1.0 - energy_norm),
        fog_fraction,
        float((window == GOAL).sum()) / n_cells,
        fog_fraction * (1.0 - energy_norm),   #  interaction term, to see if the 
        # agent learns that low energy and more fog is bad

        # Completely made with AI, as a last hail mary to get this model to work
        # directional, depth-weighted lookahead (this action's ray only)
        ray_toxic,
        ray_healthy,
        ray_goal,
        ray_fog,
        
    ]
    return np.array(features, dtype=np.float64)


class VFAAgent:
    def __init__(self, actions, featurize, w,alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999,seed=None):
        self.actions = actions
        self.featurize = featurize
        # wights vector, at start [0, ... , n]
        self.w = w
        # LR and discount factor
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        # E psilon-greedy parameters 
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self._rng = np.random.default_rng(  seed)

    # Q^​(s,a,w)=wTx(s,a).
    # Matrix multiplication
    def q_value(self, state, action):
        x = self.featurize(state, action)
        return np.dot(self.w, x)

    # Epsilon greedy
    def select_action(self, state):
        if self._rng.random() < self.epsilon:
            return self._rng.choice(self.actions)
        
        # If not epsilon then map the q values approximation for each action 
        q_values = [self.q_value(state, a) for a in self.actions]
        # print(f"q_values: {q_values}")  # Debugging line to print q_values
        
        # Return  action 
        return self._argmax_random_tiebreak(np.array(q_values))

    def update(self, state, action, reward, next_state, terminated: bool):
        x = self.featurize(state, action)

        # In case terminated then target is just reward, to not add noise 
        if terminated:
            target = float(reward)
        else:
            # α(target−Q^​(st​,at​,w))x(st​,at​)
            best_next_q = max(self.q_value(next_state, a) for a in self.actions)
            # target=rt+1​+γa′max​Q^​(st+1​,a′,w)
            target = reward + self.gamma * best_next_q


        
        td_error = target - self.q_value(state, action)
        #Δw=α(target−Q^​(st_​,a_t​,w))x(s_t​,a_t​)
        self.w += self.alpha * td_error * x

    def _argmax_random_tiebreak(self, values: np.ndarray) -> int:
        max_value = values.max()
        candidates = np.flatnonzero(values == max_value)
        
        #Take the actual action
        idx = int(self._rng.choice(candidates))
        return self.actions[idx]
