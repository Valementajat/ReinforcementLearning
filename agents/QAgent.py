
import numpy as np


class QAgent():
    """Tabular Q-learning agent for environments with discrete state and
    action spaces.

    Parameters
    ----------
    n_states, n_actions : int
        Sizes of the state and action spaces. For a Gymnasium environment
        with ``Discrete`` spaces these are ``env.observation_space.n`` and
        ``env.action_space.n``.
    alpha : float
        Learning rate (step size) of the Q-learning update.
    gamma : float
        Discount factor in [0, 1].
    epsilon_start, epsilon_min : float
        Initial and floor values of the epsilon-greedy exploration rate.
    epsilon_decay : float
        Multiplicative decay applied to ``epsilon`` at the end of each
        episode: ``epsilon <- max(epsilon_min, epsilon * epsilon_decay)``.
    initial_q : float
        Constant value used to initialise the Q-table. ``0.0`` is the
        textbook choice; positive values implement *optimistic
        initialisation*, which encourages early exploration.
    seed : int, optional
        Seed of the agent's internal random number generator, used for
        epsilon-greedy tie-breaking and exploratory action sampling.
    """
        
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999, seed=None):
        self.actions = actions
        # LR and discount factor
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        # E psilon-greedy parameters 
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self._rng = np.random.default_rng(seed)
        # A dictionary Q-table to store Q-values for state-action pairs
        # Chosen because we would potentianally have a large state space, 
        # due to action space, energy bins and fog window
        self.q_table = {}
        
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool):

        # Greedy action 
        best_next_q = max(self.get_q_value(next_state, a) for a in self.actions)
        
        bootstrap = 0.0 if terminated else best_next_q
        td_target = reward + self.gamma * bootstrap

        td_error = td_target - self.get_q_value(state, action)
        # Q-learning update rule 
        new_q = self.get_q_value(state, action) + self.alpha * ( td_error)
        self.q_table[(state, action)] = new_q

    def select_action(self, state):
        # Epsilon-greedy 
        if self._rng.random() < self.epsilon:
            return self._rng.choice(self.actions)
        # Get the Q-values for all actions in the current state
        q_values = [self.get_q_value(state, a) for a in self.actions]
        
        #Return tiebrake action
        return self._argmax_random_tiebreak(np.array(q_values))
    
    def _argmax_random_tiebreak(self, values: np.ndarray) -> int:
        max_value = values.max()
        candidates = np.flatnonzero(values == max_value)
        
        #Take the actual action
        idx = int(self._rng.choice(candidates))
        return self.actions[idx]
