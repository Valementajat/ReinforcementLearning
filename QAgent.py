class QAgent():
    def __init__(self, actions, alpha=0.1, gamma=0.9):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        best_next_q = max(self.get_q_value(next_state, a) for a in self.actions)
        new_q = (1 - self.alpha) * self.get_q_value(state, action) + self.alpha * (reward + self.gamma * best_next_q)
        self.q_table[(state, action)] = new_q

    def choose_action(self, state):
        q_values = [self.get_q_value(state, a) for a in self.actions]
        max_q = max(q_values)
        best_actions = [a for a in self.actions if self.get_q_value(state, a) == max_q]
        return best_actions[0]  # Choose the first best action for simplicity