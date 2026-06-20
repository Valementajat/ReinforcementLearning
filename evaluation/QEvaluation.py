from helper import save_episode, show_frame


def evaluate_q_learning(env, agent):
    """
    Evaluate the trained Q-learning agent in the environment.

    Parameters
    ----------
    env   : gym.Env — the environment to evaluate in
    agent : QAgent — the trained Q-learning agent

    Returns
    -------
    None
    """
    obs, info = env.reset()
    state = tuple(obs)
    done = False
    ep_step = 0
    while not done: 
        agent.epsilon = 0.0  # No exploration during evaluation
        action = agent.select_action(state)
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = tuple(obs)
        
        
        state = next_state
        done = terminated or truncated
        ep_step += 1
        # Renderer
        show_frame(
            env.render(),
            title=(f" step={ep_step} "
                f"energy={env.unwrapped.energy} "
                f"reward={reward:.1f} "
                f"return={env.unwrapped._episode_return:.1f} "
                f"ε={agent.epsilon:.3f}"
                ),
        )
    save_episode("evaluation", "q_evaluation")