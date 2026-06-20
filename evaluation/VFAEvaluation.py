from helper import EpisodeLogger, save_episode, show_frame



def evaluate_vfa(env, agent):
    # ── Evaluation (greedy, render last episode) ────────────────────────
    agent.epsilon = 0.0
    obs, info = env.reset()
    state = obs
    done = False
    ep_step = 0
    logger = EpisodeLogger() 

    while not done:
        action = agent.select_action(state)
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = obs
        ep_step += 1
        done = terminated or truncated

        totalReturn=env.unwrapped._episode_return
        q_values = [agent.q_value(state, a) for a in agent.actions]
        logger.log_step(
                    step=ep_step, state=state, action=action, reward=reward,
                    next_state=next_state, totalReturn=totalReturn, info=info, q_values = q_values,
                    agent_pos=env.unwrapped._agent_pos, energy=env.unwrapped.energy,
                )

        state = next_state

        show_frame(
            env.render(),
            title=(f"EVAL step={ep_step} "
                f"energy={env.unwrapped.energy} "
                f"reward={reward:.1f} "
                f"return={env.unwrapped._episode_return:.1f}")
        )

    logger.save(f"output/vfa_evaluation/episode_evaluation_log.txt")

    save_episode("vfa_evaluation", "vfa_evaluation")