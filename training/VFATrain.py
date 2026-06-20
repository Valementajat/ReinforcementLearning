import numpy as np

from helper import EpisodeLogger, save_episode, show_frame




def train_vfa(env, agent, num_episodes, max_training_steps):

    n_goal = 0
    for episode in range(num_episodes):
        obs, info = env.reset()
        state = obs                
        done = False
        ep_step = 0
        """ healthy_ahead_count = 0 """


        log_this_episode = episode > num_episodes - 3
        logger = EpisodeLogger() if log_this_episode else None



        while not done and ep_step < max_training_steps:
            action = agent.select_action(state)
            """x = agent.featurize(state, action)
             if x[2] == 1.0:
                healthy_ahead_count += 1 """
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = obs

            q_values = [agent.q_value(state, a) for a in agent.actions] if log_this_episode else None
            totalReturn=env.unwrapped._episode_return
            if log_this_episode:
                logger.log_step(
                    step=ep_step, state=state, action=action, reward=reward,
                    next_state=next_state, totalReturn=totalReturn, info=info, q_values=q_values,
                    agent_pos=env.unwrapped._agent_pos, energy=env.unwrapped.energy,
                )

            agent.update(state, action, reward, next_state, terminated)

            state = next_state
            ep_step += 1
            done = terminated or truncated

              # Renderer for the last 2 episodes
            if episode  > num_episodes - 3:
                show_frame(
                    env.render(),
                    title=(f" step={ep_step} "
                        f"energy={env.unwrapped.energy} "
                        f"reward={reward:.1f} "
                        f"return={env.unwrapped._episode_return:.1f} "
                        f"ε={agent.epsilon:.3f}"
                    ),
                )


        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        reached_goal = (env.unwrapped._agent_pos == env.unwrapped.goal_pos)
        if episode % 500 == 0:
            reached_goal = (env.unwrapped._agent_pos == env.unwrapped.goal_pos)
            print(f"ep={episode:5d} | steps={ep_step:4d} | "
                f"return={env.unwrapped._episode_return:.1f} | "
                f"ε={agent.epsilon:.3f} | reached_goal={reached_goal}")
            print(f"  weights: {np.round(agent.w, 2)}")
            """ print(f"  healthy_ahead fired: {healthy_ahead_count} times this episode") """


        # Save GIFs of the last 2 episodes
        if log_this_episode:
            logger.save(f"output/vfa_training/episode_{episode}_log.txt")
            save_episode(episode, "vfa_training")
        
        if reached_goal: n_goal += 1


    print("Learned weights:", agent.w)
   
    print(f"reached_goal: {n_goal}/{num_episodes} ({100*n_goal/num_episodes:.1f}%)")