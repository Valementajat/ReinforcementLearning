from helper import save_episode, show_frame


def train_q_learning(env, agent, num_episodes=1000, max_training_steps=100):

    for episode in range(num_episodes):
        (obs, info) = env.reset()
        state = tuple(obs)
        done = False
        ep_step = 0

        while not done and ep_step < max_training_steps:
            action = agent.select_action(state)
            obs, reward, terminated, truncated, info  = env.step(action)
            next_state = tuple(obs)
            
            agent.update_q_value(state, action, reward, next_state, terminated)
            ep_step += 1
            state = next_state
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

        # Print where we are in training every 500 episodes
        if episode % 500 == 0:
            reached_goal = (env.unwrapped._agent_pos == env.unwrapped.goal_pos)
            print(f"ep={episode:5d} | steps={ep_step:4d} | "
                f"return={env.unwrapped._episode_return:.1f} | "
                f"ε={agent.epsilon:.3f} | reached_goal={reached_goal}")
            
        # Save GIFs of the last 2 episodes
        if episode  > num_episodes - 3:
            save_episode(episode, "q_training")

        print("Q-learning q_table. ", len(agent.q_table), "entries", end="\r" )