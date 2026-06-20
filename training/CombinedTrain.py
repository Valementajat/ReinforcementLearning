# To save time, this part was made with Claude AI based on the both training files
# For better evaluation this file was created to train both models on the same maps

"""
CombinedTrain.py

Runs tabular Q-learning and linear VFA training side by side, episode by
episode, on the SAME randomly generated map each episode (the same seed is
passed to both agents' env.reset() calls for a given episode index). This
keeps the comparison between the two algorithms fair: any difference in
convergence speed, sample efficiency, or final policy quality reflects the
algorithms themselves, not which agent happened to draw an easier map.

Drop this file in your `training/` folder next to QTrain.py and VFATrain.py.

Call it from main.py the same way you'd call train_q_learning / train_vfa:

    from training.CombinedTrain import train_combined

    metrics = train_combined(
        env, Qagent, VFAagent,
        num_episodes=num_episodes,
        max_training_steps=max_training_steps,
    )

`metrics` is a dict: {"q": {...}, "vfa": {...}}, each holding numpy arrays
"returns", "steps", "success" (one entry per episode) — useful for plotting
convergence curves afterward. A CSV with the same data is also written to
output/comparison/training_metrics.csv.
"""

import csv
import os

import numpy as np

from helper import EpisodeLogger, save_episode, show_frame


def _run_q_episode(env, agent, map_seed, max_steps, render, episode):
    obs, info = env.reset(seed=map_seed)
    state = tuple(obs)
    done = False
    ep_step = 0
    ep_return = 0.0

    while not done and ep_step < max_steps:
        action = agent.select_action(state)
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = tuple(obs)

        agent.update_q_value(state, action, reward, next_state, terminated)

        state = next_state
        ep_step += 1
        ep_return += reward
        done = terminated or truncated

        if render:
            show_frame(
                env.render(),
                title=(f" step={ep_step} "
                       f"energy={env.unwrapped.energy} "
                       f"reward={reward:.1f} "
                       f"return={env.unwrapped._episode_return:.1f} "
                       f"ε={agent.epsilon:.3f}"),
            )

    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
    reached_goal = (env.unwrapped._agent_pos == env.unwrapped.goal_pos)

    if render:
        save_episode(episode, "q_training")

    return ep_return, ep_step, reached_goal


def _run_vfa_episode(env, agent, map_seed, max_steps, render, episode, logger):
    obs, info = env.reset(seed=map_seed)
    state = obs
    done = False
    ep_step = 0
    ep_return = 0.0

    while not done and ep_step < max_steps:
        action = agent.select_action(state)
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = obs

        if logger is not None:
            q_values = [agent.q_value(state, a) for a in agent.actions]
            logger.log_step(
                step=ep_step, state=state, action=action, reward=reward,
                next_state=next_state, totalReturn=env.unwrapped._episode_return,
                info=info, q_values=q_values,
                agent_pos=env.unwrapped._agent_pos, energy=env.unwrapped.energy,
            )

        agent.update(state, action, reward, next_state, terminated)

        state = next_state
        ep_step += 1
        ep_return += reward
        done = terminated or truncated

        if render:
            show_frame(
                env.render(),
                title=(f" step={ep_step} "
                       f"energy={env.unwrapped.energy} "
                       f"reward={reward:.1f} "
                       f"return={env.unwrapped._episode_return:.1f} "
                       f"ε={agent.epsilon:.3f}"),
            )

    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
    reached_goal = (env.unwrapped._agent_pos == env.unwrapped.goal_pos)

    if logger is not None:
        logger.save(f"output/vfa_training/episode_{episode}_log.txt")
    if render:
        save_episode(episode, "vfa_training")

    return ep_return, ep_step, reached_goal


def train_combined(env, q_agent, vfa_agent, num_episodes, max_training_steps,
                    map_seed_base=0,
                    metrics_path="output/comparison/training_metrics.csv"):
    """
    Train a tabular Q-learning agent and a linear VFA agent episode-by-episode
    on identical maps, and collect per-episode metrics for both.

    Parameters
    ----------
    env             : the shared gym environment (same one used for both agents)
    q_agent         : a QAgent instance
    vfa_agent       : a VFAAgent instance
    num_episodes    : number of episodes to train each agent for
    max_training_steps : per-episode step cap
    map_seed_base   : offset added to the episode index to form each
                      episode's map seed (lets you shift the whole sequence
                      of maps without overlapping a previous run's seeds)
    metrics_path    : where to write the per-episode comparison CSV

    Returns
    -------
    dict with keys "q" and "vfa", each mapping to a dict of numpy arrays:
        {"returns": ..., "steps": ..., "success": ...}
    """
    q_returns = np.zeros(num_episodes)
    q_steps = np.zeros(num_episodes, dtype=int)
    q_success = np.zeros(num_episodes, dtype=bool)

    vfa_returns = np.zeros(num_episodes)
    vfa_steps = np.zeros(num_episodes, dtype=int)
    vfa_success = np.zeros(num_episodes, dtype=bool)

    q_goal_count = 0
    vfa_goal_count = 0

    for episode in range(num_episodes):
        map_seed = map_seed_base + episode
        render = episode > num_episodes - 3
        log_this_episode = render

        # --- Q-learning episode -------------------------------------------------
        ep_ret, ep_steps, reached = _run_q_episode(
            env, q_agent, map_seed, max_training_steps, render, episode
        )
        q_returns[episode] = ep_ret
        q_steps[episode] = ep_steps
        q_success[episode] = reached
        q_goal_count += int(reached)

        # --- VFA episode, exact same map -----------------------------------------
        logger = EpisodeLogger() if log_this_episode else None
        ep_ret, ep_steps, reached = _run_vfa_episode(
            env, vfa_agent, map_seed, max_training_steps, render, episode, logger
        )
        vfa_returns[episode] = ep_ret
        vfa_steps[episode] = ep_steps
        vfa_success[episode] = reached
        vfa_goal_count += int(reached)

        if episode % 500 == 0:
            print(f"ep={episode:5d} | "
                  f"Q: steps={q_steps[episode]:4d} return={q_returns[episode]:7.1f} "
                  f"ε={q_agent.epsilon:.3f} goal={q_success[episode]} | "
                  f"VFA: steps={vfa_steps[episode]:4d} return={vfa_returns[episode]:7.1f} "
                  f"ε={vfa_agent.epsilon:.3f} goal={vfa_success[episode]}")

    print(f"\nQ-learning   reached_goal: {q_goal_count}/{num_episodes} "
          f"({100*q_goal_count/num_episodes:.1f}%)  |  "
          f"q_table entries: {len(q_agent.q_table)}")
    print(f"VFA          reached_goal: {vfa_goal_count}/{num_episodes} "
          f"({100*vfa_goal_count/num_episodes:.1f}%)")
    print(f"VFA learned weights: {np.round(vfa_agent.w, 3)}")

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "map_seed",
                          "q_return", "q_steps", "q_success",
                          "vfa_return", "vfa_steps", "vfa_success"])
        for i in range(num_episodes):
            writer.writerow([
                i, map_seed_base + i,
                q_returns[i], q_steps[i], int(q_success[i]),
                vfa_returns[i], vfa_steps[i], int(vfa_success[i]),
            ])
    print(f"Saved per-episode comparison metrics: {metrics_path}")

    return {
        "q": {"returns": q_returns, "steps": q_steps, "success": q_success},
        "vfa": {"returns": vfa_returns, "steps": vfa_steps, "success": vfa_success},
    }