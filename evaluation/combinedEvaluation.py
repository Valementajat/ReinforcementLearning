# To save time, this part was made with Claude AI based on the both training files
# For better evaluation this file was created to train both models on the same maps


"""
CombinedEvaluation.py

Evaluates the trained Q-learning agent and the trained VFA agent
side-by-side, greedily (epsilon=0), on the SAME set of randomly generated
maps -- same seed per evaluation episode for both agents -- so that
"policy quality across different map configurations" (per the project
brief) is measured on identical test conditions for both algorithms.

Drop this file in your `evaluation/` folder next to QEvaluation.py and
VFAEvaluation.py.

Call it from main.py the same way you'd call evaluate_q_learning / evaluate_vfa:

    from evaluation.CombinedEvaluation import evaluate_combined

    eval_metrics = evaluate_combined(env, Qagent, VFAagent)

Returns a dict: {"q": {...}, "vfa": {...}}, each holding numpy arrays
"returns", "steps", "success" (one entry per evaluation episode/map).
A CSV with the same data is written to
output/comparison/evaluation_metrics.csv.

Only the first `render_first_n` evaluation episodes are rendered to a GIF
(and, for VFA, logged step-by-step). With num_eval_episodes possibly in the
dozens, rendering every single one would be slow and create a lot of files
for little benefit -- by default only one representative episode per
algorithm is saved visually, while every episode still contributes to the
aggregate success-rate / return / steps statistics.
"""

import csv
import os

import numpy as np

from helper import EpisodeLogger, save_episode, show_frame


def _eval_q_episode(env, agent, map_seed, max_steps, render, tag):
    obs, info = env.reset(seed=map_seed)
    state = tuple(obs)
    done = False
    ep_step = 0
    ep_return = 0.0

    while not done and ep_step < max_steps:
        action = agent.select_action(state)
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = tuple(obs)

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
                       f"return={env.unwrapped._episode_return:.1f}"),
            )

    reached_goal = (env.unwrapped._agent_pos == env.unwrapped.goal_pos)

    if render:
        save_episode(tag, "q_evaluation")

    return ep_return, ep_step, reached_goal


def _eval_vfa_episode(env, agent, map_seed, max_steps, render, tag, logger):
    obs, info = env.reset(seed=map_seed)
    state = obs
    done = False
    ep_step = 0
    ep_return = 0.0

    while not done and ep_step < max_steps:
        action = agent.select_action(state)
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = obs
        ep_step += 1
        ep_return += reward
        done = terminated or truncated

        if logger is not None:
            q_values = [agent.q_value(state, a) for a in agent.actions]
            logger.log_step(
                step=ep_step, state=state, action=action, reward=reward,
                next_state=next_state, totalReturn=env.unwrapped._episode_return,
                info=info, q_values=q_values,
                agent_pos=env.unwrapped._agent_pos, energy=env.unwrapped.energy,
            )

        state = next_state

        if render:
            show_frame(
                env.render(),
                title=(f"EVAL step={ep_step} "
                       f"energy={env.unwrapped.energy} "
                       f"reward={reward:.1f} "
                       f"return={env.unwrapped._episode_return:.1f}"),
            )

    reached_goal = (env.unwrapped._agent_pos == env.unwrapped.goal_pos)

    if logger is not None:
        logger.save(f"output/vfa_evaluation/episode_{tag}_log.txt")
    if render:
        save_episode(tag, "vfa_evaluation")

    return ep_return, ep_step, reached_goal


def evaluate_combined(env, q_agent, vfa_agent, num_eval_episodes=20,
                       max_eval_steps=300, map_seed_base=10_000,
                       render_first_n=1,
                       metrics_path="output/comparison/evaluation_metrics.csv"):
    """
    Greedily evaluate a Q-learning agent and a VFA agent on the same set of
    `num_eval_episodes` randomly generated maps.

    Parameters
    ----------
    env               : the shared gym environment
    q_agent, vfa_agent: trained agents
    num_eval_episodes : how many maps to evaluate on
    max_eval_steps    : per-episode step cap (safety net; episodes normally
                         end via goal / energy depletion well before this)
    map_seed_base     : offset for evaluation map seeds -- kept well above
                         typical training seed ranges so eval maps don't
                         coincide with maps the agents trained on
    render_first_n    : number of evaluation episodes (per algorithm) to
                         render to GIF / log to file
    metrics_path      : where to write the per-episode comparison CSV

    Returns
    -------
    dict with keys "q" and "vfa", each mapping to a dict of numpy arrays:
        {"returns": ..., "steps": ..., "success": ...}
    """
    q_epsilon_backup = q_agent.epsilon
    vfa_epsilon_backup = vfa_agent.epsilon
    q_agent.epsilon = 0.0
    vfa_agent.epsilon = 0.0

    q_returns = np.zeros(num_eval_episodes)
    q_steps = np.zeros(num_eval_episodes, dtype=int)
    q_success = np.zeros(num_eval_episodes, dtype=bool)

    vfa_returns = np.zeros(num_eval_episodes)
    vfa_steps = np.zeros(num_eval_episodes, dtype=int)
    vfa_success = np.zeros(num_eval_episodes, dtype=bool)

    for i in range(num_eval_episodes):
        map_seed = map_seed_base + i
        render = i < render_first_n

        ep_ret, ep_steps, reached = _eval_q_episode(
            env, q_agent, map_seed, max_eval_steps, render, tag=f"evaluation_{i}"
        )
        q_returns[i] = ep_ret
        q_steps[i] = ep_steps
        q_success[i] = reached

        logger = EpisodeLogger() if render else None
        ep_ret, ep_steps, reached = _eval_vfa_episode(
            env, vfa_agent, map_seed, max_eval_steps, render, tag=f"evaluation_{i}",
            logger=logger,
        )
        vfa_returns[i] = ep_ret
        vfa_steps[i] = ep_steps
        vfa_success[i] = reached

    q_agent.epsilon = q_epsilon_backup
    vfa_agent.epsilon = vfa_epsilon_backup

    print(f"\nEvaluation over {num_eval_episodes} maps (seeds "
          f"{map_seed_base}..{map_seed_base + num_eval_episodes - 1}):")
    print(f"  Q-learning : success={q_success.sum()}/{num_eval_episodes} "
          f"({100*q_success.mean():.1f}%)  "
          f"avg_return={q_returns.mean():.1f}  avg_steps={q_steps.mean():.1f}")
    print(f"  VFA        : success={vfa_success.sum()}/{num_eval_episodes} "
          f"({100*vfa_success.mean():.1f}%)  "
          f"avg_return={vfa_returns.mean():.1f}  avg_steps={vfa_steps.mean():.1f}")

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_episode", "map_seed",
                          "q_return", "q_steps", "q_success",
                          "vfa_return", "vfa_steps", "vfa_success"])
        for i in range(num_eval_episodes):
            writer.writerow([
                i, map_seed_base + i,
                q_returns[i], q_steps[i], int(q_success[i]),
                vfa_returns[i], vfa_steps[i], int(vfa_success[i]),
            ])
    print(f"Saved per-episode evaluation metrics: {metrics_path}")

    return {
        "q": {"returns": q_returns, "steps": q_steps, "success": q_success},
        "vfa": {"returns": vfa_returns, "steps": vfa_steps, "success": vfa_success},
    }