import numpy as np
from collections import defaultdict
import config

def run_simulation(env, agent, num_episodes, steps_per_episode, is_dqn=False, train_dqn=False):
    """Runs the simulation for a given agent."""
    results = defaultdict(list)
    all_episode_rewards = []

    for e in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_stats = {
            'handovers': 0,
            'failures': 0,
            'ping_pongs': 0,
            'latency': [],
            'bandwidth': [],
            'serving_rsrp': []
        }

        if not is_dqn: # Reset traditional agent state per episode
            agent.reset()

        for step in range(steps_per_episode):
            if is_dqn:
                action = agent.choose_action(state)
            else: # Traditional agent needs current serving cell info
                action = agent.choose_action(state, env.serving_gnb_idx)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Store experience for DQN
            if is_dqn:
                agent.remember(state, action, reward, next_state, done)

            # Collect stats from info dict
            if info.get('handover_occurred', False):
                episode_stats['handovers'] += 1
            if info.get('handover_failed', False):
                episode_stats['failures'] += 1
            if info.get('ping_pong', False):
                episode_stats['ping_pongs'] += 1
            episode_stats['latency'].append(info.get('latency', 0))
            episode_stats['bandwidth'].append(info.get('bandwidth', 0))
            episode_stats['serving_rsrp'].append(info.get('serving_rsrp', -150))

            state = next_state

            # Train DQN agent
            if is_dqn and train_dqn and len(agent.memory) > config.BATCH_SIZE:
                 loss = agent.replay(config.BATCH_SIZE)
                 # Optional: print loss or log it

            if done:
                break

        # --- End of Episode ---
        all_episode_rewards.append(episode_reward)

        # Calculate episode averages/totals
        results['total_handovers'].append(episode_stats['handovers'])
        results['total_failures'].append(episode_stats['failures'])
        results['total_ping_pongs'].append(episode_stats['ping_pongs'])
        results['avg_latency'].append(np.mean(episode_stats['latency']) if episode_stats['latency'] else 0)
        results['avg_bandwidth'].append(np.mean(episode_stats['bandwidth']) if episode_stats['bandwidth'] else 0)
        results['avg_rsrp'].append(np.mean(episode_stats['serving_rsrp']) if episode_stats['serving_rsrp'] else -150)
        # Calculate rates (simple version: per episode)
        total_ho_attempts = episode_stats['handovers'] + episode_stats['failures']
        results['failure_rate'].append(episode_stats['failures'] / total_ho_attempts if total_ho_attempts > 0 else 0)
        # Ping pong rate relative to successful handovers
        results['ping_pong_rate'].append(episode_stats['ping_pongs'] / episode_stats['handovers'] if episode_stats['handovers'] > 0 else 0)


        print(f"Episode {e+1}/{num_episodes} - Reward: {episode_reward:.2f}"
              f" - HOs: {episode_stats['handovers']}"
              f" - Fails: {episode_stats['failures']}"
              f" - PPs: {episode_stats['ping_pongs']}"
              f" - Epsilon: {agent.epsilon:.4f}" if is_dqn else "")
        if is_dqn and train_dqn and e % 10 == 0: # Save weights periodically
             agent.save(f"dqn_handover_weights_ep{e}.weights.h5")


    print(f"Simulation finished for {'DQN' if is_dqn else 'Traditional'} agent.")
    return results, all_episode_rewards