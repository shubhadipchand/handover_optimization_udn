import numpy as np
import config # Ensure config is imported
from environment import HandoverEnv
from dqn_agent import DQNAgent
from traditional_agent import TraditionalAgent
from simulation import run_simulation
from plotting import plot_results, plot_rewards
import os
import tensorflow as tf # Import tensorflow

def main():
    # Optional: GPU memory growth configuration
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # --- Initialize UDN Environment ---
    print("--- Initializing UDN Environment ---")
    # Add a seed for reproducibility of the environment if gNBs are random
    env_seed = 42
    env_udn = HandoverEnv(
        area_width=config.AREA_WIDTH,
        area_height=config.AREA_HEIGHT,
        num_gnbs=config.NUM_GNBS_UDN,
        deployment_type=config.GNB_DEPLOYMENT_TYPE,
        ue_start_pos=config.UE_START_POS_UDN,
        ue_end_pos=config.UE_END_POS_UDN,
        ue_speed=config.UE_SPEED,
        steps_per_episode=config.STEPS_PER_EPISODE,
        seed=env_seed
    )
    state_size = env_udn.observation_space.shape[0]
    action_size = env_udn.action_space.n
    print(f"UDN Environment: State size: {state_size}, Action size: {action_size}")
    print(f"gNB locations ({config.GNB_DEPLOYMENT_TYPE}):\n{env_udn.gnb_locations[:5]}...") # Print first 5


    # --- Initialize Agents ---
    dqn_agent_udn = DQNAgent(state_size, action_size)
    traditional_agent_udn = TraditionalAgent(
        num_gnbs=action_size,
        hysteresis_db=config.HYSTERESIS_DB,
        time_to_trigger_steps=config.TIME_TO_TRIGGER
    )

    # --- Train DQN Agent for UDN ---
    print("\n--- Training DQN Agent on UDN ---")
    # Set train_dqn=True to enable learning
    dqn_results_train_udn, dqn_rewards_udn = run_simulation(
        env_udn, dqn_agent_udn, config.NUM_EPISODES, config.STEPS_PER_EPISODE, is_dqn=True, train_dqn=True
    )
    model_save_path_udn = "dqn_handover_udn_final.weights.h5"
    if hasattr(dqn_agent_udn, 'save'): # Check if save method exists
        dqn_agent_udn.save(model_save_path_udn)
        print(f"Saved UDN DQN model to {model_save_path_udn}")
    plot_rewards(dqn_rewards_udn, title=f"DQN Training Rewards (UDN - {config.NUM_GNBS_UDN} gNBs)")


    # --- Evaluate Trained DQN Agent on UDN ---
    print("\n--- Evaluating Trained DQN Agent on UDN ---")
    if hasattr(dqn_agent_udn, 'load') and os.path.exists(model_save_path_udn):
         dqn_agent_udn.load(model_save_path_udn)
         print(f"Loaded UDN DQN model from {model_save_path_udn} for evaluation.")
    dqn_agent_udn.epsilon = 0.0 # Turn off exploration for evaluation
    num_eval_episodes = max(10, config.NUM_EPISODES // 5) # Fewer episodes for eval
    dqn_results_eval_udn, _ = run_simulation(
        env_udn, dqn_agent_udn, num_eval_episodes, config.STEPS_PER_EPISODE, is_dqn=True, train_dqn=False
    )

    # --- Evaluate Traditional Agent on UDN ---
    print("\n--- Evaluating Traditional Agent on UDN ---")
    traditional_results_udn, _ = run_simulation(
        env_udn, traditional_agent_udn, num_eval_episodes, config.STEPS_PER_EPISODE, is_dqn=False
    )

    # --- Compare and Plot UDN Results ---
    print("\n--- Plotting UDN Comparison ---")
    plot_results(dqn_results_eval_udn, traditional_results_udn, num_eval_episodes)

    print("\nUDN Simulation and comparison complete. Check generated plots: dqn_rewards.png and comparison_results.png")
    
    # Close environment if it has a visualizer
    env_udn.close()

if __name__ == "__main__":
    main()