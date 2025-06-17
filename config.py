# Simulation Parameters
NUM_EPISODES = 75 # Increased episodes for potentially harder learning
STEPS_PER_EPISODE = 250 # May need more steps if area is larger or speed is lower

# UDN Specific Parameters
AREA_WIDTH = 150  # meters for a moderately dense area
AREA_HEIGHT = 150 # meters
NUM_GNBS_UDN = 20  # Number of gNBs for UDN (e.g., 4x5 grid or random)
GNB_DEPLOYMENT_TYPE = 'grid' # 'grid' or 'random'

# Path for UE (traversing a significant portion of the UDN)
UE_START_POS_UDN = (10, AREA_HEIGHT / 2) # Start near one edge
UE_END_POS_UDN = (AREA_WIDTH - 10, AREA_HEIGHT / 2) # End near opposite edge
UE_SPEED = 2.0 # meters per step (adjust based on area size and steps)

# Channel Model Parameters (Simplified Log-Distance)
PATH_LOSS_EXPONENT = 3.2 # Urban environments might have slightly different exponents
REFERENCE_DISTANCE = 1.0 # meter
REFERENCE_LOSS = 40.0 # dB
NOISE_STD_DEV = 3.5 # dB, slightly more variability

# Traditional Agent Parameters
HYSTERESIS_DB = 2.5 # May need finer tuning for UDN
TIME_TO_TRIGGER = 2 # steps (TTT might be shorter due to rapid signal changes)

# DQN Parameters
LEARNING_RATE = 0.00075
DISCOUNT_FACTOR = 0.98
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 15000 # Adjusted based on num_episodes * steps_per_episode
REPLAY_BUFFER_SIZE = 15000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 150 # steps

# Reward Function Weights
RSRP_REWARD_WEIGHT = 0.7 # Prioritize good signal more
FAILURE_PENALTY = -120
PINGPONG_PENALTY = -60
HANDOVER_EXECUTION_PENALTY = -8