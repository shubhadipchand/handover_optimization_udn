import numpy as np
import gymnasium as gym
from gymnasium import spaces
import config # Make sure this imports the updated config

class HandoverEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 10} # Added human render mode

    def __init__(self,
                 area_width,
                 area_height,
                 num_gnbs,
                 deployment_type, # 'grid' or 'random'
                 ue_start_pos,
                 ue_end_pos,
                 ue_speed,
                 steps_per_episode,
                 seed=None): # Added seed for reproducibility
        super().__init__()

        if seed is not None:
            np.random.seed(seed)


        self.area_width = area_width
        self.area_height = area_height
        self.num_gnbs = num_gnbs

        if deployment_type == 'grid':
            self.gnb_locations = self._generate_grid_gnb_locations(num_gnbs, area_width, area_height)
        elif deployment_type == 'random':
            self.gnb_locations = self._generate_random_gnb_locations(num_gnbs, area_width, area_height, seed=seed)
        else:
            raise ValueError("Invalid gNB deployment type specified in config.")

        self.ue_start_pos = np.array(ue_start_pos, dtype=float)
        self.ue_end_pos = np.array(ue_end_pos, dtype=float)
        self.ue_speed = ue_speed
        self.total_steps = steps_per_episode
        self.current_step = 0

        total_distance = np.linalg.norm(self.ue_end_pos - self.ue_start_pos)
        # Ensure steps_needed is at least 1, even if distance is 0 or very small
        self.steps_needed = max(1, int(np.ceil(total_distance / self.ue_speed))) if self.ue_speed > 0 else self.total_steps

        # Adjust total_steps if path is shorter than STEPS_PER_EPISODE
        self.total_steps = min(steps_per_episode, self.steps_needed)

        if self.steps_needed > 0 and total_distance > 0:
             self.move_vector = (self.ue_end_pos - self.ue_start_pos) / self.steps_needed
        else: # If no movement (start=end or speed=0 or steps_needed = 0)
             self.move_vector = np.zeros_like(self.ue_start_pos)


        self.ue_pos = np.copy(self.ue_start_pos)
        # _get_best_initial_gnb relies on gnb_locations being set
        self.serving_gnb_idx = self._get_best_initial_gnb()

        self.observation_space = spaces.Box(low=-150, high=0, shape=(self.num_gnbs,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_gnbs)

        self.last_serving_gnb = -1 # Stores the GNB index *before* the current serving GNB, if a HO just occurred
        self.previous_serving_gnb_for_pingpong = -1 # Stores the GNB index that UE was connected to before the last HO
        self.steps_on_current_gnb = 0 # Renamed from steps_since_last_ho for clarity
        self.ping_pong_window = 5 # steps, e.g., if HO to A -> B -> A within 5 steps, it's a ping-pong

        # For rendering
        self.viewer = None
        self.ue_history = []


    def _generate_grid_gnb_locations(self, num_gnbs, width, height):
        locations = []
        cols = int(np.ceil(np.sqrt(num_gnbs)))
        rows = int(np.ceil(num_gnbs / cols))

        x_spacing = width / (cols + 1) if cols > 0 else width
        y_spacing = height / (rows + 1) if rows > 0 else height
        
        count = 0
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                if count < num_gnbs:
                    locations.append((c * x_spacing, r * y_spacing))
                    count += 1
                else: break
            if count >= num_gnbs: break
        return np.array(locations)

    def _generate_random_gnb_locations(self, num_gnbs, width, height, seed=None):
        # Use a local RNG for this method if a seed is provided
        rng = np.random.default_rng(seed) if seed is not None else np.random
        x_coords = rng.uniform(width * 0.05, width * 0.95, num_gnbs) # Avoid edges
        y_coords = rng.uniform(height * 0.05, height * 0.95, num_gnbs)
        return np.array(list(zip(x_coords, y_coords)))

    def _calculate_rsrp(self, ue_pos, gnb_idx):
        dist = np.linalg.norm(ue_pos - self.gnb_locations[gnb_idx])
        dist = max(dist, config.REFERENCE_DISTANCE) # Clamp distance to avoid log(0) or issues with dist < ref_dist

        path_loss = config.REFERENCE_LOSS + 10 * config.PATH_LOSS_EXPONENT * np.log10(dist / config.REFERENCE_DISTANCE)
        # Add random shadowing/fading effect
        rsrp = -path_loss + np.random.normal(0, config.NOISE_STD_DEV)
        return max(-150.0, min(-20.0, rsrp)) # Clamp RSRP to a realistic range

    def _get_state(self):
        state = np.array([self._calculate_rsrp(self.ue_pos, i) for i in range(self.num_gnbs)], dtype=np.float32)
        return state

    def _get_best_initial_gnb(self):
        if not hasattr(self, 'gnb_locations') or self.gnb_locations is None or len(self.gnb_locations) == 0:
             # This can happen if num_gnbs is 0 or gnb_locations isn't populated.
             # Fallback or raise error. For now, if no gNBs, default to 0 (which is invalid but handled by action space).
             print("Warning: No gNBs found or gNB locations not initialized. Defaulting initial gNB to 0.")
             return 0 if self.num_gnbs > 0 else -1 # Or handle more gracefully
        rsrps = [self._calculate_rsrp(self.ue_start_pos, i) for i in range(self.num_gnbs)]
        return np.argmax(rsrps)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for gym.Env
        if seed is not None: # Re-seed the environment's main RNG if provided at reset
            np.random.seed(seed)
            # If gNBs are random, you might want to regenerate them here if the seed affects their placement
            # For now, assuming gNB locations are fixed per instantiation unless explicitly changed.

        self.ue_pos = np.copy(self.ue_start_pos)
        self.serving_gnb_idx = self._get_best_initial_gnb()
        self.current_step = 0
        
        self.last_serving_gnb = -1
        self.previous_serving_gnb_for_pingpong = self.serving_gnb_idx # Initialize
        self.steps_on_current_gnb = 0
        self.ue_history = [np.copy(self.ue_pos)]


        observation = self._get_state()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.current_step += 1
        current_rsrp_state = self._get_state() # RSRPs *before* HO decision and movement

        target_gnb_idx = action
        handover_occurred = False
        handover_failed = False
        is_ping_pong = False
        reward = 0

        # Store current serving gNB before any change for ping-pong detection
        gnb_before_potential_ho = self.serving_gnb_idx

        if target_gnb_idx != self.serving_gnb_idx: # Handover attempt
            handover_occurred = True
            rsrp_of_target = current_rsrp_state[target_gnb_idx]

            # Simple failure: target RSRP too low (e.g., below Radio Link Failure threshold)
            if rsrp_of_target < -115: # Example threshold
                handover_failed = True
                reward += config.FAILURE_PENALTY
                # HO failed, remain on the current gNB. Target not changed.
            else:
                # Successful HO logic
                reward += config.HANDOVER_EXECUTION_PENALTY
                self.last_serving_gnb = self.serving_gnb_idx # Store the one we are leaving
                self.serving_gnb_idx = target_gnb_idx     # Connect to new gNB

                # Ping-Pong Check: if new gNB is the one we were connected to *before the previous* gNB,
                # and it happened quickly.
                # Current: A -> B (last_serving_gnb=A, serving_gnb_idx=B)
                # Next HO: B -> A (target_gnb_idx=A)
                # Check if target_gnb_idx == previous_serving_gnb_for_pingpong
                if self.serving_gnb_idx == self.previous_serving_gnb_for_pingpong and \
                   self.steps_on_current_gnb < self.ping_pong_window and \
                   self.last_serving_gnb != -1: # Ensure there was a previous gNB
                    is_ping_pong = True
                    reward += config.PINGPONG_PENALTY
                
                self.previous_serving_gnb_for_pingpong = gnb_before_potential_ho # Update for next step's PP check
                self.steps_on_current_gnb = 0
        else: # No handover attempt (action chose current serving gNB)
            self.steps_on_current_gnb += 1


        # Move UE
        if self.current_step < self.steps_needed: # Stop moving if end of path reached
            self.ue_pos += self.move_vector
        elif self.current_step == self.steps_needed: # Final step of movement
            self.ue_pos = np.copy(self.ue_end_pos) # Snap to exact end position
        
        self.ue_history.append(np.copy(self.ue_pos))


        # Calculate next state and reward based on post-movement conditions
        next_state = self._get_state()
        serving_rsrp_after_move = next_state[self.serving_gnb_idx]

        # Reward for RSRP of the serving cell (normalized)
        # RSRP values typically range from -120 (poor) to -70 (good), up to -40 (excellent)
        # Normalize to roughly [0, 1] for -120 to -40 dBm range
        min_rsrp_reward = -120
        max_rsrp_reward = -40
        norm_rsrp = (serving_rsrp_after_move - min_rsrp_reward) / (max_rsrp_reward - min_rsrp_reward)
        rsrp_based_reward = np.clip(norm_rsrp, 0, 1.5) # Allow slightly more than 1 for excellent signal
        reward += config.RSRP_REWARD_WEIGHT * rsrp_based_reward

        terminated = self.current_step >= self.total_steps
        truncated = False # Not using truncation based on other conditions for now

        info = self._get_info()
        info.update({
            'handover_occurred': handover_occurred and not handover_failed,
            'handover_failed': handover_failed,
            'ping_pong': is_ping_pong,
            'serving_rsrp': serving_rsrp_after_move,
            'latency': self._simulate_latency(serving_rsrp_after_move),
            'bandwidth': self._simulate_bandwidth(serving_rsrp_after_move),
            'ue_x': self.ue_pos[0],
            'ue_y': self.ue_pos[1],
        })

        return next_state, reward, terminated, truncated, info

    def _simulate_latency(self, rsrp):
        # Simplified: latency decreases as RSRP improves
        # Example: 150ms at -120dBm, 10ms at -70dBm
        latency = 10 + 140 * np.clip(1 - ((rsrp - (-120)) / (-70 - (-120))), 0, 1)
        return max(5, latency) # Min latency 5ms

    def _simulate_bandwidth(self, rsrp):
        # Simplified: bandwidth increases with RSRP
        # Example: 1 Mbps at -120dBm, 300 Mbps at -70dBm
        bandwidth = 1 + 299 * np.clip(((rsrp - (-120)) / (-70 - (-120))), 0, 1)
        return max(1, bandwidth) # Min bandwidth 1 Mbps

    def _get_info(self):
        return {"ue_pos": self.ue_pos, "serving_gnb": self.serving_gnb_idx, "steps_on_gnb": self.steps_on_current_gnb}

    def render(self, mode='human'):
        if mode == 'human':
            import matplotlib.pyplot as plt
            if self.viewer is None:
                plt.ion() # Interactive mode
                self.fig, self.ax = plt.subplots(figsize=(8,8))
                self.viewer = True
                self.gnb_scatter = self.ax.scatter(self.gnb_locations[:, 0], self.gnb_locations[:, 1], c='red', marker='^', s=100, label='gNBs')
                self.ue_plot, = self.ax.plot([], [], 'bo-', markersize=10, label='UE Path') # History
                self.current_ue_scatter = self.ax.scatter([], [], c='blue', marker='o', s=60, edgecolors='black', label='Current UE')
                self.serving_line, = self.ax.plot([], [], 'g--', label='Serving Link')

                self.ax.set_xlim(0, self.area_width)
                self.ax.set_ylim(0, self.area_height)
                self.ax.set_xlabel("X-coordinate (m)")
                self.ax.set_ylabel("Y-coordinate (m)")
                self.ax.set_title(f"5G Handover Simulation - Step: {self.current_step}")
                self.ax.legend()
                self.ax.grid(True)
                plt.show(block=False)

            # Update UE path history
            if self.ue_history:
                path_x = [p[0] for p in self.ue_history]
                path_y = [p[1] for p in self.ue_history]
                self.ue_plot.set_data(path_x, path_y)

            # Update current UE position
            self.current_ue_scatter.set_offsets(self.ue_pos)

            # Update serving link
            if self.serving_gnb_idx >= 0 and self.serving_gnb_idx < len(self.gnb_locations):
                serving_gnb_pos = self.gnb_locations[self.serving_gnb_idx]
                self.serving_line.set_data([self.ue_pos[0], serving_gnb_pos[0]],
                                           [self.ue_pos[1], serving_gnb_pos[1]])
            else:
                self.serving_line.set_data([],[])


            self.ax.set_title(f"5G UDN Handover - Step: {self.current_step}/{self.total_steps}, Serving gNB: {self.serving_gnb_idx}")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            # plt.pause(0.01) # Small pause to allow plot to update

    def close(self):
        if self.viewer is not None:
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.close(self.fig)
            self.viewer = None