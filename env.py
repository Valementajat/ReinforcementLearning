from typing import Optional

import gymnasium as gym
import numpy as np

from mapCreator import  create_map

#Penalties and rewards
STEP_COST = -1
TOXIC_PENALTY = -20
HEAL_REWARD = 20
GOAL_REWARD = 100
OUT_OF_ENERGY_PENALTY = -50
RETURNTOSTART_PENALTY = -2
WALLBUMB_PENALTY = -2

#Energy cost for actions
ENERGY_COST = 1
HEALTHY_CELL_ENERGY_REWARD = 2


# Action encoding shared with Gymnasium's CliffWalking convention.
ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3

# Row-column displacement for each action.
_DELTA = {
    ACTION_UP: (-1, 0),
    ACTION_RIGHT: (0, +1),
    ACTION_DOWN: (+1, 0),
    ACTION_LEFT: (0, -1),
}
# Cell type for unseen cells in the observation, as in the "fog of war".
FOG = 6 
MINFOG =2 

class ToxicSwampEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}


    def __init__(
        self,
        map_fn=create_map,
        toxic_pct=0.25,
        healthy_pct=0.10,
        slippery=0.0,
        fog_radius=4,
        max_energy=50,
        energy_bins=10,
        render_mode=None,
    ):        
    
        super(ToxicSwampEnv, self).__init__()

        # map_fn is the create_map function from mapCreator
        self.map_fn      = map_fn
        self.toxic_pct   = toxic_pct
        self.healthy_pct = healthy_pct
        self.slippery    = slippery
        self.fog_radius  = fog_radius
        self.max_energy  = max_energy
        self.energy_bins = energy_bins
        self.render_mode = render_mode
        
        
        self.step_cost = STEP_COST
        self.toxic_penalty = TOXIC_PENALTY
        self.wall_bump_penalty = WALLBUMB_PENALTY
        self.heal_reward = HEAL_REWARD
        self.goal_reward = GOAL_REWARD

        self.energy_cost = ENERGY_COST
        self.healthy_cell_energy_reward = HEALTHY_CELL_ENERGY_REWARD
        self.out_of_energy_penalty = OUT_OF_ENERGY_PENALTY # Penalty for running out of energy
        self.return_to_start_penalty = RETURNTOSTART_PENALTY # Penalty for returning to the start position


        # Grid width and height are determined by the map_fn when we create the map on the fly in reset() method, but we need to create the map here to get the dimensions for  observation space
        map_fn = self.map_fn
        self.grid, self.toxic_cells, self.healthy_cells, self.start_pos, self.goal_pos = map_fn(
            self.toxic_pct, self.healthy_pct, seed=1
        )
        self.grid_height, self.grid_width = self.grid.shape


        # Define action and observation space
        # Actions: 0=up, 1=right, 2=down, 3=left
        self.action_space = gym.spaces.Discrete(4)
        self._actions_delta = _DELTA
        
        # Observations: discrete states corresponding to grid positions clipped to fog radius + energy bin
        window_size = (2 * self.fog_radius + 1) ** 2   # e.g. 5x5 = 25
        self.observation_space = gym.spaces.MultiDiscrete(
            [7] * window_size   + [self.energy_bins + 1]  # 25 cell types + 1 energy bin
        )
        # Rendering required parameters
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode={render_mode!r}. "
                f"Supported: {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode

        # Initialize state
        self.state = None


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, any]] = None,
    ) -> tuple[int, dict[str, any]]:
        super().reset(seed=seed)
        map_fn = self.map_fn
        self.grid, toxic_cells, healthy_cells, self.start_pos, self.goal_pos = map_fn(
            self.toxic_pct, self.healthy_pct, seed=seed
        )
       

        self.toxic_cells   = set(map(tuple, toxic_cells))
        self.healthy_cells = set(map(tuple, healthy_cells))
        # New heihgt and width 
        self.grid_height, self.grid_width = self.grid.shape 

        # Reset energy to max at the start of each episode
        self.energy = self.max_energy

        self._agent_pos = self.start_pos
        self._episode_return = 0.0
        obs = self._get_obs()
        return obs, {}




    def _intended_landing(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        """Apply action displacement, clipping at grid boundaries."""
        dr, dc = _DELTA[action]
        r = max(0, min(self.grid_height - 1, pos[0] + dr))
        c = max(0, min(self.grid_width - 1, pos[1] + dc))
        return (r, c)

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, any]]:
        if self._agent_pos is None:
            raise RuntimeError("step() called before reset().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected 0..3.")

        
        landing = self._intended_landing(self._agent_pos, action)

        self._agent_pos = landing
        self.energy -= self.energy_cost  # Energy cost for taking an action
       
        

        if landing == self.goal_pos:
            reward = self.goal_reward
            terminated = True
        elif landing == self.start_pos:
            reward = self.return_to_start_penalty

            terminated = False
        elif landing in self.toxic_cells:
            #Only a horrible reward, but we can still continue the episode
            #as we are in a swamp and we can get out of it
            """ self._agent_pos = self.start_pos """
            
            reward = self.step_cost + self.toxic_penalty
            self.energy -= 2  # Additional energy cost for landing on a toxic cell
            terminated = False

        #Healthy cellhandling
        elif landing in self.healthy_cells:
            self.energy += self.healthy_cell_energy_reward
            
            reward = self.step_cost + self.heal_reward  # Step cost is still applied, but we get a net positive reward for the healthy cell
            self.healthy_cells.discard(landing)  # one-time bonus
            terminated = False
        else:
            
            reward = self.step_cost
            terminated = False
        
        # Ensure energy is within bounds
        self.energy = np.clip(self.energy, 0, self.max_energy)
        if self.energy == 0:
            terminated = True
            reward += self.out_of_energy_penalty  # Additional penalty for running out of energy

        truncated = False
        self._episode_return += reward
        obs = self._get_obs()
        info = {"effective_action": action}
        return obs, reward, terminated, truncated, info





    def close(self) -> None:
        pass

    def _get_obs(self):
        r, c = self._agent_pos
        max_radius = self.fog_radius
        visible_radius = max(MINFOG, int(self.fog_radius * self.energy / self.max_energy))

        window = []
        for dr in range(-max_radius, max_radius + 1):
            for dc in range(-max_radius, max_radius + 1):
                nr, nc = r + dr, c + dc
                if max(abs(dr), abs(dc)) > visible_radius:
                    window.append(FOG)              # outside current vision
                elif 0 <= nr < self.grid_height and 0 <= nc < self.grid_width:
                    window.append(int(self.grid[nr, nc]))
                else:
                    window.append(5)                # WALL

        energy_bin = int(np.clip(self.energy / self.max_energy * self.energy_bins, 0, self.energy_bins))
        return np.array(window + [energy_bin], dtype=np.int64)
        #return np.array(window + [energy_bin], dtype=np.int64)

    # ---------------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------------

    # Color palette (RGB in [0, 1]).
    _COLOR_FREE = np.array([0.96, 0.96, 0.92])    # pale background
    _COLOR_TOXIC = np.array([0.55, 0.10, 0.55])   # purple band
    _COLOR_HEALTY = np.array([0.20, 0.55, 0.20])   # nutrient green
    _COLOR_GOAL = np.array([0.85, 0.85, 0.20])    # gold
    _COLOR_START = np.array([0.85, 0.85, 0.70])   # faint marker
    _COLOR_GRID = np.array([0.30, 0.30, 0.30])    # grid lines

    # Microbe color endpoints: from "healthy" to "depleted".
    _MICROBE_HEALTHY = np.array([0.10, 0.75, 0.20])  # vivid green
    _MICROBE_DEPLETED = np.array([0.55, 0.05, 0.05])  # dark red
    _ENERGY_DEPLETION_SCALE = 100.0  # cumulative cost at which the microbe
                                     # is considered fully depleted

    _CELL_PIXELS = 32
    _MARGIN = 1

    def render(self) -> Optional[np.ndarray]:
        """Return an RGB image of the current grid as a NumPy array.

        The microbe is drawn as a filled disc whose color interpolates
        between healthy green and depleted red as a function of the
        cumulative reward collected since the last reset. Concretely,
        ``alpha = clip(-episode_return / _ENERGY_DEPLETION_SCALE, 0, 1)``
        and the microbe color is ``(1 - alpha) * healthy + alpha * depleted``.
        """
        if self.render_mode != "rgb_array":
            return None
        if self._agent_pos is None:
            raise RuntimeError("render() called before reset().")

        cell = self._CELL_PIXELS
        m = self._MARGIN
        h_px = self.grid_height * cell
        w_px = self.grid_width * cell

        # Background
        img = np.tile(self._COLOR_FREE, (h_px, w_px, 1))

        # ── Fog of war overlay ────────────────────────────────────────────
        visible_radius = max(MINFOG, int(self.fog_radius * self.energy / self.max_energy))
        ar, ac = self._agent_pos
        fog_color = np.array([0.3, 0.3, 0.3])   # dark grey

        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if max(abs(row - ar), abs(col - ac)) > visible_radius:
                    r0, r1 = row * cell, (row + 1) * cell
                    c0, c1 = col * cell, (col + 1) * cell
                    # Blend 70% fog over the existing cell colour
                    img[r0:r1, c0:c1] = (
                        0.7 * fog_color + 0.3 * img[r0:r1, c0:c1]
                    )


        # Paint special cells.
        def paint_cell(pos: tuple[int, int], color: np.ndarray) -> None:
            r, c = pos
            r0, r1 = r * cell + m, (r + 1) * cell - m
            c0, c1 = c * cell + m, (c + 1) * cell - m
            img[r0:r1, c0:c1] = color

        paint_cell(self.start_pos, self._COLOR_START)
        for tc in self.toxic_cells:
            paint_cell(tc, self._COLOR_TOXIC)
        # Healthy cell colour
        for tc in self.healthy_cells:
            paint_cell(tc, self._COLOR_HEALTY)
        paint_cell(self.goal_pos, self._COLOR_GOAL)

        # Grid lines.
        for r in range(self.grid_height  + 1):
            img[min(r * cell, h_px - 1), :] = self._COLOR_GRID
        for c in range(self.grid_width + 1):
            img[:, min(c * cell, w_px - 1)] = self._COLOR_GRID

        # Microbe color from cumulative episode return.
        # episode_return is non-positive, so -episode_return >= 0.
        alpha = float(np.clip(-self._episode_return / self._ENERGY_DEPLETION_SCALE,
                              0.0, 1.0))
        microbe_color = (1.0 - alpha) * self._MICROBE_HEALTHY + alpha * self._MICROBE_DEPLETED

        # Draw the microbe as a filled disc inside its current cell.
        ar, ac = self._agent_pos
        cy = ar * cell + cell // 2
        cx = ac * cell + cell // 2
        radius = cell // 2 - 2 * m
        yy, xx = np.ogrid[:h_px, :w_px]
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        img[disc] = microbe_color

        return (img * 255).astype(np.uint8)