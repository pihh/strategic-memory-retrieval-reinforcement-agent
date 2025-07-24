import gymnasium as gym
import numpy as np

class MemoryTaskEnv(gym.Env):
    """
    MemoryTaskEnv
    -------------
    Custom Gymnasium environment simulating a delayed memory task.

    Description:
    ------------
    The agent is shown a target bit (0 or 1) at the first timestep.
    For a configurable number of steps (`delay`), the environment presents only distractor stimuli.
    After the delay, the agent must recall and act on the original target.
    Difficulty controls whether distractors are predictable (easy) or random (hard).

    Observation Space:
        Box(low=0, high=1, shape=(3,), dtype=np.float32)
        [target/distractor_1, distractor_2, unused]
    Action Space:
        Discrete(2): 0 or 1

    Args:
        delay (int): Number of steps between target presentation and response.
        difficulty (int): 0 = easy (distractors always zero), 1 = hard (random distractors).
    """

    metadata = {"render.modes": []}

    def __init__(self, delay: int = 8, difficulty: int = 0):
        """
        Initialize MemoryTaskEnv.

        Parameters
        ----------
        delay : int
            Number of timesteps before the agent is allowed to act.
        difficulty : int
            Controls distractor sampling:
                0 - always zero (easy mode)
                1 - random 0 or 1 (hard mode)
        """
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.delay = delay
        self.difficulty = difficulty

        # Configure distractor sampling based on difficulty setting.
        # Easy: always return 0 as distractor. Hard: sample randomly.
        if self.difficulty == 0:
            self._sample_distractor = lambda: 0
        else:
            self._sample_distractor = lambda: np.random.randint(0, 2)

        self.target = None  # Target to remember
        self.t = 0         # Internal step counter

    def reset(self, seed=None, options=None):
        """
        Reset environment at the start of each episode.

        Returns
        -------
        obs : np.ndarray
            Initial observation with target shown in first slot.
        info : dict
            Empty info dictionary for API compatibility.
        """
        self.t = 0
        self.target = np.random.randint(0, 2)
        # First observation: show the target, then a distractor, and a reserved slot
        obs = np.array([self.target, self._sample_distractor(), 0.0], dtype=np.float32)
        return obs, {}

    def step(self, action):
        """
        Step the environment forward by one timestep.

        Parameters
        ----------
        action : int
            The agent's action, valid only at the last step.

        Returns
        -------
        obs : np.ndarray
            Current observation (all distractors after t=0).
        reward : float
            Sparse reward: 1 for correct recall at final step, -1 otherwise, 0 during delay.
        done : bool
            Whether the episode has finished.
        truncated : bool
            Always False (no truncation logic).
        info : dict
            Empty info dictionary for API compatibility.
        """
        self.t += 1
        # Before the final step: only present distractors, no reward.
        if self.t < self.delay:
            obs = np.array([self._sample_distractor(), self._sample_distractor(), 0.0], dtype=np.float32)
            reward = 0.0
            done = False
        else:
            # On the final step: agent must recall target, receives reward.
            correct = int(action == self.target)
            reward = 1.0 if correct else -1.0
            obs = np.array([self._sample_distractor(), self._sample_distractor(), 0.0], dtype=np.float32)
            done = True
        return obs, reward, done, False, {}

    def render(self, mode='human'):
        """
        (Optional) Render the environment.
        Not implemented for this environment.
        """
        pass
