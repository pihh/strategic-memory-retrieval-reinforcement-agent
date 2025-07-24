import time
import warnings
import pandas as pd
import numpy as np

from gymnasium.wrappers import RecordEpisodeStatistics
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from tabulate import tabulate
from tqdm import tqdm

from agent import StrategicMemoryAgent
from environments import MemoryTaskEnv
from memory import StrategicMemoryBuffer, StrategicMemoryTransformerPolicy


warnings.filterwarnings('ignore')



def build_memory_agent(env, config):
    """
    Factory for creating an StrategicMemoryAgent with proper memory and aux modules.
    Args:
        env: Gymnasium environment.
        config: dict with agent, memory, policy, and aux setup.
    Returns:
        Initialized StrategicMemoryAgent agent.
    """
    mem_dim = config.get("mem_dim", 32)
    policy_class = config.get("policy_class", StrategicMemoryTransformerPolicy)
    memory = StrategicMemoryBuffer(
        obs_dim=env.observation_space.shape[0],
        action_dim=1,
        mem_dim=mem_dim,
        max_entries=config.get("max_entries", 16),
        device=config.get("device", "cpu")
    )
    aux_modules = config.get("aux_modules", [])
    agent = StrategicMemoryAgent(
        policy_class=policy_class,
        env=env,
        memory=memory,
        memory_learn_retention=config.get("memory_learn_retention", True),    
        memory_retention_coef=config.get("memory_retention_coef", 0.01),  
        aux_modules=aux_modules,
        device=config.get("device", "cpu"),
        use_rnd=config.get("use_rnd", True),
        ent_coef=config.get("ent_coef", 0.1),
        learning_rate=config.get("learning_rate", 1e-3),
        her=config.get("her", False),
        verbose=config.get("verbose", 1),
        reward_norm=config.get("reward_norm", False),
    
    )
    return agent


class AgentPerformanceBenchmark:
    """
    Benchmark class for standardized evaluation and reporting of agent performance
    on memory-based RL tasks. Handles experiment setup, training, evaluation, and result display.
    """

    def __init__(self, env_config, memory_agent_config=None):
        """
        Initializes the benchmark runner with experiment parameters.

        Parameters
        ----------
        env_config : dict
            Environment and training configuration.
        memory_agent_config : dict or None
            If provided, specifies construction of memory-based agent.
        """
        self.env_config = env_config
        self.env = MemoryTaskEnv(
            delay=env_config["delay"],
            difficulty=env_config.get("difficulty", 0)
        )
        self.n_eval_episodes = env_config.get("n_eval_episodes", 20)
        self.verbose = env_config.get("verbose", 0)
        self.log_interval = env_config.get("log_interval", 250)
        self.learning_rate = env_config.get("learning_rate", 1e-3)
        self.total_timesteps = env_config.get("total_timesteps", 10000)
        self.eval_base = env_config.get("eval_base", False)
        self.mode_name = env_config.get(
            "mode_name", "EASY" if env_config.get("difficulty", 0) == 0 else "HARD"
        )
        self.memory_agent_config = memory_agent_config

    def print_train_results(self, reward, std, model_name):
        """Prints formatted summary of evaluation results."""
        print(
            f"[{model_name} @ MemoryTaskEnv with delay={self.env_config['delay']} in {self.mode_name} Mode]  "
            f"mean_ep_rew: {reward*100:.1f}% .  std_ep_rew: {std:.2f} in {self.n_eval_episodes} episodes"
        )

    def evaluate(self, model, model_name, deterministic=True, verbose=False):
        """
        Evaluates an agent over multiple episodes, balancing both target classes.
        Returns mean and std reward.
        """
        returns = []
        target_counter = [0, 0]
        eval_complete = False
        eval_runs = 0
        n_target_samples = int(self.n_eval_episodes / 2)

        while not eval_complete:
            if hasattr(model, "reset_trajectory") and callable(getattr(model, "reset_trajectory")):
                model.reset_trajectory()
            obs, _ = self.env.reset()
            target = int(obs[0])
            eval_runs += 1
            if eval_runs > 1000:
                print("Warning: Evaluation ran over 1000 attempts, aborting early.")
                break
            if target_counter[target] >= n_target_samples:
                continue
            target_counter[target] += 1
            done = False
            total_reward = 0.0
            while not done:
                action = model.predict(obs, deterministic=deterministic)
                if isinstance(action, tuple):
                    action = action[0]
                obs, reward, done, _, _ = self.env.step(action)
                total_reward += reward
            returns.append(total_reward)
            eval_complete = sum(target_counter) >= self.n_eval_episodes

        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if verbose:
            self.print_train_results(mean_return, std_return, model_name)
        return mean_return, std_return

    def run(self):
        """
        Runs the full training and evaluation pipeline for all specified agents.
        Returns a list of dicts with experiment metrics.
        """
        results = []
        print(
            f"\nTraining in {self.mode_name} mode with delay of {self.env_config['delay']} steps\n"
        )

        # --- Agent setup list (order matters for report) ---
        agents = []
        if self.eval_base:
            agents.append(('PPO', lambda: PPO(
                'MlpPolicy',
                RecordEpisodeStatistics(self.env),
                learning_rate=self.learning_rate,
                verbose=self.verbose
            )))
            agents.append(('RecurrentPPO', lambda: RecurrentPPO(
                "MlpLstmPolicy",
                RecordEpisodeStatistics(self.env),
                verbose=self.verbose,
                learning_rate=self.learning_rate
            )))

        # --- Add your custom memory agent here ---
        if self.memory_agent_config is not None:
            agents.append(('StrategicMemoryAgent', lambda: build_memory_agent(
                self.env, self.memory_agent_config
            )))

        # --- Training & evaluation loop ---
        with tqdm(total=len(agents) * 2 + 1, desc="Benchmark Progress", unit="step") as pbar:
            for agent_name, agent_builder in agents:
                # TRAIN
                pbar.set_description(f"Training {agent_name}")
                start_time = time.time()
                model = agent_builder()
                model.learn(
                    total_timesteps=self.total_timesteps,
                    log_interval=self.log_interval
                )
                duration = time.time() - start_time
                pbar.update(1)
                # EVALUATE
                pbar.set_description(f"Evaluating {agent_name}")
                mean, std = self.evaluate(model, agent_name, verbose=False)
                pbar.update(1)
                results.append({
                    **self.env_config,
                    'agent': agent_name,
                    'mean_return': mean,
                    'std_return': std,
                    'duration': duration
                })
            pbar.set_description("Finalizing Results")
            pbar.update(1)

        # --- Format & Print Table ---
        pdf = pd.DataFrame(results)
        pdf = pdf[['agent', 'delay', 'mode_name', 'mean_return', 'std_return', 'duration']]
        pdf.rename(
            columns={
                "agent": "Agent",
                "delay": "Delay",
                "mode_name": "Mode",
                "mean_return": "Mean Ep Rew",
                "std_return": "Std Ep Rew",
                "duration": "Duration (s)"
            },
            inplace=True
        )
        print(tabulate(pdf, headers="keys", tablefmt="rounded_outline"))
        return results
