import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tabulate import tabulate

from core_modules import RewardNormalizer, StateCounter, RNDModule
from core_calculations import compute_gae, compute_explained_variance
from callbacks import print_sb3_style_log_box

class StrategicMemoryAgent:
    """
    Proximal Policy Optimization (PPO) agent with integrated external memory retrieval.

    Features:
        - Supports auxiliary losses, HER, reward normalization, and RND-based exploration.
        - Episodic or contextual memory (passed as `memory`) for strategic RL.
        - Plug-and-play auxiliary modules (e.g., cue, event, confidence).
        - Stable training with reward normalization and intrinsic/extrinsic reward mixing.

    Args:
        policy_class (nn.Module): Policy network class (should accept obs_dim, memory, aux_modules).
        env (gym.Env): Gymnasium environment.
        verbose (int): Logging verbosity (0 = silent, 1 = logs).
        learning_rate (float): Adam optimizer learning rate.
        gamma (float): Discount factor.
        lam (float): GAE lambda.
        device (str): Torch device.
        her (bool): Enable Hindsight Experience Replay (if supported by env).
        reward_norm (bool): Normalize reward with running stats.
        intrinsic_expl (bool): Use count-based intrinsic reward.
        intrinsic_eta (float): Scaling for intrinsic bonus.
        ent_coef (float): Entropy coefficient.
        memory: Memory module for contextual/episodic learning (optional).
        aux_modules (list): List of auxiliary task modules (optional).
        use_rnd (bool): Enable Random Network Distillation intrinsic reward.
        rnd_emb_dim (int): Embedding dim for RND networks.
        rnd_lr (float): Learning rate for RND predictor.
    """


    __version__ = "1.4.0"

    def __init__(
        self, 
        policy_class, 
        env, 
        verbose=0,
        learning_rate=1e-3, 
        gamma=0.99, 
        lam=0.95, 
        ent_coef=0.01,
        device="cpu",
        her=False,
        reward_norm=False,
        intrinsic_expl=True,
        intrinsic_eta=0.01,
        memory=None,
        aux_modules=None,
        use_rnd=False, 
        rnd_emb_dim=32, 
        rnd_lr=1e-3,
        memory_learn_retention=False,      
        memory_retention_coef=0.01,
        early_stop=True,
        early_stop_n_samples=100,
        early_stop_mean_threshold=0.95,
        early_stop_std_threshold=0.05,
    ):
        self.env = env
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam = lam
        self.ent_coef = ent_coef
        self.verbose = verbose
        self.memory = memory
        self.memory_learn_retention = memory_learn_retention
        self.memory_retention_coef = memory_retention_coef
        self.aux_modules = aux_modules if aux_modules is not None else []
        self.aux = len(self.aux_modules) > 0
        self.early_stop= early_stop
        self.early_stop_n_samples=early_stop_n_samples
        self.early_stop_mean_threshold= early_stop_mean_threshold
        self.early_stop_std_threshold= early_stop_std_threshold
        # Policy: must accept obs_dim, memory, aux_modules
        self.policy = policy_class(
            obs_dim=env.observation_space.shape[0], 
            memory=memory,
            aux_modules=self.aux_modules
        ).to(self.device)

        # PATCH: include modular learning parameters to the optimizer 

        params = list(self.policy.parameters())
        if self.memory_learn_retention and hasattr(self.memory, "usefulness_parameters"):
            params += list(self.memory.usefulness_parameters())
        params = list({id(p): p for p in params}.values())  # REMOVE DUPLICATES
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

        self.training_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.her = her
        self.reward_norm = reward_norm
        self.intrinsic_expl = intrinsic_expl
        self.intrinsic_eta = intrinsic_eta
        self.reward_normalizer = RewardNormalizer()
        self.state_counter = StateCounter()
        self.use_rnd = use_rnd
        if self.use_rnd:
            self.rnd = RNDModule(env.observation_space.shape[0], emb_dim=rnd_emb_dim).to(self.device)
            self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        self.trajectory_buffer = []

    def reset_trajectory(self):
        self.trajectory_buffer = []

    def run_episode(self, her_target=None):
        obs, _ = self.env.reset()
        if her_target is not None:
            obs[0] = her_target

        done = False
        trajectory, actions, rewards, log_probs, values = [], [], [], [], []
        entropies_ep, aux_preds_list = [], []
        gate_history, memory_size_history = [], []
        attn_weights = None
        initial_cue = int(obs[0])
        aux_targets_ep = {aux.name: [] for aux in self.aux_modules}
        context_traj = []  # For memory module

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            trajectory.append(obs_t)
            traj = torch.stack(trajectory)
            action_for_mem = actions[-1].item() if len(actions) > 0 else 0
            reward_for_mem = rewards[-1].item() if len(rewards) > 0 else 0.0
            context_traj.append((obs_t.cpu().numpy(), action_for_mem, reward_for_mem))

            logits, value, aux_preds = self.policy(
                traj, obs_t,
                actions=torch.tensor([a.item() for a in actions], device=self.device) if actions else None,
                rewards=torch.tensor([r.item() for r in rewards], device=self.device) if rewards else None
            )
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            obs, reward, done, _, _ = self.env.step(action.item())

            # Intrinsic reward: count-based and/or RND
            if self.intrinsic_expl:
                reward += self.intrinsic_eta * self.state_counter.intrinsic_reward(obs)
            rnd_intrinsic = 0.0
            if self.use_rnd:
                with torch.no_grad():
                    obs_rnd = obs_t.unsqueeze(0)
                    rnd_intrinsic = self.rnd(obs_rnd).item()
                    reward += self.intrinsic_eta * rnd_intrinsic

            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
            values.append(value)
            entropies_ep.append(entropy)
            aux_preds_list.append(aux_preds)

            # Auxiliary targets (for supervised heads)
            for aux in self.aux_modules:
                if aux.name == "cue":
                    aux_targets_ep[aux.name].append(initial_cue)
                elif aux.name == "next_obs":
                    aux_targets_ep[aux.name].append(torch.tensor(obs, dtype=torch.float32))
                elif aux.name == "confidence":
                    dist = Categorical(logits=logits)
                    entropy = dist.entropy().item()
                    confidence = 1.0 - entropy  # Heuristic; can be improved
                    aux_targets_ep[aux.name].append(confidence)
                elif aux.name == "event":
                    event_flag = getattr(self.env, "event_flag", 0)
                    aux_targets_ep[aux.name].append(event_flag)
                elif aux.name == "oracle_action":
                    oracle_action = getattr(self.env, "oracle_action", None)
                    aux_targets_ep[aux.name].append(oracle_action)
                else:
                    aux_targets_ep[aux.name].append(0)

        # Store full trajectory in memory module (episodic buffer)
        if self.memory is not None:
            outcome = sum([r.item() for r in rewards])
            # Modular handling: always update episode buffer if available, else fallback
            
            if hasattr(self.memory, "episodic_buffer") and hasattr(self.memory.episodic_buffer, "add_entry"):
                self.memory.episodic_buffer.add_entry(context_traj, outcome)
            elif hasattr(self.memory, "add_entry"):
                self.memory.add_entry(context_traj, outcome)
            # Optionally: update motifs if needed (usually not online, but up to you)
            # if hasattr(self.memory, "motif_bank") and hasattr(self.memory.motif_bank, "add_entry"):
            #     self.memory.motif_bank.add_entry(context_traj, outcome)
        if self.memory is not None and hasattr(self.memory, 'get_last_attention'):
            attn_weights = self.memory.get_last_attention()

        # RND predictor update (only predictor trained)
        if self.use_rnd:
            obs_batch = torch.stack([torch.tensor(np.array(o), dtype=torch.float32, device=self.device) for o in trajectory])
            rnd_loss = self.rnd(obs_batch).mean()
            self.rnd_optimizer.zero_grad()
            rnd_loss.backward()
            self.rnd_optimizer.step()

        return {
            "trajectory": trajectory,
            "actions": actions,
            "rewards": rewards,
            "log_probs": log_probs,
            "values": values,
            "entropies": entropies_ep,
            "aux_preds": aux_preds_list,
            "aux_targets": aux_targets_ep,
            "initial_cue": initial_cue,
            "gate_history": gate_history,
            "memory_size_history": memory_size_history,
            "attn_weights": attn_weights
        }

    def get_episodic_buffer(self):
        episodic_buffer = None
        if self.memory :
            episodic_buffer = self.memory.episodic_buffer if hasattr(self.memory,"episodic_buffer") else  self.memory
        return episodic_buffer

        
        
    def learn(self, total_timesteps=2000, log_interval=100):
        steps = 0
        episodes = 0
        all_returns = []
        start_time = time.time()
        aux_losses = []
        unlock_early_stopping = len(self.episode_rewards)+self.early_stop_n_samples
        while steps < total_timesteps:
            try:
                #if hasattr(sys, 'last_traceback'):  # Quick hack: set by IPython on error/stop
                #    print("Interrupted in Jupyter (sys.last_traceback). Exiting.")
                #    break
                episode = self.run_episode()
                if self.reward_norm:
                    self.reward_normalizer.update([r.item() for r in episode["rewards"]])
                    episode["rewards"] = [
                        torch.tensor(rn, dtype=torch.float32, device=self.device)
                        for rn in self.reward_normalizer.normalize([r.item() for r in episode["rewards"]])
                    ]
    
                trajectory = episode["trajectory"]
                actions = episode["actions"]
                rewards = episode["rewards"]
                log_probs = episode["log_probs"]
                values = episode["values"]
                entropies_ep = episode["entropies"]
                aux_preds = episode["aux_preds"]
                aux_targets = episode["aux_targets"]
                T = len(rewards)
                rewards_t = torch.stack(rewards)
                values_t = torch.stack(values)
                log_probs_t = torch.stack(log_probs)
                actions_t = torch.stack(actions)
                last_value = 0.0
                advantages = compute_gae(rewards_t, values_t, gamma=self.gamma, lam=self.lam, last_value=last_value)
                returns = advantages + values_t.detach()
    
                policy_loss = -(log_probs_t * advantages.detach()).sum()
                value_loss = F.mse_loss(values_t, returns)
                entropy_mean = torch.stack(entropies_ep).mean()
                explained_var = compute_explained_variance(values_t, returns)
    
                # Auxiliary losses
                aux_loss_total = torch.tensor(0.0, device=self.device)
                aux_metrics_log = {}
                if self.aux:
                    for aux in self.aux_modules:
                        preds = torch.stack([ap[aux.name] for ap in aux_preds])
                        targets = torch.tensor(aux_targets[aux.name], device=self.device)
                        if preds.dim() != targets.dim():
                            targets = targets.squeeze(-1)
                        loss = aux.aux_loss(preds, targets)
                        aux_loss_total += loss
                        metrics = aux.aux_metrics(preds, targets)
                        aux_metrics_log[aux.name] = metrics
                    aux_losses.append(aux_loss_total.item())
    
                # Memory usefullness (if enabled) =====
                episodic_buffer = self.get_episodic_buffer()
                if (
                    self.memory_learn_retention
                    and self.memory is not None
                    and hasattr(episodic_buffer, 'get_last_attention')
                    and episodic_buffer.last_attn is not None
                    and len(episodic_buffer.usefulness_vec) == len(episodic_buffer.last_attn)
                    and len(episodic_buffer.usefulness_vec) > 0
                ):
                    total_reward = sum([r.item() for r in rewards])
                    if hasattr(self.memory,'episodic_buffer'):
                        
                        attn_tensor = torch.tensor(self.memory.episodic_buffer.last_attn, dtype=torch.float32, device=self.device)
                        mem_loss = self.memory.episodic_buffer.usefulness_loss(attn_tensor, total_reward)
                    else:
                      
                        attn_tensor = torch.tensor(self.memory.last_attn, dtype=torch.float32, device=self.device)
                        mem_loss = self.memory.usefulness_loss(attn_tensor, total_reward)
                else:
                    mem_loss = torch.tensor(0.0, device=self.device)
                    
    
                loss = (
                    policy_loss 
                    + 0.5 * value_loss 
                    + 0.1 * aux_loss_total 
                    - self.ent_coef * entropy_mean
                    + (self.memory_retention_coef * mem_loss if self.memory_learn_retention else 0.0)
                )
    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
                total_reward = sum([r.item() for r in rewards])
                self.episode_rewards.append(total_reward)
                self.episode_lengths.append(T)
                episodes += 1
                steps += T

                if self.early_stop and len(self.episode_rewards) >= unlock_early_stopping:
                    mean_rew = np.mean(self.episode_rewards[-self.early_stop_n_samples:])
                    std_rew = np.std(self.episode_rewards[-self.early_stop_n_samples:])
                    if mean_rew >self.early_stop_mean_threshold and std_rew <= self.early_stop_std_threshold:
                        mean_len = np.mean(self.episode_lengths[-log_interval:])
                        elapsed = int(time.time() - start_time)
                        ep_duration = elapsed/episodes
                        table = [
                                ["Train duration",f"{elapsed}s"],
                                ["Avg episode duration",f"{ep_duration:.2f}s"],
                                ["Rolling ep rew mean", f"{mean_rew:.2f}"],
                                ["Rolling ep rew std",f"{std_rew:.2f}"],
                                ["Rolling ep length",f"{mean_len:.2f}"],
                                ["N updates", episodes]]
                        
                        print(tabulate(table ,tablefmt="rounded_outline" , headers=["Early Stop",""]))
                        return
                    
                # LOGGING (SB3-STYLE) =====================
                if episodes % log_interval == 0 and self.verbose == 1:
                    elapsed = int(time.time() - start_time)
                    mean_rew = np.mean(self.episode_rewards[-log_interval:])
                    std_rew = np.std(self.episode_rewards[-log_interval:])
                    mean_len = np.mean(self.episode_lengths[-log_interval:])
                
                    fps = int(steps / (elapsed + 1e-8))
                    adv_mean = advantages.mean().item()
                    adv_std = advantages.std().item()
                    mean_entropy = entropy_mean.item()
                    mean_aux = np.mean(aux_losses[-log_interval:]) if aux_losses else 0.0
                    stats = [{
                        "header": "rollout",
                        "stats": dict(
                            ep_len_mean=mean_len,
                            ep_rew_mean=mean_rew,
                            ep_rew_std=std_rew,
                            policy_entropy=mean_entropy,
                            advantage_mean=adv_mean,
                            advantage_std=adv_std,
                            aux_loss_mean=mean_aux
                        )}, {
                        "header": "time",
                        "stats": dict(
                            fps=fps,
                            episodes=episodes,
                            time_elapsed=elapsed,
                            total_timesteps=steps
                        )}, {
                        "header": "train",
                        "stats": dict(
                            loss=loss.item(),
                            policy_loss=policy_loss.item(),
                            value_loss=value_loss.item(),
                            explained_variance=explained_var.item(),
                            n_updates=episodes,
                            progress=100 * steps / total_timesteps
                        )}
                    ]
                    if len(aux_metrics_log.items()) > 0:
                        aux_stats = {
                            "header": "aux_train",
                            "stats": {}
                        }
                        for aux_name, metrics in aux_metrics_log.items():
                            for k, v in metrics.items():
                                aux_stats["stats"][f"aux_{aux_name}_{k}"] = v
                        stats.append(aux_stats)
                    if self.use_rnd:
                        mean_rnd_bonus = np.mean([self.rnd(torch.tensor(np.array(o), dtype=torch.float32, device=self.device).unsqueeze(0)).item() for o in trajectory])
                        stats.append({
                            "header": "rnd_net_dist",
                            "stats": {"mean_rnd_bonus": mean_rnd_bonus}
                        })
                    if self.memory_learn_retention:
                        stats.append({
                            "header": "memory",
                            "stats": {
                                "usefulness_loss": mem_loss.item()}
                        })
                    
                    print_sb3_style_log_box(stats)
                    
            except KeyboardInterrupt:
                print("\n[Stopped by user] Gracefully exiting training loop...")
                return
            
        if self.verbose == 1:
            print(f"Training complete. Total episodes: {episodes}, total steps: {steps}")

    def predict(self, obs, deterministic=False, done=False, reward=0.0):
        """
        Computes action for a given observation, with support for memory context.

        Args:
            obs (np.ndarray): Environment observation.
            deterministic (bool): Use argmax instead of sampling.
            done (bool): If episode ended, will reset trajectory buffer.
            reward (float): Last received reward (for memory context).

        Returns:
            int: Action index.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        # Track full trajectory for memory
        if not hasattr(self, "trajectory_buffer") or self.trajectory_buffer is None:
            self.trajectory_buffer = []
        if len(self.trajectory_buffer) == 0:
            self.trajectory_buffer.append((obs_t.cpu().numpy(), 0, 0.0))
        else:
            last_action = self.last_action if hasattr(self, "last_action") else 0
            last_reward = self.last_reward if hasattr(self, "last_reward") else 0.0
            self.trajectory_buffer.append((obs_t.cpu().numpy(), last_action, last_reward))
        context_traj = self.trajectory_buffer.copy()
        actions_int = [a for _, a, _ in context_traj]
        rewards_float = [r for _, _, r in context_traj]
        obs_stack = torch.stack([torch.tensor(o, dtype=torch.float32, device=self.device) for o, _, _ in context_traj])
        logits, _, _ = self.policy(
            obs_stack, obs_t,
            actions=torch.tensor(actions_int, device=self.device),
            rewards=torch.tensor(rewards_float, device=self.device)
        )
        if deterministic:
            action = torch.argmax(logits).item()
        else:
            dist = Categorical(logits=logits)
            action = dist.sample().item()
        self.last_action = action
        self.last_reward = reward
        if done:
            self.trajectory_buffer = []
        return action

    def save(self, path="memoryppo.pt"):
        """Save policy weights to file."""
        torch.save(self.policy.state_dict(), path)

    def load(self, path="memoryppo.pt"):
        """Load policy weights from file."""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))

    def evaluate(self, n_episodes=10, deterministic=False, verbose=True):
        """
        Evaluates policy over several episodes, reporting mean/std return.

        Args:
            n_episodes (int): Number of test episodes.
            deterministic (bool): Use argmax instead of sampling.
            verbose (bool): Print results to console.

        Returns:
            mean_return (float): Average reward.
            std_return (float): Std deviation of rewards.
        """
        returns = []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            self.trajectory_buffer = []
            done = False
            total_reward = 0.0
            last_reward = 0.0
            while not done:
                action = self.predict(obs, deterministic=deterministic, reward=last_reward)
                obs, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                last_reward = reward
                if done:
                    self.trajectory_buffer = []
            returns.append(total_reward)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if verbose:
            print(f"Evaluation over {n_episodes} episodes: mean return {mean_return:.2f}, std {std_return:.2f}")
        return mean_return, std_return


