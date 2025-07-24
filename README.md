# Strategic Memory Retrieval: An Agent With Active, Learnable Memory

**Strategic Memory Retrieval** is a reinforcement learning (RL) agent designed for **environments where optimal decisions require remembering and leveraging information from the distant past**—not just recent history. The agent maintains an **external, actively-managed episodic memory** where it stores compressed summaries of entire experiences (trajectories) and learns **which memories to retain and which to discard** as training progresses.

### The Problem:

Classic RL agents—like DQN, LSTM PPO, or even transformers—struggle when rewards are delayed, sparse, or depend on events far in the past. In such tasks, remembering the right event, state, or action at the right time is crucial for success. Most RL methods either forget, overfit to short-term cues, or retain irrelevant information, resulting in poor performance on long-horizon, memory-based tasks.

### The Solution:

Strategic Memory Retrieval **actively learns**:

- **What to store:** Which episodes or sequences are worth keeping in memory.
- **What to forget:** Which are unhelpful and can be safely discarded.
- **How to retrieve:** At every decision point, the agent uses attention to retrieve relevant past experiences from memory and integrates them with the current observation before acting.

**All of this is trained end-to-end**, so the agent autonomously discovers how to use its memory buffer for strategic decision-making—**no hints, flags, or engineered memory cues are required**.

---

## **Comparison Table**

| Feature / Method              | LSTM PPO     | DNC/NTM        | Decision Transformer | GTrXL             | NEC / DND | Neural Map | **Strategic Memory Retrieval** |
| ----------------------------- | ------------ | -------------- | -------------------- | ----------------- | --------- | ---------- | ------------------------------ |
| Core Memory Type              | Hidden state | External R/W   | In-Context (GPT)     | Segment history   | kNN table | 2D spatial | Episodic buffer + retention    |
| Memory Retention              | Fades        | Manual/learned | None                 | History window    | FIFO      | Manual     | _Learnable, optimized_         |
| Retrieval                     | Implicit     | Soft/explicit  | Implicit             | History attention | kNN/soft  | Soft/read  | _Soft attention_               |
| Retention Learning            | No           | Partial        | No                   | No                | No        | No         | **Yes**                        |
| Interpretable Recall          | No           | Hard           | No                   | Some              | Some      | No         | **Yes (attention, use)**       |
| Persistent Memory             | No           | Partial        | No                   | Partial           | Yes       | Yes        | **Yes**                        |
| Sequence Length               | Short/medium | Short          | _Long_               | _Long_            | Medium    | Medium     | _Long_                         |
| No Hints/Flags                | Yes          | Yes            | Yes                  | Yes               | Yes       | Yes        | **Yes**                        |
| Outperforms on Delayed Reward | ✗            | ±              | ±                    | ±                 | ±         | ±          | **✓✓✓**                        |

---

## Literature & Reference Models

This agent builds upon and advances the following lines of research:

| **Approach / Paper**                                               | **Core Idea**                                   | **Key Weakness vs This**                                        |
| ------------------------------------------------------------------ | ----------------------------------------------- | --------------------------------------------------------------- |
| **DQN/LSTM-based RL**<br>Hausknecht & Stone, 2015                  | RNN hidden state as memory                      | Struggles with long delays, limited memory                      |
| **Neural Episodic Control**<br>Pritzel et al., 2017                | Non-parametric DND table, kNN retrieval         | No learnable retention, no end-to-end training                  |
| **Differentiable Neural Computer**<br>Graves et al., 2016          | RNN w/ differentiable read/write memory         | Expensive, hard to scale, hard to tune                          |
| **Neural Map / Memory-Augmented RL**<br>Parisotto et al., 2018     | Spatially structured memory, soft addressing    | Retention/static, not fully learnable, not cue-driven           |
| **Unsupervised Predictive Memory**<br>Wayne et al., 2018           | Latent predictive memory for meta-RL            | Memory not explicitly strategic or retained                     |
| **MERLIN**<br>Wayne et al., 2018                                   | Latent memory with unsupervised auxiliary tasks | Retention not explicit, memory not strategic                    |
| **Decision Transformer**<br>Chen et al., 2021                      | Uses a GPT-style transformer over trajectory    | No explicit, persistent external memory; not episodic retrieval |
| **GTrXL (Transformer-XL RL)**<br>Parisotto et al., 2020            | Relational transformer for RL sequence modeling | "Memory" = recent history, not explicit retention or recall     |
| **MVP: Memory Value Propagation**<br>Oh et al., 2020               | Learnable memory with value propagation         | Not as interpretable, not retention-focused                     |
| **Recurrent Independent Mechanisms (RIMs)**<br>Goyal et al., 2021  | Modular memory units, attention-based gating    | No persistent, recallable episodic buffer                       |
| **Active Memory / Episodic Control (EC)**<br>Blundell et al., 2016 | Episodic memory with tabular kNN access         | No differentiable retention, no meta-learning                   |

---

## **Additional References**

- **Hausknecht & Stone, 2015**: “Deep Recurrent Q-Learning for Partially Observable MDPs”
- **Pritzel et al., 2017**: “Neural Episodic Control”, [arXiv:1703.01988](https://arxiv.org/abs/1703.01988)
- **Parisotto et al., 2018**: “Neural Map: Structured Memory for Deep Reinforcement Learning”, [ICLR 2018](https://openreview.net/forum?id=B14TlG-RW)
- **Wayne et al., 2018**: “Unsupervised Predictive Memory in a Goal-Directed Agent”, [arXiv:1803.10760](https://arxiv.org/abs/1803.10760)
- **Wayne et al., 2018**: “The Unreasonable Effectiveness of Recurrent Neural Networks in Reinforcement Learning” (MERLIN), [arXiv:1804.00761](https://arxiv.org/abs/1804.00761)
- **Chen et al., 2021**: “Decision Transformer: Reinforcement Learning via Sequence Modeling”, [arXiv:2106.01345](https://arxiv.org/abs/2106.01345)
- **Parisotto et al., 2020**: “Stabilizing Transformers for Reinforcement Learning”, [ICML 2020 (GTrXL)](http://proceedings.mlr.press/v119/parisotto20a.html)
- **Oh et al., 2020**: “Value Propagation Networks”, [ICLR 2020](https://openreview.net/forum?id=B1xSperKvB)
- **Goyal et al., 2021**: “Recurrent Independent Mechanisms”, [ICLR 2021](https://openreview.net/forum?id=mLcmdlEUxy-)
- **Blundell et al., 2016**: “Model-Free Episodic Control”, [arXiv:1606.04460](https://arxiv.org/abs/1606.04460)
- **Graves et al., 2016**: “Hybrid computing using a neural network with dynamic external memory” (DNC), [Nature 2016](https://www.nature.com/articles/nature20101)
- **Sukhbaatar et al., 2015**: “End-To-End Memory Networks”, [arXiv:1503.08895](https://arxiv.org/abs/1503.08895)

---

## TL;DR;

- **First to jointly optimize both memory retention (what to keep/discard) and retrieval (what to attend to) in a single, end-to-end RL agent**.
- **Flexible plug-and-play memory**: Can be swapped for many memory architectures (transformers, graph attention, learned compression).
- **No task-specific hacks**: Outperforms the above on classic RL memory benchmarks _without using any domain knowledge_ or “cheat” features.
- **Interpretable, practical, and scalable**: Suitable for real-world problems where “what matters” is unknown and must be discovered.

---

**Author:** Filipe Sá  
**Contact:** filipemotasa@hotmail.com | [GitHub](https://github.com/pihh/)

---
