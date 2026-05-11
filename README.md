# 🎮 ppoagent

Train reinforcement learning agents to play Gymnasium environments using **Proximal Policy Optimization (PPO)** — built from scratch with TensorFlow and Keras.

---

## Overview

**ppoagent** provides a clean, self-contained PPO implementation that works with any discrete-action Gymnasium environment. Drop in your environment of choice, run the notebook, and watch your agent learn.

The included demo trains and evaluates an agent on **CartPole-v1**, but the agent architecture is environment-agnostic — swap in `LunarLander-v2`, `MountainCar-v0`, or any other `gym.Env` with a flat observation space and discrete action space.

---

## How It Works

PPO is a policy-gradient algorithm that improves stability by clipping the policy update ratio, preventing destructively large updates. This implementation uses:

- **Separate policy and value networks** — each a 2-layer MLP (64 → 64 → output)
- **Clipped surrogate objective** with `clip_epsilon = 0.2`
- **Entropy bonus** to encourage exploration
- **Discounted reward computation** with episode-boundary resets
- **Adam optimizers** for both networks (`lr = 3e-4`)

---

## Project Structure

```
ppoagent.ipynb
├── PPOAgent class          # Policy + value networks, action selection, training step
├── Environment setup       # Gymnasium env initialization
├── Training loop           # 1000-episode rollout + per-episode update
├── Model saving            # Persists policy and value weights as .h5 files
└── Testing loop            # Renders trained agent in real time
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/IMApurbo/ppoagent.git
cd ppoagent
```

### 2. Install dependencies

```bash
pip install tensorflow gymnasium numpy
```

> For environments that require rendering (e.g., `render_mode="human"`), also install:
> ```bash
> pip install gymnasium[classic-control]
> ```

### 3. Run the notebook

Open `ppoagent.ipynb` in Jupyter and run all cells in order:

```bash
jupyter notebook ppoagent.ipynb
```

---

## Usage

### Train on any environment

Replace `"CartPole-v1"` with any compatible Gymnasium environment:

```python
env = gym.make("LunarLander-v2")
input_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = PPOAgent(input_dim=input_dim, action_dim=action_dim)
```

Then run the training loop as-is.

### Save and load a trained agent

```python
# Save after training
agent.save("my_agent")

# Load and resume
agent.load("my_agent")
```

This saves `my_agent_policy.h5` and `my_agent_value.h5`.

### Test a trained agent with rendering

```python
test_ppo_agent(agent, env_name="CartPole-v1", episodes=10)
```

---

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `lr` | `0.0003` | Adam learning rate |
| `gamma` | `0.99` | Discount factor |
| `clip_epsilon` | `0.2` | PPO clipping range |
| `episodes` | `1000` | Training episodes |
| Hidden units | `64 × 2` | Layers in policy & value networks |

---

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Gymnasium
- NumPy

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 IMApurbo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Author

**IMApurbo** — [GitHub](https://github.com/IMApurbo)
