# Soccer Physics AI – Self-Play PPO Training

2D physics-based soccer game where two agents learn to play against each other using **Proximal Policy Optimization (PPO)** with **self-play**.  
Agents are trained completely from scratch (no human data) and reach a strong level in **1 hour** on Apple Silicon M4 Max with 36GB unified memory.

## Features

- Fully vectorized environment (recommended **10,000 parallel games** for training)
- Realistic 2D physics (body physics, collisions, restitution, friction, corner repulsion)
- Dense reward shaping (goal, ball progression, proximity, collision bonuses, stall penalty)
- Self-play with an opponent pool (similar to AlphaZero-style league training)
- Role-symmetric training (agent randomly plays as left or right player to learn both sides)

## Files

| File        | Description |
|-------------|-----------|
| `game.py`   | `SoccerGameEnv` – vectorized physics environment with parallel games for faster training (supports rendering with pygame) |
| `ppo.py`    | `ActorCriticNetwork` + full `PPOTrainer` with GAE, entropy bonus, minibatch training, opponent pool, self-play logic |
| `run.py`    | Demo script – watch two trained AIs play (or control one with keyboard, by setting the parameters) |
| `soccer_final.pth` | Trained model |

## Training
 
```bash
# Install dependencies
pip install -r requirements.txt

# Train the agent (optimal parameters already set, but feel free to modify in ppo.py)
python3 ppo.py
```

Training parameters (already tuned):
- 10,000 parallel environments
- 2,000 steps per episode → 20M timesteps per update
- 200M total timesteps (100,000 games)
- Trained in 1 hour on M4 Max (MPS) or any modern GPU

The final model is saved as `soccer_final.pth` and can be used directly in `run.py`.

## Playing / Demo

```bash
# Run a match (both sides AI by default, but can set one or both to HUMAN in run.py)
python3 run.py
```

Models at other timesteps: [Dropbox](https://www.dropbox.com/scl/fo/mq5j7lds7er7gku5ec5pv/AIZpQFxs3D9gKO_N0Yr3eEI?rlkey=ycpbn82p4cy0ryuor9ec3gyxu&st=kstgqiup&dl=0)

## Blog

In the future, I plan to write a blog post detailing the implementation and training process and the intuitions behind the design choices.