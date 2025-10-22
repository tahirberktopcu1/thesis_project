# Step 4 – RL Model Development

This step corresponds to “Implement DQN or PPO using Stable-Baselines3 or PyTorch. Train the model over multiple episodes until convergence.” We selected PPO with Stable-Baselines3 for its stability in discrete action spaces with complex reward shaping.

## 4.1 Algorithm Choice
- **PPO (Proximal Policy Optimization):** Provides clipped policy updates, robust convergence, and native support in Stable-Baselines3.
- **Why PPO over DQN:** Continuous state spaces and heavily shaped rewards are better handled by policy-gradient methods. DQN would require action discretisation for angles/motions and additional tuning.

## 4.2 Implementation Details
- **Training script:** `train.py`.
- **Environment vectorisation:** `DummyVecEnv` with a single environment builder.
- **Monitoring:** Stable-Baselines3 `Monitor` wrapper records episode rewards/lengths; TensorBoard logging enabled.
- **Evaluation loop:** `EvalCallback` runs periodic evaluations (default every 5000 steps) and saves the best model to `models/best_model.zip`.

## 4.3 Hyperparameters
- Policy network architecture: two fully connected layers with 128 units (`policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))`).
- Rollout length `n_steps = 4096`.
- Batch size `1024`.
- Learning rate `3e-4` with default Adam optimiser.
- Discount factor `γ = 0.99`.
- Generalised Advantage Estimation `λ = 0.95`.
- Clipping parameter `0.2`.
- Entropy coefficient `0.01` to encourage exploration.
- Total timesteps (default 200k, recommended 500k for complex maps).

## 4.4 Training Procedure
1. Instantiate configuration: `NavigatorConfig(randomize_obstacles=True)`.
2. Construct vectorised environment for training and evaluation.
3. Create PPO model with above hyperparameters.
4. Train via `model.learn`, logging to TensorBoard.
5. Save checkpoints (`models/ppo_linear_navigator.zip`, `models/best_model.zip`).

## 4.5 Reproducibility and Logging
- `logs/` contains TensorBoard event files (`plot_training_curve.py` converts them to PNG).
- `models/` stores final and best-performing checkpoints.
- `requirements.txt` lists package versions (Gymnasium, SB3, PyTorch, Matplotlib, Pandas).

With Step 4 completed, the PPO agent is trained and ready for comparative evaluation against classical planners.***
