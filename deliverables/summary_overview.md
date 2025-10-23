# Project Overview & Workflow Summary

This document recaps every major stage completed during the project, matching the specification steps.

## Step 1 - Problem Definition & Literature Review
- Identified the limitations of classical planners (A*, D*, RRT) vs. RL-based navigation.
- Reviewed PPO, DQN, and hybrid RL approaches.
- Summarised research gap: lack of direct comparisons between PPO and deterministic planners on identical random maps.
- References compiled (Mnih 2015, Schulman 2017, Chen 2019, Koenig & Likhachev 2002, LaValle 2006, etc.).
- Document: `report/step01_problem_definition.md`

## Step 2 - Environment Design
- Built a custom Gymnasium environment (`navigator/env.py`) with:
  - 768x576 px map.
  - Random rectangular obstacles, agent radius 12 px.
  - 24-ray proximity sensors, discrete actions (forward, turn left, turn right).
  - Safety-aware forward motion.
- Documentation: `report/step02_environment_design.md`

## Step 3 - Agent & Reward Design
- Defined observation vector (position, heading, goal distance, goal direction, sensors).
- Designed reward function balancing progress, safety, oscillation avoidance.
- Implemented in `navigator/env.py` (`step` method).
- Documentation: `report/step03_agent_reward_design.md`

## Step 4 - RL Model Development
- Implemented PPO using Stable-Baselines3 (`train.py`).
- Key hyperparameters: net_arch [128,128], `n_steps=4096`, `batch_size=1024`, `lr=3e-4`, `ent_coef=0.01`.
- Evaluation via `EvalCallback`, logging to TensorBoard (`logs/`).
- Documentation: `report/step04_model_development.md`

## Step 5 - Testing & Comparison
- Evaluation scripts:
  - PPO: `evaluate_rl.py`
  - A*: `evaluate_planners.py`
- Metrics logged to `results/baselines.csv` (success rate, path length, steps, reward, nodes expanded).
- Example run (hard difficulty, 100 episodes): PPO 94% success, A* 100% success.
- Documentation: `report/step05_testing_and_comparison.md`

## Step 6 - Visualisation & Analysis
- Generated figures:
  - `results/figures/training_curve.png` (training reward curve).
  - `results/figures/comparison_hard.png` (PPO vs A* bar chart).
  - `results/figures/trajectory_hard.png` (trajectory plot).
- Visualisation scripts: `visualize_results.py`, `plot_training_curve.py`, `plot_trajectory.py`.
- Documentation: `report/step06_visualization_analysis.md`

## Step 7 - Reporting
- Compiled the full thesis (`report/thesis_full.md`) and a documentation summary (`report/step07_documentation_presentation.md`) that records defence highlights and submission materials.
- Deliverables folder includes the thesis, presentation highlights summary, figures, README, and this overview.
- Final thesis is also copied as `deliverables/thesis_full.md`.

## Deliverables Recap
- Source code: PPO training, environment, planners, visualization scripts.
- Trained models: `models/ppo_linear_navigator.zip`, `models/best_model.zip`.
- Data: `results/baselines.csv`, TensorBoard logs (`logs/`).
- Figures: training curve, comparison chart, trajectory plot.
- Documentation: thesis text, step-by-step reports, presentation highlights summary, this overview.

With these components, the project satisfies all objectives and produces reproducible experiments, visuals, and documentation ready for submission and final defense.
