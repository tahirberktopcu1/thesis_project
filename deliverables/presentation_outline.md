# Project Highlights - Reinforcement Learning-Based Path Planning

This document lists the main talking points presented during the defence of the project "Reinforcement Learning-Based Path Planning for a Mobile Robot."

- **Project identity:** Title, author (Tahir), supervising department, and academic term.
- **Motivation:** Classical planners (A* and related methods) depend on complete maps and manual heuristics; reinforcement learning offers adaptive navigation by learning from interaction.
- **Objectives achieved:** A Gymnasium-based environment with rectangular obstacles, a PPO agent trained for autonomous navigation, comparative evaluation against an A* baseline, and reproducible documentation of all experiments.
- **Environment summary:** 768x576 pixel arena, boundary walls, 24 ray sensors covering 360 degrees, discrete actions (forward, turn left, turn right), and optional random obstacle layouts.
- **Reward structure:** Progress bonus, time penalty, heading alignment incentive, sensor-based safety penalties, oscillation discouragement, success bonus, and collision penalty.
- **Training configuration:** Stable-Baselines3 PPO with net architecture [128,128], n_steps 4096, batch size 1024, learning rate 3e-4, entropy coefficient 0.01; logs stored in `logs/`, checkpoints in `models/`.
- **Evaluation approach:** 100 randomised episodes per algorithm at "hard" difficulty, identical seeds for PPO and A*, metrics captured in `results/baselines.csv` (success rate, path length, steps, reward, nodes expanded).
- **Key quantitative results:** PPO success rate approximately 94 percent with mean path length around 758 pixels; A* success rate 100 percent with mean path length around 775 pixels and roughly 583 nodes expanded.
- **Visual evidence:** `results/figures/training_curve.png` (learning progress), `comparison_hard.png` (metric comparison), `trajectory_hard.png` (representative path).
- **Conclusions and next steps:** PPO provides shorter paths but occasional failures in tight corridors; A* remains deterministic but slightly longer. Proposed extensions include RRT baselines, curriculum learning, dynamic obstacles, and sim-to-real transfer experiments.
