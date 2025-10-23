# Reinforcement Learning-Based Path Planning for a Mobile Robot

*Author:* Tahir  
*Date:* October 2025

---

## Abstract
Autonomous mobile robots must navigate cluttered environments safely while reaching goals efficiently. Classical planners such as A* or RRT rely on handcrafted heuristics and static maps, limiting adaptability in partially known or dynamic spaces. Reinforcement Learning (RL) offers the potential to learn navigation behaviours directly from interaction, yet direct comparisons between learned policies and deterministic planners on identical randomised scenarios remain scarce.  

This thesis presents a reproducible benchmarking framework that contrasts a Proximal Policy Optimization (PPO) policy against an A* planner inside a custom Gymnasium environment populated with procedurally generated rectangular obstacles. The agent observes normalised pose, goal direction, and 360° proximity rays, and executes a discrete action set (forward, turn left, turn right). Reward shaping balances path efficiency, safety, and oscillation avoidance.  

Extensive experiments show that PPO achieves shorter average paths (≈757 px) but incurs a small failure rate (≈6 %) in tight corridors, whereas A* maintains perfect success (100 %) at the expense of higher computational effort (≈583 node expansions) and slightly longer paths (≈775 px). Visual analyses (training curves, comparative bar charts, trajectory plots) highlight the trade-off between adaptive learning and deterministic guarantees. The resulting framework provides a foundation for future extensions including sampling-based planners, curriculum learning, and sim-to-real investigations.

---

## Acknowledgements
The author thanks the supervising faculty and peers for guidance, as well as the open-source communities behind Gymnasium, Stable-Baselines3, and Pygame.

---

## Table of Contents
1. [Introduction](#1-introduction)  
2. [Literature Review](#2-literature-review)  
3. [Environment Design](#3-environment-design)  
4. [Agent and Reward Design](#4-agent-and-reward-design)  
5. [RL Model Development](#5-rl-model-development)  
6. [Experimental Setup](#6-experimental-setup)  
7. [Results and Analysis](#7-results-and-analysis)  
8. [Conclusion and Future Work](#8-conclusion-and-future-work)  
9. [References](#9-references)  
10. [Appendices](#appendices)

---

## 1. Introduction
### 1.1 Motivation
Mobile robots increasingly operate in semi-structured environments such as warehouses, hospitals, and service spaces. Planning safe and efficient routes is critical, particularly when maps are incomplete or obstacles change over time. Traditional path planners require meticulous tuning and may fail gracefully if underlying assumptions are violated. Reinforcement Learning promises adaptive navigation strategies but needs rigorous benchmarking against established methods.

### 1.2 Research Problem
The central question is whether an RL agent—specifically a PPO policy—can achieve comparable navigation performance to deterministic classical planners on procedurally generated maps while preserving safety and path efficiency. Addressing this question requires a controlled environment, shared evaluation pipelines, and reproducible metrics.

### 1.3 Contributions
1. A custom Gymnasium environment with configurable rectangular obstacles, 360° proximity sensors, and discrete motion primitives.  
2. A PPO-based navigation agent trained with reward shaping that balances progress, safety, and oscillation avoidance.  
3. A classical planner baseline (A*) with grid inflation and metric logging, plus scripts (`evaluate_rl.py`, `evaluate_planners.py`) for comparative analysis.  
4. Visualisation tools (`plot_training_curve.py`, `visualize_results.py`, `plot_trajectory.py`) supporting detailed interpretation of learning behaviour and trajectories.

---

## 2. Literature Review
### 2.1 Classical Path Planning
Graph-based planners, notably A* [1] and D* Lite [2], provide optimal or near-optimal trajectories on discretised maps but require global updates when map costs change. Sampling-based algorithms like RRT/RRT* [3] and PRM handle high-dimensional spaces yet produce suboptimal and jagged paths unless heavily post-processed. Hybrid architectures commonly combine global planners with local obstacle avoidance, but they remain sensitive to perception noise and heuristics.

### 2.2 Reinforcement Learning for Navigation
Deep Q-Networks (DQN) demonstrated end-to-end control from raw observations [4], but their discretisation of action and state spaces limits applicability to richer motion primitives. Policy gradient methods, particularly PPO [5], stabilise learning through clipped updates and have been applied successfully to navigation tasks with LiDAR and visual inputs [6][7]. Surveys [8][9] highlight RL’s potential for local decision-making, especially when combined with curriculum learning or domain randomisation.

### 2.3 Research Gap
Existing RL navigation studies rarely provide direct, quantitative comparisons with deterministic planners on diverse random scenarios. Conversely, classical planner benchmarks seldom consider learned policies. This thesis addresses the gap by systematically evaluating PPO and A* on identical procedurally generated maps, logging comparable metrics.

---

## 3. Environment Design
The environment (implemented in `navigator/env.py`) is a 768 × 576 px arena with a circular agent of radius 12 px. Obstacles are axis-aligned rectangles generated via rejection sampling; parameters control count, size, and spacing.  

### 3.1 Observations
Each timestep returns a 31-dimensional vector comprising:
- Normalised agent position and heading (cos θ, sin θ).  
- Normalised distance to the goal and goal direction unit vector.  
- 24 proximity rays evenly spaced around the agent (sensor range 240 px).  

### 3.2 Actions
Discrete action space (size 3):
1. Move forward (clamped by clearance checks).  
2. Turn left in place.  
3. Turn right in place.  
Forward moves use `forward_speed=6 px/step`, rotations use `turn_speed=π/14 rad`. 

### 3.3 Safety Logic
Before advancing, `_move_forward` evaluates available space to avoid collisions. Obstacle detection functions support both reward shaping and trajectory visualisation scripts.

---

## 4. Agent and Reward Design
Reward shaping is critical for learning stable policies:
- **Progress reward:** `0.25 × Δdistance_to_goal`.  
- **Step penalty:** `0.01` (forward) or `0.005` (turn).  
- **Alignment bonus:** `0.02 × dot(heading, goal_dir)`.  
- **Safety penalty:** if minimum ray reading < 0.35, penalise proportionally by `0.2`.  
- **Directional heuristic:** Encourage turning towards the clearer side when forward is blocked.  
- **Oscillation penalty:** Growing penalty for alternating left/right actions with negligible displacement (`idle_penalty=0.08`).  
- **Negative progress penalty:** Penalise consecutive increases in goal distance.  
- **Terminal rewards:** +1.0 on success, −1.0 on collision. Episodes truncate at 400 steps.

This shaping encourages efficient navigation, reduces dithering, and maintains safety margins.

---

## 5. RL Model Development
The PPO agent is trained via Stable-Baselines3 using the following configuration:
- Policy network: `[128, 128]` units for both policy and value heads.  
- Rollout length (`n_steps`): 4096; batch size: 1024.  
- Learning rate: 3e-4; discount factor: 0.99; GAE λ: 0.95.  
- Clipping parameter: 0.2; entropy coefficient: 0.01.  
- Evaluation: `EvalCallback` saves best checkpoints; TensorBoard logs stored in `logs/`.  

The training script `train.py` instantiates the environment, monitors episodes, and saves models (e.g., `models/ppo_linear_navigator.zip`).

---

## 6. Experimental Setup
- **Scenarios:** Random obstacle layouts matching `hard` difficulty (8 obstacles, larger size ranges).  
- **Episodes:** 100 evaluation episodes per algorithm.  
- **Seed management:** Both PPO and A* use identical seed bases (`seed + episode`).  
- **Metrics:** success rate, mean/median path length, mean steps (PPO), mean nodes expanded (A*), mean reward (PPO).  
- **Data collection:** Scripts append rows to `results/baselines.csv`.  
- **Visualisation:**  
  - `visualize_results.py` → `comparison_hard.png`.  
  - `plot_training_curve.py` → `training_curve.png`.  
  - `plot_trajectory.py` → `trajectory_hard.png`.  

---

## 7. Results and Analysis
### 7.1 Quantitative Metrics

| Algorithm | Success Rate | Mean Path (px) | Median Path (px) | Mean Steps | Mean Reward | Mean Nodes Expanded |
|-----------|--------------|----------------|------------------|------------|-------------|---------------------|
| PPO       | 0.94         | 757.73         | –                | 151.51     | 134.21      | –                   |
| A*        | 1.00         | 775.12         | 769.71           | –          | –           | 582.76              |

### 7.2 Observations
- **Success vs. Efficiency:** PPO’s learned policy finds shorter average paths but fails in 6 % of trials (typically narrow corridors). A* always succeeds but incurs higher computational effort and slightly longer routes.
- **Learning Behaviour:** The training curve (`training_curve.png`) shows reward convergence after early fluctuations, demonstrating the effectiveness of the reward shaping scheme.
- **Qualitative Trajectories:** `trajectory_hard.png` visualises a PPO rollout—smooth turns and obstacle avoidance are evident.

### 7.3 Discussion
The comparison highlights the trade-off between adaptive learning and guaranteed planning. PPO adapts to sensor feedback without grid discretisation but lacks deterministic guarantees. A* requires static rasterisation yet assures success given a viable path and inflates nodes when obstacles cluster.

---

## 8. Conclusion and Future Work
### 8.1 Conclusion
This thesis delivered a complete navigation benchmark where PPO and A* operate on identical procedurally generated maps. The PPO agent demonstrates competitive efficiency but slightly lower success rate; A* remains reliable yet computationally heavier. The developed scripts and documentation make reproduction straightforward and provide a basis for future studies.

### 8.2 Future Work
- Implement sampling-based baselines (RRT/RRT*) and compare using the same framework.  
- Explore curriculum learning to improve PPO’s success rate in dense maps.  
- Incorporate dynamic obstacles and sensor noise.  
- Investigate sim-to-real transfer by introducing domain randomisation and evaluating on physical robots or more detailed simulators (e.g., Gazebo, PyBullet).

---

## 9. References
[1] Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). *A Formal Basis for the Heuristic Determination of Minimum Cost Paths*. IEEE Transactions on Systems Science and Cybernetics.  
[2] Koenig, S., & Likhachev, M. (2002). *D* Lite*. Proceedings of AAAI.  
[3] LaValle, S. M. (2006). *Planning Algorithms*. Cambridge University Press.  
[4] Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.  
[5] Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.  
[6] Zhu, Y., et al. (2017). *Target-driven Visual Navigation in Indoor Scenes using Deep RL*. ICRA.  
[7] Long, P., et al. (2018). *Towards Optimally Decentralized Multi-Robot Collision Avoidance via Deep RL*. ICRA.  
[8] Kober, J., Bagnell, J. A., & Peters, J. (2013). *Reinforcement Learning in Robotics: A Survey*. International Journal of Robotics Research.  
[9] Chen, Y., et al. (2019). *Deep Reinforcement Learning for Motion Planning with Heterogeneous Agents*. International Journal of Robotics Research.

---

## Appendices
### Appendix A – Key Scripts
- `train.py`: PPO training loop with evaluation callback.  
- `navigator/env.py`: Environment dynamics, sensor model, reward computation.  
- `evaluate_rl.py`, `evaluate_planners.py`: Benchmarking utilities.  
- `plot_training_curve.py`, `visualize_results.py`, `plot_trajectory.py`: Visualisation scripts.

### Appendix B – Reproducibility Checklist
1. Create virtual environment and install dependencies:  
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Train PPO: `python train.py --timesteps 500000 --random-map`.  
3. Evaluate and log metrics:  
   ```bash
   python evaluate_planners.py --episodes 100 --resolution 12 --difficulty hard --seed 123 --csv results/baselines.csv --tag astar
   python evaluate_rl.py --model models/ppo_linear_navigator.zip --episodes 100 --seed 123 --difficulty hard --csv results/baselines.csv --tag ppo
   ```
4. Generate figures:  
   ```bash
   python visualize_results.py --csv results/baselines.csv --difficulty hard
   python plot_training_curve.py --logdir logs --tag rollout/ep_rew_mean
   python plot_trajectory.py --model models/ppo_linear_navigator.zip --seed 123 --difficulty hard
   ```
5. Inspect TensorBoard: `tensorboard --logdir logs`.

### Appendix C – Additional Results
- Optional evaluations with different seeds/difficulties.  
- Trajectory plots for failure cases (if recorded).  
- Ablation on reward components (e.g., removing oscillation penalty).***
