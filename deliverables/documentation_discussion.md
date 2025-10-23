# Project Documentation & Results Discussion (Single File)

## 1. Project Overview
- **Title:** Reinforcement Learning-Based Path Planning for a Mobile Robot  
- **Goal:** Train a PPO agent in a custom Gymnasium environment and compare it against an A* baseline on identical procedurally generated maps.  
- **Key Outcomes:** PPO finds shorter average paths but with small failure rate (~6 %); A* guarantees success but expands more nodes and produces slightly longer paths.

## 2. Methodology Summary (Steps 1-4)
1. **Problem Definition & Literature Review**  
   - Classical planners (A*, D*, RRT) vs. RL navigation (DQN, PPO, hybrid methods).  
   - Gap: lack of direct comparisons between PPO and deterministic planners on shared random scenarios.
2. **Environment Design**  
   - 768x576 px arena, agent radius 12 px.  
   - 24 ray sensors (360 degrees), discrete actions (forward, turn left/right).  
   - Random rectangular obstacles with clearance constraints.
3. **Agent & Reward Design**  
   - Observation vector: position, heading, goal direction, distance, sensor readings.  
   - Reward shaping: progress, time penalty, alignment bonus, safety penalty, oscillation/negative-progress penalties, terminal rewards.
4. **RL Model Development**  
   - Stable-Baselines3 PPO with net_arch [128,128], `n_steps=4096`, `batch_size=1024`, `lr=3e-4`, `ent_coef=0.01`.  
   - Training logged to TensorBoard (`logs/`), checkpoints saved in `models/`.

## 3. Experimental Setup (Step 5)
- **Scenarios:** Random obstacle layouts ("hard" difficulty: 8 obstacles, larger size range).  
- **Episodes:** 100 per algorithm with shared seed base (123).  
- **Metrics:** success rate, mean/median path length, mean steps (PPO), mean nodes expanded (A*), mean reward (PPO).  
- **Logging:** `results/baselines.csv` via `evaluate_planners.py` and `evaluate_rl.py`.

## 4. Results Discussion (Step 6)
| Algorithm | Success Rate | Mean Path Length (px) | Mean Steps | Mean Reward | Mean Nodes Expanded |
|-----------|--------------|-----------------------|------------|-------------|---------------------|
| PPO       | 0.94         | 757.73                | 151.51     | 134.21      | -                   |
| A*        | 1.00         | 775.12                | -          | -           | 582.76              |

- **Interpretation:**  
  - PPO's policy is more path-efficient on average but fails in ~6 % of episodes (tight corridors).  
  - A* maintains 100 % success but explores more nodes and yields slightly longer paths.  
- **Learning curves:** `results/figures/training_curve.png` shows episode reward convergence, indicating stable training with the chosen reward shaping.  
- **Qualitative trajectories:** `results/figures/trajectory_hard.png` illustrates PPO navigating around obstacles with smooth turns.

## 5. Documentation Deliverables
- **Thesis report:** `deliverables/thesis_full.md` (complete text ready for PDF export).  
- **Presentation highlights:** `deliverables/presentation_outline.md` (summary of defence talking points).  
- **Figures:**  
  - Training curve (`results/figures/training_curve.png`)  
  - Comparison chart (`results/figures/comparison_hard.png`)  
  - Trajectory plot (`results/figures/trajectory_hard.png`)  
- **Supporting scripts:** `train.py`, `evaluate_planners.py`, `evaluate_rl.py`, `visualize_results.py`, `plot_training_curve.py`, `plot_trajectory.py`.  
- **Metrics:** `results/baselines.csv` (ready for tables in thesis or supporting material).

## 6. Future Work (Step 7 Planning)
- Add RRT/RRT* baselines to extend comparisons.  
- Explore curriculum learning or dynamic obstacles to improve PPO success rate.  
- Investigate sim-to-real transfer via domain randomisation.  

This single document summarises the project, methodology, results discussion, and supporting materials prepared for submission. All referenced files reside in the `deliverables/` folder and project root as indicated above.
