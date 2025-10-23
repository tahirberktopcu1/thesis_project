# Thesis Draft - Reinforcement Learning-Based Path Planning for a Mobile Robot

## Abstract
This thesis investigates whether a PPO-trained policy can match or exceed classical planners on random obstacle-laden maps. We build a custom Gymnasium environment with 360 degrees proximity sensors, implement a PPO navigation agent, and compare it against an A* baseline on identical scenarios. Results show PPO achieves shorter average paths but occasionally fails in tight corridors, highlighting the trade-off between adaptivity and determinism.

## Chapter 1 - Introduction
1. **Motivation:** Autonomous navigation is central to AMRs, warehouse robots, and service robotics. Efficient path planning must balance safety with path length, often under uncertain maps.  
2. **Problem Statement:** Classical algorithms rely on precomputed maps and heuristics; RL can learn behaviours from interaction but needs thorough evaluation.  
3. **Objectives:**  
   - Design a reproducible 2D environment with configurable obstacles.  
   - Train a PPO policy and benchmark it against A*.  
   - Analyse metrics (success, path length, computation) and visualise learning behaviour.  
4. **Contributions:** (i) Gym-based navigation environment, (ii) PPO vs. A* evaluation pipeline, (iii) visualisation suite and comprehensive analysis.

## Chapter 2 - Literature Review
1. Classical planners: A*, D*, RRT, hybrid global/local systems. Strengths include optimality guarantees and deterministic behaviour; weaknesses include sensitivity to map changes.  
2. RL-based navigation: DQN variants, PPO, hybrid approaches, domain randomisation. PPO's stable updates make it suitable for navigation tasks with shaped rewards.  
3. Research gap: Few studies provide direct, quantitative comparison of PPO policies and A* on identical randomised scenes.  
4. Summary: Table summarising contributions of key works (Mnih et al., 2015; Schulman et al., 2017; Chen et al., 2019; Koenig & Likhachev, 2002).

## Chapter 3 - Environment and Agent Design
1. **Environment (Step 2):**  
   - Map geometry (768x576 px), rectangular obstacles, agent radius 12 px.  
   - Random obstacle generator with size/margin controls.  
   - Observation vector (position, heading, goal vector, 24 ray sensors).  
   - Discrete actions: forward, turn left/right.  
2. **Reward (Step 3):**  
   - Distance progress, time penalty, alignment bonus.  
   - Safety penalty (sensor threshold), directional heuristics.  
   - Oscillation and negative progress penalties plus terminal rewards.  
3. **Implementation references:** `navigator/config.py`, `navigator/env.py`.

## Chapter 4 - RL Model Development
1. **PPO configuration:** Stable-Baselines3; policy network [128,128], `n_steps=4096`, `batch_size=1024`, `lr=3e-4`, `ent_coef=0.01`.  
2. **Training pipeline:** Single-threaded `DummyVecEnv`, `EvalCallback`, TensorBoard logging.  
3. **Checkpoints:** `models/ppo_linear_navigator.zip`, `models/best_model.zip`.  
4. **Hyperparameter justification and ablation possibilities.**

## Chapter 5 - Experiments and Results
1. **Evaluation setup:** Same random seeds across PPO and A*, difficulties (`warmup`, `medium`, `hard`), 100 episodes per configuration.  
2. **Metrics:** success rate, mean/median path length, mean steps, mean nodes expanded, mean reward.  
3. **Quantitative results:** Tables generated from `results/baselines.csv`. Example (hard, seed 123): PPO success 94 %, path 757.7 px; A* success 100 %, path 775.1 px, nodes expanded 582.8.  
4. **Visualisations:**  
   - `comparison_hard.png`: bar chart comparing metrics.  
   - `training_curve.png`: episode reward vs. timesteps (raw + smoothed).  
   - `trajectory_hard.png`: PPO trajectory on sample map.  
5. **Discussion:**  
   - PPO finds shorter average paths but incurs small failure rate in tight passages.  
   - A* remains deterministic but expands more nodes and requires grid discretisation.  
   - Reward shaping and heuristics help PPO avoid oscillation.

## Chapter 6 - Conclusion & Future Work
- **Summary:** PPO delivers adaptive navigation with competitive path efficiency; classical planners still excel in guaranteed goal completion.
- **Limitations:** PPO failures in narrow corridors; evaluation limited to rectangular obstacles.  
- **Future Work:** Integrate RRT baseline, dynamic obstacles, curriculum learning, sim-to-real transfer, noise modelling.  
- **Broader impact:** Approach provides a benchmark framework for cross-evaluating learned and classical planners.

## Appendices
- **A. Implementation listings:** Key code snippets (`navigator/env.py`, reward definitions, evaluation scripts).  
- **B. Reproducibility:** Environment setup commands (`pip install -r requirements.txt`), evaluation commands (`evaluate_planners.py`, `evaluate_rl.py`, `visualize_results.py`).  
- **C. Additional plots:** Optional seeds, training curves, alternative trajectories.
