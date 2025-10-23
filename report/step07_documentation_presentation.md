# Step 7 - Documentation, Results Discussion, and Presentation Preparation

## 7.1 Thesis / Report Structure
The thesis manuscript is organised according to the chapters below, drawing on the material produced in Steps 1-6:

1. **Introduction**
   - Motivation for autonomous navigation.
   - Problem statement (summarised from Step 1).
   - Thesis contributions (e.g., custom Gym environment, PPO vs. A* benchmark suite, quantitative analysis).

2. **Literature Review**
   - Classical planners (A*, D*, RRT, hybrids) with citations (see Step 1 references).
   - RL-based navigation (DQN, PPO, sim-to-real).
   - Identified research gap and justification for the chosen comparison methodology.

3. **Methodology**
   - **Environment Design (Step 2):** map geometry, obstacle generation, observation and action spaces, reward shaping.
   - **Agent & Reward Design (Step 3):** full observation vector, discrete actions, reward formula (progress, safety, oscillation penalties).
   - **RL Model Development (Step 4):** PPO configuration (SB3), hyperparameters (policy network [128,128], `n_steps=4096`, `batch_size=1024`, learning rate `3e-4`, entropy coefficient `0.01`).
   - **Baseline Planners:** A* grid rasterisation, inflation logic, metrics.

4. **Experiments & Results (Step 5)**
   - Evaluation protocol: identical random seeds for PPO and A*, difficulty profiles, number of episodes.
   - Primary metrics table (hard difficulty, seed 123, 100 episodes):

     | Algorithm | Success Rate | Mean Path Length (px) | Mean Steps | Mean Reward | Mean Nodes Expanded |
     |-----------|--------------|-----------------------|------------|-------------|---------------------|
     | PPO       | 0.94         | 757.73                | 151.51     | 134.21      | -                   |
     | A*        | 1.00         | 775.12                | -          | -           | 582.76              |

   - CSV summary stored in `results/baselines.csv` to regenerate or extend metrics.

5. **Visualisation & Analysis (Step 6)**
   - Figures included:
     - `results/figures/training_curve.png` (episode reward vs. timesteps with smoothing).
     - `results/figures/comparison_hard.png` (PPO vs. A* metrics).
     - `results/figures/trajectory_hard.png` (representative path).
   - Discussion covers qualitative observations (PPO takes slightly shorter paths but misses about 6 % cases; A* always succeeds but expands about 583 nodes).

6. **Conclusion & Future Work**
   - Summary of findings: PPO's adaptability vs. A*'s determinism.
   - Future extensions: integrate RRT baseline, curriculum training, sim-to-real transfer, dynamic obstacles.

7. **Appendices**
   - Provide details on hyperparameters, training/evaluation scripts, reproducibility notes (command listings, environment setup).

## 7.2 Documentation Status
- The Results chapter already embeds the metrics table exported from `results/baselines.csv` (hard difficulty, 100 episodes).
- Figures `training_curve.png`, `comparison_hard.png`, and `trajectory_hard.png` are referenced with captions that explain the corresponding trends.
- Methodology sections cite the relevant modules (`navigator/env.py`, `evaluate_rl.py`, `evaluate_planners.py`) when describing implementation details.
- Bibliography entries for the core references (Mnih 2015, Schulman 2017, Chen 2019, Koenig & Likhachev 2002, LaValle 2006) are prepared in the reference list.
- Appendix material documents reproducibility steps, including virtual environment setup and the commands required to regenerate models, metrics, and figures.

## 7.3 Discussion Highlights for the Results Chapter
1. **Performance trade-off:** A* achieves 100 % success but with longer mean paths and higher computational cost (nodes expanded). PPO achieves shorter average paths but occasionally fails in tight corridors (94 % success).
2. **Learning behaviour:** Training curve shows reward stabilisation after initial fluctuations; the reward shaping balance between progress and safety informs convergence speed.
3. **Qualitative behaviour:** Trajectory plots illustrate PPO's smooth turns and corner handling; failure cases occur when the policy oscillates near narrow passages.
4. **Metric sensitivity:** Results note the impact of grid resolution (A*) and sensor range/reward weights (PPO) on the recorded metrics.

## 7.4 Submission Materials
- **Thesis manuscript:** `report/thesis_full.md` (mirrored to `deliverables/thesis_full.md`).
- **Defence highlights summary:** `deliverables/presentation_outline.md`, capturing the points presented verbally.
- **Figures and tables:** PNG files in `results/figures/` and metrics in `results/baselines.csv`, referenced throughout the report.
- **Supporting code:** Training, evaluation, and visualisation scripts in the project root, ensuring every reported result can be reproduced.

Collectively, these materials complete Step 7 of the methodology by delivering the written thesis, analysed results, and accompanying evidence.
