# Step 6 - Visualisation and Analysis

## 6.1 Objectives
- Present quantitative results (success rate, path length, computational effort) in easily interpretable plots.
- Inspect training dynamics (TensorBoard curves) and qualitative behaviour (trajectory renderings).
- Summarise findings in the written thesis with supporting figures.

## 6.2 Tools and Scripts
| Artifact | Location | Purpose |
|----------|----------|---------|
| `tensorboard --logdir logs` | TensorBoard | Inspect training curves (episode reward, value loss, policy loss). |
| `plot_training_curve.py` | Python script | Extract a specific scalar (default `rollout/ep_rew_mean`) from TensorBoard logs and save a PNG. |
| `visualize_results.py` | Python script | Generate side-by-side bar charts from CSV baselines. |
| `plot_trajectory.py` | Python script | Roll out PPO (or random policy) and overlay the resulting path with obstacles/start/goal. |
| `results/baselines.csv` | CSV file | Aggregated metrics for A* and PPO (extendable with other planners). |
| `results/figures/` | Output directory | Stores generated PNG plots (e.g., `comparison_hard.png`). |
| `enjoy.py --render-training` | Runtime tool | Produce qualitative observations or screen recordings of PPO behaviour. |

## 6.3 Quantitative Plots
1. Run the evaluation scripts to populate `results/baselines.csv` (see Step 5).  
2. Generate comparison figures:
   ```bash
   python visualize_results.py \
     --csv results/baselines.csv \
     --out-dir results/figures \
     --difficulty hard
   ```
   - Output: `results/figures/comparison_hard.png` containing success rate, mean path length, steps (PPO), and reward/nodes expanded bars.
   - Repeat with `--difficulty warmup` or `--difficulty medium` if those scenarios are evaluated.
3. Include the generated figures in the thesis (Results/Discussion chapters) and describe key observations.

## 6.4 Training Curves
- TensorBoard logs in the `logs/` directory (created during PPO training) record episode reward, value loss, KL divergence, etc.
- Quick export to PNG without opening the TensorBoard UI:
  ```bash
  python plot_training_curve.py \
    --logdir logs \
    --tag rollout/ep_rew_mean \
    --out results/figures/training_curve.png \
    --smooth 20
  ```
- The script writes `results/figures/training_curve.png`; include both raw and smoothed curves when discussing convergence speed, stability, and plateaus.

## 6.5 Qualitative Analysis
- Use `python enjoy.py --model ... --render-training` to observe the agent's behaviour in real time. Optional: screen record segments for presentation.
- Generate reproducible trajectory plots:
  ```bash
  python plot_trajectory.py \
    --model models/ppo_linear_navigator.zip \
    --seed 123 \
    --difficulty hard \
    --out results/figures/trajectory_hard.png
  ```
- Note behavioural patterns: e.g., PPO detours slightly longer than A* but avoids oscillation, or fails in specific corner cases. Use the saved trajectory figures to illustrate these behaviours in the thesis.

## 6.6 Sample Result Interpretation
Using the current hard-difficulty baseline (seed = 123, 100 episodes):
- **PPO**: 94 % success, mean path approx. 758 px, mean steps approx. 152, mean reward approx. 134.  
- **A***: 100 % success, mean path approx. 775 px, median approx. 770 px, mean nodes expanded approx. 583.

These values indicate PPO delivers shorter average paths in the sampled scenes, albeit with ~6 % failure rate (likely due to tight corridors). A* retains perfect success albeit with higher computational cost (nodes expanded). This trade-off should be discussed alongside plots and tensorboard curves to contextualise RL vs. classical methods.

## 6.7 Deliverables for Thesis
- Figures: comparison bar chart(s), TensorBoard convergence screenshot(s), optional trajectory snapshots.
- Textual analysis summarising differences in success rate, path efficiency, and computation.
- Appendix or supplementary CSV for reproducibility.



