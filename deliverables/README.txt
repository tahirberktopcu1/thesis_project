# Deliverables Checklist

## 1. Thesis Report
- File: `deliverables/thesis_full.md`
- Contents: full thesis text (abstract, introduction, literature review, methodology, results, conclusion, references, appendices).
- Notes: ready for conversion to PDF or import into a Word/LaTeX template if required by submission guidelines.

## 2. Presentation Summary
- File: `deliverables/presentation_outline.md` (key talking points captured during the defence).
- Figures available for inclusion in supporting material:
  - `results/figures/training_curve.png`
  - `results/figures/comparison_hard.png`
  - `results/figures/trajectory_hard.png`
- Metrics source: `results/baselines.csv` (success rate, path length, steps, reward, nodes expanded).

## 3. Supporting Scripts & Data
- Code repositories already contain:
  - Training script: `train.py`
  - Evaluation scripts: `evaluate_planners.py`, `evaluate_rl.py`
  - Visualisation scripts: `visualize_results.py`, `plot_training_curve.py`, `plot_trajectory.py`
- Metrics: `results/baselines.csv`
- Trained model checkpoints: `models/ppo_linear_navigator.zip`, `models/best_model.zip`

## 4. Reproducibility Commands
```
pip install -r requirements.txt
python train.py --timesteps 500000 --random-map
python evaluate_planners.py --episodes 100 --resolution 12 --difficulty hard --seed 123 --csv results/baselines.csv --tag astar
python evaluate_rl.py --model models/ppo_linear_navigator.zip --episodes 100 --seed 123 --difficulty hard --csv results/baselines.csv --tag ppo
python visualize_results.py --csv results/baselines.csv --difficulty hard
python plot_training_curve.py --logdir logs --tag rollout/ep_rew_mean
python plot_trajectory.py --model models/ppo_linear_navigator.zip --seed 123 --difficulty hard
```

## 5. Optional Future Work Notes
- Implement RRT baseline following `planners/astar.py` template.
- Introduce curriculum learning or dynamic obstacles for expanded evaluation.
- Investigate sim-to-real transfer.

**Use this list to ensure all documentation, scripts, and figures are available before submission.**
