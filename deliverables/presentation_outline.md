# Presentation Outline – Reinforcement Learning-Based Path Planning

Use the following structure when creating the slide deck (PowerPoint/Google Slides).

1. **Title Slide**
   - Project title: “Reinforcement Learning-Based Path Planning for a Mobile Robot”
   - Author name, supervisor(s), institution, date

2. **Motivation & Problem Statement**
   - Bullet: Navigation challenges in cluttered/dynamic environments
   - Bullet: Limitations of classical planners (A*, D*) – handcrafted heuristics, static maps
   - Bullet: Opportunity for adaptive policies learned via RL

3. **Objectives**
   - Design Gym-based navigation environment with configurable obstacles
   - Train PPO agent and benchmark vs. classical planner (A*)
   - Visualise learning behaviour and compare metrics (path, success, nodes expanded)

4. **Literature Snapshot**
   - Classical planning (A*, D*, RRT)
   - Deep RL: DQN vs. PPO; why PPO chosen
   - References: Koenig & Likhachev 2002, Mnih 2015, Schulman 2017, Chen 2019

5. **Environment Design**
   - Diagram/screenshot of arena (if possible)
   - Map size, obstacle generation, sensors (24 rays), discrete actions

6. **Reward & Agent Design**
   - Reward components (progress, safety, oscillation penalty, terminal reward)
   - Action space (forward, turn left/right)

7. **Model Training**
   - PPO hyperparameters (net_arch [128,128], n_steps=4096, batch_size=1024, lr=3e-4, ent_coef=0.01)
   - TensorBoard training curve (`results/figures/training_curve.png`)

8. **Evaluation Protocol**
   - Shared seeds for PPO and A*
   - 100 episodes, difficulty “hard”, metrics recorded
   - Mention A* grid resolution (12 px) and path inflation

9. **Results (Quantitative)**
   - Insert metric table or bar chart (`results/figures/comparison_hard.png`)
   - Highlight: PPO success 94 %, mean path 757 px; A* success 100 %, path 775 px, nodes 583

10. **Qualitative Behaviour**
    - Include trajectory plot (`results/figures/trajectory_hard.png`)
    - Discuss PPO avoiding oscillation / A* determinism

11. **Discussion**
    - Trade-off: PPO adaptability vs. A* determinism
    - Failure modes (PPO in tight corridors) vs. computational cost (A*)

12. **Conclusion & Future Work**
    - Summary of findings
    - Future directions: RRT baseline, curriculum learning, dynamic obstacles, sim-to-real

13. **Q&A / Contact Info**
    - Acknowledge supervisors, provide contact email

Use the thesis report (`thesis_full.md`) and experiment figures from `results/figures/` to populate detailed content on each slide.
