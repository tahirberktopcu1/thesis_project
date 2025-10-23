# Step 1 - Problem Definition and Literature Context

## 1.1 Problem Statement
Autonomous mobile robots are expected to reach designated targets while avoiding collisions, minimising travel time or distance, and adapting to changes in their operating environment. Factory floors, warehouses, hospitals, and service robots all demand robust navigation despite clutter, narrow corridors, and sensor noise. Traditional planning pipelines decompose the task into global path generation (e.g., A* on a prior map) and local obstacle avoidance (e.g., potential fields). Although such systems are deployed widely, they depend heavily on accurate maps, carefully tuned heuristics, and deterministic behaviour-leading to brittle performance when confronted with unexpected obstacles or partially known layouts. Reinforcement Learning (RL) offers an alternative: instead of hard-coding heuristics, the agent learns a policy from interaction, optimising long-term reward that combines goal achievement and safety. The primary research question in this work is: **Can a policy trained with Proximal Policy Optimization (PPO) deliver competitive or superior navigation performance compared to classical planners when both operate in identical procedurally generated 2D maps with rectangular obstacles?**

## 1.2 Classical Path Planning Background
Classical planning algorithms assume access to a map or can at least incrementally build one while exploring. Their key characteristics are:

### Graph-based planners
- **A\*** (Hart et al., 1968) and its derivatives treat the environment as a grid or graph, using heuristics (e.g., Euclidean distance) to guide search toward the goal.  
- **D\*** and **D\* Lite** [Koenig & Likhachev, 2002] extend A\* to dynamic environments by repairing paths when costs change, enabling replanning for robot navigation.  
- **Strengths:** Guarantee of optimal path given admissible heuristics; simple to implement; mature theoretical foundations.  
- **Weaknesses:** Require discretisation; computational cost grows with map size; frequent replanning is needed when new obstacles appear.

### Sampling-based planners
- **RRT / RRT\*** [LaValle, 2006] explore continuous spaces by incrementally building a tree of feasible states. RRT\* asymptotically converges to optimal paths via rewiring.  
- **Probabilistic Roadmaps (PRM)** build a roadmap by sampling random collision-free points and connecting them with local planners.  
- **Strengths:** Scale to high-dimensional configuration spaces, do not require explicit grid discretisation.  
- **Weaknesses:** Paths often contain unnecessary detours; performance is sensitive to sampling density and obstacle geometry.

### Hybrid and hierarchical systems
- Many deployed mobile robots combine a global planner (A\*, D\*) with a local controller (Dynamic Window Approach, potential fields) to react to short-term obstacles.  
- Hybrid approaches deliver predictable behaviour but require careful integration and manual tuning.

## 1.3 Reinforcement Learning for Robot Navigation
Recent years have seen significant interest in using RL to learn navigation policies end-to-end:

### Value-based methods
- **Deep Q-Networks (DQN)** [Mnih et al., 2015] approximate action-value functions for discrete actions. Extensions (Double DQN, Dueling DQN) stabilise learning.  
- Applications to navigation discretise headings or movement primitives; reward shaping (distance reduction, collision penalties) is crucial. However, DQN struggles with continuous control and may require extensive replay buffers.

### Policy gradient methods
- **PPO** [Schulman et al., 2017] uses clipped surrogate losses to maintain stable updates, supporting both discrete (turn/forward) and continuous actions.  
- Studies such as Zhu et al. (2017) and Long et al. (2019) show PPO policies navigating mazes with rich sensory inputs (LiDAR, RGB).  
- PPO naturally accommodates stochastic policies and can integrate sensor noise during training.

### RL in robotics surveys and hybrid approaches
- Surveys (Kober et al., 2013; Chen et al., 2019) highlight RL's ability to learn local navigation behaviours that complement classical planners, especially when combined with curricula, domain randomisation, or sim-to-real techniques.
- Hybrid pipelines often still rely on classical algorithms for global planning while RL handles local decision-making or dynamic obstacle avoidance.

### Sensory modalities and reward shaping
- Range sensors (LiDAR/sonar rays) are common for 2D navigation tasks: they provide compact yet informative observations.  
- Reward functions typically include: progress toward goal, collision penalties, smoothness constraints, and time penalties. Balancing these terms is essential for stable learning.

## 1.4 Research Gap and Motivation
Despite abundant work in both domains, few studies evaluate RL policies and classical planners under identical, procedurally generated conditions with quantitative metrics suitable for side-by-side comparison. Common shortcomings include:
- RL evaluations limited to single maps or small sets of manually designed mazes.
- Classical planner benchmarks that ignore learned policies or compare against heuristics only qualitatively.
- Lack of transparent metrics (success rate, path length, computational effort) recorded jointly for both approaches.

This thesis addresses the gap by:
1. Building a reproducible simulation environment where PPO policies and A\* operate on the same random obstacle layouts and identical sensor models.
2. Logging metrics for both approaches-success/failure, path length, node expansions (A\*), step count and cumulative reward (PPO)-to quantify trade-offs between deterministic planning and learned behaviour.
3. Providing scripts that produce CSV summaries to facilitate statistical analysis and plotting for the thesis report.

## 1.5 Expected Contributions of Step 1
- A clear articulation of the navigation problem, emphasising the need for adaptable and efficient planning in cluttered environments.  
- A literature analysis contrasting classical planners with RL methods, noting strengths, limitations, and representative publications.  
- Identification of a concrete research question and benchmarking framework that integrates both paradigms.  
- Establishment of reference material and citation list for subsequent writing (Methodology, Experiments, Results).

## 1.6 References
1. Mnih, V. et al., "Human-level control through deep reinforcement learning," *Nature*, 2015.  
2. Schulman, J. et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.  
3. Kober, J., Bagnell, J.A., Peters, J., "Reinforcement Learning in Robotics: A Survey," *International Journal of Robotics Research*, 2013.  
4. Chen, Y. et al., "Deep Reinforcement Learning for Motion Planning with Heterogeneous Agents," *International Journal of Robotics Research*, 2019.  
5. LaValle, S.M., *Planning Algorithms*, Cambridge University Press, 2006.  
6. Koenig, S., Likhachev, M., "D* Lite," *Proceedings of AAAI*, 2002.  
7. Hart, P.E., Nilsson, N.J., Raphael, B., "A Formal Basis for the Heuristic Determination of Minimum Cost Paths," *IEEE Transactions on Systems Science and Cybernetics*, 1968.  
8. Zhu, Y. et al., "Target-driven Visual Navigation in Indoor Scenes using Deep RL," *ICRA*, 2017.  
9. Long, P. et al., "Towards Optimally Decentralized Multi-Robot Collision Avoidance via Deep RL," *ICRA*, 2018.  
