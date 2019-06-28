# Deep Learning Soccer

## Model Architecture and Training

- Two-Tower Model
- Possession Model: Average length of a possession
- It would be nice to focus on *sparsity.* Tarrak didn't see how to deal with the sparsity problem, other than by reward shaping and upsampling. We could argue that soccer is the major sport with the most reward sparsity and our method deals with it. 
  - Are there special training methods we use to deal with it?
  - In Sloan I proposed using *conditional value* for impact: instead of using $Q_{home}$, use $Q_{home}/(Q_{home}+Q_{away})$. If this works better than plain Q values it would be another contribution for dealing with sparisity.
  - Another idea: *initialize* so that `goal` actions (or successful shots) are mapped to value 1. This helps with interpretability too, because that's what the reader expects.
- The deep mind architecture, where each action is represented as a separate node. Might help with the problem that Q(goal) isn't equal to 1.
- Using "possessing" vs. "defending" team rather than home or away team.
- Puterman's idea: train the model to learn the difference $Q_{home}-Q_{away}$. This is also recommended by Gelman. Puterman proves nice convergence properties. If we also learn $Q_{neither}$, we can recover the Q-values through normalization.
- These ideas can be combined. For example, training on the difference and initialize the model so that $Q_{home}(goal(home))-Q_{away(goal(home))}=1$ and  $Q_{home}(goal(away))-Q_{away(goal(away))}=-1$ makes a lot of sense.
- Btw, we may be able to get a theoretical convergence guarantee by adapting Puterman's ideas to TD-Sarsa.

## Model Validation

### Calibration

Features for bins:

- period (1,2)
- half (defensive, offensive)
- manpower
- score differential
- Sparsity:
  - Does $Q_{neither}$ match the number of 0-0 games in the dataset?
  - Does $Q_{neither}$ always dominate because of reward sparsity? When does it not?

### Action Values

- *We should prominently feature shots.* Why:
  - Modelling shot quality is recognized to be important. We may be able to find baseline models. For example, the caltech guys in the ghosting paper built a shot quality model.
  - We can explain how TD works: first the system learns the value of shots. Then it learns what situations lead to shots. 
  - Could calibrate against a separate shot quality model, even our own separate model perhaps.
- *Passing* is the most frequent action, another place to focus. 
  - See [Luke's paper p.6](http://www.sloansportsconference.com/wp-content/uploads/2019/02/Decomposing-the-Immeasurable-Sport.pdf) . 
  - Also various papers at [MLSA 2018](https://dtai.cs.kuleuven.be/events/MLSA18/schedule.php). 
  - There's even a [pass prediction competition](https://github.com/JanVanHaaren/mlsa18-pass-prediction) .
- If we can *output action values for the game that Tarrak analyzed*, he is willing to review them to see if they are plausible.
- The eye test: ranges we expect:
  - free kick should be high, should increase close to goal
  - throw-ins? Tarrak gets high values
  - penalty kicks should be very high, higher than free kick. About 70% success according to Ican McHale.
    - are these represented as `penalty_obtained`?
  - goal kicks should be small value
  - corner shots pretty high
  - goal kick < throw-in < corner shot < free kick < shots < penalty kick

### Ablation/Lesion/Sensitivity Analysis

Baselines comparisons

- IJCAI 2018 architecture
- van Haaren (Monte Carlo approach)
- fine-tuning for specific league

## Model Application

- Home Team Advantage
- Player Ranking
  - *correlation results with success metric*
- Action-specific Player-ranking
- Team Evaluation
  - total goal ratio as ground truth (see our [DMKD paper](http://rdcu.be/ql8n))
  - round-by-round correlation with ground truth
  - Comparison methods:
    - elo
    - goal differential
    - [tim Schwartz's handbook?](https://www.amazon.com/Handbook-Statistical-Methods-Analyses-Handbooks/dp/1498737366)

## Places To Publish

- ECML journal track?
- DAMI?
- Big Data conference, as in `[10]  Uwe Dick and Ulf Brefeld Learning to Rate Player Positioning in Soccer. *Big data*. 2019. `





