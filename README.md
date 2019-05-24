# Deep Learning Soccer

## Model Architecture and Training

- Two-Tower Model
- Possession Model: Average length of a possession
- It would be nice to focus on *sparsity.* Tarrak didn't see how to deal with the sparsity problem, other than by reward shaping and upsampling. We could argue that soccer is the major sport with the most reward sparsity and our method deals with it. 
  - Are there special training methods we use to deal with it?
  - In Sloan I proposed using conditional value for impact: instead of using $Q_{home}$, use $Q_{home}/(Q_{home}+Q_{away})$. If this works better than plain Q values it would be another contribution for dealing with sparisity.

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
- If we can output action values for the game that Tarrak analyzed, he is willing to review them to see if they are plausible.
- The eye test: ranges we expect:
  - free kick should be high, should increase close to goal
  - throw-ins? Tarrak gets high values
  - penalty kicks should be very high, higher than free kick
  - goal kicks should be small value
  - corner shots pretty high
  - goal kick < throw-in < corner shot < free kick < shots < penalty kick

### Ablation/Lesion/Sensitivity Analysis

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




