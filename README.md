# Deep Learning Soccer

## Model Architecture

- Two-Tower Model
- Possession Model: Average length of a possession

## Model Validation

### Calibration

Features for bins:

- period (1,2)
- half (defensive, offensive)
- manpower
- score differential

### Action Values

- free kick should be high, should increase close to goal
- throw-ins? Tarrak gets high values
- penalty kicks should be very high, higher than free kick
- goal kicks should be small value
- corner shots pretty high
- goal kick < throw-in < corner shot < free kick < penalty kick

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
    - tim Shwartz's handbook?





