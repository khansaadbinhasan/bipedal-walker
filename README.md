# bipedal-walker
openAI gym's bipedal walker

To solve openAI's bipedal walker, we have to make it walk from starting to end without falling and using motors in the most optimized way possible. We used Deep Deterministic Policy Gradients(DDPG) to solve this problem. Additional details can be found in the report.

This repo contains the following files:
**Report**: latex code for report.
**bipedal.mp4**: Demo of the walker.
**ddpg_agent.py**: DDPG algorithm
**model.py**: actor and critic models for DDPG.
**bipedal_ddpg.py**: Code to run bipedal robot
