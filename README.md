# AI305 Project: A3 Dual-Arm Simulation

This repository contains the A3 simulation project, research notes, and recorded experiments for AI305.

## SmolVLA evaluation results

The +40k fine-tuned checkpoint achieved **9/10 successful episodes (90%)** and a **9.2/10 mean object score** on ten paired development scenes. These results do not yet establish performance on unseen scenes.

[Read the experiment report and view the results](results/smolvla_2x5_2026-11-03/README.md). The report includes all five checkpoints, per-seed scores, evaluation records, representative videos, and the distinction between directly evaluated checkpoints and the teacher baseline reconstructed from historical trajectories.

## Project materials

- [Simulation code and installation guide](a3_dual_arm_sim/README.md)
- [Research and project documentation](docs/)
- [Project brief](midterm.pdf)

Model weights and demonstration datasets are hosted separately. The experiment report identifies the sources and the dependencies required to repeat the evaluation.
