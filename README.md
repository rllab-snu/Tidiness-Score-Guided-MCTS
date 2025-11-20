# Tidiness Score-Guided Monte Carlo Tree Search (TSMCTS)

This repository contains the official PyTorch implementation of our RA-L 2025 paper:

### **Tidiness Score-Guided Monte Carlo Tree Search for Visual Tabletop Rearrangement**  
Hogun Kee, Wooseok Oh, Minjae Kang, Hyemin Ahn, and Songhwai Oh
Robot Learning Lab, Seoul National University

---

## Abstract

We propose **Tidiness Score-Guided MCTS**, a planning framework for multi-object rearrangement tasks where a robot must tidy a cluttered scene without an explicit target arrangement.

The key idea is to use a **tidiness score**—a learned scalar measure of how close the scene is to a tidy state—and integrate it into **Monte Carlo Tree Search (MCTS)** to explore long-horizon rearrangement plans.

Our framework consists of:

1. **Tidiness Score Discriminator**  
   A learned model that evaluates scene tidiness from RGB observations.

2. **Tidying Policy**  
   A learned policy that outputs pick-n-place actions from RGB observations.

3. **Tidiness Score-Guided MCTS**  
   An MCTS planner that expands trajectories using tidiness score increments as reward.

TSMCTS produces **higher success rates**, **more efficient rearrangement paths**, and **interpretable planning traces** compared to baseline planners.

---

## Citation

If you find this repository helpful in your research, please cite:

@article{kee2025tsmcts,
  title = {Tidiness Score-Guided Monte Carlo Tree Search for Visual Tabletop Rearrangement},
  author = {Hogun Kee and Wooseok Oh and Minjae Kang and Hyemin Ahn and Songhwai Oh},
  journal = {IEEE Robotics and Automation Letters},
  year = {2025}
}


