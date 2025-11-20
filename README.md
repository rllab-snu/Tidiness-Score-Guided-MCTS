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

---

## Download Data

You can download TTU dataset from the URL below:

[TTU-Dataset](https://github.com/rllab-snu/TTU-Dataset)

TTU dataset contains the files below: 

**train_B.tar.gz**, **train_C.tar.gz**, **train_D.tar.gz**, **train_O.tar.gz**,
**test_SU.tar.gz**, **test_US.tar.gz**, **test_UU.tar.gz**


## Data Structure
  
    TTU_dataset/
      ├── train/
      │     └── {scene_id}/                          # scene_id:         (B1, B2, …, C1, C2, …, D1, D2, …, O1, O2, …)
      │         └── template_{template_num}/         # template_num:     (00001 ~ 00016)
      │             └── traj_{trajectory_num}/       # trajectory_num:   (00000 ~ 00099)
      │                 └── {frame_num}/             # frame_num:        (000 ~ 004)
      │                     ├── rgb_top.png
      │                     ├── rgb_front_top.png
      │                     ├── depth_top.npy
      │                     ├── deoth_front_top.npy
      │                     ├── seg_top.npy
      │                     ├── seg_front_top.npy
      │                     └── obj_info.json
      ├── test-seen_obj-unseen_template/
      │     └── ...
      ├── test-unseen_obj-seen_template/
      │     └── ...
      └── test-unseen_obj-unseen_template/
            └── ...

---

## Tidiness Score Training

To train the tidiness discriminator, run:

```bash
cd mcts/
python classifier_train.py --data-dir <DATA_DIR> --loss mse --label_type linspace
```

## Tidying Policy Training

To train the tidying policy, run:

```bash
cd iql/
python main.py --eval-period 2000 --data-dir <DATA_DIR> --reward-model-path ../mcts/data/classification-best/top_nobg_linspace_mse-best.pth --n-epochs 30 --policy-net resnet --q-net resnet --batch-size 32 --reward classifier
```

## TSMCTS Evaluation

To evaluate the TSMCTS method, run:

```bash
cd mcts/
python mcts.py --iteration-limit 5000  --gui-off  --tree-policy iql-policy --policynet-path ../iql/logs/<IQL_TAG>/<IQL_MODEL_NAME>.pth --threshold-prob 1e-4 --num-objects 5 --blurring 3 --data-dir <DATA_DIR> --num-scenes 10 --seed 4321 --logging

```


---

## Citation

If you find this repository helpful in your research, please cite:

@article{kee2025tsmcts,
  title = {Tidiness Score-Guided Monte Carlo Tree Search for Visual Tabletop Rearrangement},
  author = {Hogun Kee and Wooseok Oh and Minjae Kang and Hyemin Ahn and Songhwai Oh},
  journal = {IEEE Robotics and Automation Letters},
  year = {2025}
}


## License
This project is released under the MIT license, as found in the [LICENSE](LICENSE.txt) file.

