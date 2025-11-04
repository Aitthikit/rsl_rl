# RSL RL

A fast and simple implementation of RL algorithms, designed to run fully on GPU.
This code is an evolution of `rl-pytorch` provided with NVIDIA's Isaac Gym.

Environment repositories using the framework:

* **`Isaac Lab`** (built on top of NVIDIA Isaac Sim): https://github.com/isaac-sim/IsaacLab (Version 4.5.0)

The my-fix branch supports DPPO algorithm, Student-Teacher Distillation with Perception encoder for Depth camera data.

**Maintainer**: Aitthikit Kitcharoennon <br/>
**Contact**: memekhos001@gmail.com <br/>

> **Note:** The DPPO algorithm are maintrained from DPPO in `algorithms` branch. 

## DPPO algorithm

Distribution Proximal Policy Optimization algorithm (DPPO) is 1 of the Distribution Reinforcement Learning algorithm type.
* Distribution Reinforcement Learning it is the concept of changing from estimating the expected return, which is just a single value indicating how well a policy performs, to estimating the distribution of returns instead, allowing for the analysis of uncertainty and risk.

![Distribution Reinforcement Learning](https://wikidocs.net/images/page/169321/Fig_00.png)

* PPO vs DPPO the difference between 2 algorithm is the normal PPO will use some function like mean or log to compute the value from critic but DPPO will use Risk metric to quantitatively measure risk (Risk Quantification) by applying risk metrics to value distributions. The ones commonly used in robotics are distortion risk.

* Example of risk metric that i use
  * Wang metric

$$
g_{\beta}^{\text{wang}}(\tau) = \Phi\big(\Phi^{-1}(\tau) + \beta\big)
$$



## Student-Teacher Distillation with Perception encoder for Depth camera data

## Citing

**We are working on writing a white paper for this library.** Until then, please cite the following work
if you use this library for your research:

```text
@InProceedings{rudin2022learning,
  title = 	 {Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning},
  author =       {Rudin, Nikita and Hoeller, David and Reist, Philipp and Hutter, Marco},
  booktitle = 	 {Proceedings of the 5th Conference on Robot Learning},
  pages = 	 {91--100},
  year = 	 {2022},
  volume = 	 {164},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v164/rudin22a.html},
}
```

If you use the library with curiosity-driven exploration (random network distillation), please cite:

```text
@InProceedings{schwarke2023curiosity,
  title = 	 {Curiosity-Driven Learning of Joint Locomotion and Manipulation Tasks},
  author =       {Schwarke, Clemens and Klemm, Victor and Boon, Matthijs van der and Bjelonic, Marko and Hutter, Marco},
  booktitle = 	 {Proceedings of The 7th Conference on Robot Learning},
  pages = 	 {2594--2610},
  year = 	 {2023},
  volume = 	 {229},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v229/schwarke23a.html},
}
```

If you use the library with symmetry augmentation, please cite:

```text
@InProceedings{mittal2024symmetry,
  author={Mittal, Mayank and Rudin, Nikita and Klemm, Victor and Allshire, Arthur and Hutter, Marco},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  title={Symmetry Considerations for Learning Task Symmetric Robot Policies},
  year={2024},
  pages={7433-7439},
  doi={10.1109/ICRA57147.2024.10611493}
}
```