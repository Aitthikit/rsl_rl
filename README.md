# RSL RL

A fast and simple implementation of RL algorithms, designed to run fully on GPU.

Environment repositories using the framework:

* **`Isaac Lab`** (built on top of NVIDIA Isaac Sim): https://github.com/isaac-sim/IsaacLab (Version 4.5.0)

The my-fix branch supports DPPO algorithm, Student-Teacher Distillation with Perception encoder for Depth camera data.

**Maintainer**: Aitthikit Kitcharoennon <br/>
**Contact**: memekhos001@gmail.com <br/>

> **Note:** The DPPO algorithm are maintrained from DPPO in `algorithms` branch. 

## Setup

The package can be installed by cloning this repository:

```bash
git clone https://github.com/Aitthikit/rsl_rl.git -b my-fix
cd rsl_rl
pip install -e .
```

## Major Updates in the my-fix 

1. Added DPPO (Distributional Proximal Policy Optimization):
  * Introduces distributional learning capabilities
  * Supports different loss types: energy, MSE, and huber
  * Includes quantile regression functionality

2. Enhanced Architecture:
  * **`quantile_nn.py`** Implementation of quantile neural networks
  * **`quantile_distribution.py`** Support for distributional RL
  * Improved observation encoding capabilities

3. Training Improvements :
  * Additional loss functions
    * Energy-based losses
    * Huber loss
    * Enhanced MSE implementations

4. Perception Encoder :
  * Multiple Encoder Types **`obs_encoder.py`**
    * ObsEncoder (Base MLP Encoder)
    * GRUEncoder
    * ConvEncoder
    * ConvGRUEncoder

5. Enhanced Integration:
  * Better integration with PPO and DPPO algorithms
  * Support for both student and teacher models in distillation
  * Flexible observation selection and processing
  * Automatic initialization based on configuration



## DPPO algorithm

Distribution Proximal Policy Optimization algorithm (DPPO) is 1 of the Distribution Reinforcement Learning algorithm type.
* Distribution Reinforcement Learning it is the concept of changing from estimating the expected return, which is just a single value indicating how well a policy performs, to estimating the distribution of returns instead, allowing for the analysis of uncertainty and risk.

![Distribution Reinforcement Learning](https://wikidocs.net/images/page/169321/Fig_00.png)

* PPO vs DPPO the difference between 2 algorithm is the normal PPO will use mean or log to compute the value from critic but DPPO will use Risk metric to quantitatively measure risk (Risk Quantification) by applying risk metrics to value distributions. The ones commonly used in robotics are distortion risk.

![PPO vs DPPO](../../PPOvsDPPO.png)

* Example of risk metric that i use
  * Wang metric
    * Measure the reflect the robot's personal perspective on risk.

$$
g_{\beta}^{\text{wang}}(\tau) = \Phi\big(\Phi^{-1}(\tau) + \beta\big)
$$



## Student-Teacher Distillation with Perception encoder for Depth camera 

For Student-Teacher Distillation i use base Distillation from rsl_rl in main brance. But update some feature like:
1. Added Encoder model Loss compute for 2 phases training:

The reason why we have to train 2 phases is because if we train RL (reinforcement learning) by directly using hard-to-compute perception (ex. depth camera), it may be hard to convert (require many samples).

![Distillation](../../Distillation.png)




## Example

For a demo configuration of DPPO, please check the [dppo_config.py](config/dppo_config.py) file.

For a demo configuration of Distillation with Perception encoder,please check the [distillation_with_depth_config.py](config/distillation_with_depth_config.py) file.

> **Note:** Do not forget to create cfg in **`isaaclab_rl`** before use algorithm

For observation term order i recommend:
1. If you using encoder observation:

* Set term that you want to encoded at the last term and set command(with Beta range) before encoded term
```python
obs = torch.cat(
    [
        tensor
        for tensor in (
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            self._robot.data.joint_vel,
            self._actions,
            self._commands,  # <--- Set command before last term of observation
            height_data, # <--- Set Encode perception at the last term of observation
        )
        if tensor is not None
    ],
    dim=-1,
)
critic_obs = torch.cat(
    [
        tensor
        for tensor in (
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            self._robot.data.joint_vel,
            self._actions,
            self._commands[:,:3], # <--- Set command before last term of observation
            height_data, # <--- Set Encode perception at the last term of observation
        )
        if tensor is not None
    ],
    dim=-1,
)
observations = {"policy": obs , "critic": critic_obs} # policy for Actor and critic for Critic
```

2. If you didn't use encoder observation:

* Set the command(with Beta range) at last term of observation
```python
obs = torch.cat(
    [
        tensor
        for tensor in (
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            self._robot.data.joint_vel,
            height_data,
            self._actions,
            self._commands,  # <--- Set command at last term of observation
        )
        if tensor is not None
    ],
    dim=-1,
)
critic_obs = torch.cat(
    [
        tensor
        for tensor in (
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            self._robot.data.joint_vel,
            height_data,
            self._actions,
            self._commands[:,:3], # <--- Set command at last term of observation
        )
        if tensor is not None
    ],
    dim=-1,
)
observations = {"policy": obs , "critic": critic_obs} # policy for Actor and critic for Critic
```

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

If you use the library with DPPO, please cite:

```text
@InProceedings{rudin2023learning,
  author={Rudin, Nikita and Allshire, Arthur and Kuppuswamy, Naveen and Hutter, Marco},
  title={Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2023},
  note={arXiv:2309.14246},
  url={https://arxiv.org/abs/2309.14246}
}
```

If you use the library with Distillation with Perception encoder for Depth camera, please cite:

```text
@InProceedings{rudin2022advanced,
  author={Rudin, Nikita and Allshire, Arthur and Peng, Xue Bin and Hutter, Marco},
  title={Advanced Skills through Multiple Adversarial Motion Priors in Reinforcement Learning},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2022},
  note={arXiv:2211.07638},
  url={https://arxiv.org/abs/2211.07638}
}
```
