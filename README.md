# Guided Soft Actor-Critic

Reference implementation of Guided SAC, a deep reinforcement learning method for partially observable Markov decision processes.

> Haklidir, M. and Temeltas, H. (2021). Guided Soft Actor Critic: A Guided Deep Reinforcement Learning Approach for Partially Observable Markov Decision Processes. *IEEE Access*, 9, 159672-159683. https://doi.org/10.1109/ACCESS.2021.3131772

Guided SAC trains two actors. A guiding actor sees the full state and is used only during training; a control actor sees partial observations and is the policy deployed at test time. The two actors interact with the environment in alternating steps, both feed a shared critic with full-state inputs, and a distillation network transfers the guiding actor's behaviour to the control actor from the replay buffer. The control actor receives a short observation history (K = 4).

## Contents

```
guided_sac_base.py        the method as a standalone script (agent, networks, buffer, training loop)
notebooks/experiments.ipynb   the MuJoCo POMDP experiments of the paper against the SAC baseline
```

## Requirements

Python 3.10 or later, PyTorch, Gymnasium with MuJoCo. See `requirements.txt`.

## Running

```bash
python guided_sac_base.py --env Hopper-v4 --seed 0
python guided_sac_base.py --env HalfCheetah-v4 --seed 0 --total_steps 1000000
```

The partial observability of each task is defined by the inactive observation dimensions of Table 5 in the paper (`INACTIVE_DIMS` in the script). The notebook runs the same code on the five tasks, compares it with SAC on partial observations and reproduces Table 3 and the learning curves of Figure 5. Its defaults are shorter than the paper's protocol (200K steps, 3 seeds); the full protocol is 1M steps and 10 seeds per task.

## Citation

See `CITATION.cff`.

## License

MIT, see `LICENSE`.
