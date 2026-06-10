# blackjack-rl
## About The Project

This is a PyTorch implementation of Group-Relative Policy Optimization (GRPO) and Robust Policy Optimization (RPO) to solve the Blackjack-v1 environment from [Gymnasium](https://gymnasium.farama.org/environments/toy_text/blackjack/).

Demos:
- ![GRPO](demos/grpoblackjack_1.mov)
- ![GRPO](demos/grpoblackjack_2.mov)
- ![RPO](demos/rpoblackjack_1.mov)


### Installation
Note: requires **Python 3.11** 

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Train an agent using a named hyperparameter set from `hyperparameters.yml`:

```bash
cd ALGORITHM_DIR
python ALGORITHM_agent.py HYPERPARAM_SET_NAME --train
```

Watch a trained agent play (loads `runs/<set>.pt`, renders to screen):

```bash
cd ALGORITHM_DIR
python agent.py HYPERPARAM_SET_NAME
```

While training, the script writes:
- `runs/<set>.log`. The timestamped log of new best rewards.
- `runs/<set>.pt`. The best model weights so far.
- `runs/<set>.png`. Curves of mean-reward and epsilon decay.

## Roadmap
Features:
- **GRPO** and **RPO** implementation from scratch.
- **Hyperparameters** configurable per each experiment.
- **Automatic logging**, **best-model checkpointing**, and **live reward/epsilon plots**.

## License
Distributed under the project_license. See `LICENSE.txt` for more information.

## Contact
Rishi Jain - [LinkedIn](https://www.linkedin.com/in/27rjain/) - 27rishijainpersonalemail@gmail.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)


## Acknowledgments

* [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)
* [Robust Policy Optimization in Deep Reinforcement Learning](https://arxiv.org/pdf/2212.07536)

