import isaacgym
from utils.runner_mbrl import Runner

# python play_mbrl_train_policy.py --task=T1_MBRL
if __name__ == "__main__":
    runner = Runner(test=False)
    runner.train()
