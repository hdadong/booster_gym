import isaacgym
from utils.runner_mbrl_baseline import Runner

# python play_ppobaseline_train_policy.py --task=T1_MBRL_baseline
if __name__ == "__main__":
    runner = Runner(test=False)
    runner.train()
