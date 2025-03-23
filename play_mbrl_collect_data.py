import isaacgym
from utils.runner_mbrl import Runner

# python play_mbrl_collect_data.py --task=T1_MBRL
if __name__ == "__main__":
    runner = Runner(test=False)
    runner.play()
