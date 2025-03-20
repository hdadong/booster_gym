import isaacgym
from utils.runner_mbrl import Runner

#python play_mbrl.py --task=T1_MBRL --checkpoint=-1
if __name__ == "__main__":
    runner = Runner(test=False)
    runner.play()
