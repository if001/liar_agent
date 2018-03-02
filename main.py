from agent import Agent
from env import Env
from act import LiarAct
from time import sleep


class Status():
    def __init__(self, state, act, reward, next_state):
        self.state = state
        self.act = act
        self.reward = reward
        self.next_state = next_state

    def debug(self):
        print("state ", self.state)
        print("act ", self.act)
        print("next_state ", self.next_state)
        print("reward ", self.reward)
        print("--------------------------------")


def main():
    start = [5, 1]
    goal = [1, 8]
    block = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
             [1, 0, 1, 0, 1, 1, 0, 0, 1, 1],
             [1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
             [1, 0, 1, 1, 0, 0, 0, 1, 0, 1],
             [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             ]
    state_num = [len(block), len(block[0])]

    # liarの事前学習
    status = Status(state=start, act=0, reward=0, next_state=start)
    env = Env(status.state, status.next_state, block, start, goal)
    liar = Agent(act_num=4, state_num=state_num)

    while(True):
        status.act = liar.pub(status, block)
        status.state, status.next_state, status.reward = env.pub(status)
        env.debug()
        status.debug()

        liar.train(status)
        if env.is_goal():
            # init 処理
            print("init")
            del status
            del env
            status = Status(state=start, act=0, reward=0, next_state=start)
            env = Env(status.state, status.next_state, block, start, goal)
            sleep(1)
    exit(0)
    # playerの学習開始
    status = Status(state=start, act=0, reward=0, next_state=start)
    env = Env(status.state, status.next_state, block, start, goal)
    player = Agent(act_num=4, state_num=state_num)

    for i in range(10):
        liar_act = liar.pub(status)
        liar_act = LiarAct.toVec(liar_act)

        status.act = player.pub(status, liar_act)
        status.state, status.next_state, status.reward = env.pub(status)
        env.debug()
        status.debug()
        player.train(status)
        if env.is_goal():
            # init 処理
            print("init")


if __name__ == "__main__":
    main()
