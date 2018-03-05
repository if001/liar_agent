from agent import Agent
from env import Env
from act import LiarAct
from act import PlayerAct
from time import sleep
import pylab as plt


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
    step = 0
    # while(True):
    for _ in range(10000):
        status.act = liar.pub(status, block)
        status.state, status.next_state, status.reward = env.pub(status)
        print(len(liar.goal_steps))
        env.debug()
        status.debug()
        liar.train(status)
        step += 1
        if env.is_goal():
            del status
            del env
            status = Status(state=start, act=0, reward=0, next_state=start)
            env = Env(status.state, status.next_state, block, start, goal)
            liar.goal_steps.append(step)
            step = 0

    with open("./pre_learning.txt", "o") as f:
        f.writelines()

    # i = int(len(liar.goal_steps) / 3)
    # t = range(len(liar.goal_steps[:i]))
    # plt.plot(t, liar.goal_steps[:i], label="liar", color="b")

    shortest_step = liar.goal_steps[-1]

    # playerの学習開始
    status = Status(state=start, act=0, reward=0, next_state=start)
    env = Env(status.state, status.next_state, block, start, goal)
    player = Agent(act_num=4, state_num=state_num)
    step = 0
    liar_reward = 0
    for i in range(10000):
        liar_act = liar.pub(status, block)
        liar_act_vec = LiarAct.toVec(liar_act)

        status.act = player.pub(status, block, liar_act_vec)
        status.state, status.next_state, status.reward = env.pub(status)
        print(len(player.goal_steps))
        env.debug()
        status.debug()
        print("liar", liar_act, "  player ", status.act)
        player.train(status)
        player_act_vec = PlayerAct.toVec(status.act)
        trust_degree = liar.cal_trust_degree(liar_act_vec, player_act_vec)

        liar_reward = 0
        if status.reward > 0:
            liar_reward = step - shortest_step
        liar.train_liar(status, liar_reward, trust_degree)
        step += 1
        if env.is_goal():
            del status
            del env
            status = Status(state=start, act=0, reward=0, next_state=start)
            env = Env(status.state, status.next_state, block, start, goal)
            player.goal_steps.append(step)

            step = 0
        # key = input()
    with open("./player.txt", "o") as f:
        f.writelines()

    # i = int(len(liar.goal_steps) / 3)
    # t = range(len(player.goal_steps[:i]))
    # plt.plot(t, player.goal_steps[:i], label="player", color="r")
    # plt.show()


if __name__ == "__main__":
    main()
