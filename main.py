from agent import Agent
from env import Env
from act import LiarAct
from act import PlayerAct
from time import sleep
import pylab as plt
import sys


class Status():
    def __init__(self, state, act, reward, next_state):
        self.state = state
        self.act = act
        self.reward = reward
        self.next_state = next_state

    def debug(self, d_flag=False):
        if d_flag:
            print("state ", self.state)
            print("act ", self.act)
            print("next_state ", self.next_state)
            print("reward ", self.reward)
            print("--------------------------------")


def save_data(v, fname):
    with open(fname, "a") as f:
        v = ','.join(str(e) for e in v)
        f.writelines(v + "\n")


def main():
    d_flag = False
    if sys.argv[-1] == "-d":
        d_flag = True

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
    pre_liar_status = Status(state=start, act=0, reward=0, next_state=start)
    env = Env(pre_liar_status.state,
              pre_liar_status.next_state, block, start, goal)
    liar = Agent(act_num=4, state_num=state_num)
    step = 0
    while(True):
        pre_liar_status.act = liar.pub(pre_liar_status, block)
        pre_liar_status.state, pre_liar_status.next_state, pre_liar_status.reward = env.pub(
            pre_liar_status)
        env.debug(d_flag)
        pre_liar_status.debug(d_flag)
        liar.train(pre_liar_status)
        step += 1
        liar.reward_hist.append(pre_liar_status.reward)
        if env.is_goal():
            del pre_liar_status
            del env
            pre_liar_status = Status(
                state=start, act=0, reward=0, next_state=start)
            env = Env(pre_liar_status.state,
                      pre_liar_status.next_state, block, start, goal)
            liar.goal_steps.append(step)
            step = 0
        if len(liar.goal_steps) >= 100:
            break

    save_data(liar.goal_steps[::], "./data/pre_learning.txt")
    # save_data(liar.reward_hist, "./data/pre_learning_reward_histry.txt")

    shortest_step = liar.goal_steps[-1]
    # playerの学習開始
    player_status = Status(state=start, act=0, reward=0, next_state=start)
    liar_status = Status(state=start, act=0, reward=0, next_state=start)
    env = Env(player_status.state, player_status.next_state, block, start, goal)
    player = Agent(act_num=4, state_num=state_num)
    step = 0
    liar_reward = 0
    while(True):
        liar_status.act = liar.pub(liar_status, block)
        liar_act_vec = LiarAct.toVec(liar_status.act)

        player_status.act = player.pub(player_status, block, liar_act_vec)
        player_status.state, player_status.next_state, player_status.reward = env.pub(
            player_status)
        liar_status.state = player_status.state
        liar_status.next_state = player_status.next_state
        liar_status.reward = 0
        if player_status.reward > 0:
            liar_status.reward = step - shortest_step

        env.debug(d_flag)
        player_status.debug(d_flag)
        liar_status.debug(d_flag)

        player.train(player_status)
        liar.train(liar_status)

        player.reward_hist.append(player_status.reward)
        step += 1
        if env.is_goal() or step >= 10000:
            del player_status
            del env
            player_status = Status(
                state=start, act=0, reward=0, next_state=start)
            env = Env(player_status.state,
                      player_status.next_state, block, start, goal)
            if step < 10000:
                player.goal_steps.append(step)

            step = 0
        if len(player.goal_steps) >= 100:
            break
        if d_flag:
            key = input()

    save_data(player.goal_steps[::], "./data/player.txt")
    # save_data(player.reward_hist[::], "./data/player_reward_histry.txt")


if __name__ == "__main__":
    main()
