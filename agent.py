import numpy as np
from act import PlayerAct
import math


class Agent():
    def __init__(self, act_num, state_num):
        self.act_num = act_num
        self.state_num = state_num
        self.actor = [[[1.0 for i in range(act_num)] for j in range(
            state_num[1])] for k in range(state_num[0])]

        self.value = [[0 for i in range(state_num[1])]
                      for j in range(state_num[0])]
        self.td = 0

        self.ALPHA = 0.8
        self.BETA = 0.8
        self.discount_rate = 0.8
        self.reversed_c = 0.7
        self.td_rate = 0.2  # trust degree rate

        self.goal_steps = []
        self.trust_deg = 0  # ひとまず状態の関数にはしない
        self.reward_hist = []

    def softmax(self, v):
        v = np.array(v[::])
        # print(np.exp(v * self.reversed_c), sum(np.exp(v * self.reversed_c)))
        return np.exp(v * self.reversed_c) / sum(np.exp(v * self.reversed_c))

    def entropy(self, v):
        v = np.array(v[::])
        return -1 * sum(v * np.log2(v))

    def act_constraints(self, p, state, block):
        p = p[::]
        # 行動束縛
        up = block[state[0] - 1][state[1]]
        right = block[state[0]][state[1] + 1]
        down = block[state[0] + 1][state[1]]
        left = block[state[0]][state[1] - 1]
        if up == 1:
            p[0] = -10000000000.0
        if right == 1:
            p[1] = -10000000000.0
        if down == 1:
            p[2] = -10000000000.0
        if left == 1:
            p[3] = -10000000000.0
        return p

    def pub(self, status, block, advice=None):
        __state = status.next_state[::]
        __actor = self.actor[__state[0]][__state[1]][::]

        if advice is None:
            __p = np.array(__actor)
        else:
            __p = np.array(__actor) + np.array(advice) * \
                self.entropy(self.softmax(__actor))
        __p = self.act_constraints(__p, __state, block)

        policy = self.softmax(__p)
        act = np.random.choice(self.act_num, 1, p=policy)[0]
        return act

    def renew_critic(self, status):
        state = status.state[::]
        next_state = status.next_state[::]
        reward = status.reward

        self.td = reward + self.discount_rate * \
            self.value[next_state[0]][next_state[1]] - \
            self.value[state[0]][state[1]]
        self.value[state[0]][state[1]] += self.ALPHA * self.td
        if self.value[state[0]][state[1]] > 100:
            self.value[state[0]][state[1]] = 100

    def renew_actor(self, status):
        state = status.state[::]
        act = status.act
        self.actor[state[0]][state[1]][act] += self.BETA * self.td
        if self.actor[state[0]][state[1]][act] > 100:
            self.actor[state[0]][state[1]][act] = 100

    def train(self, status):
        self.renew_critic(status)
        self.renew_actor(status)

    def train_liar(self, status, reward):
        # print("reward: ", reward, "trust_degree:", trust_degree)
        # self.renew_critic_liar(status, reward, trust_degree)
        self.renew_critic(status)
        self.renew_actor(status)

    def renew_critic_liar(self, status, reward, trust_degree):
        state = status.state
        next_state = status.next_state

        self.td = reward + trust_degree + self.discount_rate * \
            self.value[next_state[0]][next_state[1]] - \
            self.value[state[0]][state[1]]
        self.value[state[0]][state[1]] += self.ALPHA * self.td
        if self.value[state[0]][state[1]] > 100:
            self.value[state[0]][state[1]] = 100

    def cal_trust_degree(self, liar_act_vec, player_act_vec):
        beta = 0.09
        # v1 = np.array(liar_act_vec)
        # v2 = np.array(player_act_vec)
        # cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if liar_act_vec == player_act_vec:
            cos = 1.0
        else:
            cos = -1.0
        # print("cos", cos)
        self.trust_deg += beta * cos
        # print("trust", self.trust_deg)
        return self.trust_deg
