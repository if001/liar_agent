from act import PlayerAct


class StateQueue():
    def __init__(self, state, next_state):
        self.state_queue = [state, next_state]

    def push(self, state):
        self.state_queue.pop(0)
        self.state_queue.append(state)

    def get(self):
        return self.state_queue[::]


class Env():
    def __init__(self, init_state, init_next_state, block, start, goal):
        self.stateQueue = StateQueue(init_state, init_next_state)
        self.start = start
        self.goal = goal
        self.block = block

    def pub(self, status):
        __pre_state, __state = self.stateQueue.get()
        __act = status.act

        __next_state = [0, 0]
        if __act == PlayerAct.up.value:
            __next_state = [__state[0] - 1, __state[1]]
        if __act == PlayerAct.right.value:
            __next_state = [__state[0], __state[1] + 1]
        if __act == PlayerAct.down.value:
            __next_state = [__state[0] + 1, __state[1]]
        if __act == PlayerAct.left.value:
            __next_state = [__state[0], __state[1] - 1]

        # # 移動先がブロックなら元いた場所に戻す
        # is_hit_block = False
        # if self.block[__next_state[0]][__next_state[1]] == 1:
        #     __next_state = __state
        #     is_hit_block = True

        self.stateQueue.push(__next_state)
        reward = self.get_reward(__next_state)
        return __state, __next_state, reward

    def get_reward(self, state):
        reward = 0.0
        if state == self.goal:
            reward = 10
        return reward

    def is_goal(self):
        _, state = self.stateQueue.get()
        if state == self.goal:
            return True
        else:
            return False

    def debug(self):
        for x in range(len(self.block)):
            for y in range(len(self.block[0])):
                if self.block[x][y] == 1:
                    print("壁", end='')
                else:
                    _, state = self.stateQueue.get()
                    if state == [x, y]:
                        print("え", end='')
                    else:
                        print("　", end='')
            print()
        print()
