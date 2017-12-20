#!/usr/bin/env python

import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout


class Field:
    N = 5
    WIN_COMPINATION_LENGTH = 3

    HUMAN_FIELDS = {
        -1: '0',
        1: 'X',
        0: ' '
    }

    def __init__(self, combination):
        assert isinstance(combination, (list, tuple, np.ndarray)) and len(combination) == self.N*self.N
        self._field = np.array(combination).reshape(self.N, self.N)

    def reward(self):
        """
        :return:
        1 - x win
        0 - 0 win
        0.5 - draw or unknown
        """
        for i in range(self.N):
            x = self._field[:, i]
            y = self._field[i, :]

            # Checking rows and columns
            for j in range(self.N - self.WIN_COMPINATION_LENGTH+1):
                for arr in (x, y):
                    s = sum(arr[j:j+self.WIN_COMPINATION_LENGTH])

                    if s == -self.WIN_COMPINATION_LENGTH:
                        return 0

                    if s == self.WIN_COMPINATION_LENGTH:
                        return 1

            # Checking diagonals
            diag1 = np.diagonal(self._field, offset=i)
            diag2 = np.diagonal(self._field, offset=-i)

            _reversed = np.flip(self._field, 0)

            diag3 = np.diagonal(_reversed, offset=i)
            diag4 = np.diagonal(_reversed, offset=-i)

            for j in range(len(diag1) - self.WIN_COMPINATION_LENGTH+1):
                for diag in (diag1, diag2, diag3, diag4):
                    s = sum(diag[j:j+self.WIN_COMPINATION_LENGTH])

                    if s == -self.WIN_COMPINATION_LENGTH:
                        return 0

                    if s == self.WIN_COMPINATION_LENGTH:
                        return 1

        return 0.5

    def get_actions(self):
        """
        :return:
        list of pairs (i,j)
        """
        actions = list()
        for i in range(self.N):
            for j in range(self.N):
                if self._field[i, j] == 0:
                    actions.append((i, j))

        return actions

    def update(self, action, player):
        assert player in (1, 2)
        assert self.is_action_possible(action)

        self._field[action] = 1 if player == 1 else -1

    def is_action_possible(self, action):
        i, j = action
        if i not in range(self.N) or j not in range(self.N):
            return False

        return self._field[action] == 0

    def apply_action(self, action, player):
        """
        :param action: (i,j)
        :return:
        """
        assert player in (1, 2)

        vector = self._field.copy()
        vector[action] = 1 if player == 1 else -1

        return vector.reshape(1, self.N*self.N)

    def hash_with_action(self, action):
        return hash(tuple(self._field.reshape(self.N*self.N))) + hash(action)

    def __str__(self):
        return '\n'.join(
            [
                '|'.join(
                        map(
                            lambda x: self.HUMAN_FIELDS[x],
                            self._field[i, :]
                        )
                    )
                for i in range(self.N)
            ]
        )



class StateRegistry:
    """
    This is Q[s,a] in Q-learning algorithm
    """
    def __init__(self):
        # field -> reward
        self.states = dict()

    def add(self, field, action, reward):
        assert isinstance(field, Field)

        k = field.hash_with_action(action)

        self.states[k] = reward

    def get(self, field, action, player):
        k = field.hash_with_action(action)
        if k in self.states:
            return self.states[k]

        # default
        return Field(field.apply_action(action, player).reshape(Field.N * Field.N)).reward()


if __name__ == '__main__':
    NUM_SAMPLES_TO_LEARN = 20000

    # learning rate
    LR = 0.1
    # discount factor
    DF = 0.1

    print('training network on {} random samples'.format(NUM_SAMPLES_TO_LEARN))

    registry = StateRegistry()

    n = Field.N * Field.N

    for i in range(NUM_SAMPLES_TO_LEARN):
        state = Field([random.randint(-1, 1) for _ in range(n)])
        cur_reward = state.reward()

        # Choose random action
        actions = state.get_actions()

        if len(actions) == 0 or state.reward() in (-1, 1):
            continue

        action = actions[random.randint(0, len(actions)-1)]

        #assert False, type(state.apply_action(action, 1)[0])
        new_state = Field(state.apply_action(action, 1)[0])
        new_state_actions = new_state.get_actions()

        # Q[s',a'] = Q[s',a'] + LF * (r + DF * MAX(Q,s) — Q[s',a'])
        m = 0
        for new_action in new_state_actions:
            q = registry.get(new_state, new_action, 1)
            if q > m:
                m = q

        cur_reward = (1-LR)*cur_reward + LR*(new_state.reward() + DF*m)

        registry.add(state, action, cur_reward)

        state.update(action, 1)

    """
    model = Sequential()
    model.add(Dense(12, input_dim=n, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='rmsprop',
        loss='mse'
    )

    samples = list()
    rewards = list()

    for i in range(NUM_SAMPLES_TO_LEARN):
        sample = [random.randint(-1, 1) for _ in range(n)]
        reward = Field(sample).reward()

        samples.append(sample)
        rewards.append(reward)

    model.fit(samples, rewards, batch_size=32)
    """

    print("Now let's play ...")

    f = Field([0]*(Field.N*Field.N))

    print(f)

    computer_number = random.randint(1, 2)

    while True:
        is_finished = False

        for player in (1, 2):
            actions = f.get_actions()

            if not actions:
                print('Draw')
                is_finished = True
                break

            if player == computer_number:
                print('Player {}:'.format(player))

                best_reward = 0 if player == 1 else 2
                best_action = actions[0]

                for action in actions:
                    #possible_field = f.apply_action(action, player)
                    #reward = model.predict(possible_field)
                    reward = registry.get(f, action, player)

                    #print('Action: {} Reward: {}'.format(possible_field, reward))

                    if (player == 1 and reward > best_reward) or (player == 2 and reward < best_reward) or (reward == best_reward and random.randint(0,2) == 2):
                        best_reward = reward
                        best_action = action

                f.update(best_action, player)
            else:
                print('Your turn: enter two numbers for position: ')
                i, j = None, None

                while i is None:
                    try:
                        i, j = map(int, list(input().replace(' ', '')))
                        if not f.is_action_possible((i, j)):
                            print('This cell is already filled! Again')
                            i, j = None, None
                    except ValueError:
                        print('Please enter 2 number separated by space (position start from zero). Example: 1 1')

                f.update((i, j), player)

            print(f, f.reward())

            cur_reward = f.reward()

            if cur_reward == 1:
                print('Player 1 win')
                is_finished = True
                break

            if cur_reward == 0:
                print('Player 2 win')
                is_finished = True
                break

        if is_finished:
            break

    print('Done')
