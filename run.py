#!/usr/bin/env python

import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout


class Field:
    N = 3

    _win_combination = np.array([1, 1, 1])
    _lose_combination = np.array([-1, -1, -1])

    HUMAN_FIELDS = {
        -1: '0',
        1: 'X',
        0: ' '
    }

    def __init__(self, combination):
        assert isinstance(combination, (list, tuple)) and len(combination) == self.N*self.N
        self._field = np.array(combination).reshape(self.N, self.N)

    def reward(self):
        for i in range(self.N):
            if np.array_equal(self._field[:, i], self._lose_combination) or np.array_equal(self._field[i, :], self._lose_combination):
                return 0

            if np.array_equal(self._field[:, i], self._win_combination) or np.array_equal(self._field[i, :], self._win_combination):
                return 1

            if np.array_equal(np.diag(self._field), self._lose_combination):
                return 0

            if np.array_equal(np.diag(self._field), self._win_combination):
                return 1

            _reversed = np.flip(self._field, 0)
            if np.array_equal(_reversed, self._lose_combination):
                return 0

            if np.array_equal(_reversed, self._win_combination):
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


if __name__ == '__main__':
    NUM_SAMPLES_TO_LEARN = 100

    print('training network on {} random samples'.format(NUM_SAMPLES_TO_LEARN))

    model = Sequential()
    model.add(Dense(12, input_dim=9, activation='relu'))
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
        sample = [random.randint(-1, 1) for _ in range(9)]
        reward = Field(sample).reward()

        samples.append(sample)
        rewards.append(reward)

    model.fit(samples, rewards, batch_size=32)

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
                    possible_field = f.apply_action(action, player)
                    reward = model.predict(possible_field)
                    print('Action: {} Reward: {}'.format(possible_field, reward))

                    if (player == 1 and reward > best_reward) or (player == 2 and reward < best_reward):
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

            print(f)

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