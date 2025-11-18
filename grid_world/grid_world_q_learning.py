import numpy as np

# global variables
BOARD_ROWS = 5
BOARD_COLS = 5
WIN_STATE = (5, 5)
# LOSE_STATE, lose state will be determined by the accumulated reward
# Agent loses if rewards is greater than 10
START = (2, 1)
# BLOCKED STATES, AGENT CAN NOT MOVE TO THESE SPACES
BLOCKED_STATES = [(3, 3), (3, 4), (4, 3), (3, 5)]
DETERMINISTIC = True


class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        # set initial state to start
        self.state = state
        # set flag isEnd, marks the completion of the episode
        self.isEnd = False
        # set initial reward to initial state
        self.rewards = 0
        # set Deterministic, board controls its actions
        self.deterministic = DETERMINISTIC

    def give_rewards(self):
        if self.state == WIN_STATE:
            return 10
        return 0

    def next_position(self, action):
        r, c = self.state
        if self.deterministic:
            if action == "up":
                candidate = (r - 1, c)
            elif action == "down":
                candidate = (r + 1, c)
            elif action == "right":
                candidate = (r, c + 1)
            elif action == "left":
                candidate = (r, c - 1)
            else:
                candidate = (r, c)
        else:
            # For non-deterministic
            candidate = (r, c)

            # check boundaries
        if not (1 <= candidate[0] <= BOARD_ROWS and 1 <= candidate[1] <= BOARD_COLS):
            return self.state  # invalid move -> remain in same state

            # check blocked states
        if candidate in BLOCKED_STATES:
            return self.state

        return candidate

    def move_to_next_state(self, action):
        # actions are up, down , right and left
        new_pos = self.next_position(action)
        self.state = new_pos
        # update reward if needed (accumulate or set)
        self.rewards = self.give_rewards()
        # update end flag
        self.setEnd()
        return self.state

    def setEnd(self):
        if self.state == WIN_STATE:
            self.isEnd = True
        else:
            self.isEnd = False


class Agent:

    def __init__(self):
        # initialise the state object of the agent
        self.state = State()
        # initialise all actions
        self.actions = ["up", "down", "right", "left"]
        # set exploration rate
        self.exploration_rate = 0.7
        # set learning_rate
        self.learning_rate = 0.5
        # list of all states recorded by agent
        self.states = []
        # initialise the state values
        self.state_values = {}

        # set initial state values to 0
        for r in range(1, BOARD_ROWS + 1):
            for c in range(1, BOARD_COLS + 1):
                self.state_values[(r, c)] = 0.0

    def choose_action(self):
        if np.random.uniform(0, 1) <= self.exploration_rate:
            # pick only valid actions
            valid_actions = [a for a in self.actions if self.state.next_position(a) != self.state.state]
            if valid_actions:
                return np.random.choice(valid_actions)
            else:
                return np.random.choice(self.actions)
        else:
            # pick action with max state value among valid moves
            maximum_next_reward = float("-inf")
            best_actions = []
            for a in self.actions:
                next_pos = self.state.next_position(a)
                next_reward = self.state_values[next_pos]
                if next_reward > maximum_next_reward:
                    best_actions = [a]
                    maximum_next_reward = next_reward
                elif next_reward == maximum_next_reward:
                    best_actions.append(a)
            return np.random.choice(best_actions)

    def take_action(self, action):
        """Apply the action to the current state (mutates the same State object)."""
        return self.state.move_to_next_state(action=action)

    def play(self, rounds=10):
        # rounds - the number of episodes
        for episode in range(rounds):
            # reset environment for new episode
            self.reset_to_start()
            self.states = []
            step = 0
            while True:
                # if current state is terminal, do the backpropagation / update and break
                if self.state.isEnd:
                    reward = self.state.give_rewards()
                    # explicitly assign end state's value
                    self.state_values[self.state.state] = reward
                    # Backpropagate value to previous visited states (simple backup)
                    for s in reversed(self.states):
                        # V(s) <- V(s) + alpha * (V(next) - V(s))
                        next_val = reward
                        old_val = self.state_values[s]
                        updated = old_val + self.learning_rate * (next_val - old_val)
                        self.state_values[s] = round(updated, 3)
                        # for multi-step episodes, next_val becomes updated V(s) for earlier steps
                        reward = self.state_values[s]
                    print(f"Episode {episode + 1} ended in {self.state.state} reward={self.state.rewards}.")
                    break

                # choose action and step
                action = self.choose_action()
                # record the current state's next-state (the state we will end up in)
                current = self.state.state
                self.states.append(current)
                nxt = self.take_action(action)
                print(f"... pos={self.state.state} action={action} -> next={nxt}")

                step += 1
                # safety: prevent infinite loops
                if step > 100:
                    print("Episode aborted: too many steps")
                    break

    def reset_to_start(self):
        # reset environment to start position
        self.state = State(state=START)

    def show_values(self):
        for r in range(1, BOARD_ROWS + 1):
            print('----------------------------------')
            out = '| '
            for c in range(1, BOARD_COLS + 1):
                out += str(self.state_values[(r, c)]).ljust(6) + ' | '
            print(out)
        print('----------------------------------')


if __name__ == '__main__':
    ag = Agent()
    ag.play(50)
    print(ag.show_values())

