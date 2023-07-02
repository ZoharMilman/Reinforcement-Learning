import numpy as np
import itertools
import random
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, job_costs, jobs, prob):
        self.job_costs = np.array(job_costs)
        self.jobs = np.array(jobs)
        self.prob = prob
        self.states = self.get_state_space()
        self.costs = np.zeros(len(self.states))
        self.costs[-1] = 0
        for i in range(len(self.states) - 1):
            self.costs[i] = self.get_state_cost(self.states[i].astype(int))

    def get_state_space(self):
        jobs = range(1, len(self.jobs) + 1)
        states = []
        for L in range(len(jobs) + 1):
            for subset in itertools.combinations(jobs, L):
                states.append(np.array(subset))
        states.reverse()
        return states

    def get_state_cost(self, state):
        return np.sum(self.job_costs[state - 1])

    def get_state_reward(self, state):
        return 1 / self.get_state_cost(state)

    def get_next_state(self, origin_state, action):
        if action in origin_state:
            return origin_state[origin_state != action]
        else:
            raise ValueError('Illegal action: action not valid for the given state')

    def get_state_index(self, state):
        for i in range(len(self.states)):
            if np.array_equal(self.states[i], state):
                return i

    def get_tran_prob_mat(self, pi):
        P = np.zeros([len(self.states), len(self.states)])
        for i in range(len(self.states) - 1):
            a = int(pi[i])
            next_state = self.get_next_state(self.states[i], a)
            P[i][i] = 1 - self.prob[a - 1]
            P[i][self.get_state_index(next_state)] = self.prob[a - 1]
        P[-1][-1] = 1
        return P

    def get_value_function_with_policy(self, pi):
        value = np.zeros(len(self.states))
        prob_matrix = self.get_tran_prob_mat(pi)
        mat = np.identity(len(pi)) - prob_matrix[:-1, :-1]
        inv_mat = np.linalg.inv(mat)
        value[:-1] = np.dot(inv_mat, self.costs[:-1])
        return value

    def policy_iteration(self, initial_value, print_flag=0):
        if print_flag == 1:
            initial_state_val = []
            step_count = 0
        next_value = initial_value
        pi = np.zeros(len(self.states) - 1)
        delta = 1
        while (delta > 0.0001):
            current_value = next_value
            if print_flag == 1:
                initial_state_val.append(current_value[0])
            for i in range(len(self.states) - 1):
                state = self.states[i]
                action_value = np.inf
                cost = self.costs[self.get_state_index(state)]
                for a in state:
                    next_state = self.get_next_state(state, a)
                    tran_prob = self.prob[a - 1]
                    new_action_value = cost + tran_prob * current_value[self.get_state_index(next_state)] + (
                            1 - tran_prob) * current_value[self.get_state_index(state)]
                    if new_action_value < action_value:
                        action_value = new_action_value
                        pi[i] = a
            next_value = self.get_value_function_with_policy(pi)
            delta = max(current_value - next_value)
            if print_flag == 1:
                step_count = step_count + 1

        if print_flag == 1:
            print('converged in ' + str(step_count))
            initial_state_val.append(initial_state_val[-1])
            plt.plot(range(1, step_count + 2), initial_state_val)
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            plt.title('Value of initial state over policy iteration')
            plt.show()

        return pi, next_value

    def simulator(self, state, action):
        tran_prob = self.prob[action - 1]
        state_cost = self.costs[self.get_state_index(state)]

        outcomes = [True, False]
        probabilities = [tran_prob, 1 - tran_prob]
        job_success = random.choices(outcomes, probabilities)[0]

        if job_success:
            next_state = self.get_next_state(state, action)
        else:
            next_state = state

        return state_cost, next_state

    # Only use self.states and functions
    def TD_zero(self, pi, alpha_func, gamma=1, episodes=1000, return_err=False, actual_value=0):
        if return_err:
            max_error = np.zeros(episodes)
            state0_error = np.zeros(episodes)

        value = np.zeros(len(self.states))
        visits = np.zeros(len(self.states))
        for i in range(episodes):
            state = random.choice(self.states[:-1])  # Randomly select a non-terminal state
            while len(state) > 0:
                visits[self.get_state_index(state)] += 1
                action = pi[self.get_state_index(state)]  # Use the policy to determine the action
                cost, next_state = self.simulator(state, action)
                alpha = alpha_func(state, visits)
                value[self.get_state_index(state)] += alpha * (
                        cost + gamma * value[self.get_state_index(next_state)] - value[self.get_state_index(state)])

                state = next_state

            if return_err:
                max_error[i] = max(abs(value - actual_value))
                state0_error[i] = abs(value[0] - actual_value[0])

            if i % 1000 == 0:
                print('Iteration ', i, ' Of ', episodes)

        if return_err:
            return value, max_error, state0_error

        return value

    def TD_lambda(self, pi, alpha_func, lambda_val, gamma=1, episodes=100, return_err=False, actual_value=0):
        if return_err:
            max_error = np.zeros(episodes)
            state0_error = np.zeros(episodes)

        value = np.zeros(len(self.states))
        visits = np.zeros(len(self.states))
        e = np.zeros(len(self.states))
        for i in range(episodes):
            state = random.choice(self.states[:-1])  # Randomly select a non-terminal state
            while len(state) > 0:
                visits[self.get_state_index(state)] += 1
                # Observe next state
                action = pi[self.get_state_index(state)]
                cost, next_state = self.simulator(state, action)

                # Get e and the current delta
                delta = cost + gamma * value[self.get_state_index(next_state)] - value[self.get_state_index(state)]
                e *= lambda_val * gamma
                e[self.get_state_index(state)] += 1
                alpha = alpha_func(state, visits)

                value += alpha * delta * e

                state = next_state

            if return_err:
                max_error[i] = max(abs(value - actual_value))
                state0_error[i] = abs(value[0] - actual_value[0])

            if i % 1000 == 0:
                print('Iteration ', i, ' Of ', episodes)
        if return_err:
            return value, max_error, state0_error

        return value

    # This is a helper function for the Q-learning algorithm which calculates the epsilon greedy policy

    def epsilon_greedy(self, epsilon, state, Q):
        # Get all optimal and non-optimal actions
        state_index = self.get_state_index(state)
        action_space = state
        subQ = Q[state_index, action_space - 1]
        subQ_a_hat_value = np.min(subQ)
        subQ_a_hat_indices = np.where(subQ == subQ_a_hat_value)
        a_hat = action_space[subQ_a_hat_indices]
        non_optimal_actions = np.delete(action_space, np.where(np.isin(action_space, a_hat)))

        # Special case - all actions are optimal
        if a_hat.size == action_space.size:
            return np.random.choice(a_hat)

        # This has a 1-epsilon probability of occuring
        if random.random() > epsilon:
            return np.random.choice(a_hat)
        else:
            return np.random.choice(non_optimal_actions)

    def get_pi_from_Q(self, Q):
        pi = np.zeros(len(self.states) - 1)
        for state in self.states[:-1]:
            state_index = self.get_state_index(state)
            action_space = state
            subQ = Q[state_index, action_space - 1]
            subQ_a_hat_value = np.min(subQ)
            subQ_a_hat_indices = np.where(subQ == subQ_a_hat_value)
            a_hat = action_space[subQ_a_hat_indices]
            pi[self.get_state_index(state)] = np.random.choice(a_hat)

        return pi

    def sarsa_q_learning(self, alpha_func, epsilon, gamma, optimal_value=0, episodes=1000, return_err=False,
                         return_err_every=1):
        if return_err:
            pi_Q = [np.random.choice(self.states[i]) for i in range(len(self.states) - 1)]
            max_error = np.zeros(episodes)
            state0_error = np.zeros(episodes)

        Q = np.zeros((len(self.states) - 1, len(self.jobs)))
        visits = np.zeros(len(self.states))

        for i in range(episodes):
            state = random.choice(self.states[:-1])  # Randomly select a non-terminal state
            action = np.random.choice(state)  # Use the policy to determine the initial action

            while len(state) > 0:
                next_action = action
                visits[self.get_state_index(state)] += 1
                cost, next_state = self.simulator(state, action)
                alpha = alpha_func(state, visits)

                state_index = self.get_state_index(state)
                next_state_index = self.get_state_index(next_state)

                if len(next_state) > 0:
                    next_action = self.epsilon_greedy(epsilon, next_state, Q)
                    # final_pi[next_state_index] = next_action
                    if return_err:
                        pi_Q[next_state_index] = self.epsilon_greedy(0, next_state, Q)

                    Q[state_index, action - 1] += alpha * (
                            cost + gamma * Q[next_state_index, next_action - 1] - Q[state_index, action - 1])

                else:
                    Q[state_index, action - 1] += alpha * (cost - Q[state_index, action - 1])

                state = next_state
                action = next_action

            if return_err:
                pi = self.get_pi_from_Q(Q)
                V_Q = self.get_value_function_with_policy(pi)
                max_error[int(i / return_err_every)] = max(abs(optimal_value - V_Q))
                state0_error[int(i / return_err_every)] = abs(optimal_value[0] - np.min(Q[0]))

            if i % 100 == 0:
                print('Iteration:', i)

        if return_err:
            return pi, max_error, state0_error

        return pi
