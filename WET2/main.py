import numpy as np
import itertools
import matplotlib.pyplot as plt
import random


def getStateSpace(job_num):
    jobs = range(1, job_num + 1)
    states = []
    for L in range(len(jobs) + 1):
        for subset in itertools.combinations(jobs, L):
            states.append(np.array(subset))
    states.reverse()
    return states


def getStateCost(state, job_costs):
    return np.sum(job_costs[state - 1])


def getStateReward(state, job_costs):
    return 1 / getStateCost(state, job_costs)


def getNextState(origin_state, action):
    # We associate action=1 with taking out job 1 etc. we dont assign an action to the empty state because its
    # meaningless
    if action in origin_state:
        return origin_state[origin_state != action]
    else:
        print('Illegal action')


def getStateIndex(states, state):
    for i in range(len(states)):
        if np.array_equal(states[i], state):
            return i


def getTranProbMat(states, pi, prob):
    P = np.zeros([len(states), len(states)])
    # We add -1 to skip the empty state as an original state
    for i in range(len(states) - 1):
        a = int(pi[i])
        next_state = getNextState(states[i], a)
        # The probability of staying at the given state
        P[i][i] = 1 - prob[a - 1]
        # The probability of moving to the next state
        P[i][getStateIndex(states, next_state)] = prob[a - 1]

    # The probability of staying in the last state is 1
    P[-1][-1] = 1
    return P


job_costs = np.array([1, 4, 6, 2, 9])
jobs = np.array([1, 2, 3, 4, 5])
prob = [0.6, 0.5, 0.3, 0.7, 0.1]
states = getStateSpace(5)

costs = np.zeros(len(states))
costs[-1] = 0
for i in range(len(states) - 1):
    costs[i] = getStateCost(states[i].astype(int), job_costs)


# print(states)
# P = getTranProbMat(states, [3, 3, 3, 2, 3, 2, 1], prob)
# print(P)


# state = getNextState(states[0], 3)
# print(state)

# -------------------------------------------------Section b------------------------------------

def getVFWithPolicy(states, pi, prob, costs):
    value = np.zeros(len(states))
    prob_matrix = getTranProbMat(states, pi, prob)
    # Since V=0 for the empty state, we dont need to know the transition probabilities to it to calculate V of the
    # empty state
    mat = np.identity(len(pi)) - prob_matrix[:-1, :-1]
    inv_mat = np.linalg.inv(mat)
    value[:-1] = np.dot(inv_mat, costs[:-1])
    return value


# -------------------------------------------------Section c------------------------------------

# Get the policy which chooses the job with the maximal cost, again, we dont associate an action to the empty state
pi = np.zeros(len(states)-1).astype(int)

for i in range(len(states)-1):
    state = states[i]
    relevent_jobs = jobs[state-1]
    pi[i] = relevent_jobs[np.argmax(job_costs[state-1])]

print(pi)

value = getVFWithPolicy(states, pi, prob, costs)

plt.plot(range(len(value)), value)  # Plotting the bars
plt.xticks(range(len(value)), states, rotation='vertical')    # Set x-axis ticks, this ticks off some warning
plt.xlabel('State Values')
plt.ylabel('Value')
plt.title('Value Function For Max Cost pi')
plt.show()

print(value)

# -------------------------------------------------Section d------------------------------------

def policyIteration(states, prob, costs, initial_value, print_flag=0):
    if print_flag == 1:
        initial_state_val = []
        step_count = 0
    next_value = initial_value
    pi = np.zeros(len(states) - 1)
    delta = 1  # Delta represents the maximal difference between the current value function, and the next one
    while (delta > 0.0001):
        # Update pi using the current value function
        current_value = next_value
        if print_flag == 1:
            initial_state_val.append(current_value[0])
        for i in range(len(states) - 1):
            state = states[i]
            action_value = np.inf
            cost = costs[getStateIndex(states, state)]  # The cost of the current state
            for a in state:  # Possible actions are only what is in the state.
                next_state = getNextState(state, a)
                tran_prob = prob[a - 1]  # Probability of success at taking out the job associated with a
                # There are always 2 states we can go to, the state determined by the action and the same state.
                new_action_value = cost + tran_prob * current_value[getStateIndex(states, next_state)] + (
                        1 - tran_prob) * current_value[getStateIndex(states, state)]
                if new_action_value < action_value:
                    action_value = new_action_value
                    pi[i] = a
        #  Update the value function with the new pi
        next_value = getVFWithPolicy(states, pi, prob, costs)
        delta = max(current_value - next_value)
        if print_flag == 1:
            step_count = step_count + 1

    if print_flag == 1:
        print('converged in ' + str(step_count))
        # We dont actually get this iteration, we add this value to show in the plot that the value converged
        initial_state_val.append(initial_state_val[-1])
        plt.plot(range(1, step_count + 2), initial_state_val)
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title('Value of initial state over policy iteration')
        plt.show()

    return pi, next_value


# Get the policy which chooses the job with the maximal cost, again, we dont associate an action to the empty state
pi = np.zeros(len(states) - 1).astype(int)

for i in range(len(states) - 1):
    state = states[i]
    relevent_jobs = jobs[state - 1]
    pi[i] = relevent_jobs[np.argmax(job_costs[state - 1])]

print(pi)

value = getVFWithPolicy(states, pi, prob, costs)
pi, final_val = policyIteration(states, prob, costs, value, 1)

print(states)
print(pi)

# -------------------------------------------------Section e------------------------------------

# We get the pi from the cu law
cu = job_costs*prob

cu_pi = np.zeros(len(states) - 1).astype(int)

for i in range(len(states) - 1):
    state = states[i]
    relevent_jobs = jobs[state - 1]
    cu_pi[i] = relevent_jobs[np.argmax(cu[state - 1])]

# Get the policy iteration pi
max_pi = np.zeros(len(states) - 1).astype(int)

for i in range(len(states) - 1):
    state = states[i]
    relevent_jobs = jobs[state - 1]
    max_pi[i] = relevent_jobs[np.argmax(job_costs[state - 1])]

value = getVFWithPolicy(states, max_pi, prob, costs)
PI_pi, final_val = policyIteration(states, prob, costs, value)

# Plotting the 2 pi arrays
plt.plot(range(len(PI_pi)), PI_pi, '+', markersize=10)
plt.plot(range(len(cu_pi)), cu_pi, '.')
plt.legend(['PI pi', 'cu pi'])
plt.xticks(range(len(PI_pi)), states[:-1], rotation='vertical')    # Set x-axis ticks, this ticks off some warning
plt.xlabel('State')
plt.ylabel('Action')
plt.title('PI pi and cu PI over the states')
plt.show()

# Plotting the value functions
plt.plot(range(len(final_val)), final_val)
plt.plot(range(len(value)), value)
plt.legend(['PI value', 'MAX value'])
plt.xticks(range(len(final_val)), states, rotation='vertical')    # Set x-axis ticks, this ticks off some warning
plt.xlabel('State')
plt.ylabel('Value')
plt.title('PI value and MAX value over the states')
plt.show()

# -------------------------------------------------Section f------------------------------------

def simulator(state, action, costs, states):
    tran_prob = prob[action - 1]
    state_cost = costs[getStateIndex(states, state)]

    # Get the state to return according to the probability
    outcomes = [True, False]
    probabilities = [tran_prob, 1 - tran_prob]
    job_success = random.choices(outcomes, probabilities)[0]

    if job_success:
        next_state = getNextState(state, action)
    else:
        next_state = state

    return state_cost, next_state


# c, s = simulator(states[0], 3, costs, states)
# print(c)
# print(s)

# -------------------------------------------------Section g------------------------------------\
# Get the policy that chooses the job with the maximal cost
max_pi = np.zeros(len(states) - 1).astype(int)

for i in range(len(states) - 1):
    state = states[i]
    relevent_jobs = jobs[state - 1]
    max_pi[i] = relevent_jobs[np.argmax(job_costs[state - 1])]

def TD0(pi, costs, states):
    value = np.zeroes(states)
    step_sizes = np.zeros(len(states), len(states[0]))


    for state in states:


