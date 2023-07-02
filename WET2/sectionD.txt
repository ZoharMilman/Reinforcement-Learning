import numpy as np
from environment import Environment

job_costs = [1, 4, 6, 2, 9]
jobs = [1, 2, 3, 4, 5]
prob = [0.6, 0.5, 0.3, 0.7, 0.1]

env = Environment(job_costs, jobs, prob)

# Calculate pi
pi = np.zeros(len(env.states) - 1).astype(int)
for i in range(len(env.states) - 1):
    state = env.states[i]
    relevant_jobs = env.jobs[state - 1]
    pi[i] = relevant_jobs[np.argmax(env.job_costs[state - 1])]


# Calculate value function with policy
value = env.get_value_function_with_policy(pi)

# Perform policy iteration
pi, final_val = env.policy_iteration(value, print_flag=1)