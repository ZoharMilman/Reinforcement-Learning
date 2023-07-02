import numpy as np
import matplotlib.pyplot as plt
from environment import Environment

job_costs = np.array([1, 4, 6, 2, 9])
jobs = np.array([1, 2, 3, 4, 5])
prob = [0.6, 0.5, 0.3, 0.7, 0.1]

env = Environment(job_costs, jobs, prob)

# Calculate cu pi
cu = env.job_costs * env.prob
cu_pi = np.zeros(len(env.states) - 1).astype(int)
for i in range(len(env.states) - 1):
    state = env.states[i]
    relevant_jobs = env.jobs[state - 1]
    cu_pi[i] = relevant_jobs[np.argmax(cu[state - 1])]

# Get the policy iteration pi
max_pi = np.zeros(len(env.states) - 1).astype(int)
for i in range(len(env.states) - 1):
    state = env.states[i]
    relevant_jobs = env.jobs[state - 1]
    max_pi[i] = relevant_jobs[np.argmax(env.job_costs[state - 1])]

# Calculate value function with max_pi and policy iteration pi
value = env.get_value_function_with_policy(max_pi)
PI_pi, final_val = env.policy_iteration(value)

# print(final_val)

# Plotting the 2 pi arrays
plt.plot(range(len(PI_pi)), PI_pi, '+', markersize=10)
plt.plot(range(len(cu_pi)), cu_pi, '.')
plt.legend(['PI pi', 'cu pi'])
plt.xticks(range(len(PI_pi)), env.states[:-1], rotation='vertical')
plt.xlabel('State')
plt.ylabel('Action')
plt.title('PI pi and cu PI over the states')
plt.show()

# Plotting the value functions
plt.plot(range(len(final_val)), final_val)
plt.plot(range(len(value)), value)
plt.legend(['PI value', 'MAX value'])
plt.xticks(range(len(final_val)), env.states, rotation='vertical')
plt.xlabel('State')
plt.ylabel('Value')
plt.title('PI value and MAX value over the states')
plt.show()
