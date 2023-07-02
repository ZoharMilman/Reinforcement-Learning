import matplotlib.pyplot as plt
from environment import Environment
import numpy as np

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
print(pi)

# Calculate value function with policy
value = env.get_value_function_with_policy(pi)
print(value)

# Plot value function
plt.plot(range(len(value)), value)
plt.xticks(range(len(value)), env.states, rotation='vertical')
plt.xlabel('State Values')
plt.ylabel('Value')
plt.title('Value Function For Max Cost pi')
plt.show()
