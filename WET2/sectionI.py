import numpy as np
import matplotlib.pyplot as plt
from environment import Environment

job_costs = np.array([1, 4, 6, 2, 9])
jobs = np.array([1, 2, 3, 4, 5])
prob = [0.6, 0.5, 0.3, 0.7, 0.1]

# job_costs = np.array([1, 4, 9])
# jobs = np.array([1, 2, 3])
# prob = [0.6, 0.5, 0.6]
#
# job_costs = np.array([1, 4])
# jobs = np.array([1, 2])
# prob = [0.6, 0.5]
env = Environment(job_costs, jobs, prob)
#

def alphaI(state, visits):
    return 1 / visits[env.get_state_index(state)]


def alphaII(state, visits):
    return 0.01

def alphaIII(state, visits):
    return 10 / (100 + visits[env.get_state_index(state)])


# Calculate cu pi
cu = env.job_costs * env.prob
cu_pi = np.zeros(len(env.states) - 1).astype(int)
for i in range(len(env.states) - 1):
    state = env.states[i]
    relevant_jobs = env.jobs[state - 1]
    cu_pi[i] = relevant_jobs[np.argmax(cu[state - 1])]
# print(cu_pi)
optimal_value_cu = env.get_value_function_with_policy(cu_pi)

# Perform Q-learning algorithm with different alpha functions
piI, errorI, state0_errorI = env.sarsa_q_learning(alphaI, 0.1, 1, optimal_value=optimal_value_cu, episodes=100000, return_err=True, return_err_every=1)
piII, errorII, state0_errorII = env.sarsa_q_learning(alphaII, 0.1, 1, optimal_value=optimal_value_cu, episodes=100000, return_err=True, return_err_every=1)
piIII, errorIII, state0_errorIII = env.sarsa_q_learning(alphaIII, 0.1, 1, optimal_value=optimal_value_cu, episodes=100000, return_err=True, return_err_every=1)
print(piI)
print(piII)
print(piIII)
print(cu_pi)
# # Plot the value functions
# plt.plot(range(len(valueI)), valueI, label='Alpha I')
# plt.plot(range(len(valueII)), valueII, label='Alpha II')
# plt.plot(range(len(valueIII)), valueIII, label='Alpha III')
# plt.plot(range(len(actual_max_value)), actual_max_value, label='Actual Value')
# plt.legend()
# plt.xlabel('State')
# plt.ylabel('Value')
# plt.title('Value Functions with Different Alpha Functions')
# plt.show()

# Create subplots for errors and state 0 errors
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Plot the errors
axes[0].plot(range(len(errorI)), errorI, label='Alpha I')
axes[0].plot(range(len(errorII)), errorII, label='Alpha II')
axes[0].plot(range(len(errorIII)), errorIII, label='Alpha III')
axes[0].legend()
axes[0].set_xlabel('Episode/100')
axes[0].set_ylabel('Max Error')
axes[0].set_title('Max Error during SARSA Q-Learning with Different Alpha Functions')

# Plot the state0_errors
axes[1].plot(range(len(state0_errorI)), state0_errorI, label='Alpha I')
axes[1].plot(range(len(state0_errorII)), state0_errorII, label='Alpha II')
axes[1].plot(range(len(state0_errorIII)), state0_errorIII, label='Alpha III')
axes[1].legend()
axes[1].set_xlabel('Episode/100')
axes[1].set_ylabel('State 0 Error')
axes[1].set_title('State 0 Error during SARSA Q-Learning with Different Alpha Functions')

# Adjust spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()

