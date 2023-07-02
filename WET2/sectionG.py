import numpy as np
import matplotlib.pyplot as plt
from environment import Environment

job_costs = np.array([1, 4, 6, 2, 9])
jobs = np.array([1, 2, 3, 4, 5])
prob = [0.6, 0.5, 0.3, 0.7, 0.1]

env = Environment(job_costs, jobs, prob)

def alphaI(state, visits):
    return 1/visits[env.get_state_index(state)]

def alphaII(state, visits):
    return 0.01

def alphaIII(state, visits):
    return 10/(100+visits[env.get_state_index(state)])

# Calculate pi
pi = np.zeros(len(env.states) - 1).astype(int)
for i in range(len(env.states) - 1):
    state = env.states[i]
    relevant_jobs = env.jobs[state - 1]
    pi[i] = relevant_jobs[np.argmax(env.job_costs[state - 1])]

actual_max_value = env.get_value_function_with_policy(pi)

# Perform TD(0) algorithm with different alpha functions
valueI, errorI, state0_errorI = env.TD_zero(pi, alphaI, episodes=100000, return_err=1, actual_value=actual_max_value)
valueII, errorII, state0_errorII = env.TD_zero(pi, alphaII, episodes=100000, return_err=1, actual_value=actual_max_value)
valueIII, errorIII, state0_errorIII = env.TD_zero(pi, alphaIII, episodes=100000, return_err=1, actual_value=actual_max_value)

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

# Create subplots for error and state 0 error
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot the errors
axes[0].plot(range(len(errorI)), errorI, label='Alpha I')
axes[0].plot(range(len(errorII)), errorII, label='Alpha II')
axes[0].plot(range(len(errorIII)), errorIII, label='Alpha III')
axes[0].legend()
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Max Error')
axes[0].set_title('Max Error during TD(0) with Different Alpha Functions')

# Plot the state0_errors
axes[1].plot(range(len(state0_errorI)), state0_errorI, label='Alpha I')
axes[1].plot(range(len(state0_errorII)), state0_errorII, label='Alpha II')
axes[1].plot(range(len(state0_errorIII)), state0_errorIII, label='Alpha III')
axes[1].legend()
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('State 0 Error')
axes[1].set_title('State 0 Error during TD(0) with Different Alpha Functions')

# Adjust spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()








