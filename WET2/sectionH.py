import numpy as np
import matplotlib.pyplot as plt
from environment import Environment

job_costs = np.array([1, 4, 6, 2, 9])
jobs = np.array([1, 2, 3, 4, 5])
prob = [0.6, 0.5, 0.3, 0.7, 0.1]

env = Environment(job_costs, jobs, prob)

def alphaIII(state, visits):
    return 10/(100+visits[env.get_state_index(state)])

# Define lambda values to test
# lambda_values = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

lambda_values = [0.1, 0.3, 0.7]
# Calculate pi
pi = np.zeros(len(env.states) - 1).astype(int)
for i in range(len(env.states) - 1):
    state = env.states[i]
    relevant_jobs = env.jobs[state - 1]
    pi[i] = relevant_jobs[np.argmax(env.job_costs[state - 1])]

actual_max_value = env.get_value_function_with_policy(pi)

# Perform TD(lambda) algorithm for different lambda values
num_runs = 20
avg_max_errors = []
avg_max_state0_errors = []
for i in range(len(lambda_values)):
    lambda_val = lambda_values[i]
    max_error_sum = 0
    state0_error_sum = 0
    for run in range(num_runs):

        value, error, state0_error = env.TD_lambda(pi, alphaIII, lambda_val, episodes=10000, return_err=1, actual_value=actual_max_value)
        max_error_sum += error
        state0_error_sum += state0_error

    avg_max_error = max_error_sum/num_runs
    avg_state0_error = state0_error_sum/num_runs

    avg_max_errors.append(avg_max_error)
    avg_max_state0_errors.append(avg_state0_error)

# Create subplots for average max error and average state 0 error
fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# Plot the average max errors
for i in range(len(lambda_values)):
    axes[0].plot(range(len(avg_max_errors[i])), avg_max_errors[i], label=(str(lambda_values[i]) + ' Max Error'))
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Average Max Error')
axes[0].set_title('TD(lambda) Average Max Error for Different Lambda Values')
axes[0].legend()

# Plot the average state 0 errors
for i in range(len(lambda_values)):
    axes[1].plot(range(len(avg_max_state0_errors[i])), avg_max_state0_errors[i], label=(str(lambda_values[i]) + ' State 0 Error'))
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Average State 0 Error')
axes[1].set_title('TD(lambda) Average State 0 Error for Different Lambda Values')
axes[1].legend()

# Adjust spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()

