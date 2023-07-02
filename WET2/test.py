import numpy as np
import matplotlib.pyplot as plt
from environment import Environment

# Define the job costs, jobs, and probabilities
# job_costs = [1, 4, 6, 2, 9]
# jobs = [1, 2, 3, 4, 5]
# prob = [0.6, 0.5, 0.3, 0.7, 0.1]
job_costs = np.array([1, 4])
jobs = np.array([1, 2])
prob = [0.6, 0.5]
# Create an instance of the Environment
env = Environment(job_costs, jobs, prob)

print(env.states)
# Create a test scenario
Q = np.array([[1, 2],
              [0, 2],
              [1, 0]]) # Example Q-values

pi = env.get_pi_from_Q(Q)
print(pi)


# state = env.states[0]  # Example state
# print(state)
# print(Q[0])
# epsilon = 0.1
# # Test the epsilon-greedy function
# a=0
# b=0
# for i in range(100000):
#     action = env.epsilon_greedy(epsilon, state, Q)
#     if action == 2 or action == 3:
#         a = a + 1
#     if action == 1:
#         b = b + 1
#
# print(a/100000)
# print(b/100000)


#
