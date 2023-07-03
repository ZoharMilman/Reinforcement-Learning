import numpy as np
import time

from matplotlib import animation
from data_transformer import DataTransformer
from mountain_car_with_data_collection import MountainCarWithResetEnv
from radial_basis_function_extractor import RadialBasisFunctionExtractor
import matplotlib.pyplot as plt


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

class Solver:
    def __init__(self, number_of_kernels_per_dim, number_of_actions, gamma, learning_rate):
        # Set max value for normalization of inputs
        self._max_normal = 1
        # get state \action information
        self.data_transformer = DataTransformer()
        state_mean = [-3.00283763e-01,  5.61618575e-05]
        state_std = [0.51981243, 0.04024895]
        self.data_transformer.set(state_mean, state_std)
        self._actions = number_of_actions
        # create RBF features:
        self.feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
        self.number_of_features = self.feature_extractor.get_number_of_features()
        # the weights of the q learner
        self.theta = np.random.uniform(-0.001, 0, size=number_of_actions * self.number_of_features)
        # discount factor for the solver
        self.gamma = gamma
        self.learning_rate = learning_rate

    def _normalize_state(self, s):
        return self.data_transformer.transform_states(np.array([s]))[0]

    def get_features(self, state):
        normalized_state = self._normalize_state(state)
        features = self.feature_extractor.encode_states_with_radial_basis_functions([normalized_state])[0]
        return features

    def get_q_val(self, features, action):
        theta_ = self.theta[action*self.number_of_features: (1 + action)*self.number_of_features]
        return np.dot(features, theta_)

    def get_all_q_vals(self, features):
        all_vals = np.zeros(self._actions)
        for a in range(self._actions):
            all_vals[a] = solver.get_q_val(features, a)
        return all_vals

    def get_max_action(self, state):
        sparse_features = solver.get_features(state)
        q_vals = solver.get_all_q_vals(sparse_features)
        return np.argmax(q_vals)

    def get_state_action_features(self, state, action):
        state_features = self.get_features(state)
        all_features = np.zeros(len(state_features) * self._actions)
        all_features[action * len(state_features): (1 + action) * len(state_features)] = state_features
        return all_features

    def update_theta(self, state, action, reward, next_state, done):
        if done:
            # There is no next Q value
            target = reward
        else:
            # Get the approximation for the next Q value
            max_action = self.get_max_action(next_state)
            next_features = self.get_features(next_state)
            next_q = self.get_q_val(next_features, max_action)
            target = reward + self.gamma * next_q

        # Get the error
        features = self.get_features(state)
        current_q = self.get_q_val(features, action)
        error = target - current_q

        # Update the weights for the specific action
        action_weights_start = action * self.number_of_features
        action_weights_end = (action + 1) * self.number_of_features
        self.theta[action_weights_start:action_weights_end] += self.learning_rate * error * features

        return error


def modify_reward(reward):
    reward -= 1
    if reward == 0.:
        reward = 100.
    return reward


def run_episode(env, solver, is_train=True, epsilon=None, max_steps=200, render=False):
    frames = []
    episode_gain = 0
    deltas = []
    if is_train:
        start_position = np.random.uniform(env.min_position, env.goal_position - 0.01)
        start_velocity = np.random.uniform(-env.max_speed, env.max_speed)
    else:
        start_position = -0.5
        start_velocity = np.random.uniform(-env.max_speed / 100., env.max_speed / 100.)
    state = env.reset_specific(start_position, start_velocity)
    step = 0
    if render:
        frames.append(env.render(mode="rgb_array"))
        time.sleep(0.1)
    while True:
        if epsilon is not None and np.random.uniform() < epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            action = solver.get_max_action(state)
        if render:
            frames.append(env.render(mode="rgb_array"))
            time.sleep(0.1)
        next_state, reward, done, _ = env.step(action)
        reward = modify_reward(reward)
        step += 1
        episode_gain += reward
        if is_train:
            deltas.append(solver.update_theta(state, action, reward, next_state, done))
        if (done or step == max_steps) and not render:
            return episode_gain, np.mean(deltas)
        if (done or step == max_steps) and render:
            return episode_gain, np.mean(deltas), frames

        state = next_state

if __name__ == "__main__":
    env = MountainCarWithResetEnv()
    seeds = [123, 234, 345]
    # seed = 234
    # seed = 345

    for seed in seeds:
        np.random.seed(seed)
        env.seed(seed)

        gamma = 0.99
        learning_rate = 0.01
        epsilon_current = 0.1
        epsilon_decrease = 1.
        epsilon_min = 0.05

        max_episodes = 10000

        rewards = np.zeros(max_episodes)
        success_rates = np.zeros(max_episodes)
        state_values = np.zeros(max_episodes)
        avg_errors = np.zeros(int(max_episodes/100))
        error_window = np.zeros(100)

        solver = Solver(
            # learning parameters
            gamma=gamma, learning_rate=learning_rate,
            # feature extraction parameters
            number_of_kernels_per_dim=[7, 5],
            # env dependencies (DO NOT CHANGE):
            number_of_actions=env.action_space.n,
        )

        for episode_index in range(1, max_episodes + 1):

            episode_gain, mean_delta = run_episode(env, solver, is_train=True, epsilon=epsilon_current)
            error_window[episode_index % 100 - 1] = mean_delta
            rewards[episode_index-1] = episode_gain
            state_features = solver.get_features([-0.5, 0.])
            state_values[episode_index-1] = np.mean(solver.get_all_q_vals(state_features))

            # reduce epsilon if required
            epsilon_current *= epsilon_decrease
            epsilon_current = max(epsilon_current, epsilon_min)

            print(f'after {episode_index}, reward = {episode_gain}, epsilon {epsilon_current}, average error {mean_delta}')

            if episode_index % 100 == 99:
                avg_errors[int((episode_index+1)/100)-1] = np.mean(error_window)
                error_window = np.zeros(100)

            # termination condition:
            if episode_index % 10 == 9:
                test_gains = [run_episode(env, solver, is_train=False, epsilon=0.)[0] for _ in range(10)]
                num_of_success = np.count_nonzero(test_gains != -200)
                success_rate = num_of_success/10
                success_rates[episode_index - 1] = success_rate
                mean_test_gain = np.mean(test_gains)
                print(f'tested 10 episodes: mean gain is {mean_test_gain}')
                if mean_test_gain >= -75.:
                    print(f'solved in {episode_index} episodes')
                    break

        import matplotlib.pyplot as plt

        # Trim the errors
        rewards = np.trim_zeros(rewards)
        success_rates = np.trim_zeros(success_rates)
        state_values = np.trim_zeros(state_values)
        avg_errors = np.trim_zeros(avg_errors)

        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        # Plot 1: Total reward in the training episode vs. training episodes
        axs[0, 0].plot(range(1, len(rewards) + 1), rewards)
        axs[0, 0].set_xlabel('Training Episodes')
        axs[0, 0].set_ylabel('Total Reward')
        axs[0, 0].set_title('Total Reward vs. Training Episodes For Seed: ' + str(seed))

        # Plot 2: Performance vs. training episodes
        axs[0, 1].plot(range(1, len(success_rates) + 1), success_rates)
        axs[0, 1].set_xlabel('Training Episodes')
        axs[0, 1].set_ylabel('Performance')
        axs[0, 1].set_title('Performance vs. Training Episodes For Seed ' + str(seed))

        # Plot 3: Approximate value of the state at the bottom of the hill vs. training episodes
        axs[1, 0].plot(range(1, len(state_values) + 1), state_values)
        axs[1, 0].set_xlabel('Training Episodes')
        axs[1, 0].set_ylabel('Approximate Value')
        axs[1, 0].set_title('Approximate Value vs. Training Episodes For Seed' + str(seed))

        # Plot 4: Total Bellman error of the episode, averaged over most recent 100 episodes
        axs[1, 1].plot(range(len(avg_errors)), avg_errors)
        axs[1, 1].set_xlabel('Training Episodes/100')
        axs[1, 1].set_ylabel('Average Bellman Error')
        axs[1, 1].set_title('Average Bellman Error vs. Training Episodes For Seed=' + str(seed))

        # Adjust spacing between subplots
        plt.tight_layout()

        # Display the plot
        # plt.show()

        plt.savefig(str(seed) + '_results')

        _, _, frames = run_episode(env, solver, is_train=False, render=True)
        save_frames_as_gif(frames, filename=(str(seed) + '.gif'))








