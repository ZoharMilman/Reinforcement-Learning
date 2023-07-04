import matplotlib.pyplot as plt
import numpy as np

# Generate x values
x = np.linspace(-10, 10, 100)

# Define the linear functions
m_values = [1, -2, 0.5, -1.5]  # Slope values
b_values = [2, -3, 1, -2]  # Intercept values

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Plot each linear function
for i, ax in enumerate(axs.flat):
    m = m_values[i]
    b = b_values[i]
    y = m * x + b
    ax.plot(x, y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'y = {m}x + {b}')

# Adjust spacing between subplots
plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.2)
plt.suptitle('Balls', fontsize=16, fontweight='bold')
# Display the plot
plt.show()
