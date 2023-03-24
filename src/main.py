import numpy as np
import matplotlib.pyplot as plt
import NeurNet

# Generate random data points
x_train = np.random.rand(10) * 10
y_train = x_train + np.random.randn(10) * 3

# Train neural network on generated data points
input_dim = 1
output_dim = 1
nn = NeurNet.StraightLineNN(input_dim, output_dim)
nn.train(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

# Evaluate loss function on training data points
hidden_layer_output, y_train_pred = nn.forward(x_train.reshape(-1, 1))
loss = np.mean((y_train_pred - y_train) ** 2)
print(f"Training Loss = {loss}")

# Get line parameters
slope, y_intercept = nn.get_line_parameters()
print(f"Slope = {slope}, Y-intercept = {y_intercept}")
plt.scatter(x_train, y_train, label="Training Data")
x_range = np.linspace(0, 10, 100)
y_range = slope * x_range + y_intercept
plt.plot(x_range, y_range, label="Fitted Line", color="red")
plt.legend()
plt.show()
