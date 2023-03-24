import numpy as np

# Define neural network architecture
class StraightLineNN:
    def __init__(self, input_dim, output_dim, hidden_dim=10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weights1 = np.absolute(np.floor(np.random.randn(self.input_dim, self.hidden_dim)))
        self.biases1 = np.absolute(np.floor(np.random.randn(self.hidden_dim)))
        self.weights2 = np.absolute(np.floor(np.random.randn(self.hidden_dim, self.output_dim)))
        self.biases2 = np.absolute(np.floor(np.random.randn(self.output_dim)))
        self.d_slope = None
        self.d_y_intercept = None
        self.learning_rate = None

    def forward(self, x):
        hidden_layer_output = np.dot(x, self.weights1) + self.biases1
        hidden_layer_output = np.maximum(0, hidden_layer_output)  # ReLU activation function
        output_layer_output = np.dot(hidden_layer_output, self.weights2) + self.biases2
        return hidden_layer_output, output_layer_output

    def backward(self, x, y, hidden_layer_output, output_layer_output):
        error = output_layer_output - y
        d_weights2 = np.dot(hidden_layer_output.T, error)
        d_biases2 = np.sum(error, axis=0)
        d_hidden_layer_output = np.dot(error, self.weights2.T)
        d_hidden_layer_output[hidden_layer_output <= 0] = 0  # derivative of ReLU
        d_weights1 = np.dot(x.T, d_hidden_layer_output)
        d_biases1 = np.sum(d_hidden_layer_output, axis=0)
        return d_weights1, d_biases1, d_weights2, d_biases2

    def train(self, x, y, learning_rate=0.0001, num_epochs=1000):
        self.learning_rate = learning_rate
        for epoch in range(num_epochs):
            hidden_layer_output, output_layer_output = self.forward(x)
            loss = np.mean((output_layer_output - y) ** 2)
            d_weights1, d_biases1, d_weights2, d_biases2 = self.backward(x, y, hidden_layer_output, output_layer_output)
            # Calculate slope and y-intercept corrections
            self.d_slope = 2 * np.mean((output_layer_output - y) * hidden_layer_output)
            self.d_y_intercept = 2 * np.mean(output_layer_output - y)
            # Limit the size of slope and y-intercept corrections
            max_correction = 10
            if self.d_slope > max_correction:
                self.d_slope = max_correction
            elif self.d_slope < -max_correction:
                self.d_slope = -max_correction
            if self.d_y_intercept > max_correction:
                self.d_y_intercept = max_correction
            elif self.d_y_intercept < -max_correction:
                self.d_y_intercept = -max_correction
            # Apply corrections to weights and biases
            self.weights1 -= learning_rate * d_weights1
            self.biases1 -= learning_rate * d_biases1
            self.weights2 -= learning_rate * d_weights2
            self.biases2 -= learning_rate * d_biases2
            self.weights1[0][0] -= learning_rate * self.d_slope
            self.biases2[0] -= learning_rate * self.d_y_intercept
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss}")
                print(f"Slope {self.weights1[0][0] * self.weights2[0][0]}: y_intercept = {self.biases2[0]}")

    def get_line_parameters(self):
        # Get slope and y-intercept of the fitted straight line
        w1, b1, w2, b2 = self.weights1, self.biases1, self.weights2, self.biases2
        slope = w1[0][0] * w2[0][0] - self.learning_rate * self.d_slope
        y_intercept = b2[0] - self.learning_rate * self.d_y_intercept
        # Account for ReLU activation in the hidden layer
        if w1[0][0] < 0:
            slope *= -1
            y_intercept += b1[0] * w2[0][0]
        return slope, y_intercept
