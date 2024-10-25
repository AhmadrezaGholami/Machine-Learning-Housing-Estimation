import numpy as np

class SGDRegressor_Scratch:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
    
    def initialize_params(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def fit(self, X, y):
        # Convert y to a numpy array to avoid KeyError when shuffling
        y = np.array(y)

        # Get number of samples and features
        n_samples, n_features = X.shape
        # Initialize weights
        self.initialize_params(n_features)

        # Stochastic Gradient Descent loop
        for epoch in range(self.n_epochs):
            # Shuffle the dataset at each epoch
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]

            # Loop over each sample
            for i in range(n_samples):
                X_i = X[i, :].reshape(1, -1)  # Ensure it's 2D
                y_i = y[i]

                # Prediction for the i-th sample
                y_pred = self.predict(X_i)

                # Compute gradient for weights and bias
                gradient_w = -(2 / n_samples) * X_i.T.dot(y_i - y_pred)
                gradient_b = -(2 / n_samples) * (y_i - y_pred)

                # Update weights and bias using the gradients
                self.weights -= self.learning_rate * gradient_w.ravel()
                self.bias -= self.learning_rate * gradient_b
                
class NormalEquationRegressor_Scratch:
    def __init__(self):
        self.theta = None
    
    def fit(self, X, y):
        # Add bias term (column of ones) to X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of ones for the bias term
        
        # Compute the Normal Equation: theta = (X^T * X)^(-1) * X^T * y
        self.theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    def predict(self, X):
        # Add bias term to X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Make predictions: y_pred = X_b * theta
        return X_b.dot(self.theta)