## Simple Linear regression implementation with gradient descent and ordinary least squares as error function

import random

def scaler(x, min_x = None, max_x = None):
    """
    Implement a simple Min-Max scaler to normalize all the values of x.
    This is done to reduce the impact to huge gradient updates taking place. It is cause by the `xi` term being multipled to the
    partial weight derivative which will scale the gradient update by much larger values compared to bias which has no additional xi to scale.

    Normalization will pull down all values of x between 0 and 1 reducing overall effect on scaling grading update to huge values. 
    """
    min_x = float("inf") if min_x is None else min_x
    max_x = float("-inf") if max_x is None else max_x
    if min_x == float("inf"):
        for xi in x:
            if xi > max_x:
                max_x = xi
            if xi < min_x:
                min_x = xi
    scaled_vals = []
    for xi in x:
        scaled_vals.append((xi-min_x)/(max_x - min_x))
    return scaled_vals, min_x, max_x

class SimpleLinearRegression: 
    def __init__(self, learning_rate = 0.01, num_epochs = 500, weight = None, bias = None):
        self.learning_rate = learning_rate
        self.epochs = num_epochs
        self.weight = random.gauss(mu = 0, sigma = 1) if weight is None else weight   
        self.bias = random.gauss(mu = 0, sigma = 1) if bias is None else bias

    def calculate_prediction(self, x):
        """Calculates forward pass in linear regression"""
        y_pred = self.weight * x + self.bias
        return y_pred
    
    def calculate_error(self, x, y_true) -> float:
        """
        Calcualtes error between labels and predictions using ordinary least squares methods (Mean squared error).

        It is required as without squaring errors, positive and negative errors would cancel each other out.
        """
        total_error = 0
        for xi, yi in zip(x, y_true):
            total_error += (yi - self.calculate_prediction(xi))**2
        return total_error / len(y_true)  

    def calculate_gradient_weight(self, x, y_true):
        """
        Calculates the gradient of the loss function with respect to the weight.
        """
        total_weight_gradient = 0
        for xi,yi in zip(x, y_true):
            total_weight_gradient += -xi  * (yi - (self.weight * xi) - self.bias)
        weight_partial_derivative = (2/(len(y_true))) * total_weight_gradient
        return weight_partial_derivative


    def calculate_gradient_bias(self, x, y_true):
        """
        Calculates the gradient of the loss function with respect to the bias using partial derivative of the loss function wrt bias term (self.bias).
        """
        total_bias_gradient = 0
        for xi,yi in zip(x, y_true):
            total_bias_gradient += -(yi - (xi * self.weight) - self.bias )
        bias_partial_derivative = (2/len(y_true)) * total_bias_gradient
        return bias_partial_derivative
    
    def fit(self, x, y_true):
        """Run the training loop trying to minimize the loss function by running gradient updates for the defined epochs."""
        self.loss_history = []
        for i in range(self.epochs):
            error = self.calculate_error(x, y_true)
            weight_grad = self.calculate_gradient_weight(x, y_true)
            bias_grad = self.calculate_gradient_bias(x, y_true)
            self.weight = self.weight - (self.learning_rate * weight_grad)
            self.bias = self.bias - (self.learning_rate * bias_grad)
            self.loss_history.append(error)
        return self

    def predict(self, x):
        """Run predicition on overall dataset"""
        return [self.calculate_prediction(xi) for xi in x]
    



if __name__ == "__main__":
    linear_regression = SimpleLinearRegression(num_epochs=5000)
    x = [1, 3, 5, 10, 15]
    x_test = [2,4,6,8,12]
    y_true = [3, 7, 11, 21, 31]
    x_new, min_x, max_x = scaler(x)
    print(x_new)
    model = linear_regression.fit(x_new,y_true)
    x_test_new, _, _ = scaler(x_test, min_x, max_x)
    predictions = model.predict(x_test_new)
    print(model.weight)
    print(model.bias)
    print(predictions)
            



        