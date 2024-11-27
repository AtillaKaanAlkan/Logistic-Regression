import numpy as np
import matplotlib.pyplot as plt
import random

class LogisticRegression:

    def __init__(self, nb_iter, alpha):
        
        self.n_samples, self.n_features = X.shape
        self.nb_iter = nb_iter
        self.alpha = alpha # alpha is the learning rate
        self.w = None
        self.bias = None
        self.losses = []

    # Compute the linear relation between X and Z
    def compute_Z(self, X):
        Z = np.dot(X, self.w) + self.bias
        return  Z

    # Sigmoid activation function
    def _sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    # Compute the loss
    def compute_loss(self, y_true, y_pred):
        y_1 = y_true * np.log(y_pred)
        y_2 = (1-y_true) * np.log(1-y_pred)
        #L = (-1/2)*(y_1 + y_2)
        L = - np.mean(y_1 + y_2)
        return L

    def train(self, X, y):

        # initizalize w and b randomly 
        self.w = np.zeros(self.n_features)
        self.bias = 0

        # gradient descent 
        for i in range(self.nb_iter):

            Z = self.compute_Z(X)
            a = self._sigmoid(Z) # the prediction

            # Compute the loss to compare the predicted value with the real value
            loss = self.compute_loss(y, a)
            self.losses.append(loss)           

            # compute the gradients
            dz = a - y 
            dW = (1/self.n_features) * np.dot(X.T, dz)
            db = (1/self.n_features) * np.sum(dz)
                
            # Update the parameters
            self.w = self.w - self.alpha * dW
            self.bias = self.bias - self.alpha * db

    def predict(self, X):

        threshold = 0.5
        z  = np.dot(X, self.w) + self.bias
        y_pred = self._sigmoid(z)

        y_pred_classe = [1 if i > threshold else 0 for i in y_pred]

        return y_pred_classe

        
if __name__ == '__main__':

    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from sklearn.metrics import confusion_matrix
    
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target

    X, y = dataset.data, dataset.target 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state = 1234
    )

    regressor = LogisticRegression(nb_iter = 1000,
                                   alpha = 0.0001)
    
    regressor.train(X_train, y_train)
    predictions = regressor.predict(X_test)
   
    # Compute metrics
    cm = confusion_matrix(y_test, predictions)
