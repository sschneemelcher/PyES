import numpy as np
from inspect import isfunction
class ES:
    # ES takes following arguments:
    # - loss:       one of ["mse","acc",custom_function]
    #               the custom function has to take predictions and 
    #               target and must return some kind of number as a score
    # - predict:    a custom function that takes dna and input data and
    #               returns a prediction
    # - npop:       population size
    # - sigma:      randomness factor for the mutations
    # - lr:         learning rate
    # - epochs:     no of epochs for training

    def __init__(self, loss, predict, npop, sigma, lr, epochs):
        self.loss       = loss
        self.predict    = predict
        self.npop       = npop
        self.sigma      = sigma
        self.lr         = lr
        self.epochs     = epochs

    def mse(self, y_true, y_pred):
        return -np.mean((y_true - y_pred)**2)

    def accuracy(self, y_true, y_pred):
        return np.mean((y_true == y_pred))

    def score(self, y_true, y_pred):
        if isfunction(self.loss):
            return self.loss(y_true, y_pred)
        elif self.loss == "mse":
            return self.mse(y_true, y_pred)
        elif self.loss == "acc":
            return self.accuracy(y_true, y_pred)
        else:
            raise ValueError('Unknown loss')

    def fit(self, dna, x_train, y_train, verbosity=True):
        for e in range(self.epochs):
            fitness = np.zeros(self.npop)
            noise = np.random.randn(self.npop, len(dna))
            for n in range(self.npop):
                predictions = []
                model = dna + self.sigma * noise[n]
                for x in x_train:
                    predictions.append(self.predict(model, x))
                fitness[n] = self.score(y_train, predictions)
            d = (fitness - np.mean(fitness)) / np.std(fitness)
            dna = dna + self.lr / (self.npop * self.sigma) * np.dot(noise.T, d)
            if verbosity:
                print("epoch %d: fitness = %f" % (e, np.amax(fitness)))
        return dna
