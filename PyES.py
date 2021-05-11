import numpy as np
import multiprocessing as mp
import redis
import os
import sys
import struct
from inspect import isfunction
class ES:
    # ES takes following arguments:
    # - loss:        one of ["mse","acc",custom_function]
    #                the custom function has to take predictions and 
    #                target and must return some kind of number as a score
    # - predict:     a custom function that takes dna and input data and
    #                returns a prediction
    #- predict_args: list of extra arguments that the predict function can take

    def __init__(self, loss, predict, predict_args = [], server = ""):
        self.loss         = loss
        self.predict      = predict
        self.predict_args = predict_args
        self.distributed  = server != ""
        self.server       = server
    
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

    def get_fit(self, q, npop, dna, sigma, lr, x_batch, y_batch):
        noise = np.random.randn(npop, len(dna))
        fitness = np.zeros(npop)
        for p in range(npop):
            predictions = []
            model = dna + sigma * noise[p]
            for x in x_batch:
                predictions.append(self.predict(model, x, self.predict_args))
            fitness[p] = self.score(y_batch, predictions)
        q.put([fitness, noise])

    
    # - npop:       population size
    # - sigma:      randomness factor for the mutations
    # - lr:         learning rate
    # - epochs:     no of epochs for training
    def fit(self, dna, x_train, y_train, batch_size=0, shuffle=False, lr=0.05, sigma=0.1, npop=50, epochs=10, verbosity=2):

        if self.distributed:
            r = redis.Redis(host=self.server.split(":")[0], port=self.server.split(":")[1], db=0)
            dna_str = str(list(dna))[1:-1]
            dna_hash = r.get('hash')
            if dna_hash is None: # synchronize the dna with the redis server
                r.set('hash', hash(dna_str))
                r.set('dna', dna_str)
            else:
                dna_hash = r.get('hash')
                dna_server = r.get('dna')  
                while dna_server is None: # be sure that there is dna on the server
                    dna_server = r.get('dna')
                dna = np.array(dna_server.split(b','), dtype=np.float32)

        cores = mp.cpu_count()
        print("detected %d cores.." % (cores))
        data_len = len(x_train)
        if batch_size < 1:
            batch_size = data_len
        batch_count    = data_len // batch_size
        

        for e in range(epochs):
            if shuffle:
                indices = np.random.permutation(data_len)
            else:
                indices = range(data_len)
            
            for b in range(batch_count):
                x_batch = x_train[indices[b*batch_size:(b+1)*batch_size]]
                y_batch = y_train[indices[b*batch_size:(b+1)*batch_size]]
                q = mp.Queue()

                procs = []
                for i in range(cores): # give every process only a share of the whole population
                    proc = mp.Process(target=self.get_fit, args=(q, npop//cores, dna, sigma, lr, x_batch, y_batch))
                    procs.append(proc)
                    proc.start()

                fitness, noise = q.get()
                for i in range(len(procs) - 1):
                    content = q.get()
                    fitness = np.append(fitness, content[0])
                    noise   = np.append(noise, content[1], axis=0)

                std = np.std(fitness)
                if std == 0:
                    std = 10**-16
                d = (fitness - np.mean(fitness)) / std
                dna += (lr / (npop * sigma) * np.dot(noise.T, d))

                for p in procs: 
                    p.join()

                if self.distributed:
                    dna_str = str(list(dna))[1:-1]
                    ##### check if your dna is still the newest
                    while dna_hash != r.get('hash'):
                        print("hash mismatch")
                        dna_hash = r.get('hash')
                        dna = np.array(r.get('dna').split(b','), dtype=np.float32) + d
                        dna_str = str(list(dna))[1:-1]
                    print("shared new dna")
                    r.set('hash', hash(dna_str))
                    dna_hash = r.get('hash')

                if verbosity == 2:
                    if b == batch_count-1:
                        prog = int(np.ceil(b/batch_count)*20)
                    else: 
                        prog = (b*20//batch_count)
                    sys.stdout.write("[%s] epoch %d: batch %d: fitness = %f\n" % (prog*'#' + (20-prog)*' ', e, b, np.amax(fitness)))
            if verbosity == 1:
                print("epoch %d: fitness = %f" % (e, np.amax(fitness)))
            elif verbosity == 2:
                sys.stdout.write("\n")
        return dna


