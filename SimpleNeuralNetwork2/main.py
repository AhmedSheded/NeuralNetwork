import numpy as np

class NeuralNetwork(object):
    def __init__(self):
        # parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
# wights
        self.w1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.w2 = np.random.randn(self.hiddenSize, self.outputSize)
        # print(self.w1, "\n dd", self.w2)

    def forward(self, x):
        self.z = np.dot(x, self.w1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.w2)
        o = self.sigmoid(self.z3)
        return o

    def sigmoid(self, s):
        return 1/(1-np.exp(-s))

    def sigmoidPrime(self, s):
        return s * (1-s)

    def backward(self, x, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error*self.sigmoidPrime(o)

        self.z2_error = self.o_delta.dot(self.w2.T)
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)

        self.w1 += x.T.dot(self.z2_delta)
        self.w2 += self.z2.T.dot(self.o_delta)

    def train(self, X, Y):
        o = self.forward(X)
        self.backward(X, Y, o)

    def saveweights(self):
        np.savetxt("w1.txt", self.w1, fmt="%s")
        np.savetxt("W2", self.w2, fmt="%s")

    def predict(self):
        print("input (scaled): \n", str(xPredicted))
        print("output: \n" + str(self.forward(xPredicted)))

# X = (hours of styding, hours of sleeping), y= score on test , xpredicted = 4 hours styding & 8 hours sleeping
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
Y = np.array(([92], [86], [89]), dtype=float)
xPredicted = np.array(([4, 8]), dtype=float)

# scale units
X = X/np.amax(X, axis=0)
xPredicted = xPredicted/np.amax(xPredicted, axis=0)
Y = Y/100

NN = NeuralNetwork()
for i in range(10):
    # print("loss \n", + str(np.mean(np.square(Y - NN.forward(X)))))
    NN.train(X, Y)
NN.saveweights()
NN.predict()