import logging
from Network import Neurons as NN
from Network.Functions import *
import numpy as np


class Optimizers:
    class Optimizer:
        def __init__(self, lr=.01, stepsToUpdate=1, clipping=None):
            self.lr = lr
            self.stepsToUpdate = stepsToUpdate
            self.clipping = clipping

        def calcUpdate(self, partialDerivative):
            raise NotImplementedError

        def clip(self, derivative):
            if self.clipping is not None:
                return np.clip(derivative, -self.clipping, self.clipping)
            return derivative

        def setError(self, error, stepsToUpdate = 1):
            self.stepsToUpdate = stepsToUpdate
            # negative value will increase a parameter's magnitude
            self.error = error

        def getError(self):
            return self.error

        def setLearningRate(self, lr):
            self.lr = lr

    class GD(Optimizer):
        def __init__(self, lr=.01, clipping = None):
            super().__init__(lr = lr, clipping=clipping)

        def calcUpdate(self, partialDerivative):
            return np.divide((np.multiply(-self.lr, partialDerivative)),self.stepsToUpdate)



    class GDWithSigmoid(Optimizer):
        def __init__(self, lr=.01, clipping = None):
            super().__init__(lr = lr, clipping=clipping)

        def calcUpdate(self, partialDerivative):
            return np.divide((np.multiply(-self.lr, partialDerivative)),self.stepsToUpdate)

        def clip(self, derivative):
            if self.clipping is not None:
                return self.clipping * (sigmoidFunc(derivative) - .5 * np.ones_like(derivative))
            return derivative

    class GDWithTanh(Optimizer):
        def __init__(self, lr=.01, clipping=None):
            super().__init__(lr=lr, clipping=clipping)

        def calcUpdate(self, partialDerivative):
            return np.divide((np.multiply(-self.lr, partialDerivative)), self.stepsToUpdate)

        def clip(self, derivative):
            if self.clipping is not None:
                # print('before', derivative)
                # print('after', self.clipping * np.tanh(derivative))
                return self.clipping * np.tanh(derivative)
            return derivative


class Computegraph:

    def __init__(self, loss=None, maxForwardOffset=0, maxBackwardOffset=0, optimizer=Optimizers.GD(lr=.01)):
        self.optimizer = optimizer
        self.loss = loss
        self.neurons = []
        self.computeForwardQ = []
        self.computeBackwardQ = []
        self.t = 0
        self.b = 0
        self.neuronDict = {}
        self.maxForwardOffset = maxForwardOffset
        self.maxBackwardOffset = maxBackwardOffset

    def GD(partialDeriv):
        pass

    def getNeuronByName(self, name):
        if name not in self.neuronDict:
            return None
        return self.neuronDict[name]

    def addNeuron(self, neuron):
        for n in neuron.getNeurons():
            self.neurons.append(n)
            if n.name is not None:
                if n.name in self.neuronDict:
                    logging.debug('A neuron with the same name is in the dictionary: {}'.format(n.name))
                else:
                    self.neuronDict[n.name] = neuron

    def getStdNeuronArguments(self):
        return {'maxForwardOffset': self.maxForwardOffset, 'maxBackwardOffset': self.maxBackwardOffset}

    def printQ(self, backward=False):
        print('_______Compute Q start_________')
        if backward:
            for i in self.computeBackwardQ:
                print(i.name)
        else:
            for i in self.computeForwardQ:
                print(i.name)
        print('_______Compute Q end_________')

    def restartTime(self):
        self.t = 0
        for n in self.neurons:
            n.reset()

    def isTimeInSync(self):
        inSync = True
        time = self.t
        for n in self.neurons:
            inSync = inSync and time == n.t
        return inSync

    def computeForward(self):
        self.enqueueForward([n for n in self.neurons if n.forwardQ == 0])
        # self.printQ()
        NN.logger.debug('\n\n********forward: current timestep: {}'.format(self.t))
        while len(self.computeForwardQ) > 0:
            neuron = self.computeForwardQ[-1]
            self.computeForwardQ.pop()
            if (neuron.t == self.t):
                NN.logger.debug('     >name: {} time {}'.format(neuron.name, neuron.t))
                self.enqueueForward(neuron.forward())
                NN.logger.debug(
                    '     >output {} from output index {}'.format(neuron.getOutput(), neuron.calcOutputOffset(1)))
                NN.logger.debug('_____________________')
        NN.logger.debug('********end timestep: {}\n'.format(self.t))
        self.t += 1

    def computeBackward(self):
        self.b = 0
        while (self.b < self.optimizer.stepsToUpdate and self.b < self.t):
            self.enqueueBackward([self.loss])
            # self.printQ(backward=True)
            NN.logger.debug('\n\n********backward: timestep being updated: {}'.format(self.t - self.b - 1))
            while len(self.computeBackwardQ) > 0:
                neuron = self.computeBackwardQ[-1]
                self.computeBackwardQ.pop()
                if (neuron.b == self.b):
                    NN.logger.debug('     >name: {} original time: {} time being updated: {}'
                                    .format(neuron.name, neuron.t - 1, neuron.t - neuron.b - 1))
                    self.enqueueBackward(neuron.backward(self.optimizer))
                    NN.logger.debug(
                        '     >derivatives {} from derivatives index {}'.format(neuron.getDerivative(),
                                                                                neuron.calcBackPropOffset(1)))
                    NN.logger.debug('_____________________')

            self.b += 1
        self.updateParameters()

    def updateParameters(self):
        for n in self.neurons:
            n.updateParameters()

    def enqueueForward(self, neurons):
        for n in neurons:
            if n.t == self.t:
                self.computeForwardQ.insert(0, n)

    def enqueueBackward(self, neurons):
        for n in neurons:
            if n.b == self.b:
                self.computeBackwardQ.insert(0, n)
