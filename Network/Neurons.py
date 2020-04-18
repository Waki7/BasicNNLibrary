from sklearn.utils.extmath import softmax
from Network import Computegraph as CG
from Network.ANeuron import *
from Network.Functions import *
import logging

# class NeuronBunch(Neuron):
#     def __init__(self, neurons = None):
#         self.outshpe = (1,1)
#         for n in neurons:
#             self.outshpe[1] += n.outshpe[1]
#
#     def add(self, neuron):
#         self.outshpe[1] += neuron.outshpe[1]
#
#     def addAll(self, neurons):
#         for n in neurons:
#             self.outshpe[1] += n.outshpe[1]

logger = logging.getLogger('-')


def calcInshpe(inputs=None, numbers=0, shapes=(0, 0)):
    add = numbers
    if isinstance(numbers, tuple):
        add = sum(numbers)
    if isinstance(shapes[0], tuple):
        add += sum([x[1] for x in shapes])
    else:
        add += shapes[1]
    if isinstance(inputs, tuple):
        return (1, sum(n.outshpe[1] for n in inputs) + add)
    if inputs is not None:
        return (1, inputs.outshpe[1] + add)
    return (1, add)


class ConstantNeuron(Neuron):

    def __init__(self, inshpe, maxForwardOffset=0, maxBackwardOffset=0, input=None, timeOffset=0, name=''):
        super().__init__(inshpe, outshpe=inshpe,
                         maxForwardOffset=maxForwardOffset, maxBackwardOffset=maxBackwardOffset,
                         name=name)
        if input is not None:
            if not isinstance(input, Neuron):
                self.input = input
                self.inputs = [np.copy(self.input)] * self.derivativeArrayLength
            else:
                if input is not None:
                    self.takeInputsFrom(n=input, timeOffset=timeOffset)
        else:
            self.inputs = [np.zeros(self.inshpe)] * self.derivativeArrayLength

    def setInput(self, input, timeOffset= 0):
        if np.shape(input) != self.inshpe:
            raise ValueError('Shape mismatch')
        if timeOffset == 0:
            self.input = input
        self.inputs[self.calcInputOffset(timeOffset)] = input


class Subset(Neuron):

    def __init__(self, inshpe, maxForwardOffset=0, maxBackwardOffset=0, input=None,
                 inputSubset = None, timeOffset=0, name=''):
        super().__init__(inshpe, outshpe=(1, inputSubset[1]-inputSubset[0]), input=input, timeOffset=timeOffset,
                         maxForwardOffset=maxForwardOffset, maxBackwardOffset=maxBackwardOffset,
                         name=name)
        self.inputSubset = inputSubset

    def calcOutput(self):
        return self.input[:,self.inputSubset[0]:self.inputSubset[1]]

    def calcDerivative(self):
        partial = np.zeros_like(self.input)
        partial[:,self.inputSubset[0]:self.inputSubset[1]] = self.backInput
        return partial

class NeuronSplitter(Neuron):

    def __init__(self, inshpe, splits, maxForwardOffset=0, maxBackwardOffset=0, input=None, timeOffset=0, name='',
                 names=('',)):
        super().__init__(inshpe, outshpe=inshpe, maxForwardOffset=maxForwardOffset, maxBackwardOffset=maxBackwardOffset,
                         input=input, timeOffset=timeOffset, name=name)
        self.splits = splits
        self.splitInput = np.split(self.input, splits, axis=1)
        self.segments = []
        for i in range(0, len(self.splitInput)):
            if i == 0:
                start = 0
            else:
                start = splits[i-1]
            if i == len(self.splitInput)-1:
                end = self.inshpe[1]
            else:
                end = splits[i]
            newSegment = Subset(inshpe=self.inshpe,
                                        inputSubset=(start, end),
                                        maxForwardOffset=maxForwardOffset,
                                        maxBackwardOffset=maxBackwardOffset,
                                        name=name + 'segment' + str(i) + '_' + names[i % len(names)])
            newSegment.takeInputsFrom(self, 0)
            self.segments.append(newSegment)

    def getNeurons(self, includeSelf=True):
        if includeSelf:
            return self.segments + [self]
        return self.segments

    def getSegment(self, index):
        return self.segments[index]

    def getSegments(self):
        return self.segments


class Affine(Neuron):
    # numHU= number of hidden units
    def __init__(self, inshpe, numHU, maxForwardOffset=0, maxBackwardOffset=0, input=None, timeOffset=0, bias=False,
                 name=''):
        if bias:
            inshpe = (inshpe[0], inshpe[1] + 1)
        super().__init__(inshpe, outshpe=(inshpe[0], numHU), maxForwardOffset=maxForwardOffset,
                         maxBackwardOffset=maxBackwardOffset, input=input, timeOffset=timeOffset, name=name)
        if bias:
            self.input[:, -1] = 1
        self.bias = bias
        self.numHU = numHU
        self.w = 2 * np.random.random((self.inshpe[1], self.numHU)) - 1
        self.wUpdate = np.zeros_like(self.w)
        self.inputs = [np.copy(self.input)] * self.derivativeArrayLength

    def calcOutput(self):
        return np.dot(self.input, self.w)

    def calcDerivative(self):
        dydx = np.dot(self.backInput, np.transpose(self.w))
        return dydx

    def incrementalUpdate(self, optimizer=None):
        input = self.getInput(postForward=True)
        Neurons.logger.debug(
            '     >input before \n{}'.format(input))
        Neurons.logger.debug(
            '     >weights before \n{}'.format(self.w))
        partialW = np.dot(np.transpose(input), self.backInput)
        if optimizer is None:
            self.wUpdate += CG.Optimizers.GD().calcUpdate(partialW)
        else:
            self.wUpdate += optimizer.calcUpdate(partialW)
        Neurons.logger.debug(
            '     >weights delta {}'.format(self.wUpdate))

    def updateParameters(self, optimizer=None):
        self.w = np.add(self.w, self.wUpdate)
        self.wUpdate = np.zeros_like(self.w)
        return super().updateParameters()


class LogicalOrOperator(Neuron):  # TODO: add support for more than two logical envInputs
    # numHU= number of hidden units
    def __init__(self, inshpe, maxForwardOffset=0, maxBackwardOffset=0, input=None, timeOffset=0, name=''):
        self.inLength = inshpe[1]
        self.halfInLength = int(self.inLength / 2)
        super().__init__(inshpe, outshpe=(1, self.halfInLength), maxForwardOffset=maxForwardOffset,
                         maxBackwardOffset=maxBackwardOffset, input=input, timeOffset=timeOffset, name=name)
        self.inputs = [np.zeros(self.inshpe)] * self.derivativeArrayLength

    def calcOutput(self):
        return np.logical_or(self.input[:, :-self.halfInLength], self.input[:, self.halfInLength:]).astype(float)

    def calcDerivative(self):
        input = self.getInput(postForward=True)
        partial = np.asarray([[input[0,i] if input[0,i] == input[0,(i +self.halfInLength) % self.inLength]
                              else .5 for i in range(self.inLength)]])
        derivative = np.multiply(partial, np.concatenate([self.backInput]*2, axis=1))
        return derivative


class AffRelSplitNeuron(Neuron):
    # Container neuron, placement is ahead of the neurons it contains
    def __init__(self, inshpe, numAff, numRel, maxForwardOffset=0, maxBackwardOffset=0, input=None, timeOffset=0,
                 bias=False, name=''):
        self.Aff = Affine(inshpe=inshpe, maxForwardOffset=maxForwardOffset, maxBackwardOffset=maxBackwardOffset,
                          numHU=numAff, timeOffset=timeOffset, bias=bias, name=name + '.Affine')
        self.AffAct = Affine(inshpe=inshpe, maxForwardOffset=maxForwardOffset, maxBackwardOffset=maxBackwardOffset,
                             numHU=numRel, timeOffset=timeOffset, bias=bias, name=name + '.ActivatedAffine')
        self.Rel = RelU(inshpe=self.AffAct.outshpe, maxForwardOffset=maxForwardOffset,
                        maxBackwardOffset=maxBackwardOffset, name=name + '.RelU')
        self.Rel.takeInputsFrom(self.AffAct, 0)
        shape = (1, self.Rel.outshpe[1] + self.Aff.outshpe[1])
        super().__init__(inshpe=shape, outshpe=shape,
                         maxForwardOffset=maxForwardOffset, maxBackwardOffset=maxBackwardOffset,
                         input=input, timeOffset=timeOffset, name=name)
        super().takeInputsFrom(n=self.Aff, timeOffset=0)
        super().takeInputsFrom(n=self.Rel, timeOffset=0)

    def getNeurons(self):
        return self, self.Aff, self.AffAct, self.Rel

    def takeInputsFrom(self, n, timeOffset):
        self.Aff.takeInputsFrom(n=n, timeOffset=timeOffset)
        self.AffAct.takeInputsFrom(n=n, timeOffset=timeOffset)

    def printWeights(self):
        print(self.name, '\nAffine\n', self.Aff.w, '\nActivatedAffine\n', self.AffAct.w,
              '\n ______________________________________________\n')


class RelU(Neuron):

    def __init__(self, inshpe, maxForwardOffset=0, maxBackwardOffset=0, input=None, timeOffset=0, name=''):
        super().__init__(inshpe, outshpe=inshpe, maxForwardOffset=maxForwardOffset,
                         maxBackwardOffset=maxBackwardOffset,
                         input=input, timeOffset=timeOffset, name=name)
        self.inputs = [np.zeros(self.inshpe)] * self.derivativeArrayLength

    def calcOutput(self):
        return np.maximum(self.input, np.zeros(self.inshpe))

    def calcDerivative(self):
        partial = np.ones(self.outshpe)
        input = self.getInput(postForward=True)
        partial[input < 0] = 0
        partial[input == 0] = .5
        derivative = np.multiply(self.backInput, partial)
        return derivative


class Mapping(Neuron):

    def __init__(self, inshpe, map, outshpe=None, maxForwardOffset=0, maxBackwardOffset=0, input=None, timeOffset=0,
                 name=''):
        if outshpe is None:
            outshpe = (1, len(map))

        super().__init__(inshpe, outshpe=outshpe, maxForwardOffset=maxForwardOffset,
                         maxBackwardOffset=maxBackwardOffset, input=input, timeOffset=timeOffset, name=name)
        self.map = map

    def calcOutput(self):
        output = np.zeros(self.outshpe)
        for x, y in self.map.items():
            output[:,y] = self.input[:,x]
        return np.asarray(output)

    def calcDerivative(self):
        derivative = np.zeros(self.inshpe)
        for x, y in self.map.items():
            derivative[:,x] = self.backInput[:,y]
        return np.asarray(derivative)


class MaxOneHotEncoding(Neuron):

    def __init__(self, inshpe, maxForwardOffset=0, maxBackwardOffset=0, input=None, timeOffset=0, name=''):
        super().__init__(inshpe, outshpe=inshpe, maxForwardOffset=maxForwardOffset,
                         maxBackwardOffset=maxBackwardOffset, input=input, timeOffset=timeOffset, name=name)
        self.inputs = [np.zeros(self.inshpe)] * self.derivativeArrayLength

    def calcOutput(self):
        out = np.zeros(self.inshpe)
        out[0, self.input.argmax(1)] = 1
        return out

    def calcDerivative(self):
        input = self.getInput(postForward=True)
        partial = np.ones(self.inshpe)
        partial[0, np.argmax(input, axis=1)] = 1
        derivative = np.multiply(partial, self.backInput)
        return derivative


class Loss(Neuron):
    def __init__(self, inshpe, optimizer, maxForwardOffset=0, maxBackwardOffset=0,
                 input=None, timeOffset=0, name=''):
        super().__init__(inshpe, maxForwardOffset=maxForwardOffset,
                         maxBackwardOffset=maxBackwardOffset,
                         input=input, timeOffset=timeOffset, name=name)
        self.inputs = [np.zeros(self.inshpe)] * self.derivativeArrayLength
        self.optimizer = optimizer

    def calcDerivative(self):
        if self.inshpe != np.shape(self.optimizer.getError()):
            raise ValueError('optimizer and loss function error shapes don\'t match')
        if self.b == 0:
            return self.optimizer.getError()
        else:
            return np.zeros(self.inshpe)

    def getPredictions(self):
        return self.getInput(postForward=True)


class Sigmoid(Neuron):
    def __init__(self, inshpe, maxForwardOffset=0, maxBackwardOffset=0, input=None, timeOffset=0, name=''):
        super().__init__(inshpe=inshpe, outshpe=inshpe, maxForwardOffset=maxForwardOffset,
                         maxBackwardOffset=maxBackwardOffset, input=input, timeOffset=timeOffset,
                         name=name)
        self.inputs = [np.zeros(self.inshpe)] * self.derivativeArrayLength

    def calcOutput(self):
        return sigmoidFunc(self.input)

    def calcDerivative(self):
        input = self.getInput(postForward=True)
        sig = sigmoidFunc(input)
        diff = 1 - sig
        prod = np.multiply(sig, diff)
        dydx = np.multiply(self.backInput,prod)
        return dydx
# class Noramlize(Neuron):

class SoftMax(Neuron):
    # unfinished
    def __init__(self, inshpe, maxForwardOffset=0, maxBackwardOffset=0, input=None, timeOffset=0, name=''):
        super().__init__(inshpe=inshpe, outshpe=inshpe, maxForwardOffset=maxForwardOffset,
                         maxBackwardOffset=maxBackwardOffset, input=input, timeOffset=timeOffset,
                         name=name)
        self.inputs = [np.zeros(self.inshpe)] * self.derivativeArrayLength


    def calcOutput(self):
        return softmax(self.input)

    def calcDerivative(self):
        input = self.getInput(postForward=True)
        sx = softmax(input)
        s = sx.reshape(-1, 1)
        partial = np.diagflat(s) - np.dot(s, s.T)
        dydx = np.dot(self.backInput, partial)
        return dydx


class ProbArgMax(Neuron):
    # unfinished
    def __init__(self, inshpe, maxForwardOffset=0, maxBackwardOffset=0, input=None, timeOffset=0, name=''):
        super().__init__(inshpe, outshpe=inshpe, maxForwardOffset=maxForwardOffset,
                         maxBackwardOffset=maxBackwardOffset, input=input, timeOffset=timeOffset, name=name)
