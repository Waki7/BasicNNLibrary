import numpy as np
from Network import Neurons


class Neuron(object):
    def __init__(self, inshpe, outshpe=None, maxForwardOffset=0,
                 maxBackwardOffset=0, input=None, timeOffset=0, name=''):
        # time offset from envInputs, used for forward prop to get envInputs from correct index of incoming neurons
        self.inTimeOffset = []
        self.inEdges = []
        self.inEdgesDataIndex = [0]  # keep track of inedges locations in input vector (0 - end of array)

        self.outTimeOffset = []
        self.outEdges = []
        self.outEdgesMappedIndex = []  # keeps track output index in input of each out edge

        self.maxForwardOffset = maxForwardOffset
        self.outputArrayLength = maxForwardOffset + 1
        self.forwardQ = 0
        self.forwardQCapacity = 0
        self.t = 0

        self.maxBackwardOffset = maxBackwardOffset
        self.derivativeArrayLength = maxBackwardOffset + 1
        self.backwardQ = 0
        self.backwardQCapacity = 0
        self.b = 0

        # self.defaultOutput
        # self.defaultInput
        # self.defaultDeriv

        self.name = name
        self.inshpe = inshpe
        self.input = np.zeros(self.inshpe)
        self.outshpe = outshpe
        self.backInput = np.zeros(self.outshpe)
        self.derivatives = [np.zeros(self.inshpe)] * self.derivativeArrayLength
        self.outputs = [np.zeros(self.outshpe)] * self.outputArrayLength
        self.inputs = None
        if input is not None:
            self.takeInputsFrom(n=input, timeOffset=timeOffset)

    def reset(self):
        #as of now, only resets time and q's
        self.t = 0
        self.b = 0
        self.forwardQ = self.forwardQCapacity
        self.backwardQ = self.backwardQCapacity

    def getNeurons(self):
        return self,

    def calcOutputOffset(self, timeOffset):
        # this method is always called form the later neuron
        return (self.t - timeOffset + self.maxForwardOffset) % self.outputArrayLength

    def calcBackPropOffset(self, timeOffset):
        # this method is always called from the earlier neuron
        return (-self.b + timeOffset + self.maxBackwardOffset) % self.derivativeArrayLength

    def calcInputOffset(self, timeOffset):
        # if being called post forward pass then set timeoffset to 1
        return (self.t - timeOffset - self.b + self.maxBackwardOffset) % self.derivativeArrayLength

    def addToInputs(self):
        if self.inputs is not None:
            self.inputs[self.calcInputOffset(0)] = self.input

    def getInput(self, postForward=False):
        if self.inputs is not None:
            return self.inputs[self.calcInputOffset(postForward)]
        else:
            return self.input

    def takeInputsFrom(self, n, timeOffset):
        if isinstance(n, tuple) or isinstance(n, list):
            if isinstance(timeOffset, tuple):
                for i in range(0, len(n)):
                    self.takeInputFrom(n=n[i], timeOffset=timeOffset[i])
            else:
                for i in n:
                    self.takeInputFrom(n=i, timeOffset=timeOffset)
        else:
            if isinstance(timeOffset, tuple):
                for i in timeOffset:
                    self.takeInputFrom(n=n, timeOffset=i)
            else:
                self.takeInputFrom(n=n, timeOffset=timeOffset)

    def takeInputFrom(self, n, timeOffset):
        # allows node coming in to know what index it's in
        n.outEdgesMappedIndex.append(len(self.inEdges))
        self.inEdges.append(n)
        # keep track of what index the start of each incoming edge's output will be at
        self.inEdgesDataIndex.append(self.inEdgesDataIndex[-1] + n.outshpe[1])
        n.outEdges.append(self)
        n.outTimeOffset.append(timeOffset)
        self.inTimeOffset.append(timeOffset)
        if timeOffset == 0:
            self.forwardQCapacity += 1
            self.forwardQ += 1
            n.backwardQCapacity += 1
            n.backwardQ += 1

    def forward(self):
        self.forwardQ = self.forwardQCapacity
        for i in range(0, len(self.inEdges)):
            inEdge = self.inEdges[i]
            partialInput = inEdge.getPartialOutput(self, i)
            self.updateInput(partialInput, i)
        readyToCompute = []
        for i in range(0, len(self.outEdges)):
            outEdge = self.outEdges[i]
            timeOffset = self.outTimeOffset[i]
            if timeOffset == 0:
                outEdge.forwardQ -= 1
            if outEdge.forwardQ == 0:
                readyToCompute.append(outEdge)
        self.addToInputs()
        output = self.calcOutput()
        self.setOutput(output)
        self.t += 1
        return readyToCompute

    def getPartialOutput(self, output, index):
        outputIndex = output.calcOutputOffset(output.inTimeOffset[index])
        Neurons.logger.debug(
            '     >index from output array {} ___ input :  {} coming in from  {}'.format(outputIndex,
                                                                                         self.outputs[outputIndex],
                                                                                         self.name))
        return self.outputs[outputIndex]

    def calcOutput(self):
        return self.input

    def setOutput(self, val):
        if val is not None:
            self.outputs[self.calcOutputOffset(0)] = val

    def updateInput(self, partialInput, index):
        self.input[:, self.inEdgesDataIndex[index]:self.inEdgesDataIndex[index + 1]] = partialInput

    def backward(self, optimizer):
        self.backInput = np.zeros(self.outshpe)
        for i in range(0, len(self.outEdges)):
            if self.outTimeOffset[i] <= self.b:
                outEdge = self.outEdges[i]
                partialDerivative = outEdge.getPartialDerivative(self, i)
                self.updatePartial(partialDerivative)
        readyToCompute = []
        for i in range(0, len(self.inEdges)):
            inEdge = self.inEdges[i]
            timeOffset = self.inTimeOffset[i]
            if timeOffset == 0:
                inEdge.backwardQ -= 1
            if inEdge.backwardQ == 0:
                readyToCompute.append(inEdge)
        derivative = self.calcDerivative()
        derivative = optimizer.clip(derivative)
        self.setDerivative(derivative)
        self.incrementalUpdate(optimizer)
        self.backwardQ = self.backwardQCapacity
        self.b += 1
        return readyToCompute

    def getPartialDerivative(self, input, index):
        timeIndex = input.calcBackPropOffset(input.outTimeOffset[index])
        inputIndex = input.outEdgesMappedIndex[index]
        partialDerivative = self.derivatives[timeIndex][:,
                            self.inEdgesDataIndex[inputIndex]:self.inEdgesDataIndex[inputIndex + 1]]
        Neurons.logger.debug(
            '     >index from derivatives array {} , backinput :  {} coming in from  {}'.format(timeIndex,
                                                                                                partialDerivative,
                                                                                                self.name))
        return partialDerivative

    def calcDerivative(self):
        return self.backInput

    def setDerivative(self, val):
        if val is not None:
            self.derivatives[self.calcBackPropOffset(0)] = val

    def updatePartial(self, partialDerivative):
        self.backInput += partialDerivative


    def incrementalUpdate(self, optimizer):
        pass

    def updateParameters(self, optimizer=None):
        self.b = 0


    def getDerivative(self):
        index = self.calcBackPropOffset(1)
        ret = self.derivatives[index]
        return ret

    def getOutput(self):
        index = self.calcOutputOffset(1)
        ret = self.outputs[index]
        return ret

    def initOutput(self, val, timeOffset):
        if val is not None:
            self.outputs[self.calcOutputOffset(timeOffset)] = val
