import Network.Neurons as NN
import Network.Computegraph as CG
import numpy as np
import sys
import logging


def simpleUpdates():
    mF = 3
    mB = 7
    gd = CG.Optimizers.GD(lr=.01)
    testGraph = CG.Computegraph(maxForwardOffset=mF, maxBackwardOffset=mB)
    args = testGraph.getStdNeuronArguments()
    sampleInput = np.asarray([[1, 1]])
    l1 = NN.ConstantNeuron(inshpe=(1, 2), **args, name='L-1')
    l2 = NN.Affine(numHU=2, **args, inshpe=NN.calcInshpe(l1), input=l1, timeOffset=0,
                   name='L-2')
    l3 = NN.SoftMax(inshpe=NN.calcInshpe(l2), input=l2, **args, name='Max')
    l2.w = np.asarray([[0.5, 0.0], [0.5, 0.5]])
    l1.takeInputsFrom(l2, 1)
    l2.initOutput(sampleInput, 1)
    loss = NN.Loss(inshpe=NN.calcInshpe(l3), input=l3, optimizer=gd, **args, timeOffset=0, name='loss')
    testGraph.loss = loss
    testGraph.optimizer = gd
    testGraph.addNeuron(l1)
    testGraph.addNeuron(l2)
    print(l2.w)
    testGraph.computeForward()
    testGraph.computeForward()

    gd.stepsToUpdate = 1
    gd.clipping = np.asarray([-1, 1])
    output = l3.getOutput()
    target = np.zeros_like(output)
    target[0, 1] = 1
    target = target.astype(np.int)
    error = gd.cross_entropy(pred=output, target=target)
    gd.setError(error)
    testGraph.computeBackward()
    print(l2.w)


if __name__ == "__main__":
    np.random.seed(1)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    simpleUpdates()

# if __name__ == '__main__':
#     unittest.main()
