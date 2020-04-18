# import unittest
from Network import Neurons as NN
from Network import Computegraph as CG
from RLTasks.LifeSim import Simulation as Sim
import numpy as np
import sys
import logging



# class TestStringMethods(unittest.TestCase):
#     def setUp(self):
#         mF = 3
#         mB = 7
#         self.testGraph = CG.Computegraph(maxForwardOffset=mF, maxBackwardOffset=mB)
#
#     def test_upper(self):
#         self.assertEqual('foo'.upper(), 'FOO')
#
#     def test_isupper(self):
#         self.assertTrue('FOO'.isupper())
#         self.assertFalse('Foo'.isupper())
#
#     def test_split(self):
#         s = 'hello world'
#         self.assertEqual(s.split(), ['hello', 'world'])
#         # check that s.split fails when the separator is not a string
#         with self.assertRaises(TypeError):
#             s.split(2)

def testingSimpleUpdates():
    mF = 3
    mB = 7
    gd = CG.Optimizers.GD(lr=.01)
    testGraph = CG.Computegraph(maxForwardOffset=mF, maxBackwardOffset=mB)
    args = testGraph.getStdNeuronArguments()
    sampleInput = np.asarray([[1, 1]])
    l1 = NN.ConstantNeuron(inshpe=(1,2), **args, name='L-1')
    l2 = NN.Affine(numHU=2, **args, inshpe=NN.calcInshpe(l1), input=l1, timeOffset=0,
                              name='L-2')
    l3 = NN.MaxOneHotEncoding(inshpe=NN.calcInshpe(l2), input = l2, **args, name='Max')
    l2.w = np.asarray([[0.5, 0.0], [0.5, 0.5]])
    l1.takeInputsFrom(l2, 1)
    l2.initOutput(sampleInput, 1)
    loss = NN.Loss(inshpe=NN.calcInshpe(l3), input=l3, **args, timeOffset=0, name='loss')
    testGraph.loss = loss
    testGraph.optimizer = gd
    testGraph.addNeuron(l1)
    testGraph.addNeuron(l2)
    print(l2.w)
    testGraph.computeForward()
    testGraph.computeForward()

    gd.stepsToUpdate=1
    gd.clipping=(-1,1)
    testGraph.computeBackward(stepsToUpdate=2)
    print(l2.w)


def testing():
    mF = 3
    mB = 7
    gd = CG.Optimizers.GD(lr=.01)
    testGraph = CG.Computegraph(maxForwardOffset=mF, maxBackwardOffset=mB)
    args = testGraph.getStdNeuronArguments()
    sampleInput = np.asarray([[1, 1, 1, 1]])
    sampleInput2 = np.asarray([[2, 2, 2, 2]])

    l1 = NN.ConstantNeuron(inshpe=np.shape(sampleInput), input=sampleInput, **args, name='L-1')
    l1.setInput(sampleInput2, 1)
    l2 = NN.AffRelSplitNeuron(numAff=2, numRel=2, **args, inshpe=NN.calcInshpe(l1, 2 * 2 * 2), input=(l1), timeOffset=2,
                           name='L-2')

    l2.takeInputsFrom(l2, 1)
    l2.takeInputsFrom(l2, 2)
    l3 = NN.AffRelSplitNeuron(numAff=2, numRel=2, **args, inshpe=NN.calcInshpe((l1, l2)), input=(l1, l2), timeOffset=0,
                           name='L-3')

    # splitter = NeuronSplitter(inshpe=calcInshpe((l1, l2)), input=(l1, l2),
    #                           splits=[5], **args, name='L-split')
    # sm1 = SoftMax(**args, inshpe=calcInshpe(splitter.getSegment(0)),
    #               input=splitter.getSegment(0), timeOffset=0,
    #               name='L-SM1')
    # sm2 = SoftMax(**args, inshpe=calcInshpe(splitter.getSegment(1)),
    #               input=splitter.getSegment(1),
    #               timeOffset=0,
    #               name='L-SM2')

    loss = NN.Loss(inshpe=NN.calcInshpe((l3, l2)), input=(l3, l2), **args, timeOffset=0, name='loss')
    testGraph.loss = loss

    l2.printWeights()
    l3.printWeights()

    print('-----')
    testGraph.addNeuron(l1)
    testGraph.addNeuron(l2)
    testGraph.addNeuron(l3)
    testGraph.addNeuron(loss)

    print(testGraph.neuronDict['L-3.Affine'].derivatives[7])

    # testGraph.addNeuron(splitter)
    # testGraph.addNeuron(sm1)
    # testGraph.addNeuron(sm2)

    testGraph.computeForward()
    testGraph.computeForward()
    testGraph.computeBackward()


def testingSimple():
    pass

if __name__ == "__main__":
    np.random.seed(1)
    print(Sim.channels)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    testingSimpleUpdates()

# if __name__ == '__main__':
#     unittest.main()