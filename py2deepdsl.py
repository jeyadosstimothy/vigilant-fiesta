from subprocess import call as syscall
from os import chdir

class CodeGenerator:
    def __init__(self):
        self.code = ''

    def setDimensions(self):
        self.batchSize = self.inputs.batchSize
        self.numClasses = self.layers[-1].numNodes
        self.dims = list(map(str, self.inputs.shape))

    def setNetwork(self, layers):
        self.inputs = layers[0]
        self.layers = layers[1:]
        self.setDimensions()

    def setDecay(self, decay):
        self.decay = decay

    def setMomentum(self, momentum):
        self.momentum = momentum

    def setLearnRate(self, learnrate):
        self.learnrate = learnrate

    def setLoss(self, loss):
        self.loss = loss

    def setTrainEpochs(self, epochs):
        self.trainEpochs = epochs

    def generateImports(self):
        code = '''
        package deepdsl.derivation

        import deepdsl.analysis._
        import deepdsl.ast.{Lmdb, Mnist, _}
        import deepdsl.derivation.MemoryAnalysis._
        import deepdsl.lang._
        import deepdsl.layer._
        import deepdsl.run._
        import org.junit.Test
        '''

        return code

    def generateDimensions(self):
        N = '''
                val N = batch_size'''.format(batchSize=self.batchSize)
        dims = '''
                val dim = List(N, ''' + ', '.join(self.dims) + ''')'''
        K = '''
                val K = {numClasses}'''.format(numClasses=self.numClasses)

        return N + dims + K

    def generateDataset(self):
        dataset = self.inputs.dataset.strip().lower()
        datasetCode = ''
        if dataset == 'mnist':
            datasetCode = '''

                val dataset = Mnist(dim)
                val y = T._new("Y", List(N))
                val x = T._new("X", dim)
            '''
        return datasetCode

    def generateNetwork(self):
        visited = {}
        layerCode = ''
        networkCode = []

        for layer in self.layers:
            if type(layer) is Full:
                if 'full' in visited:
                    visited['full'] += 1
                else:
                    visited['full'] = 0
                count = visited['full']
                numNodes = layer.numNodes
                layerCode += '''
                val fc{count} = Layer.full("fc{count}", {numNodes})'''.format(count=count, numNodes=numNodes)
                networkCode.append('fc{count}'.format(count=count))
            elif type(layer) is Convolv:
                if 'conv' in visited:
                    visited['conv'] += 1
                else:
                    visited['conv'] = 0
                count = visited['conv']
                kernelSize = layer.kernelSize
                numKernels = layer.numKernels # CudaLayer.convolv("cv1", 5, 20)
                layerCode += '''
                val cv{count} = CudaLayer.convolv("cv{count}", {kernel}, {numNodes})'''.format(count=count, kernel=kernelSize, numNodes=numKernels)
                networkCode.append('cv{count}'.format(count=count))

            elif type(layer) is MaxPool:
                if 'mp' in visited:
                    visited['mp'] += 1
                else:
                    visited['mp'] = 0
                count = visited['mp']
                kernelSize = layer.kernelSize
                layerCode += '''
                val mp{count} = CudaLayer.max_pool({kernel})'''.format(count=count, kernel=kernelSize)
                networkCode.append('mp{count}'.format(count=count))

            elif type(layer) is Flatten:
                if 'fl' in visited:
                    visited['fl'] += 1
                else:
                    visited['fl'] = 0
                count = visited['fl']
                numDims = layer.numDims
                cuts = layer.cuts
                layerCode += '''
                val fl{count} = Layer.flatten({numDims}, {cuts})'''.format(count=count, numDims=numDims, cuts=cuts)
                networkCode.append('fl{count}'.format(count=count))

            if layer.activation and layer.activation.strip().lower() == 'relu':
                if type(layer) is Convolv:
                    if 'relu4' not in visited:
                        visited['relu4'] = True
                        layerCode += '''
                val relu4 = CudaLayer.relu(4)'''
                    networkCode.append('relu4')
                else:
                    if 'relu2' not in visited:
                        visited['relu2'] = True
                        layerCode += '''
                val relu2 = CudaLayer.relu(2)'''
                    networkCode.append('relu2')
            elif layer.activation and layer.activation.strip().lower() == 'softmax':
                if 'softmax' not in visited:
                    visited['softmax'] = True
                    layerCode += '''
                val softmax = CudaLayer.softmax'''
                networkCode.append('softmax')

        networkCode.reverse()
        networkCode = '''

                val network = ''' + ' o '.join(networkCode)

        return layerCode + networkCode

    def generateLossFunction(self):
        code = '''
                println(typeof(network))

                val xCuda = x.asCuda
                val yCuda = y.asIndicator(K).asCuda'''
        if self.loss == 'log_loss' or self.loss == 'categorical_crossentropy':
            code += '''
                val loss = (Layer.log_loss(yCuda) o network)(xCuda)'''
        code += '''
                val accuracy = network(xCuda)'''

        return code

    def generateSolver(self):
        code = '''
                val param = loss.freeParam.toList
                val solver = Train(name, train_iter, test_iter, learn_rate, momentum, decay, 0)
                val loop = Loop(loss, accuracy, dataset, (x, y), param, solver)

                runtimeMemory(loop.train.lst)
                parameterMemory(param, momentum)
                CudaCompile(path).print(loop)'''

        return code

    def generateNetworkFunction(self):
        code = ('''
            private def mynet(batch_size: Int, learn_rate: Float, momentum: Float, decay: Float, train_iter: Int, test_iter: Int, name: String) {
            '''
            + self.generateDimensions()
            + self.generateDataset()
            + self.generateNetwork()
            + self.generateLossFunction()
            + self.generateSolver()
            + '''
            }''')

        return code

    def testCode(self):
        code = '''
            @Test
            def testMyNet = mynet({batchSize}, {learnrate}f, {momentum}f, {decay}f, {trainEpochs}, {testEpochs}, "MyNet")
            '''.format(batchSize=self.batchSize, learnrate=self.learnrate, momentum=self.momentum, decay=self.decay, trainEpochs=self.trainEpochs, testEpochs=1)

        return code

    def generateClass(self):
        self.code = (self.generateImports()
        + '''
        class TestNetworkNew {
            val path = "deepdsl/gen"
        '''
        + self.generateNetworkFunction()
        + self.testCode()
        + '''
        }
        ''')

    def compile(self):
        self.generateClass()
        with open('deepdsl/deepdsl-java/src/test/scala/deepdsl/derivation/TestNetworkNew.scala', 'w+') as f:
            f.write(self.code)


class Model:
    def __init__(self, layers):
        self.layers = layers
        self.codeGenerator = CodeGenerator()

    def compile(self, decay, momentum, learnrate, loss, epochs):
        self.codeGenerator.setNetwork(self.layers)
        self.codeGenerator.setDecay(decay)
        self.codeGenerator.setMomentum(momentum)
        self.codeGenerator.setLearnRate(learnrate)
        self.codeGenerator.setLoss(loss)
        self.codeGenerator.setTrainEpochs(epochs)
        self.codeGenerator.compile()
        print('Generated DeepDSL code...')
        chdir('deepdsl/deepdsl-java')
        syscall(['mvn', '-Plinux64', 'clean'])
        syscall(['mvn', '-Plinux64', 'test', '-Dtest=TestNetworkNew#testMyNet'])

class Layer:
    def __init__(self, numNodes, activation):
        self.numNodes = numNodes
        self.activation = activation

class Input(Layer):
    def __init__(self, dataset, shape, batchSize=16):
        self.dataset = dataset
        self.shape = shape
        self.batchSize = batchSize

class Full(Layer):
    pass

class Convolv(Layer):
    def __init__(self, kernelSize, numKernels, activation=None):
        self.numKernels = numKernels
        self.kernelSize = kernelSize
        self.activation = activation

class MaxPool(Layer):
    def __init__(self, kernelSize, activation=None):
        self.kernelSize = kernelSize
        self.activation = activation

class Flatten(Layer):
    def __init__(self, numDims, cuts, activation=None):
        self.numDims = numDims
        self.cuts = cuts
        self.activation = activation
