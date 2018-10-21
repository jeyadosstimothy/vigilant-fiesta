from py2deepdsl import *

layers = [
            Input('mnist', shape=(1, 28, 28), batchSize=256),
            Convolv(kernelSize=3, numKernels=64, activation='relu'),
            Convolv(kernelSize=3, numKernels=64, activation='relu'),
            MaxPool(kernelSize=2),
            Convolv(kernelSize=3, numKernels=128, activation='relu'),
            Convolv(kernelSize=3, numKernels=128, activation='relu'),
            MaxPool(kernelSize=2),
            Convolv(kernelSize=3, numKernels=256, activation='relu'),
            Convolv(kernelSize=3, numKernels=256, activation='relu'),
            Convolv(kernelSize=3, numKernels=256, activation='relu'),
            MaxPool(kernelSize=2),
            Convolv(kernelSize=3, numKernels=512, activation='relu'),
            Convolv(kernelSize=3, numKernels=512, activation='relu'),
            Convolv(kernelSize=3, numKernels=512, activation='relu'),
            MaxPool(kernelSize=2),
            Flatten(numDims=4, cuts=1),
            Full(numNodes=4096, activation='relu'),
            Full(numNodes=4096, activation='relu'),
            Full(numNodes=10, activation='softmax')
        ]

model = Model(layers=layers)
model.compile(decay=0.0005, momentum=0.1, learnrate=0.01, loss='categorical_crossentropy', epochs=10)
