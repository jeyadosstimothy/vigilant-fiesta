from py2deepdsl import *

layers = [
            Input('mnist', shape=(1, 28, 28), batchSize=512),
            Convolv(kernelSize=5, numKernels=20),
            MaxPool(kernelSize=2),
            Convolv(kernelSize=5, numKernels=20),
            MaxPool(kernelSize=2),
            Flatten(numDims=4, cuts=1),
            Full(numNodes=500, activation='relu'),
            Full(numNodes=10, activation='softmax')
        ]

model = Model(layers=layers)
model.compile(decay=0.0005, momentum=0.1, learnrate=0.01, loss='categorical_crossentropy', epochs=50)
