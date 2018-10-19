import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
import pandas as pd

#dados de treinamento
trainDf = pd.read_csv("data/dota2Train.csv", ",")
trainWinners = trainDf.winnerTeam
trainHeroPicks = np.array(trainDf.values)
trainHeroPicks = trainHeroPicks[:,4:]

#dados de teste
testDf = pd.read_csv("data/dota2Test.csv", ",")
testWinners = testDf.winnerTeam
testHeroPicks = np.array(testDf.values)
testHeroPicks = testHeroPicks[:,4:]

 
model = models.Sequential()
# Input - Layer
model.add(layers.Dense(50, activation = "relu", input_shape=(113, )))
# Hidden - Layers
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
# Output- Layer
model.add(layers.Dense(1, activation = "sigmoid"))

model.summary()

# compiling the model
model.compile(
 optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"]
)

fitResult = model.fit(trainHeroPicks, trainWinners, epochs=10, batch_size=32, verbose=1)

predicted = model.predict(testHeroPicks)
print(predicted)
