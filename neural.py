import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.models import load_model
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def winnerValuesTreatment(winners):
    treated = list()
    for item in winners:
        if item == -1:
            treated.append([0,1]) 
            
        else:
            treated.append([1,0])
    return treated

def neuralNetwork():
    #dados de treinamento
    trainDf = pd.read_csv("data/dota2Train.csv", ",")
    trainWinners = np.array(winnerValuesTreatment(trainDf.winnerTeam))
    trainHeroPicks = np.array(trainDf.values)
    trainHeroPicks = trainHeroPicks[:,4:]

    #dados de teste
    testDf = pd.read_csv("data/dota2Test.csv", ",")
    testWinners = np.array(winnerValuesTreatment(testDf.winnerTeam))
    testHeroPicks = np.array(testDf.values)
    testHeroPicks = testHeroPicks[:,4:]

    
    model = models.Sequential()
    # Input - Layer
    model.add(layers.Dense(1024, activation = "relu", input_shape=(113, )))
    # Hidden - Layers
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(512, activation = "softmax"))
    model.add(layers.Dense(512, activation = "relu"))
    model.add(layers.Dense(512, activation = "relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(256, activation = "relu"))
    # Output- Layer
    model.add(layers.Dense(2, activation = "softmax"))
    
    
    #model.summary()

    #compiling the model
    #model.compile(
    #optimizer = "adam",
    #loss = "categorical_crossentropy",
    #metrics = ["accuracy"]
    #)

    model = load_model('models/categoricalModel.h5')

    fitResult = model.fit(trainHeroPicks, trainWinners, epochs=1, batch_size=1024, verbose=1)

    predicted = model.predict(testHeroPicks)
    print(predicted)
    
    model.save('models/categoricalModel.h5')

    predict = list()
    test_labels = list()
    for value in predicted:
        predict.append(0 if value[0] < value[1] else 1)
    for value in testWinners:
        test_labels.append(0 if value[0] < value[1] else 1)


    mat = confusion_matrix(test_labels, predict)
    plt.figure(figsize=(8, 8))
    sns.set()
    sns.heatmap(mat.T, square=True,
                xticklabels=np.unique(test_labels),
                yticklabels=np.unique(test_labels))
    # annot=True,
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    # Save confusion matrix to outputs in Workbench
    # plt.savefig(os.path.join('.', 'outputs', 'confusion_matrix.png'))
    print(mat)
    plt.show()

    return predicted

