import csv
import random

# load dataset (.csv) file
def loadData(filename, split, trainingSet=[] , testSet=[]):
  with open(filename) as csvfile:
    # csv.reader object
    lines = csv.reader(csvfile)
    # each csv line turns into an array
    dataset = list(lines)
    for x in range(len(dataset)-1):
      for y in range(4):
        # converting string -> float in order to work in knn
        dataset[x][y] = float(dataset[x][y])
        # randomly divide dataset into train and test according to given ratio
        if random.random() < split:
          trainingSet.append(dataset[x])
        else:
          testSet.append(dataset[x])

trainingSet=[]
testSet=[]
loadData('iris.csv', 0.66, trainingSet, testSet)
# sum / 4 -> csv lines
print('Train: ' + repr(len(trainingSet)))
print('Test: ' + repr(len(testSet)))