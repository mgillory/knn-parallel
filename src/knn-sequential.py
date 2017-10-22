import csv
import random
import math
import operator

def load_data(filename, ratio, training_set=[], test_set=[]):
  """
    Loads iris dataset (.csv) and randomly divide dataset into training and tests
    sets in order to perform the knn algorithm
  """
  with open(filename) as csvfile:
    lines = csv.reader(csvfile)
    dataset = list(lines)
    for x in range(len(dataset)-1):
      for y in range(4):
        dataset[x][y] = float(dataset[x][y])
        if random.random() < ratio:
          training_set.append(dataset[x])
        else:
          test_set.append(dataset[x])

def euclidean_distance(first, second, size):
  """ 
    Calculates the euclidean distance between two given arrays 
    <first> and <second> of size <size> 
  """
  distance = 0
  for x in range(size):
    distance += (first[x] - second[x]) ** 2

  return math.sqrt(distance)

def get_neighbors(training_set, test_obj, k):
  """ 
    Get <k> neighbors of <test_obj> analyzing the <training_set>
  """
  distances = []
  length = len(test_obj)-1
  for x in range(len(training_set)):
	  dist = euclidean_distance(test_obj, training_set[x], length)
	  distances.append((training_set[x], dist))

  distances.sort(key = operator.itemgetter(1))
  neighbors = []
  for x in range(k):
	  neighbors.append(distances[x][0])
  return neighbors

def get_response(neighbors):
  """ 
    Extracts the appropriate class according to the neighbors
  """
  class_votes = {}
  for x in range(len(neighbors)):
	  response = neighbors[x][-1]
	  if response in class_votes:
		  class_votes[response] += 1
	  else:
		  class_votes[response] = 1
  sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
  return sorted_votes[0][0]

trainingSet=[]
testSet=[]
ratio = 0.67
load_data('iris.csv', ratio, trainingSet, testSet)
print('Train set: ' + repr(len(trainingSet)))
print('Test set: ' + repr(len(testSet)))

k = 3
for x in range(len(testSet)):
  neighbors = get_neighbors(trainingSet, testSet[x], k)
  result = get_response(neighbors)
  print(x)
  print(result)
