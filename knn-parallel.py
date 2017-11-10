import csv, sys, random, math, operator, time, threading

class bcolors:
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

testSetSize = 0
wrongPredictionsNumber = 0

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

def get_accuracy(test_set, predictions):
  """ 
    Gets the accuracy of our knn algorithm
  """
  correct = 0
  for x in range(len(test_set)):
    if test_set[x][-1] == predictions[x]:
      correct += 1
  return str((correct/float(len(test_set))) * 100.0)

def make_prediction(testValue, trainingSet):
  k = 3
  neighbors = get_neighbors(trainingSet, testValue, k)
  return get_response(neighbors)

def print_result(predicted, actual, pos):
  space = ''
  maxSpacingSize = len(repr(testSetSize))
  actualSpacingSize = len(repr(pos))
  for i in range(maxSpacingSize - actualSpacingSize):
    space = space + '0'  	
  if repr(predicted) == repr(actual):
    print(space + repr(pos) + '> predicted=' + repr(predicted) + ', actual=' + repr(actual))
  else: 
    print(space + bcolors.WARNING + repr(pos) + '> predicted=' + repr(predicted) + ', actual=' + repr(actual) + bcolors.ENDC)	

def make_all_predictions(testSet,trainingSet):
  predictions = []
  # generate predictions
  for x in range(testSetSize):
    result = make_prediction(testSet[x], trainingSet)
    predictions.append(result)
    if repr(result) != repr(testSet[x][-1]):
      global wrongPredictionsNumber
      wrongPredictionsNumber += 1
    print_result(result, testSet[x][-1], x+1)
  return predictions
  
def main():
	
  # checkin command line arguments
  if(len(sys.argv) != 2):
	  print(bcolors.FAIL + 'Invalid command line arguments.\nMust be: <program_name> <thread_count>' + bcolors.ENDC)
	  sys.exit()
	  
  thread_number = int(float(sys.argv[1]))
  threaList = []
  	
  # prepare data
  trainingSet=[]
  testSet=[]
  ratio = 0.67
  load_data('iris.csv', ratio, trainingSet, testSet)
  global testSetSize
  testSetSize = len(testSet)
  
  start = time.time()
  predictions = make_all_predictions(testSet,trainingSet)
  acuracy = get_accuracy(testSet, predictions)
  
  end = time.time()
  print('Train set: ' + repr(len(trainingSet)))
  print('Test set: ' + repr(testSetSize))
  print('Wrong Predictions: ' + repr(wrongPredictionsNumber))
  print('Accuracy: ' + acuracy + '%')
  print('Elapsed Time: ' + repr(end-start))
	
main()
