import csv, sys, random, math, operator, time, threading

class bcolors:
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

test_set_size = 0
wrong_predictions = 0

"""
    Loads iris dataset (.csv) and randomly divide dataset into training and tests
    sets in order to perform the knn algorithm
"""
def load_data(filename, ratio, training_set=[], test_set=[]):
  with open(filename) as csvfile:
    lines = csv.reader(csvfile)
    dataset = list(lines)
    for x in range(len(dataset)-1):
      for y in range(9):
        dataset[x][y] = int(dataset[x][y])
        if random.random() < ratio:
          training_set.append(dataset[x])
        else:
          test_set.append(dataset[x])

""" 
    Calculates the euclidean distance between two given arrays 
    <first> and <second> of size <size> 
"""
def euclidean_distance(first, second, size):
  distance = 0
  for x in range(size):
    distance += (first[x] - second[x]) ** 2

  return math.sqrt(distance)

""" 
    Get <k> neighbors of <test_obj> analyzing the <training_set>
"""
def get_neighbors(training_set, test_obj, k):
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

""" 
    Extracts the appropriate class according to the neighbors
"""
def get_response(neighbors):
  class_votes = {}
  for x in range(len(neighbors)):
    response = neighbors[x][-1]
    if response in class_votes:
      class_votes[response] += 1
    else:
      class_votes[response] = 1
  sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
  return sorted_votes[0][0]

def make_prediction(test_value, training_set):
  k = 3
  neighbors = get_neighbors(training_set, test_value, k)
  return get_response(neighbors)

def print_result(predicted, actual, pos):
  space = ''
  max_spacing_size = len(repr(test_set_size))
  actual_spacing_size = len(repr(pos))
  for i in range(max_spacing_size - actual_spacing_size):
    space = space + '0'     
  if repr(predicted) == repr(actual):
    print(space + repr(pos) + '> predicted=' + repr(predicted) + ', actual=' + repr(actual))
  else: 
    print(space + bcolors.WARNING + repr(pos) + '> predicted=' + repr(predicted) + ', actual=' + repr(actual) + bcolors.ENDC) 

def sequential_knn(test_set, training_set):
  predictions = []
  # generate predictions
  wrong = 0
  right = 0
  for x in range(test_set_size):
    result = make_prediction(test_set[x], training_set)
    predictions.append(result)
    if repr(result) != repr(test_set[x][-1]):
      global wrong_predictions
      wrong_predictions += 1
    print_result(result, test_set[x][-1], x+1)
  
  return predictions
  
def main():
  
  # prepare data
  training_set=[]
  test_set=[]
  ratio = 0.4
  load_data('breast-cancer-wisconsin.csv', ratio, training_set, test_set)
  global test_set_size
  test_set_size = len(test_set)
  
  start = time.time()
  predictions = sequential_knn(test_set,training_set)
  end = time.time()

  print('Training set: ' + repr(len(training_set)))
  print('Test set: ' + repr(test_set_size))
  print('Wrong Predictions: ' + repr(wrong_predictions))
  print('Right Predictions: ' + repr(test_set_size - wrong_predictions))
  print('Accuracy: ' + ((test_set_size - wrong_predictions)/float(test_set_size)) + '%')
  print('Elapsed Time: ' + repr(end-start))
  
main()