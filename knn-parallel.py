from threading import Thread
import csv, sys, random, math, operator, time

class bcolors:
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

predictions = {}
predictions_index = 0
thread_number = 0
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
        dataset[x][y] = float(dataset[x][y])
        if random.random() < ratio:
          training_set.append(dataset[x])
        else:
          test_set.append(dataset[x])

""" 
  Calculates the euclidean distance between two given arrays 
  <first> and <second> of length <size> 
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

""" 
  Gets the accuracy of our knn algorithm
"""
def get_accuracy(test_set):
  correct = 0
  for x in range(len(test_set)):
    if test_set[predictions[x][0]][-1] == predictions[x][1]:
      correct += 1
  return ((correct/float(len(test_set))) * 100.0)

def make_prediction(testValue, training_set):
  k = 3
  neighbors = get_neighbors(training_set, testValue, k)
  return get_response(neighbors)

def print_result(predicted, actual, pos):
  space = ''
  maxSpacingSize = len(repr(test_set_size))
  actualSpacingSize = len(repr(pos))
  for i in range(maxSpacingSize - actualSpacingSize):
    space = space + '0'  	
  if repr(predicted) == repr(actual):
    print(space + repr(pos) + '> predicted=' + repr(predicted) + ', actual=' + repr(actual))
  else: 
    print(space + bcolors.WARNING + repr(pos) + '> predicted=' + repr(predicted) + ', actual=' + repr(actual) + bcolors.ENDC)	

def parallel_knn(test_set,training_set, current_thread, test_set_size):
  # parallelization divide logic
  block_size = int(test_set_size / thread_number)
  index = int(block_size * current_thread)
  end = index + block_size
  wrong = 0
  if((current_thread == thread_number - 1) and (test_set_size % thread_number != 0)):
    end += test_set_size % thread_number
  # generate predictions
  while index < end:
    result = make_prediction(test_set[index], training_set)
    global predictions, predictions_index
    predictions[predictions_index] = [index, result]
    predictions_index += 1
    if repr(result) != repr(test_set[index][-1]):
	  global wrong_predictions
	  wrong_predictions += 1
    
    #print_result(result, test_set[index][-1], index+1)
    index += 1
  
  #wrong_predictions = wrong
  
def main():
	
  # checkin command line arguments
  if(len(sys.argv) != 2):
	  print(bcolors.FAIL + 'Invalid command line arguments.\nMust be: <program_name> <thread_count>' + bcolors.ENDC)
	  sys.exit()
	
  global thread_number
  thread_number = int(float(sys.argv[1]))
  threadList = []
  	
  # prepare data
  training_set = []
  test_set = []
  ratio = 0.2
  load_data('breast-cancer-wisconsin.csv', ratio, training_set, test_set)
  global test_set_size
  test_set_size = len(test_set)
  threadList = [] 
  
  start = time.time()
  for i in range(thread_number):
    thread = Thread(target=parallel_knn,args=(test_set, training_set, i, test_set_size))
    thread.start()
    threadList.append(thread)
  for thread in threadList:  
    thread.join()	    
  end = time.time()

  #accuracy = get_accuracy(test_set)
  
  print('\nTrain set: ' + repr(len(training_set)))
  print('Test set: ' + repr(test_set_size))
  print('Wrong Predictions: ' + repr(wrong_predictions))
  print('Correct Predictions: ' + repr(test_set_size - wrong_predictions))
  #print('Accuracy: ' + repr(accuracy) + '%')
  print('Elapsed Time: ' + repr(end-start))
	
main()
