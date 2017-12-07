# knn-parallel
A parallel K-Nearest Neighbors algorithm implementation in python.

## Command line arguments
```
<program name> <dataset location> <number of colums> <ratio> <number of neighbors> <thread number>
```
- **program name**: script name.
- **dataset location**: full location of dataset.
- **number of colums**: total number of dataset columns minus class column. Example:
```
column 1,column 2,column 3,class column
```
In this case would be 3.
- **ratio**: decimal number to divide training and test set. Range from [0-1].
- **number of neighbors**: integer number of neighbors.
- **thread number**: integer number of threads.

### Running example
#### [Jython](http://www.jython.org/)
```
java -jar jython.jar knn-parallel.py /fullpath/dataset.csv 5 0.5 3 4
```
#### [Python](https://www.python.org/)
```
python knn-parallel.py /fullpath/dataset.csv 5 0.5 3 4
```
#### [Iron python](http://ironpython.net/)
```
ipy knn-parallel.py /fullpath/dataset.csv 5 0.5 3 4
