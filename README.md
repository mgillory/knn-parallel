# knn-parallel
A parallel K-Nearest Neighbors algorithm implementation in python.

## Prerequisites
- Python. 

Implemented in Python 2.7.13. If speedup is a concern, GIL (Global Interpreter Lock) workaround is necessary. [Jython](http://www.jython.org/) recommended.

## Branches design

Master | Sequential | Parallel
------------ | ------------- | ------------- 
Dataset samples | Sequential knn version | Parallel knn version
Project structure | Sequential guide | Parallel guide

More datasets: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.html)

## License
[MIT](https://github.com/mgillory/knn-parallel/blob/master/LICENSE)
