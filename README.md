# knn-parallel
A parallel K-Nearest Neighbors algorithm implementation in python.

## Prerequisites
- Any Python implementation. If speedup is a concern, GIL (Global Interpreter Lock) workaround is necessary. [Jython](http://www.jython.org/) recommended.

## Branches design

Master | Sequential | Parallel
------------ | ------------- | ------------- 
Dataset samples | Sequential knn version | Parallel knn version
Structure design | Sequential guide | Parallel guide
