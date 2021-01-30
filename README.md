# COMP3308
Coursework for COMP3308: Introduction to Artificial Intelligence class at University of Sydney

## Assignment 1 - Search Methods
Program that outputs the path from A to B based on a chosen search algorithm.<br>
Points A and B are defined by 3 digit numbers, 000-999. One step contitutes a move of ±1 along one axis. <br>
It is possible to exclude given points from the path.<br>

### Input
Input consists of a search algorithm and a text file specifing points A and B as well as points to be ommited. <br>
#### Algorithm abbreviations
1. B - Breadth First Search
2. D - Depth First Search
3. I - Iterative Deepening Search
4. G - Greedy
5. A - A*
6. H - Hill-Climbing
#### Sample text file
```
111 #start
789 #goal
222,443,112 #omit
```

### How to run
```console
(base) AnetasMacBook2: python ThreeDigits.py [SEARCH ALGORITHM] [DOCUMENT]
```
#### Example
```console
(base) AnetasMacBook2: python ThreeDigits.py A test_input.txt
```

## Assignment 2 - Classification
Program classifing samples using a chosen algorithm. It takes a training set with class labels and a new/testing without labels. It outputs classifications for the second set. It can be used for classification and validation.

### Input
Input consists of two csv files without headers. The first is the training file, with final column determining class. The second file has all parameter columns and no class column.<br>

#### Algorithm abbreviations
1. k-NN - K-Nearest Neighbours, where k is an integer
2. NB - Naive Bayes

### How to run
```console
(base) AnetasMacBook2: python MyClassifier.py [TRAINING SET] [UNCATEGORIZED SET] [ALGORITHM]
```
#### Example
```console
(base) AnetasMacBook2: python MyClassifier.py train.csv new.csv 5-NN
```
