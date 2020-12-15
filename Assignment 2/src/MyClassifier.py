import re
import sys
import numpy as np
import math

def main():
    # input
    with open(sys.argv[1]) as raw_train:
        train = read_train(raw_train)

    
    with open(sys.argv[2]) as raw_test:
        test = read_test(raw_test)
    

    classifier = sys.argv[3]
    # output
    
    if bool(re.search(r'NB', classifier, re.IGNORECASE)) :
        for element in test:
            print(nb(train, element))

    elif bool(re.search(r'\d+NN', classifier, re.IGNORECASE)):
        k = int(classifier[:-2])
        for i in range(len(test)):
            print(knn(k,train,test[i]))
    else:
        print("Classifier not recognized. Use Naive Bayes (NB) or k-Nearest Neighbours(kNN)")

def read_train(raw_data): #converts csv into a list of lists holding the data
    lines = raw_data.read().splitlines()
    lines = [line.split(",") for line in lines]
    lines = list(map(lambda line: list(map(float, line[:-1])) + [line[-1]], lines))
    return lines

def read_test(raw_data): #converts csv into a list of lists holding the data
    lines = raw_data.read().splitlines()
    lines = [line.split(",") for line in lines]
    for i in range(len(lines)):
        lines[i] = [float(j) for j in lines[i]]
    return lines


def nb(train, test): # Naive Bayes function
    # split on class, assume binary classification
    train_y = []
    train_n = []

    for i in range(len(train)):
        if train[i][-1] == 'yes':
            train_y.append(train[i][:-1])
        elif train[i][-1] == 'no':
            train_n.append(train[i][:-1])

    # convert class list to arrays
    train_y = np.array(train_y)
    train_n = np.array(train_n)

    # compute prior class probabilities
    p_y = len(train_y)/(len(train_y)+len(train_n))
    p_n = len(train_n)/(len(train_y)+len(train_n))

    # models for class 1 and 2
    model_y = model(train_y)
    model_n = model(train_n)

    #classify
    attribute_ps_y = []
    attribute_ps_n = []

    for i in range(len(test)):
        attribute_ps_y.append(normal_pdf(test[i], model_y[i][0], model_y[i][1]))
        attribute_ps_n.append(normal_pdf(test[i], model_n[i][0], model_n[i][1]))

    prob_if_y = np.prod(attribute_ps_y)
    prob_if_n = np.prod(attribute_ps_n)

    prob = (prob_if_y*p_y)/(prob_if_y*p_y+prob_if_n*p_n)

    #If there is ever a tie between the two classes, choose class yes
    if prob >= 0.5 and prob <= 1:
        return "yes"
    elif prob < 0.5 and prob >= 0:
        return "no"
    


def knn(k,train,test): # K-NN function

    # calculate distances from all training samples
    distances = [] #list of (index, distance, class)
    for i in range(len(train)):
        distances.append((i, heuristic(test, train[i][:-1]), train[i][-1]))
    
    distances.sort(key=lambda x: x[1])
    neighbours = distances[:k]
    votes_yes = len([i for i in neighbours if i[2] == 'yes'])
    votes_no = len([i for i in neighbours if i[2] == 'no'])

    if votes_no > votes_yes:
        return "no"
    elif votes_yes >= votes_no: #If there is ever a tie between the two classes, choose class yes
        return "yes"
    
def heuristic(x,y): #Euclidean distance as the distance measure
    if len(x) != len(y):
        raise ValueError('Dimension mismatch, the input vectors have different number of attributes')
    x = np.array(x)
    y = np.array(y)
    diffs = np.array([x[i]-y[i] for i in range(len(x))])
    return np.sqrt(np.sum(diffs**2))    
    


def model(data): # takes in an array and outputs a list of tuples of (mean, sd) following gauss pdf for each attribute
    means = np.mean(data, axis = 0)
    sds = np.std(data, axis = 0)
    model = list(zip(means,sds))
    return model


def normal_pdf(x,mean,sd):
    return (1/(sd*np.sqrt(2*np.pi)))*np.exp(-.5*((x-mean)/sd)**2)
    
def accuracy(classifier, train, test): #takes a list of lists and returns a scalar value, test values have to be labelled
    test_unlabelled = [element[:-1] for element in test]
    test_labels = [element[-1] for element in test]

    predictions = []

    if classifier == 'NB':
        for element in test_unlabelled:
            predictions.append(nb(train, element))
    elif bool(re.search(r'\d+NN', classifier, re.IGNORECASE)):
        k = int(classifier[:-2])
        for i in range(len(test_unlabelled)):
            predictions.append(knn(k,train,test_unlabelled[i]))
    
    count = 0
    accuracy = None
    if len(predictions) == len(test_unlabelled):
        for i in range(len(test_unlabelled)):
            if test_labels[i] == predictions[i]:
                count += 1
    accuracy = count/len(test_unlabelled)
    
    return accuracy
    
def confusion_matrix(classifier, train, test): #outputs array [[tp,fp],[fn,tn]]
    test_unlabelled = [element[:-1] for element in test]
    test_labels = [element[-1] for element in test]

    predictions = []

    if classifier == 'NB':
        for element in test_unlabelled:
            predictions.append(nb(train, element))
    elif bool(re.search(r'\d+NN', classifier, re.IGNORECASE)):
        k = int(classifier[:-2])
        for i in range(len(test_unlabelled)):
            predictions.append(knn(k,train,test_unlabelled[i]))
    
    confusion_matrix = np.zeros((2,2))
    
    if len(predictions) == len(test_unlabelled):
        for i in range(len(test_unlabelled)):
            if predictions[i] == 'yes':
                if test_labels[i] == 'yes':
                    confusion_matrix[0][0] +=1
                elif test_labels[i] == 'no':
                    confusion_matrix[0][1] +=1
                else:
                    raise ValueError("Class not recognised")
            elif predictions[i] == 'no':
                if test_labels[i] == 'yes':
                    confusion_matrix[1][0] +=1
                elif test_labels[i] == 'no':
                    confusion_matrix[1][1] +=1
                else:
                    raise ValueError("Class not recognised")
                
    return confusion_matrix



if __name__ == "__main__":
    main()
