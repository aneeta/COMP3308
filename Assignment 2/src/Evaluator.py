import MyClassifier as MC
import numpy as np
import sys
import itertools


def main():
    # input
    folds = dict()
    
    for i in range(1,len(sys.argv)):
        with open(sys.argv[i]) as raw_train:
            folds[i] = MC.read_train(raw_train)
    
    

    tests = [] 
    for i in range(1,len(sys.argv)):
        tests.append(folds[i])

    trains = []
    
    for i in range(len(sys.argv)-1):
        combined = []
        for j in range(len(sys.argv)-1):
            if j == i:
                continue
            combined += tests[j]
        trains.append(combined)

    #list of (train, test)
    train_and_test = list(zip(trains, tests))
    
    classifier = ["NB","1NN","3NN","5NN"]
    avg_accuracy = []
    avg_m = []

    for element in classifier:
        accs = []
        matrices = []
        for i in range(len(sys.argv)-1):
            accs.append(MC.accuracy(element, train_and_test[i][0], train_and_test[i][1]))
            matrices.append(MC.confusion_matrix(element, train_and_test[i][0], train_and_test[i][1]))
        accs = np.array(accs)
        avg_acc = np.mean(accs)
        avg_accuracy.append(avg_acc)
        avg_mat = np.mean(np.array(matrices), axis=0)
        avg_m.append(avg_mat)    

    results = list(zip(classifier, avg_accuracy, avg_m))
    print("Classifier and avg accuracy: \n", results)


if __name__ == "__main__":
    main()
