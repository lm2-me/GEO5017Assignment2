import numpy as np

# Overall accuracy
def overallAccuracy(y_true, y_pred):
    n = len(y_true)
    accuracy = 0
    unique = np.unique(y_true)

    correct = []
    for l in range(len(unique)):
        correct.append(0)

    classif = 0
    for u in unique:
        for i, t in enumerate(y_true):
            if u == t:
                if y_pred[i] == y_true[i]:
                    correct[classif] +=1
            else:
                continue
        classif +=1

    for c in correct:
        accuracy += c

    oa = accuracy / n
    
    return oa

# Mean per-class accuracy
def meanPerClassAccuracy(y_true, y_pred):
    accuracy = 0
    unique = np.unique(y_true)
    c = len(unique)

    correct = []
    total = []
    for l in range(len(unique)):
        correct.append(0)
        total.append(0)

    classif = 0
    for u in unique:
        for i, t in enumerate(y_true):
            if u == t:
                total[classif] += 1
                if y_pred[i] == y_true[i]:
                    correct[classif] +=1
            else:
                continue
        classif +=1

    for index, corr in enumerate(correct):
        classaccuracy = corr / total[index]
        accuracy += classaccuracy

    mpa = accuracy / c
    
    return mpa


# Confusion Matrix
def confusionMatrix(y_true, y_pred):

    unique = np.unique(y_true)
    matrix = np.zeros(shape=(len(unique),len(unique)))

    total = []

    for l in range(len(unique)):
        total.append(0)

    for i, u in enumerate(unique):
        for j, p in enumerate(y_pred):
            if u == p:
                total[i] += 1
                if y_pred[j] == y_true[j]:
                    matrix[i, i] += 1
                else:
                    for k, w in enumerate(unique):
                        if y_true[j] == w:
                            matrix[k, i] += 1
            else:
                continue
        
    cm = np.vstack([[unique], matrix])
    labels = np.transpose(np.concatenate(([' '],unique)))
    cm2 = np.c_[labels, cm]

    cm_formatted = '\n '.join(['[' + ''.join([str(v).ljust(10) for v in t]) + ']' for t in cm2])

    
    return cm_formatted
    
