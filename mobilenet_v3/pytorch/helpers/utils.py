def print_confusion_matrix(cm):
    print("Confusion matrix:")
    
    for i in reversed(range(2)):
        for j in reversed(range(2)):
            print(cm[i][j], end=' ')
        print('\n', end='')
    print("----------------------------------------------------------------------")