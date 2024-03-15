import os

def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def print_confusion_matrix(cm):
    print("Confusion matrix:")
    
    for i in reversed(range(2)):
        for j in reversed(range(2)):
            print(cm[i][j], end=' ')
        print('\n', end='')