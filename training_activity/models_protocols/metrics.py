def accuracy_score(y_true, y_pred):
    correct_pred = 0

    for index, pred in enumerate(y_pred):
        if pred == y_true[index]:
            correct_pred += 1

    return round((correct_pred / len(y_pred)), 2)


def confusion_matrix(y_true, y_pred):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for index, pred in enumerate(y_pred):
        if pred == 1 and y_true[index] == 1:
            true_positives += 1
        elif pred == 1 and y_true[index] == 0:
            false_positives += 1
        elif pred == 0 and y_true[index] == 1:
            false_negatives += 1
        elif pred == 0 and y_true[index] == 0:
            true_negatives += 1

    return [[true_negatives, false_positives], [false_negatives, true_positives]]