from sklearn import multiclass, svm


def sklearn_multiclass_prediction(mode, X_train, y_train, X_test):
    """
    Use Scikit Learn built-in functions multiclass.OneVsRestClassifier
    and multiclass.OneVsOneClassifier to perform multiclass classification.

    Arguments:
        mode: one of 'ovr', 'ovo' or 'crammer'.
        X_train, X_test: numpy ndarray of training and test features.
        y_train: labels of training data, from 0 to 9.

    Returns:
        y_pred_train, y_pred_test: a tuple of 2 numpy ndarrays,
                                   being your prediction of labels on
                                   training and test data, from 0 to 9.
    """
    y_pred_train = None
    y_pred_test = None
    # using random_state=12345 for reproductivity
    svm_model = svm.LinearSVC(random_state=12345)
    if mode == 'ovr':
        ovr_model = multiclass.OneVsRestClassifier(svm_model)
        ovr_model.fit(X_train, y_train)
        y_pred_train = ovr_model.predict(X_train)
        y_pred_test = ovr_model.predict(X_test)
    elif mode == 'ovo':
        ovo_model = multiclass.OneVsOneClassifier(svm_model)
        ovo_model.fit(X_train, y_train)
        y_pred_train = ovo_model.predict(X_train)
        y_pred_test = ovo_model.predict(X_test)
    elif mode == 'crammer':
        # using random_state=12345 for reproductivity
        crammer_singer_model = svm.LinearSVC(multi_class='crammer_singer', random_state=12345)
        crammer_singer_model.fit(X_train, y_train)
        y_pred_train = crammer_singer_model.predict(X_train)
        y_pred_test = crammer_singer_model.predict(X_test)
    else:
        print("Invalid mode. Mode should be 'ovr', 'ovo' or 'crammer'.")

    return y_pred_train, y_pred_test
