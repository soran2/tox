from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import svm
from surprise import SVD, SVDpp, Reader, accuracy, Dataset
from surprise.model_selection.search import GridSearchCV, RandomizedSearchCV
from surprise.model_selection import KFold

def cnn_model(X_train, y_train, X_test, y_test, X_pred, nepochs, shape):
    """ Apply CNN. The CNN model is a VGG-like model similar to the one in keras documentation page:
    [https://keras.io/getting-started/sequential-model-guide/]
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=nepochs)

    return model.predict(X_test), model.predict(X_pred)


def svm_model(df, unknown_idx, important_features, X_train, y_train, X_test, X_unknown):
    """ Apply SVM.
    """    
    svm_model = svm.SVC(probability = True, random_state = 75, class_weight = 'balanced')   
        
    svm_model.fit(X_train, y_train)
    
    test_probs = svm_model.predict_proba(X_test)[:, 1]

    result = df.iloc[unknown_idx]
    result['SVM_ToxicProb'] = svm_model.predict_proba(X_unknown[:,important_features])[:, 1]
    result = result[['smiles', 'SVM_ToxicProb']]

    return test_probs, result


def svd_model(df):
    """ Apply SVD.
    """
    df = pd.melt(df, id_vars='smiles', 
                 value_vars=list(df.columns[1:]),
                 var_name='Target', 
                 value_name='TargetValue')

    mark = df.TargetValue.isna()
    unknown = df.loc[mark]
    known = df.loc[~mark]


    reader = Reader(rating_scale=(0,1))
    data = Dataset.load_from_df(known[['smiles', 'Target', 'TargetValue']], reader)

    kf = KFold(n_splits=3, random_state=57)

    algo = SVDpp(n_factors=12, reg_all = 0.003, lr_all = 0.006, random_state=132)

    for trainset, testset in kf.split(data):

        algo.fit(trainset)
        predictions = algo.test(testset)
        
        rmse = round(accuracy.rmse(predictions, verbose=True), 3)

        print('RMSE of SVD model for cross validation'+str(rmse))

    result = unknown.copy()
    result['ToxicProb'] = result.apply(lambda x: algo.predict(x.smiles, x.Target).est, axis = 1)
    result = result.drop(columns='TargetValue')
    
    return result
