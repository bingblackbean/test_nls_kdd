from sklearn.metrics import roc_auc_score
import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn
from imblearn.over_sampling import SMOTE, ADASYN,BorderlineSMOTE,SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
"""
functions
"""


def read_data():
    field_names_df = pd.read_csv(field_name_file,header=None,names=['name','data_type'])
    field_names = field_names_df['name'].tolist()
    field_names+=['label', 'label_code']
    df = pd.read_csv(train_file, header=None, names=field_names)
    df_test = pd.read_csv(test_file, header=None, names=field_names)
    attack_type_df = pd.read_csv(attack_type_file, sep=' ',header=None, names=['name', 'attack_type'])
    attack_type_dict = dict(zip(attack_type_df['name'].tolist(), attack_type_df['attack_type'].tolist()))
    df.drop('label_code',axis=1,inplace=True)
    df_test.drop('label_code',axis=1, inplace=True)
    df['label'].replace(attack_type_dict,inplace=True)
    df_test['label'].replace(attack_type_dict, inplace=True)
    return df,df_test



def label_encoder(data):
    labelencoder = LabelEncoder()
    for col in data.columns:
        data[col] = labelencoder.fit_transform(data[col])
    return data


def split_category(data, columns):
    cat_data = data[columns]
    rest_data = data.drop(columns, axis=1)
    return rest_data, cat_data


def one_hot_cat(data):
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data, columns=[data.name])
    out = pd.DataFrame([])
    for col in data.columns:
        one_hot_cols = pd.get_dummies(data[col], prefix=col)
        out = pd.concat([out, one_hot_cols], axis=1)
    out.set_index(data.index)
    return out

def log_transform(data,log_cols):
    for col in log_cols:
        data[col] = np.log(data[col]+1) # add 1 to avoid log 0
    return data


def create_feture_sel_model(X, Y):
    model = ExtraTreesClassifier(n_estimators=250,
                                 random_state=0)
    model.fit(X, Y)
    return model


def selectKImportance(model, X, k=5):
    return X.iloc[:, model.feature_importances_.argsort()[::-1][:k]]


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):

    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(
            new_actual_class,
            new_pred_class,
            average=average)
        roc_auc_dict[per_class] = roc_auc
    return roc_auc_dict


# oversampling
def resample_method(X, Y, method='SMOTE'):
    if method == 'SMOTE':
        oversample = SMOTE()
        X, Y = oversample.fit_resample(X, Y)
        return X, Y
    elif method == 'SMOTE_NUM':
        oversample = SMOTE(sampling_strategy='minority')
        X, Y = oversample.fit_resample(X, Y)
        return X, Y
    elif method == 'BorderlineSMOTE':
        oversample = BorderlineSMOTE()
        X, Y = oversample.fit_resample(X, Y)
        return X, Y
    elif method == 'SVMSMOTE':
        oversample = SVMSMOTE()
        X, Y = oversample.fit_resample(X, Y)
        return X, Y
    elif method == 'ADASYN':
        oversample = ADASYN(sampling_strategy='all')
        X, Y = oversample.fit_resample(X, Y)
        return X, Y
    elif method == 'SMOTEENN':
        oversample = SMOTEENN()
        X, Y = oversample.fit_resample(X, Y)
        return X, Y
    elif method == 'SMOTETomek':
        oversample = SMOTETomek()
        X, Y = oversample.fit_resample(X, Y)
        return X, Y
    else:
        return X, Y



def pipe_line(
        add_catgory=False,
        resampling='SMOTETomek',
        top_k=38,
        class_w=None):
    # read data form csv
    df,df_test = read_data()
    c = Counter(df['label'])
    print(f'original df label is {c}')
    c = Counter(df_test['label'])
    print(f'original df_test label is {c}')

    # split data to X and Y
    Y = df['label']
    Y_test = df_test['label']
    X = df.drop('label', axis=1)
    X_test = df_test.drop('label', axis=1)



    # log transform data
    log_cols = ['src_bytes','dst_bytes']
    X = log_transform(X,log_cols)
    X_test = log_transform(X_test,log_cols)
    print('log transform large data')

    # categorical_columns
    categorical_columns = ['protocol_type', 'service', 'flag']

    # first label_encoder to allow resampling
    X[categorical_columns] = label_encoder(X[categorical_columns])
    X_test[categorical_columns] = label_encoder(X_test[categorical_columns])



    # resampling data
    X, Y = resample_method(X, Y, method=resampling)
    X, X_cat = split_category(X, categorical_columns)
    X_test, X_test_cat = split_category(X_test, categorical_columns)
    c = Counter(Y)
    print(f'after oversampling df label is {c}')
    c = Counter(Y_test)
    print(f'after oversampling df label is {c}')
    # feature selecting
    if top_k is not None:
        feature_sel_model = create_feture_sel_model(X, Y)
        X = selectKImportance(feature_sel_model, X, k=top_k)
        X_test = selectKImportance(feature_sel_model, X_test, k=top_k)
        print(f'select {top_k} features')
    else:
        print(f'use all features')
    if add_catgory:
        # convert to one-hot
        X_cat_one_hot_cols = one_hot_cat(X_cat)
        X_test_cat_one_hot_cols = one_hot_cat(X_test_cat)
        # align train to test
        X_cat_one_hot_cols, X_test_cat_one_hot_cols = X_cat_one_hot_cols.align(
            X_test_cat_one_hot_cols, join='inner', axis=1)
        X_cat_one_hot_cols.fillna(0, inplace=True)
        X_test_cat_one_hot_cols.fillna(0, inplace=True)
        X = pd.concat([X, X_cat_one_hot_cols], axis=1, ignore_index=True)
        X_test = pd.concat([X_test, X_test_cat_one_hot_cols],
                           axis=1, ignore_index=True)
        print(f'add one-hot features')
    else:
        print(f'no one-hot features')


    scaler = preprocessing.Normalizer().fit(X)
    X = scaler.transform(X)
    X_test = scaler.transform(X_test)
    print(f'Normalize data')

    Y_val = Y
    X_val = X
    print(f'split dataset to train and validate')
    print(Counter(Y))
    print(Counter(Y_val))
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        disable_default_eval_metric=1)
    if class_w is None:
        class_w = 'balanced'

    sample_w = compute_sample_weight(class_weight=class_w, y=Y)
    print(f'compute_sample_weight done')

    hist = model.fit(
        X,
        Y,
        eval_metric='logloss',
        sample_weight=sample_w,
        verbose=True,
        # eval_set=(
        #     X_val,
        #     Y_val)
    )
    print(f'fitting done')
    y_pred = model.predict(X_test)

    auc_score = roc_auc_score_multiclass(Y_test, y_pred)
    print(auc_score)
    print(f'auc score is {accuracy_score(Y_test, y_pred)}')
    print(sklearn.metrics.confusion_matrix(Y_test, y_pred))
    print(sklearn.metrics.classification_report(Y_test, y_pred, digits=3))


if __name__ == '__main__':
    train_file = 'data/KDDTrain+.csv'
    test_file = 'data/KDDTest+.csv'
    field_name_file = 'data/Field Names.csv'
    attack_type_file = 'data/attack_types.txt'

    class_w = {
        'normal': 1,
        'dos': 1,
        'probe': 1,
        'r2l': 1,
        'u2r': 1}
    pipe_line(add_catgory=True, resampling='ADASYN', top_k=None,class_w=class_w)
