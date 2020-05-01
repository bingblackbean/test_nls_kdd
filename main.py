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


# list columns names
col_names = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label"]

label_to_num_dict = {'normal': 'normal',
                     'neptune': 'cls1',
                     'back': 'cls1',
                     'land': 'cls1',
                     'pod': 'cls1',
                     'smurf': 'cls1',
                     'teardrop': 'cls1',
                     'mailbomb': 'cls1',
                     'apache2': 'cls1',
                     'processtable': 'cls1',
                     'udpstorm': 'cls1',
                     'worm': 'cls1',
                     'ipsweep': 'cls2',
                     'nmap': 'cls2',
                     'portsweep': 'cls2',
                     'satan': 'cls2',
                     'mscan': 'cls2',
                     'saint': 'cls2',
                     'ftp_write': 'cls3',
                     'guess_passwd': 'cls3',
                     'imap': 'cls3',
                     'multihop': 'cls3',
                     'phf': 'cls3',
                     'spy': 'cls3',
                     'warezclient': 'cls3',
                     'warezmaster': 'cls3',
                     'sendmail': 'cls3',
                     'named': 'cls3',
                     'snmpgetattack': 'cls3',
                     'snmpguess': 'cls3',
                     'xlock': 'cls3',
                     'xsnoop': 'cls3',
                     'httptunnel': 'cls3',
                     'buffer_overflow': 'cls4',
                     'loadmodule': 'cls4',
                     'perl': 'cls4',
                     'rootkit': 'cls4',
                     'ps': 'cls4',
                     'sqlattack': 'cls4',
                     'xterm': 'cls4'}


# label_to_num_dict = {'normal': 'normal',
#                      'neptune': 'cls',
#                      'back': 'cls',
#                      'land': 'cls',
#                      'pod': 'cls',
#                      'smurf': 'cls',
#                      'teardrop': 'cls',
#                      'mailbomb': 'cls',
#                      'apache2': 'cls',
#                      'processtable': 'cls',
#                      'udpstorm': 'cls',
#                      'worm': 'cls',
#                      'ipsweep': 'cls',
#                      'nmap': 'cls',
#                      'portsweep': 'cls',
#                      'satan': 'cls',
#                      'mscan': 'cls',
#                      'saint': 'cls',
#                      'ftp_write': 'cls',
#                      'guess_passwd': 'cls',
#                      'imap': 'cls',
#                      'multihop': 'cls',
#                      'phf': 'cls',
#                      'spy': 'cls',
#                      'warezclient': 'cls',
#                      'warezmaster': 'cls',
#                      'sendmail': 'cls',
#                      'named': 'cls',
#                      'snmpgetattack': 'cls',
#                      'snmpguess': 'cls',
#                      'xlock': 'cls',
#                      'xsnoop': 'cls',
#                      'httptunnel': 'cls',
#                      'buffer_overflow': 'cls',
#                      'loadmodule': 'cls',
#                      'perl': 'cls',
#                      'rootkit': 'cls',
#                      'ps': 'cls',
#                      'sqlattack': 'cls',
#                      'xterm': 'cls'}


def pipe_line(
        add_catgory=False,
        resampling='SMOTETomek',
        top_k=38,
        label_cat=False):
    # read data form csv
    df = pd.read_csv("KDDTrain+_2.csv", header=None, names=col_names)
    df_test = pd.read_csv("KDDTest+_2.csv", header=None, names=col_names)

    # handle the Y label
    df['label'] = df['label'].replace(label_to_num_dict)
    df_test['label'] = df_test['label'].replace(label_to_num_dict)

    c = Counter(df['label'])
    print(f' original df label is {c}')

    c = Counter(df_test['label'])
    print(f' original df test label is {c}')

    # split data to X and Y
    Y = df['label']
    Y_test = df_test['label']
    X = df.drop('label', axis=1)
    X_test = df_test.drop('label', axis=1)

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
    print(f' after oversampling df label is {c}')

    c = Counter(Y_test)
    print(f' after oversampling df label is {c}')
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

    class_w = {
        'normal': 1,
        'cls1': 1,
        'cls2': 1,
        'cls3': 4 ,
        'cls4': 4}
    # class_w = {
    #     'normal': 1,
    #     'cls': 4}
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
    pipe_line(add_catgory=True, resampling='ADASYN', top_k=None)
