#!/usr/bin/env python
# coding: utf-8


import json
import numpy as np
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, auc, balanced_accuracy_score, classification_report, precision_score, recall_score, roc_curve, auc, precision_recall_curve

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import warnings
warnings.filterwarnings('ignore')


# Retrieve data from MySQL
def retreive_loan_data():
    with open('db_config.json', 'r') as json_file:
        db_config = json.load(json_file)

    try:
        # Attempt to connect to the MySQL server
        connection = mysql.connector.connect(**db_config)

        # Check if the connection is successful
        if connection.is_connected():
            print("Connected to MySQL Server")
        
        df = pd.read_sql("SELECT * FROM matured_loan LIMIT 300000", connection)

        # Close the connection
        connection.close()
        
        return df

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        print("Failed to connect to MySQL Server")
        
        return


# drop irrelevant columns
def drop_irrelevant_columns(df):
    irrelevant_columns = [
        'id', 'initial_list_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt',
        'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'last_pymnt_to_income',
        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt',
        'is_matured_loan', 'calculated_installment', 'ROI', 'IRR', 'int_rate_bin', 'annual_inc_bin'
    ] + [
        'funded_amnt', 'funded_amnt_inv', 'debt_settlement_flag',
        'last_pymnt_to_income', 'last_fico_range_high', 
        'last_fico_range_low', 'FICO_change']


    nlp_columns = ['purpose', 'title', 'zip_code', "emp_title"]


    datetime = ['issue_d']


    df = df.drop(irrelevant_columns + nlp_columns + datetime, axis=1)
    return df


def transform(df):
    # add a new column for emp_length
    df['emp_length_int'] = df['emp_length'].replace(
        ['4 years', '2 years', '10+ years', 
         '3 years', '5 years', '6 years',
         '1 year', '7 years', '< 1 year', 
         '9 years', '8 years'], [
            4, 2, 10, 3, 5, 6, 1, 7, 1, 9, 8
        ])


    # turn categorical into one-hot encoding variable
    categorical = [
        "term", "grade", "sub_grade", "emp_length", "home_ownership",
        "verification_status", "addr_state", "application_type", "hardship_flag"
    ]


    df = pd.get_dummies(df, columns=categorical, prefix=categorical) #, drop_first=True)


    # target
    df["loan_status"].replace(["Fully Paid", "Charged Off"], [0, 1], inplace=True)
    
    return df


# reduce mem usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def plot_roc_curve(y_test, y_prob, name):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.savefig(f'plots/roc_{name}.jpg')
    
    
def plot_precision_recall_curve(y_test, y_prob, name):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    average_precision = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AP={average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig(f'plots/pr_{name}.jpg')


def plot_feature_importances(estimator, columns, chosen, name):
    f_imp = estimator.feature_importances_
    colors = np.array(["b" if c else "r" for c in chosen])
    
    idx = np.argsort(f_imp)
    f_imp = np.take_along_axis(f_imp, idx, axis=0)
    colors = np.take_along_axis(colors, idx, axis=0)
    columns = np.take_along_axis(columns, idx, axis=0)
    
    plt.figure(figsize=(30, 40))
    plt.barh(columns, f_imp, color=colors)
    plt.title('Feature Importances')
    plt.xlabel('Score')
    plt.ylabel('Feature')
    plt.savefig(f'plots/fimp_{name}.jpg')


def main():
    df = retreive_loan_data()
    df = drop_irrelevant_columns(df)
    df = transform(df)

    df = reduce_mem_usage(df)

    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    # for reproducible
    random_seed = 42
    
    # train test split
    X, X_test, y, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_seed, stratify=y)

    # create a k fold object
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=random_seed)

    # classifiers that works well as a baseline model
    classifiers = [
        ('lr', LogisticRegression(random_state=random_seed)),
        ('lgbm', LGBMClassifier(n_estimators=300, random_state=random_seed)),
        ('xgb', XGBClassifier(n_estimators=300, random_state=random_seed))
        # ('rf', RandomForestClassifier(n_estimators=300, random_state=random_seed))
    ]

    # SMOTE
    smote = SMOTE(random_state=random_seed)
    X, y = smote.fit_resample(X, y)

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Create a new MLflow Experiment
    mlflow.set_experiment("PD Modelling")
    
    # iter over possible classifiers
    for name, estimator in classifiers:
        # start experiment with mlflow
        with mlflow.start_run(run_name="model_" + name):
            mlflow.log_params({"model": name})
            
            # make a pipeline
            model = Pipeline([
                ('scaler', RobustScaler()),
                ('selector', SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, random_state=random_seed))),
                ('estimator', estimator)
            ])
            
            # metrics
            roc_auc_m = 0
            
            # fit for KFold
            print('Model:', name)
            for i, (train_index, val_index) in enumerate(rskf.split(X, y)):
                print('KFold:', i)
                X_train = X.iloc[train_index]
                X_val = X.iloc[val_index]
                y_train = y.iloc[train_index]
                y_val = y.iloc[val_index]

                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_val)[:,1]
                roc_auc_m += roc_auc_score(y_val, y_prob)
            
            mlflow.set_tag("Training Info", "Baseline")
            mlflow.log_metric("I.S. ROC AUC Score", roc_auc_m / (i + 1))
            
            # out-of-sample prediction
            y_prob = model.predict_proba(X_test)[:,1]
            mlflow.log_metric("O.O.S. ROC AUC Score", roc_auc_score(y_test, y_prob))
            for j in range(5, 35, 5):
                y_pred = np.where(y_prob > j*0.01, 1, 0)
                mlflow.log_metric(f"O.O.S. F1 Score at threshold {j*0.01}", f1_score(y_test, y_pred))
                mlflow.log_metric(f"O.O.S. Precision at threshold {j*0.01}", precision_score(y_test, y_pred))
                mlflow.log_metric(f"O.O.S. Recall at threshold {j*0.01}", recall_score(y_test, y_pred))
            
            # record feature selection result
            chosen = model[1].get_support()
            plot_feature_importances(model[1].estimator_, X_train.columns, chosen, name)
            
            # plot performance
            plot_roc_curve(y_test, y_prob, name)
            plot_precision_recall_curve(y_test, y_prob, name)
            
            # save plots as artifacts
            mlflow.log_artifact(f"plots/pr_{name}.jpg", artifact_path="plot")
            mlflow.log_artifact(f"plots/roc_{name}.jpg", artifact_path="plot")
            mlflow.log_artifact(f"plots/fimp_{name}.jpg", artifact_path="plot")

            # Log the model
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="baseline_" + name,
            )

if __name__ == "__main__":
    main()