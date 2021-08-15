# library doc string

"""
Functions to proceess machine learning on a dataset
Author: Data Science Learner
Date: Aug 21, 2021
"""

# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

# Functions


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth)
        return df
    except FileNotFoundError as err:
        raise err


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    print('starting plotting')

    # distribution of churn
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    ax.hist(df['Churn'])
    fig.savefig("./images/eda/churn_dist.png")

    # distribution of customer age
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.hist(df['Customer_Age'])
    fig.savefig("./images/eda/customerAge_dist.png")


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that
        could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    for category in category_lst:
        print('category:', category)
        lst = []
        groups = df.groupby(category).mean()[response]

        for val in df[category]:
            lst.append(groups.loc[val])
            categoryName = category + '_' + response
        df[categoryName] = lst

    # return the result
    df = df[df.columns[~df.columns.isin(category_lst)]]
    df = df[df.columns[df.columns != "Attrition_Flag"]]
    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    print(df.dtypes)
    X = df.loc[:, df.columns != response]
    y = df[response]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # Save image - RF
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.text(0.01, 1.25, str('Random Forest Train'), {
            'fontsize': 10}, fontproperties='monospace')
    ax.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    ax.text(0.01, 0.6, str('Random Forest Test'), {
            'fontsize': 10}, fontproperties='monospace')
    ax.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    ax.axis('off')
    fig.savefig("./images/results/results_randomForest.png")

    # Save image - LR
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.text(0.01, 1.25, str('Logistic Regression Train'),
            {'fontsize': 10}, fontproperties='monospace')
    ax.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    ax.text(0.01, 0.6, str('Logistic Regression Test'), {
            'fontsize': 10}, fontproperties='monospace')
    ax.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    ax.axis('off')
    fig.savefig("./images/results/results_logisticRegression.png")


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    ax.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    fig.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    # Training
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)
    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_test)

    lrc = LogisticRegression()
    lrc.fit(X_train, y_train)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Reporting
    classification_report_image(
        y_train, y_test,
        y_train_preds_lr, y_train_preds_rf,
        y_test_preds_lr, y_test_preds_rf)

    # Feature importance
    feature_importance_plot(
        rfc, X_train, './images/results/features_randomForest.png')
    #feature_importance_plot(lrc, X_train, './images/results/features_logisticRegression.png')

    # ROC Curves
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    plot_roc_curve(lrc, X_test, y_test, ax=ax)
    plot_roc_curve(rfc, X_test, y_test, ax=ax)
    fig.savefig("./images/results/results_ROCcurves.png")

    # Save Models
    joblib.dump(rfc, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == '__main__':

    pth = './data/bank_data.csv'
    df = import_data(pth)

    perform_eda(df)

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    df = encoder_helper(df, cat_columns, "Churn")

    X_tra, X_tes, y_tra, y_tes = perform_feature_engineering(df, "Churn")

    train_models(X_tra, X_tes, y_tra, y_tes)
