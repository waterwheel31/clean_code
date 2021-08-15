"""
This is a test script for churn library funtions
"""
import os
import logging
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    pth = './data/bank_data.csv'
    df = cls.import_data(pth)
    perform_eda(df)
    try:
        assert os.path.isfile("./images/eda/churn_dist.png")
        assert os.path.isfile("./images/eda/customerAge_dist.png")
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: ERROR. an image has not been created properly")
        raise err

def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    pth = './data/bank_data.csv'
    df = cls.import_data(pth)
    cls.perform_eda(df)
    df = encoder_helper(df, cat_columns, "Churn")
    
    try:
        assert "Attrition_Flag" not in df.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: ERROR. \
            There is still a column named 'Attrition_Flag'")
        raise err

def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''

    pth = './data/bank_data.csv'
    df = cls.import_data(pth)
    cls.perform_eda(df)
    response = 'Churn'
    
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, response)
    
    try:
        assert len(X_test) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err: 
        logging.error("Testing perform_feature_engineering: ERROR")
        raise err

def test_train_models(train_models):
    '''
    test train_models
    '''
    
    pth = './data/bank_data.csv'
    df = cls.import_data(pth)
    cls.perform_eda(df)
    df = cls.encoder_helper(df, cat_columns, "Churn")
    response = 'Churn'
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, response)
    train_models(X_train, X_test, y_train, y_test)
    try:
        assert os.path.isfile('./images/results/features_randomForest.png')
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err: 
        logging.error("Testing train_models: ERROR. The feature importance plot is not created")
        raise err


if __name__ == "__main__":
    
    cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'     
    ]
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
    