import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle
import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from joblib import parallel_backend


def load_data(database_filepath):
    """
    Import processed data from database.
    
    Args:
    database_filepath: String - Path to database file.
    Returns:
    X: Series - Series of messages to categorize.
    Y: DataFrame - DataFrame containing class labels.
    category_names: Index - List of strings containing category names.
    """
    engine = create_engine(r'sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('CleanData', engine)
    X = df.message
    Y = df[df.columns[4:]]
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    """
    Normalize and tokenize message strings.
    
    Args:
    text: String - message text to process
    Returns:
    clean_tokens: list of strings - list of tokens from the message
    """
    # normalize case and remove punctuation
    text = text = re.sub('\W', ' ', text.lower())
    
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    
    # Reduce words to their stems
    clean_tokens = [PorterStemmer().stem(tok).strip() for tok in tokens if tok not in stop_words]

    return clean_tokens

def report_results(Y_test, Y_pred, category_names):
    """
    Report precision, recall  and f1_score for the Machine Learning Model.
    
    Args:
    Y_test: DataFrame - test-dataset containing actual categories for messages.
    Y_pred: DataFrame - dataset containing predicted categories for messages.
    category_names: Index - List of strings containing category names.
    Returns:
    results: DataFrame - DataFrame containing precision, recall  and f1_score for each category aswell as median values for the dataset.
    """
        
    results = pd.DataFrame(columns= ['category', 'precision', 'recall', 'f1-score'])
        
    for i in range(len(category_names)):
        y_true = Y_test.iloc[:,i].values
        y_pred = Y_pred[:,i]
        
        row = {'category':category_names[i], 
               'precision':precision_score(y_true, y_pred, zero_division=0, average='macro'), 
               'recall':recall_score(y_true, y_pred, zero_division=0, average='macro'), 
               'f1-score':f1_score(y_true, y_pred, zero_division=0, average='macro')}
        results = results.append(row, ignore_index=True)
    
    median_values = {'category':'median_values', 
               'precision':results['precision'].median(), 
               'recall':results['recall'].median(), 
               'f1-score':results['f1-score'].median()}
    results = results.append(median_values, ignore_index=True)
    
    return results

def f1_scorer(y_true, y_pred):
    """
    Calculate median F1-Score to measure model performance.

    Args:
    y_true: DataFrame containing the actual labels
    y_pred: Array containing the predicted labels
    Returns:
    f1_score: Float representing the median F1-Score for the model.
    """
    scores = []
        
    for i in range(y_pred.shape[1]):
        scores.append(f1_score(np.array(y_true)[:,i], y_pred[:,i], zero_division=0, average='macro'))
    score = np.median(scores)
    return score

def build_model():
    """
    Creates GridSearchCV object with custom f1-scorer function.

    Args:
    None
    Returns:
    model: GridSearchCV object - GridSearchCV object that looks for the optimal parameter set.
    """

    pipeline = Pipeline([
               ('vect', CountVectorizer(tokenizer=tokenize) ),
               ('tfidf', TfidfTransformer() ),
               ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight='balanced', n_jobs=-1)) )
                ])
    
    parameters = {
        'clf__estimator__min_samples_leaf':[5, 25, 50],
        'clf__estimator__n_estimators': [100, 200, 400]}

    scorer = make_scorer(f1_scorer)
    model = GridSearchCV(pipeline, param_grid=parameters, cv=3, scoring=scorer, verbose=3)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Predicts labels for test-dataset on returns precision, recall  and f1_score for these predictions.

    Args:
    model: GridSearchCV object - GridSearchCV object with the optimal parameter set.
    X_test: Series - test-dataset containing messages to predict relevant labels.
    Y_test: DataFrame - test-dataset containing actual categories for messages.
    category_names: Index - List of strings containing category names.
    Returns:
    None
    """
    Y_pred = model.predict(X_test)
    results = report_results(Y_test, Y_pred, category_names)

    print(results)

def save_model(model, model_filepath):
    """
    Stores fitted model in pickle file.

    Args:
    model: GridSearchCV object - GridSearchCV object with the optimal parameter set.
    model_filepath:  String - Path to store pickle file.
    Returns:
    None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()