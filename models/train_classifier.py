import pickle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')


def load_data(database_filepath):
    """Load the data
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_query('select * from messages', engine)
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X, Y, Y.columns


def tokenize(text):
    """Tokenize the messages
    """
    wordTokens = word_tokenize(text)
    wordNetLemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for t in wordTokens:
        clean_token = wordNetLemmatizer.lemmatize(t).lower().strip()
        clean_tokens.append(clean_token)
    return clean_tokens


def build_model():
    """Build the model for classify messages
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), ('tfidf', TfidfTransformer(
    )), ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': [True, False],
        'tfidf__norm': ['l1', 'l2']
    }
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model
    """
    Y_preds = model.predict(X_test)
    for i in range(len(category_names)):
        print(category_names)
        print(classification_report(Y_test.values[:, i], Y_preds[:, i]))


def save_model(model, model_filepath):
    """Export the model
    """
    with open(model_filepath, 'wb') as outfile:
        pickle.dump(model, outfile)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
