import json
import joblib
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly.express as px
# from plotly.graph_objs import Bar
# from plotly.graph_objects import Histogram

from sqlalchemy import create_engine

from models.train_classifier import tokenize, f1_scorer

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('CleanData', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    data = df.iloc[:,4:]
    data_fig1 = data.sum().sort_values(ascending=False)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    figures = [
        px.bar(data_fig1, x=data_fig1.keys(), y=data_fig1.values,
                title="Distribution of Message Categories",
                labels={"x": "categories",  "y": "count"}),

        px.histogram(x= data.sum(axis='columns'),
                title="Distribution of relevant categories per row",
                labels={
                    "x": "total number of relevant categories",
                    "y": "frequency"})
    ]
    
    # encode plotly graphs in JSON
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]
    figuresJSON  = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, figuresJSON=figuresJSON )


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)

if __name__ == '__main__':
    main()