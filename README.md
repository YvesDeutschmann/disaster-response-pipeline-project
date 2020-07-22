# Disaster Response Web App

## Installation
The code contained in this repository was written in HTML and Python 3. The following packages are requied to run the app:

- json
- re
- pickle
- plotly
- pandas
- numpy
- scikit-learn
- nltk
- flask
- sql-alchemy
- warnings

The full list of installed packages in the virtual environment can be found in the requirements.txt file.

## Project Overview
This repository contains code for a web app that classifies incoming messages during a disaster into several categories. The categorization can then be used to redirect that message  to the relevant aid agencies.

The app uses a Machine Learning Model to categorize any new messages received. It is possible to read in new datasets and train th emodel on this new data with the contained modules.

## File Descriptions
- process_data.py: Creates table 'CleanData' in a SQL databse from messages and categories textfiles.
- train_classifier.py: Reads in the 'CleanData' table and creates, fits and saves the model in a pickle file. 
- ETL Pipeline Preparation.ipynb: Notebook preparing the data cleaning process that is later automated in `process_data.py`.
- ML Pipeline Preparation.ipynb: Notebook preparing the model creation and fitting that is later automated in `train_classifier.py`
- BalancedPipeline.ipynb: Investigation on alternative ML model that was rejected.
- Read ML Results.ipynb: Notebook the evaluate the results of all run models due to remote execution in this case.
- data: folder containing sample messages and categories datasets in csv format.
- app: folder containing the files necessary to run and render the web app.

## Running Instructions
### Run process_data.py
- From the root of the programm run the following code:

`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
### Run train_classifier.py
- From the root of the programm run the following code:

`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
### Run web app
- Naviagte to the app directory and run the following code: 

`python run.py`

- Open http://127.0.0.1:5000/ in your browser

## Results
Altohough the overall score was acceptable the model lacks in predicitve value when it comes to underrepresented categories. the category 'child_alone' for example does not have a single positive classification in this dataset. This means that the ML model is unable to learn what are neccessary feature to predict this label. Hence the web app is not able to predict this category even when we directly enter 'child alone.'

To improve this behaviour we have to balance our dataset and try to feed the model roughly the same amount of data for every single category we try to predict. Until then we have to be careful with predictions for underepresented categories.   

Licensing, Authors, Acknowledgements
This app was completed as part of the Udacity Data Scientist Nanodegree. Code templates and data were provided by Udacity. The data was originally sourced by Udacity from Figure Eight.