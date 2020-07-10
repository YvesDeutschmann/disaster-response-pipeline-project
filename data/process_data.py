import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the data.
    
    Args:
        messages_filepath: String - csv file containing disaster messages.
        categories_filepath: String - csv file containing categories for each disaster message.
    Returns:
        df: DataFrame containing messages and categories.
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories)

    return df

def clean_data(df):
    """
    Clean up the message dataframe.
    
    Args:
        df: DataFrame containing messages and categories.
    Returns:
        df: cleaned DataFrame containing messages and categories.
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # extract column names for categories from first row
    row = categories.iloc[0,:]
    colnames = [column[:-2] for column in row.values]
    categories.columns = colnames

    # Convert category values to 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Replace categories column in df with new category columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # remove duplicates
    df.drop_duplicates(inplace=True)
    # remove rows with a value of 2 in 'related' column
    df.drop(df[df.related==2].index, inplace=True)

    return df


def save_data(df, database_filename):
    """
    Store data in database.
    
    Args:
        df: cleaned DataFrame containing messages and categories.
        database_filename: String - Name of Database the DataFrame is stored in.
    """
    engine = create_engine(r'sqlite:///{}'.format(database_filename))
    df.to_sql('CleanData', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()