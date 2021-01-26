import sys
import os

import click

import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

@click.command()
@click.option('--input_path',
              type=click.STRING,
              default='/Users/philipphaslhofer/Data/TechnicalPhysics/Master/DarkMatter_Proj/data/pheno_ML/sm/atop_10fb.csv',
          help='The path to the csv file.')

@click.option('--save_path',
              type=click.STRING,
            default='/Users/philipphaslhofer/Data/TechnicalPhysics/Master/DarkMatter_Proj/data/pheno_ML/sm/',
              help='The name to the pkl file.')

@click.option('--batch_size', type=click.STRING, default='all' , help='Batch size. If not provided or two big 1 million lines are read.')

@click.option('--object', type=click.STRING, default='all', help='Provide particle object specifier. If not provided all objects in the given dataset will be converted.')

def csv_to_df(input_path, save_path, batch_size, object):

    if(batch_size == 'all'):
        print('Processing whole dataset!')
        default = '0-1000000'
        batch_size = 'whole'
        start, end = default.split('-')
    else:
        start, end = batch_size.split('-')
        print('Batch to process:', batch_size)

    data = []
    linecount = 0

    print('\nReading data @:', input_path)
    with open(input_path, 'r') as file:
        for line in file.readlines()[int(start):int(end)]:
            line = line.replace(';', ',')
            line = line.rstrip(',\n')
            line = line.split(',')
            data.append(line)
            linecount += 1
            if linecount > int(1e7):
                print('Stopped for Memory!')
                break

    #Find the longest line in the data
    longest_line = max(data, key = len)
    #Set the maximum number of columns
    max_col_num = len(longest_line)
    #Set the columns names
    col_names = ['event_ID', 'process_ID', 'event_weight', 'MET', 'MET_Phi']

    for i in range(1, (int((max_col_num-5)/5))+1):
        col_names.append('obj'+str(i))
        col_names.append('E'+str(i))
        col_names.append('pt'+str(i))
        col_names.append('eta'+str(i))
        col_names.append('phi'+str(i))

    #Create a dataframe from the list, using the column names from before
    print('\nProcessing the data ...')
    df = pd.DataFrame(data, columns=col_names)
    df.fillna(value=0, inplace=True)
    df = df.drop(columns=col_names[:5])
    x = df.values.reshape([df.shape[0]*df.shape[1]//5,5])
    lst = []
    for i in range(x.shape[0]):
        if (x[i] == 0).all():
            lst.append(i)
    data = np.delete(x, lst, 0)
    col_names = ['obj', 'E', 'pt', 'eta', 'phi']
    del df

    ## Change datatypes of the dataframe columns. float32 is enough for machine learning and normalization use.
    df = pd.DataFrame(data, columns=col_names)
    df = df.astype({'obj':'string', 'E':'float32', 'pt':'float32', 'eta':'float32', 'phi':'float32'})
    ## Choose the objects in the dataset to be used. Currently only handels single label or all objects.
    if(object == 'all'):
        print('Using all objects in the dataset!')
        print(df.head())
    else:
        print(f'Using only {object}-object/jet/particle')
        df = df[df['obj'] == object]
        print(df.head())

    df = df.drop(columns='obj')
    variables = df.keys()

    ## Create the train-test split
    train_array, test_array = train_test_split(df.values, shuffle = True, random_state = 42, test_size = 0.1)

    data_train_df = pd.DataFrame(train_array, columns = variables)
    data_test_df = pd.DataFrame(test_array, columns = variables)

    print(f'\nWriting data: {save_path}-{batch_size}-{object}_4D_***.pkl')
    data_train_df.to_pickle(save_path + '_' + batch_size + '_' + object + '_4D_train.pkl')
    data_test_df.to_pickle(save_path + '_' + batch_size + '_' + object + '_4D_test.pkl')

    return

if __name__=='__main__':
    csv_to_df()
