import pandas as pd
import os

def rename_coutries(x):
    if x in ['Argentina','Argentine']:
        return 'Argentina'
    elif x in ['Brasile','Brasilien','Brazil','Brésil']:
        return 'Brazil'
    elif x in ['Chile','Chili']:
        return 'Chile'
    elif x in ['Fransk Guyana','French Guiana']:
        return 'French Guiana'
    else:
        return x

def format_csvs_to_df(path):
    file_names = os.listdir(path)

    frames = []
    for n in file_names:
        coordinates = [float(x) for x in n[:-4].split('_')]
        df = pd.read_csv(path + n, sep=';')
        df['coordinates'] = [coordinates] * len(df)
        frames.append(df)
    df_total = pd.concat(frames)
    return df_total

