import pandas as pd
import os

def format_csvs_to_df(path):
    file_names = os.listdir('../data/Copy of dev_south_america_merged/')

    frames = []
    for n in file_names:
        coordinates = [float(x) for x in n[:-4].split('_')]
        df = pd.read_csv('../data/Copy of dev_south_america_merged/' + n, sep=';')
        df['coordinates'] = [coordinates] * len(df)
        frames.append(df)
    df_total = pd.concat(frames)
    return df_total