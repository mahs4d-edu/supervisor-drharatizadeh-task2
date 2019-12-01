from os import path
import pandas as pd

def load_dataset():
    file_path = path.join(path.abspath(path.dirname(__file__)), '../data/movielens100k_ratings.csv')
    dataset = pd.read_csv(file_path)
