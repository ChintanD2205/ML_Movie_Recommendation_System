# ML project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

song_data = pd.read_csv("C:\\Users\\chint\\Downloads\\triplets_file.csv")
# print(song_data)

""" The song_data dataset contains the user id, song and listen count
for that particular song
"""

metadata = pd.read_csv("C:\\Users\\chint\\Downloads\\song_data (1).csv")
# print(metadata)
""" The metadata dataset contains details pertaining to the song
about its singer and many more details"""

""" We load these two datasets so that we can work on the prediction system"""