import pandas as pd

df_100 = pd.read_csv("ant_v4_results_100.csv")
df_1000 = pd.read_csv("ant_v4_results_1000.csv")
df_5000 = pd.read_csv("ant_v4_results_5000.csv")
df_10000 = pd.read_csv("ant_v4_results_10000.csv")
df_50000 = pd.read_csv("ant_v4_results_50000.csv")
df_100000 = pd.read_csv("ant_v4_results_100000.csv")
df_500000 = pd.read_csv("ant_v4_results_500000.csv")
df_1000000 = pd.read_csv("ant_v4_results_1000000.csv")

for df in [df_100, df_1000, df_5000, df_10000, df_50000, df_100000, df_500000, df_1000000]:
    print("Combined - Single Sum", df['Combined - Single'].sum())
    positive = df['Combined - Single'].apply(lambda x: x > 0)
    print("Number of points with positive difference", positive.sum())
    print("reward ctrl multi - single Sum", df['reward ctrl multi - single'].sum())
    print("Number of points with positive difference", df['reward ctrl multi - single'].apply(lambda x: x > 0).sum())
    print()