from Helpers.preprocessingHelpers import find_peaks, getData
import pandas as pd
import matplotlib.pyplot as plt

# x1, x2, y1, y2 = getData('newCombos.txt')
# print(x1[0])



df = pd.read_csv("./data/peakGraph/graphs/1/mz_base.csv")
df.insert(1, "mz_av", pd.read_csv("./data/peakGraph/graphs/1/mz_av.csv")["mz_av"])
df = df.set_index("mz_base")

# print(df.index)
# print(df["mz_av"])

# plt.plot(df.index, df['mz_av'])
# plt.show()

find_peaks("./data/peakGraph/graphs/1/mz_base.csv", "./data/peakGraph/graphs/1/mz_av.csv", "peakData", True)