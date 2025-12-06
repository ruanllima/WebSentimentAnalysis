import pandas as pd

data = pd.read_csv("./data/dataset.csv")
df = pd.DataFrame(data)
df = df.sample(frac=0.3)
print(df["Positive"].value_counts())