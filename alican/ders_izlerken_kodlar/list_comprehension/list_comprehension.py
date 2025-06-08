# Görev 1:
import seaborn as sns
df = sns.load_dataset("car_crashes")
["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]

# Görev 2:
import seaborn as sns
df = sns.load_dataset("car_crashes")
[col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]


# Görev 3:
import seaborn as sns
df = sns.load_dataset("car_crashes")
og_list = ["abbrev", "no_previous"]
new_cols = [col for col in df.columns if col not in og_list]
new_df = df[new_cols]