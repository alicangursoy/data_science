import seaborn as sns

# Görev 1
df = sns.load_dataset("titanic")
df.head()

# Görev 2
df["sex"].value_counts()

# Görev 3
df.nunique()

# Görev 4
df["pclass"].nunique()

# Görev 5
df["pclass"].nunique()
df["parch"].nunique()

# Görev 6
df["embarked"].dtypes
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtypes

# Görev 7
df.loc[df["embarked"] == "C", :].head()

# Görev 8
df.loc[df["embarked"] != "S", :].head()

# Görev 9
df.loc[(df["age"] < 30) & (df["sex"] == "female"), :].head()

# Görev 10
df.loc[(df["fare"] > 500) | (df["age"] > 70), :].head()

# Görev 11
df.isnull().sum()

# Görev 12
df.drop("age", axis=1, inplace=True)

# Görev 13
deck_mode = df["deck"].mode()[0]
df.loc[df["deck"].isnull(), "deck"] = deck_mode

# Görev 14
age_median = df["age"].median()
df.loc[df["age"].isnull(), "age"] = age_median

# Görev 15
df.groupby(["sex", "pclass"]).agg({
    "survived": ["sum", "count", "mean"]})


# Görev 16
def getCategoryValue(age):
    if age < 30:
        return 1
    return 0


df["age_flag"] = df["age"].apply(lambda x: getCategoryValue(x))


# Görev 17
df = sns.load_dataset("tips")
df.head()

# Görev 18
df.groupby(["time"]).agg({
    "total_bill": ["sum", "min", "max", "mean"]})

# Görev 19
df.groupby(["time", "day"]).agg({
    "total_bill": ["sum", "min", "max", "mean"]})

# Görev 20
df.loc[(df["sex"] == "Female") & (df["time"] == "Lunch"), :].groupby(["day"]).agg({
    "total_bill": ["sum", "min", "max", "mean"],
    "tip": ["sum", "min", "max", "mean"]})

# Görev 21
df.loc[(df["size"] < 3) & (df["total_bill"] > 10), "total_bill"].mean()

# Görev 22
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]

# Görev 23
df2 = df.sort_values(by=["total_bill_tip_sum"], ascending=False).head(n=30)