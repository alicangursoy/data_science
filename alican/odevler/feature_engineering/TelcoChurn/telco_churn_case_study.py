import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: '%.3f' % x)
pd.set_option("display.width", 500)


##########################################
# Görev 1: Keşifçi Veri Analizi
##########################################


# Adım 1: Genel resmi inceleyiniz.
df = pd.read_csv("Telco-Customer-Churn.csv")
df.head()
df.shape
df.describe().T
df.info()

# Kolon isimleri büyük harf yapılıyor.
df.columns = [col.upper() for col in df.columns]


# Adım 2: Nümerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"length of cat_cols: {len(cat_cols)}")
    print(f"length of num_cols: {len(num_cols)}")
    print(f"length of cat_but_car: {len(cat_but_car)}")
    print(f"length of num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Adım 3: Nümerik ve kategorik değişkenlerin analizini yapın.
def num_summary(dataframe, numerical_cols):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_cols].describe(quantiles).T)


for col in num_cols:
    num_summary(df, col)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("############################################################")


for col in cat_cols:
    cat_summary(df, col)


# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

############################################################
# Hedef değişkenin kategorik değişkenlere göre ortalaması
############################################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))


# Hedef değişkenin ortalamasını almak için, Yes/No'yu 1/0'a çeviriyoruz.
df["CHURN"] = df.apply(lambda x: 1 if x["CHURN"] == "Yes" else 0, axis=1)

for col in cat_cols:
    target_summary_with_cat(df, "CHURN", col)


############################################################
# Hedef değişkene göre nümerik değişkenlerin ortalaması
############################################################
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")


for col in num_cols:
    target_summary_with_num(df, "CHURN", col)


# Adım 5: Aykırı gözlem analizi yapınız.
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + (1.5 * interquantile_range)
    low_limit = quartile1 - (1.5 * interquantile_range)
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))


# Adım 6: Eksik gözlem analizi yapınız.
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df, True)


# Adım 7: Korelasyon analizi yapınız.
corr = df[num_cols].corr()


##########################################
# Görev 2 : Feature Engineering
##########################################

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız.

# Eksik değer yok. Fakat, boşluk karakteriyle doldurulanlar var. Dolayısıyla boşluk karakterlerini NULL ile değiştiriyoruz.
df_not_targets = df.drop(["CHURN"], axis=1)
df_not_targets.replace(' ', np.nan, inplace=True)

# Hedef değişkenin kolonu çıkarılmış data frame ile hedef değişkeni içeren dataframe parçası birleştiriliyor.
df = pd.concat([df_not_targets, df["CHURN"]], axis=1)

# Sayısal değerleri median değeri ile dolduruluyor.
df["TOTALCHARGES"] = df["TOTALCHARGES"].astype(float)
df = df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

# Kategorik değerleri mode değeri ile dolduruluyor.
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)



# Aykırı değer yok.
for col in num_cols:
    print(col, check_outlier(df, col))

# Adım 2: Yeni değişkenler oluşturunuz.

# TENURE'e göre yeni değişkenler
df.loc[df["TENURE"] < 9, "NEW_TENURE_CAT"] = "VERYSHORT"
df.loc[((df["TENURE"] >= 9) & (df["TENURE"] < 21)), "NEW_TENURE_CAT"] = "SHORT"
df.loc[((df["TENURE"] >= 21) & (df["TENURE"] < 33)), "NEW_TENURE_CAT"] = "REGULAR"
df.loc[((df["TENURE"] >= 33) & (df["TENURE"] < 55)), "NEW_TENURE_CAT"] = "LONG"
df.loc[df["TENURE"] >= 55, "NEW_TENURE_CAT"] = "VERYLONG"

# MONTHLYCHARGES'a göre yeni değişkenler
df.loc[df["MONTHLYCHARGES"] < 21, "NEW_MONTHLYCHARGES_CAT"] = "VERYLOW"
df.loc[((df["MONTHLYCHARGES"] >= 21) & (df["MONTHLYCHARGES"] < 35)), "NEW_MONTHLYCHARGES_CAT"] = "LOW"
df.loc[((df["MONTHLYCHARGES"] >= 35) & (df["MONTHLYCHARGES"] < 65)), "NEW_MONTHLYCHARGES_CAT"] = "STANDARD"
df.loc[((df["MONTHLYCHARGES"] >= 65) & (df["MONTHLYCHARGES"] < 89)), "NEW_MONTHLYCHARGES_CAT"] = "HIGH"
df.loc[df["MONTHLYCHARGES"] >= 90, "NEW_MONTHLYCHARGES_CAT"] = "VERYHIGH"

# TOTALCHARGES'a göre yeni değişkenler
df.loc[df["TOTALCHARGES"] < 21, "NEW_TOTALCHARGES_CAT"] = "VERYLOW"
df.loc[((df["TOTALCHARGES"] >= 21) & (df["TOTALCHARGES"] < 402)), "NEW_TOTALCHARGES_CAT"] = "LOW"
df.loc[((df["TOTALCHARGES"] >= 402) & (df["TOTALCHARGES"] < 1398)), "NEW_TOTALCHARGES_CAT"] = "STANDARD"
df.loc[((df["TOTALCHARGES"] >= 1398) & (df["TOTALCHARGES"] < 3794)), "NEW_TOTALCHARGES_CAT"] = "HIGH"
df.loc[df["TOTALCHARGES"] >= 3794, "NEW_TOTALCHARGES_CAT"] = "VERYHIGH"


# Adım 3: Encoding işlemlerini gerçekleştiriniz.
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe


cat_cols, num_cols, cat_but_car = grab_col_names(df)
# cat_cols'tan hedef değişkeni çıkarttım.
cat_cols = [col for col in cat_cols if "CHURN" not in col]
df = one_hot_encoder(df, cat_cols)
df.columns = [col.upper() for col in df.columns]

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
# Nümerik değişkenleri, yeni değişkenler üreterek kategorik hale çevirdiğim için, standartlaştırma işlemi yapmaya gerek yok.


# Adım 5: Model oluşturunuz.
y = df["CHURN"]
X = df.drop(["CHURN", "CUSTOMERID", "TENURE", "MONTHLYCHARGES", "TOTALCHARGES"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=90)  # 1990'lı olduğum için 90 :)


rf_model = RandomForestClassifier(random_state=90).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)  # accuracy_score = 0.7802385008517888
