import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: '%.3f' % x)
pd.set_option("display.width", 500)


##########################################
# Görev 1: Keşifçi Veri Analizi
##########################################

df = pd.read_csv("telco_customer_churn/Telco-Customer-Churn.csv")

# Kolon isimleri büyük harf yapılıyor.
df.columns = [col.upper() for col in df.columns]


# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
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
df.info()


# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

# Sadece Yes ve No değerlerini içeren kategorik değişkenlerde, Yes'e 1 No'ya 0 atanıyor.
def convert_yes_no_to_one_zero(dataframe, categorical_col):
    if len(dataframe[categorical_col].value_counts().index) == 2:
        dataframe[categorical_col] = dataframe.apply(lambda x: 1 if x[categorical_col] == "Yes" else (0 if x[categorical_col] == "No" else x[categorical_col]), axis=1)


for col in cat_cols:
    convert_yes_no_to_one_zero(df, col)

# TOTALCHARGES değişkeni dışında diğer değişkenlerde bir sorun görmedim. TOTALCHARGES değişkeninde 11 adet boşluk karakteri olduğu için aslında nümerik olması gereken bu değişken,
# kategorikmiş gibi değerlendirilmiş. Öncelikle 11 adet boşluk karakterini silip yerlerine median değerini yazacağız.
df["TOTALCHARGES"].replace(' ', np.nan, inplace=True)
df["TOTALCHARGES"] = df["TOTALCHARGES"].astype(float)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Sayısal değerler median değeri ile dolduruluyor.
df = df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)


# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
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


# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


target = "CHURN"
for col in cat_cols:
    target_summary_with_cat(df, target, col)


# Adım 5: Aykırı gözlem var mı inceleyiniz.
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

# Aykırı değer yok.


# Adım 6: Eksik gözlem var mı inceleyiniz.
# Eksik gözlem başta da yoktu fakat TOTALCHARGES'ta boşluk karakteri vardı. Onu düzelttim. Tekrar bir sorgulayalım eksik değer var mı?
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df, True)
# Eksik gözlem yok.


##########################################
# Görev 2 : Feature Engineering
##########################################
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
# Eksik veya aykırı değer yok.

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
cat_cols = [col for col in cat_cols if target not in col]
df = one_hot_encoder(df, cat_cols)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
# Nümerik değişkenleri, yeni değişkenler üreterek kategorik hale çevirdiğim için, standartlaştırma işlemi yapmaya gerek yok.


##########################################
# Görev 3 : Modelleme
##########################################

# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.

# Hedef değişken ve bağımlı değişkenler ayrıştırılıp train/test olarak veri setleri ayrıştırılıyor.
y = df["CHURN"]
X = df.drop(["CHURN", "CUSTOMERID", "TENURE", "MONTHLYCHARGES", "TOTALCHARGES"], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=90)  # 1990'lı olduğum için 90 :)

# Model 1: Random Forest
rf_model = RandomForestClassifier(random_state=90).fit(X, y)
cv_results = cross_validate(rf_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.7745303124395122
cv_results["test_f1"].mean()  # 0.5303611746023076
cv_results["test_roc_auc"].mean()  # 0.7994201055746234


# Model 2: Logistic Regression
log_model = LogisticRegression().fit(X, y)
cv_results = cross_validate(log_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.8042033961868507
cv_results["test_f1"].mean()  # 0.5867185331091191
cv_results["test_roc_auc"].mean()  # 0.8474560580469899

# Model 3: KNN Model
knn_model = KNeighborsClassifier().fit(X, y)
cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.768993463287954
cv_results["test_f1"].mean()  # 0.5485975363096212
cv_results["test_roc_auc"].mean()  # 0.7853562486579027

# Model 4: LightGBM
light_gbm_model = LGBMClassifier()
cv_results = cross_validate(light_gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.7959704053487322
cv_results["test_f1"].mean()  # 0.5794439442778554
cv_results["test_roc_auc"].mean()  # 0.8336772281794221

# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile modeli tekrar kurunuz.
# En iyi sonuçları Logistic Regression Model ve LightGBM verdi. LightGBM'de hyperparametre optimizasyonu ile skorları daha yukarı çekebileceğimizi düşündüğüm için
# LightGBM'i seçtim.
light_gbm_params = {"learning_rate": [0.001, 0.01, 0.1],
                   "n_estimators": [300, 500, 700]}
gs_best = GridSearchCV(light_gbm_model, light_gbm_params, cv=5, n_jobs=-1, verbose=False).fit(X, y)
light_gbm_final_model = light_gbm_model.set_params(**gs_best.best_params_)
cv_results = cross_validate(light_gbm_final_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.799803011484612
cv_results["test_f1"].mean()  # 0.5824607166439133
cv_results["test_roc_auc"].mean()  # 0.8407794040141751

# Hyperparametre optimizasyonuna rağmen, LightGBM'in skorları Logistic Regression Model'in skorlarını geçemedi.
# Dolayısıyla 4 modelden en iyisi Logistic Regression Model gibi görünmektedir.



