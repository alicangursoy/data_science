#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
matplotlib.use('tkagg')

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: '%.3f' % x)
pd.set_option("display.width", 500)


def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data


dff = load_application_train()
dff.head()


def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data


df = load()
df.head()

#############################################
# 1. Outliers (Aykırı Değerler)
#############################################


#############################################
# Aykırı Değerleri Yakalama
#############################################

#############################################
# Grafik Teknikle Aykırı Değerler
#############################################

sns.boxplot(x=df["Age"])
plt.show()


#############################################
# Aykırı Değerler Nasıl Yakalanır?
#############################################

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)

iqr = q3 - q1
up = q3 + (1.5 * iqr)
low = q1 - (1.5 * iqr)

df[(df["Age"] > up) | (df["Age"] < low)]
df[(df["Age"] > up) | (df["Age"] < low)].index

#############################################
# Aykırı Değer Var Mı Yok Mu?
#############################################
df[(df["Age"] > up) | (df["Age"] < low)].any(axis=None)  # Aykırı herhangi bir değer var mı?

df[~((df["Age"] > up) | (df["Age"] < low))]  # Aykırı olmayanları getir. ~ işareti NOT görevi görüyor.

df[~((df["Age"] > up) | (df["Age"] < low))].any(axis=None)  # Aykırı olmayan var mı? (Tabii ki var :))

df[(df["Age"] < low)].any(axis=None)

# Yaptıklarımızın özeti:
# 1. Eşik değer belirledik.
# 2. Aykırılara eriştik.
# 3. Hızlıca aykırı değer var mı yok mu diye sorduk.


#############################################
# İşlemleri Fonksiyonlaştırma
#############################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + (1.5 * interquantile_range)
    low_limit = quartile1 - (1.5 * interquantile_range)
    return low_limit, up_limit


outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

low, up = outlier_thresholds(df, "Age")

df[(df["Age"] < low) | (df["Age"] > up)]


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


check_outlier(df, "Age")
check_outlier(df, "Fare")


########################################
# grab_col_names
########################################

df.head()


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, nümerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine nümerik görünümlü kategorik değişkenler de dahildir.

    :param dataframe: Değişken isimleri istenilen dataframe
    :param cat_th: int, optional: Nümerik fakat kategorik olan değişkenler için sınıf eşik değeri
    :param car_th: int, optional: Kategorik fakat kardinal değişkenler için sınıf eşik değeri
    :return:
        cat_cols: list: Kategorik değişken listesi
        num_cols: list: Nümerik değişken listesi
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
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

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col))

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]
for col in num_cols:
    print(col, check_outlier(dff, col))


##########################################
# Aykırı Değerlerin Kendilerine Erişmek
##########################################
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


grab_outliers(df, "Age")
age_index = grab_outliers(df, "Age", True)


##########################################
# Aykırı Değer Problemini Çözme
##########################################


##############
# Silme
##############
low, up = outlier_thresholds(df, "Fare")
df.shape

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((df[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0]


#######################################
# Baskılama Yöntemi (re-assignment with thresholds)
#######################################

low, up = outlier_thresholds(df, "Fare")

df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]

df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]

df.loc[(df["Fare"] > up), "Fare"] = up

df.loc[(df["Fare"] < low), "Fare"] = low


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))


############################
# Recap
############################
df = load()
outlier_threshholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", index=True)
remove_outlier(df, "Age")
replace_with_thresholds(df, "Age")
check_outlier(df, "Age")


################################################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
################################################################

df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=["float64", "int64"])
df = df.dropna()
df.head()

for col in df.columns:
    print(col, check_outlier(df, col))

low, up = outlier_threshholds(df, "carat")
df[((df["carat"] < low) | (df["carat"] > up))].shape

low, up = outlier_threshholds(df, "depth")
df[((df["depth"] < low) | (df["depth"] > up))].shape

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores[0:5]
# df_scores = -df_scores  # df_scores değerleri negatif geliyor. Pozitiflerle çalışmak istersek bu satırı çalıştırmalıyız.

np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()
# Threshold belirlerken, dirsek yöntemi ile belirleyebiliriz.
# Dirsek yöntemi, iki nokta arasındaki değişimin en son yüksek olduğu yer noktayı threshold olarak almaya dayanıyor.
# Bu noktadan sonra değişimler, bu değişime göre çok daha küçüktür.
# Bu örnek için 3.index'teki (baştan 4.eleman yani) noktayı threshold kabul ettik.

th = np.sort(df_scores)[3]

df[df_scores < th]

df[df_scores < th].shape

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)


############################################
# Missing Values (Eksik Değerler)
############################################

############################################
# Eksik Değerlerin Yakalanması
############################################

df = load()
df.head()

# eksik değer var mı yok mu gözlem sorusu
df.isnull().values.any()

# değişkenlerdeki eksik değer sayısı
df.isnull().sum()

# değişkenlerdeki tam değer sayısı
df.notnull().sum()

# veri setindeki toplam eksik değer sayısı
df.isnull().sum().sum()

# en az bir tane eksik değere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)
missing_values_table(df, True)


############################################
# Eksik Değer Problemini Çözme
############################################
missing_values_table(df)


########################################
# Çözüm 1: Hızlıca silmek
########################################

df.dropna().shape


################################################
# Çözüm 2: Basit Atama Yöntemleri İle Doldurmak
################################################

df["Age"].fillna(df["Age"].mean())

df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

# df.apply(lambda x: x.fillna(x.mean()), axis=0)  # Hata alıyor, çünkü kolonlarda sayısal olmayan kolonlar var. Bunlarda, ortalama alamaz.

df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()
# isnull().sum()'ı, öncesindeki kodun düzgümn çalışıp çalışmadığını gözlemlemek için yazıyoruz. 0 olursa sonuç, işlem başarılı.

df["Embarked"].fillna("missing")

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()


######################################
# Kategorik Değişken Kırılımında Değer Atama
######################################
df.groupby("Sex")["Age"].mean()
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().mean()

df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.groupby("Sex")["Age"].mean()["male"]
df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()


################################################
# Çözüm 3: Tahmine Dayalı Atama İle Doldurma
################################################
df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
dff.head()

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# knn'in (bir makine öğrenmesi çeşidi) uygulanması
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
# inverse_transform yapma sebebimiz, scaler ile 0 ile 1 arasına scale etmiştik tüm kolonları.
# Age kolonundaki değerlerin gerçek değerlerini n_neighbors = 5 ile yani 5 komşusunun ortalama değeri ile doldurma komutu ile boşları doldurmuştuk.
# Age 0 ile 1 arasındayken düzgün anlaşılmıyor. O yüzden scaler'ın inverse_transform metodunu kullanarak 0-1 aralığına almadan önceki haline döndürebiliyoruz.
dff.head()

df["age_imputted_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(), ["Age", "age_imputted_knn"]]




##################
# Recap
##################


########################################################
# Gelişmiş Analizler
########################################################

########################################
# Eksik Veri Yapısının İncelenmesi
########################################

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

# heatmap, boş olan değerlere sahip kolonların, birbirleriyle olan correlation'ını ortaya çıkarmayı hedefler.
# Eğer, değerler pozitif veya negatif 0.60'dan fazlaysa, o zaman birbirleriyle bağlantıları olma ihtimali yüksektir.
# titanic veri setinde, 3 adet kolonda eksiklik var, alttaki heatmap çalıştırınca bunların birbirleriyle olan correlation değerleri
# +0.1, 0, -0.1 şeklinde olduğu için bunların null olma sebebi birbiriyle bağlantılıdır diyemeyiz.
msno.heatmap(df)
plt.show()


#####################################################################
# Eksik Değerlerin Bağımlı Değişken İle İlişkisinin Değerlendirilmesi
#####################################################################

na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "COUNT": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Survived", na_cols)


# Veriyi yükledik.
df = load()
# missing table
missing_values_table(df)
# sayısal değişkenleri direkt median ile doldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile doldurma
missing_vs_target(df, "Survived", na_cols)



################################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
################################################


################################################
# Label Encoding & Binary Encoding
################################################

df = load()
df.head()
df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]  # le.fit_transform, değerlere alfabetik olarak değer verir. female = 0, male = 1
le.inverse_transform([0, 1])  # le.inverse_transform'da inputta verilen değerleri, hangi orijinal değerler için verdiğini yazıyor.


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


df = load()
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
for col in binary_cols:
    label_encoder(df, col)


df = load_application_train()
df.shape
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
df[binary_cols].head()
for col in binary_cols:
    label_encoder(df, col)


############################################
# One-Hot Encoding
############################################

df = load()
df.head()
df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"]).head()
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()
# İlk sınıfı (alfabetik olarak) uçurmamız gerekiyor, nedeni ise eğer uçurmazsak dummy değişkenler yüksek correlation oluşturuyor.
# Kendi eklediğimiz verinin yüksek correlation oluşturarak veri analizini bozmasını istemeyiz.
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()

pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe


df = load()
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols).head()


#####################################################
# Rare Encoding
#####################################################

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi
# 3. Rare encoding yazacağız.


########################################################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi
########################################################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("############################################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)


##################################################################################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi
##################################################################################
df["NAME_INCOME_TYPE"].value_counts()
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()


def rare_analyzer(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyzer(df, "TARGET", cat_cols)


##########################################################
# 3. Rare encoder'ın yazılması.
##########################################################
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df


new_df = rare_encoder(df, 0.01)
new_df.head()

rare_analyzer(new_df, "TARGET", cat_cols)
df["OCCUPATION_TYPE"].value_counts()



############################################################
# Feature Scaling (Özellik Ölçeklendirme)
############################################################

######################################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
######################################
df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()


######################################
# RobustScaler: Medyanı çıkar iqr'a böl. Aykırı değerlerden, StandardScaler'a göre daha az etkileniyor.
######################################
rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T


######################################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
######################################
mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T


age_cols = [col for col in df.columns if "Age" in col]


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in age_cols:
    num_summary(df, col, plot=True)


######################################
# Numeric to Categorical: Sayısal Değişkenleri Kategorik Değişkenlere Çevirme
# Binning
######################################

df["Age_qcut"] = pd.qcut(df["Age"], 5)  # qcut metodu, değerleri küçükten büyüğe sıralar ve çeyrek değerlere göre böler.
df.head()


############################################################
# Feature Extraction (Özellik Çıkarımı)
############################################################

#################################################
# Binary Features: Flag, Bool, True-False
#################################################

df = load()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],
                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]]
                                      )

print("Test Stat = %.4f, p-value: %.4f" % (test_stat, pvalue))

df.loc[((df["SibSp"] + df["Parch"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SibSp"] + df["Parch"]) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],
                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]]
                                      )

print("Test Stat = %.4f, p-value: %.4f" % (test_stat, pvalue))




######################################################################
# Text'ler Üzerinden Özellik Türetmek
######################################################################
df = load()
df.head()


################################################
# Letter Count
################################################

df["NEW_NAME_COUNT"] = df["Name"].str.len()


################################################
# Word Count
################################################

df["NEW_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

################################################
# Özel Yapıları Yakalamak
################################################

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df.groupby("NEW_NAME_DR").agg({"Survived": "mean"})


################################################
# Regex İle Değişken Türetmek
################################################
df.head()
df["NEW_TITLE"] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)  # df.Name = df["Name"]
df[["NEW_TITLE", "Survived", "Age"]].groupby("NEW_TITLE").agg({"Survived": "mean", "Age": ["count", "mean"]})


################################################
# Date Değişkenleri Üretmek
################################################
dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
dff.info()

dff["Timestamp"] = pd.to_datetime(dff["Timestamp"], format="ISO8601")

# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
dff["month_diff"] = (date.today().year - dff["Timestamp"].dt.year) * 12 + (date.today().month - dff["Timestamp"].dt.month)

# day name
dff["day_name"] = dff["Timestamp"].dt.day_name()
dff.head()

df.loc[(df["Sex"] == "female") & ((df["Age"] > 21) & (df["Age"] <= 50)), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["Sex"] == "female") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniorfemale"
df.head()

df.groupby("NEW_SEX_CAT")["Survived"].mean()


############################################################
# Titanic Uçtan Uca Feature Engineering & Data Preprocessing
############################################################
df = load()
df.shape
df.head()
df.columns = [col.upper() for col in df.columns]

#############################################################
# 1. Feature Engineering (Değişken Mühendisliği)
#############################################################
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')

df["NEW_NAME_COUNT"] = df["NAME"].str.len()

df["NEW_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))

df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df["NEW_TITLE"] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)  # df.Name = df["Name"]

df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1

df.loc[((df["SIBSP"] + df["PARCH"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SIBSP"] + df["PARCH"]) == 0), "NEW_IS_ALONE"] = "YES"

df.loc[df["AGE"] < 18, "NEW_AGE_CAT"] = "young"
df.loc[((df["AGE"] >= 18) & (df["AGE"] < 56)), "NEW_AGE_CAT"] = "mature"
df.loc[df["AGE"] > 56, "NEW_AGE_CAT"] = "senior"


df.loc[(df["SEX"] == "male") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngmale"
df.loc[(df["SEX"] == "male") & ((df["AGE"] > 21) & (df["AGE"] <= 50)), "NEW_SEX_CAT"] = "maturemale"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniormale"

df.loc[(df["SEX"] == "female") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngfemale"
df.loc[(df["SEX"] == "female") & ((df["AGE"] > 21) & (df["AGE"] <= 50)), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniorfemale"

df.shape
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]


#############################################################
# 2. Outliers (Aykırı Değerler)
#############################################################

for col in num_cols:
    print(col, check_outlier(df, col))

df.head()
# year
dff["year"] = dff["Timestamp"].dt.year

# month
dff["month"] = dff["Timestamp"].dt.month

# year diff
dff["year_diff"] = date.today().year - dff["Timestamp"].dt.year


#############################################
# Feature Interactions (Özellik Etkileşimleri)
#############################################
df = load()
df.head()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'

df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df.head()

df.groupby("NEW_SEX_CAT")["Survived"].mean()


#############################################
# Titanic Uçtan Uca Feature Engineering & Data Preprocessing
#############################################

df = load()
df.shape
df.head()

df.columns = [col.upper() for col in df.columns]

#############################################
# 1. Feature Engineering (Değişken Mühendisliği)
#############################################

# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

#############################################
# 2. Outliers (Aykırı Değerler)
#############################################

for col in num_cols:
    print(col, check_outlier(df, col))
for col in num_cols:
    replace_with_thresholds(df, col)
for col in num_cols:
    print(col, check_outlier(df, col))

#############################################
# 3. Missing Values (Eksik Değerler)
#############################################

missing_values_table(df)

df.drop("CABIN", inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)


df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))


df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)


#############################################################
# 4. Label Encoding
#############################################################

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

df.head()

for col in binary_cols:
    df = label_encoder(df, col)


################################################
# 5. Rare Encoding
################################################

rare_analyzer(df, "SURVIVED", cat_cols)

df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts()


################################################
# 6. One-Hot Encoding
################################################

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)
df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "PASSENGERID" not in col]
rare_analyzer(df, "SURVIVED", cat_cols)


useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# df.drop(useless_cols, axis=1, inplace=True)  # Bize çok bir anlam ifade etmeyen kolonları silmeyi tercih edebiliriz.


################################################
# 7. Standard Scaler
################################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()

df.head()
df.shape


################################################
# 8. Model
################################################

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
# X_train: kurulacak modelin kullanacağı veri seti
# X_test: test için kullanılacak veri seti

from sklearn.ensemble import RandomForestClassifier
# Ağaç temelli yöntem

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)  # accuracy_score = 0.8059701492537313


####################################################
# Hiçbir işlem yapılmadan elde edilecek skor?
####################################################
dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)  # accuracy_score = 0.7090909090909091

# Veri ön işleme işlemleri yapılarak, daha isabetli tahminlerde bulunulabiliyor.
# Eğer boş veriler silinmezse, RandomForestClassifer çalışmıyor. Veri setinde boş değer olmamalı.
# Label encoding yapılmazsa, makine öğrenmesi yine anlamıyor.


# Yeni ürettiğimiz değişkenler ne alemde?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


plot_importance(rf_model, X_train)


var1 = [-1,-3,4,5,2,-6]
var2 = [0,-1,0,-3,9,4]
var3 = [0,3,45,78,21,5]
var4 = [45,27,4,4,2.34]

np.log(var4)



dfa = pd.DataFrame({"date": ['2014-05-06', '2014-05-13', '2014-05-09' ]})
dfa["date"].str.extract(r"(\d{4})", expand=True)

bdatetime_series = pd.Series(pd.date_range("2015-07-04", periods=4, freq="M"))
df = pd. DataFrame({"date": bdatetime_series})
df["day"] = df.date.dt.day_name()


