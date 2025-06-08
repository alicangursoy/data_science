##################################################################
# PYTHON İLE VERİ ANALİZİ (DATA ANALYSIS WITH PYTHON)
##################################################################
# - NumPy
# - Pandas
# - Veri Görselleştirme: Matplotlib & Seaborn
# - Gelişmiş Fonksiyonel Keşifçi Veri Analizi (Advanced Functional Exploratory Data Analysis)


##################################################################
# NUMPY
##################################################################

# Neden NumPy? (Why NumPy?)
# NumPy Array'i Oluşturmak (Creating NumPy Arrays)
# NumPy Array Özellikleri (Attributes of NumPy Arrays)
# Yeniden Şekillendirme (Reshaping)
# Index Seçimi (Index Selection)
# Slicing
# Fancy Index
# NumPy'da Koşullu İşlemler (Conditions on NumPy)
# Matematiksel İşlemler (Mathematical Operations)

import numpy as np

a = [1, 2, 3, 4]
b = [2, 3, 4, 5]
ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b

##################################################################
# NumPy Array'i Oluşturmak (Creating NumPy Arrays)
##################################################################
import numpy as np

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))

np.zeros(10, dtype=int)
np.random.randint(0, 10, size=10)
np.random.normal(10, 4, (3, 4))

##################################################################
# NumPy Array Özellikleri (Attributes of NumPy Arrays)
##################################################################

# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtpye: array veri tipi

a = np.random.randint(0, 10, 5)
a.ndim
a.shape
a.size
a.dtype

##################################################################
# Yeniden Şekillendirme (Reshaping)
##################################################################
import numpy as np

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3, 3)

##################################################################
# Index Seçimi (Index Selection)
##################################################################
import numpy as np

a = np.random.randint(10, size=10)
a[0]
a[0:5]  # slicing de denir.
a[0] = 999

m = np.random.randint(10, size=(3, 5))
m[0, 0]
m[1, 1]
m[2, 3]
m[2, 3] = 999
m[2, 3] = 2.9  # float sayısı int'e çevirip kaydeder. Dolayısıyla burdaki değer 2 gelir.

m[:, 0]  # Tüm satırların 0.elemanlarını getir.
m[1, :]  # 1.satırın tüm sütunlarını getir.
m[0:2, 0:3]

##################################################################
# Fancy Index
##################################################################
import numpy as np

v = np.arange(0, 30, 3)
v[1]
v[4]
catch = [1, 2, 3]
v[catch]

##################################################################
# NumPy'da Koşullu İşlemler (Conditions on NumPy)
##################################################################
import numpy as np

v = np.array([1, 2, 3, 4, 5])

################
# Klasik döngü ile
################
ab = []
for i in v:
    if i < 3:
        ab.append(i)

#################
# NumPy ile
#################
v < 3
v[v < 3]
v[v > 3]
v[v != 3]
v[v == 3]

##################################################################
# Matematiksel İşlemler (Mathematical Operations)
##################################################################
import numpy as np

v = np.array([1, 2, 3, 4, 5])
v / 5
v * 5 / 10
v ** 2
v - 1
np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)

###############################################
# NumPy ile İki Bilinmeyenli Denklem Çözümü
###############################################
import numpy as np

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10
a = np.array([[5, 1], [1, 3]])
b = np.array([12, 10])
np.linalg.solve(a, b)

###############################################
# PANDAS
###############################################

# Pandas Series
# Veri Okuma (Reading Data)
# Veriye Hızlı Bakış (Quick Look At Data)
# Pandas'ta Seçim İşlemleri (Selection in Pandas)
# Toplulaştırma ve Gruplama (Aggregation & Grouping)
# Apply & Lambda
# Birleştirme (Join) İşlemleri


###############################################
# Pandas Series
###############################################
import pandas as pd

s = pd.Series([10, 77, 12, 4, 5])
type(s)
s.index
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head(3)
s.tail(3)

###############################################
# Veri Okuma (Reading Data)
###############################################
import pandas as pd

df = pd.read_csv("datasets/advertising.csv")
df.head()
# pandas cheatsheet


###############################################
# Veriye Hızlı Bakış (Quick Look at Data)
###############################################
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe()
df.describe().T  # .T: transpozunu alır.
df.isnull().values.any()  # Herhangi bir veri null mı?
df.isnull().sum()  # Her bir değişkende kaç tane eksik değer olduğu bilgisi
df["sex"].head()
df["sex"].value_counts()  # Her bir sex değerinden kaçar tane var?

###############################################
# Pandas'ta Seçim İşlemleri (Selection in Pandas)
###############################################
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.index
df[0:13]
df.drop(0, axis=0).head()

delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis=0).head(10)

# df = df.drop(delete_indexes, axis=0)
# df.drop(delete_indexes, axis=0, inplace=True)

############################
# Değişkeni Indexe Çevirmek
############################

df["age"].head()
df.age.head()

df.index = df["age"]
df.drop("age", axis=1).head()
df.drop("age", axis=1, inplace=True)
df.head()

############################
# Indexi Değişkene Çevirmek
############################
df.index

df["age"] = df.index

df.head()
df.drop("age", axis=1, inplace=True)

# df.reset_index().head()
df = df.reset_index()
df.head()

##############################
# Değişkenler Üzerine İşlemler
##############################
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

"age" in df

df["age"].head()
df.age.head()

df["age"].head()
type(df["age"].head())

df[["age"]].head()
type(df[[
    "age"]].head())  # NOT: Burda, [[]] kullandığımızda Data Frame elde ediyoruz. Eğer [] kullanılsaydı Serie elde edecektik.

df[["age", "alive"]]

col_names = ["age", "adult_male", "alive"]

df[col_names]

df["age2"] = df["age"] ** 2
df.head()
df["age3"] = df["age"] / df["age2"]
df.head()

df.drop("age3", axis=1).head()

df.drop(col_names, axis=1).head()

df.loc[:, ~df.columns.str.contains("age")].head()

##############################
# iloc & loc: integer based selection (iloc) & label based selection (loc)
##############################
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

# iloc: integer based selection
df.iloc[0:3]  # 0, 1 ve 2.index (satır) seçilir.
df.iloc[0, 0]

# loc: label based selection
df.loc[0:3]  # 0, 1, 2 ve 3.index (satır) seçilir.

# df.iloc[0:3, "age"]  # ValueError: Location based indexing can only have [integer, integer slice (START point is INCLUDED, END point is EXCLUDED), listlike of integers, boolean array] types
df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]

##############################
# Koşullu Seçim (Conditional Selection)
##############################
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head()
df[df["age"] > 50]["age"].count()

df.loc[df["age"] > 50, ["age", "class"]].head()
df_new = df.loc[(df["age"] > 50)
                & (df["sex"] == "male")
                & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
                ["age", "class", "embark_town"]]
df_new["embark_town"].value_counts()

##############################
# Toplulaştırma ve Gruplama (Aggreagation & Grouping)
##############################

# - count()
# - first()
# - last()
# - mean()
# - median()
# - min()
# - max()
# - std()
# - var()
# - sum()
# - pivot table
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()
df.groupby("sex")["age"].mean()
df.groupby("sex").agg({"age": "mean"})
df.groupby("sex").agg({"age": ["mean", "sum"]})

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "embark_town": "count"})

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town"]).agg({"age": ["mean", "sum"],
                                        "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean", "sum"],
                                                 "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({
    "age": ["mean", "sum"],
    "survived": "mean",
    "sex": "count"
    })


##############################
# Pivot Table
##############################
import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns",None)
df = sns.load_dataset("titanic")
df.head()
df.pivot_table("survived", "sex", "embarked")  # embarked yerine göre cinsiyetlere bağlı olarak hayatta kalma ortalaması
df.pivot_table("survived", "sex", "embarked", aggfunc="std")  # embarked yerine göre cinsiyetlere bağlı olarak hayatta kalmanın standart sapması
df.pivot_table("survived", "sex", ["embarked", "class"])
df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])  # cut ve qcut fonksiyonu, sayısal bir değişkeni kategorik bir değişkene dönüştürüyor.
df.head()
df.pivot_table("survived", "sex", ["new_age", "class"])
pd.set_option("display.width", 500)


##############################
# Apply & Lambda
##############################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 5
(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()
for col in df.columns:
    if "age" in col:
        print(col)

for col in df.columns:
    if "age" in col:
        print((df[col] / 10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col] / 10

df.head()

df[["age", "age2", "age3"]].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()


def standard_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()


df.loc[:, df.columns.str.contains("age")].apply(lambda x: standard_scaler(x)).head()
df.loc[:, df.columns.str.contains("age")].apply(standard_scaler).head()
df.loc[:, ["age", "age2", "age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standard_scaler)
df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standard_scaler)
df.head()


##############################
# Birleştirme (Join) İşlemleri
##############################
import numpy as np
import pandas as pd
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99
df1.head()
df2.head()
pd.concat([df1, df2])
pd.concat([df1, df2], ignore_index=True)


##############################
# Merge ile Birleştirme İşlemleri
##############################

df1 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'],
                    'group': ['accounting', 'engineering', 'engineering', 'hr']})

df2 = pd.DataFrame({'employees': ['mark', 'john', 'dennis', 'maria'],
                    'start_date': [2010, 2009, 2014, 2019]})

pd.merge(df1, df2)
pd.merge(df1, df2, on="employees")

# Amaç: Her çalışanın müdür bilgisine erişmek istiyoruz.
df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({'group': ['accounting', 'engineering', 'hr'],
                    'manager': ['Caner', 'Mustafa', 'Berkcan']})
df5 = pd.merge(df3, df4)
df5 = pd.merge(df3, df4, on='group')
df5.head()


##############################
# VERİ GÖRSELLEŞTİRME: MATPLOTLIB & SEABORN
##############################

##############################
# MATPLOTLIB
##############################

# Kategorik bir değişken ise: Sütun grafik ile görselleştiriyoruz. countplot bar
# Sayısal bir değişken ise: hist (historgram), boxplot


##############################
# Kategorik Değişken Görselleştirme
##############################
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind='bar')
plt.show()



##############################
# Sayısal Değişken Görselleştirme
##############################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()


##############################
# Matplotlib'in Özellikleri
##############################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

##############################
# plot
##############################

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()


##############################
# marker
##############################

y = np.array([13, 28, 11, 100])
plt.plot(y, marker='o')
plt.show()
plt.plot(y, marker='*')
plt.show()
markers = ['o', '*', '.', ',', 'x', 'X', '+', 'P', 's', 'D', 'd', 'p', 'H', 'h']


##############################
# line
##############################
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

y = np.array([13, 28, 11, 100])
plt.plot(y)
plt.show()

plt.plot(y, linestyle="dashed")
plt.show()

plt.plot(y, linestyle="dashdot")
plt.show()

plt.plot(y, linestyle="dashdot", color="r")
plt.show()


##############################
# Multiple Lines
##############################
x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()


##############################
# Labels
##############################
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x, y)
# Başlık
plt.title("Bu ana başlık")
# X eksenini isimlendirme
plt.xlabel("X ekseni isimlendirmesi")
# Y eksenini isimlendirme
plt.ylabel("Y ekseni isimlendirmesi")
plt.grid()  # Izgaralara ayırıyor çizelgeyi.
plt.show()


##############################
# Subplots
##############################
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 2, 1)
plt.title("1")
plt.plot(x, y)
plt.show()

# plot 2
x = np.array([8, 8, 9, 9, 10, 10, 11, 11, 12, 12])
y = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33])
plt.subplot(1, 2, 2)
plt.title("2")
plt.plot(x, y)
plt.show()


##############################
# SEABORN
##############################
import seaborn as sns
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt

df = sns.load_dataset("tips")
df.head()
df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show()

##############################
# Sayısal Değişkenleri Görselleştirme
##############################
sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()

sns.scatterplot(x=df["tip"], y=df["total_bill"], hue=df["smoker"], data=df)
plt.show()



####################################
# Kategorik Değişken Analizi
#####################################
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df["survived"].value_counts()
100 * df["survived"].value_counts() / len(df)

# Bizim için kategorik değişkenler (yani belli sayıda değer alabilen değişkenler) önemli bir belirteç veri biliminde.

# Bunun için önce kategorik değişkenleri buluyoruz. (cat_cols)

# Buna ek olarak, tipi nümerik olan fakat kategorik değişken gibi davranan değişkenleri buluyoruz. (num_but_cat)

# Bunları birbirine ekliyoruz.

# Kategorik değişkenlerden de belli bir kardinaliteden (farklı değer) fazla kardinaliteye sahip olanlar
# aslında bizim için kategorik değişken olarak değerlendiremeyeceğimiz için bunları listemizden çıkarmalıyız. (cat_but_car)

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["bool", "category", "object"]]  # str içinde vermek gerekiyor kategorik değişkenlerin tiplerini almak istediğimizde.
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
cat_cols = cat_cols + num_but_cat
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
cat_cols = [col for col in cat_cols if col not in cat_but_car]
df[cat_cols]


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#####################################")


cat_summary(df, "survived")


for col in cat_cols:
    cat_summary(df, col)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#####################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


cat_summary(df, "sex", plot=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("Not plottable")
    else:
        cat_summary(df, col, plot=True)


for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
    cat_summary(df, col, plot=True)


####################################
# Sayısal Değişken Analizi
#####################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df[["age", "fare"]].describe().T

num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
num_cols = [col for col in num_cols if col not in cat_cols]


def num_summary(dataframe, numerical_cols):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_cols].describe(quantiles).T)


num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)



####################################
# Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi
#####################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()


# docstring
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, nümerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe
    cat_th: int, float
        nümerik fakat kategorik olan değişkenker için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri
    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Nümerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi
    Notes
    -------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_car cat_cols'un içerisinde
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols
    """
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["bool", "category",
                                                                     "object"]]  # str içinde vermek gerekiyor kategorik değişkenlerin tiplerini almak istediğimizde.
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#####################################")


cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)


def num_summary(dataframe, numerical_cols, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_cols].describe(quantiles).T)
    if plot:
        dataframe[numerical_cols].hist()
        plt.xlabel(numerical_cols)
        plt.title(numerical_cols)
        plt.show(block=True)


num_summary(df, "age")

for col in cat_cols:
    num_summary(df, col, plot=True)


# bonus
df = sns.load_dataset("titanic")
df.info()
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)  # Burdaki amaç bool değişkenleri yakalayıp plotta 'bool' alan tipinden hata almasının önüne geçmek.

    cat_cols, num_cols, cat_but_car = grab_col_names(df)


####################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
#####################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)


# docstring
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, nümerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe
    cat_th: int, float
        nümerik fakat kategorik olan değişkenker için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri
    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Nümerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi
    Notes
    -------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_car cat_cols'un içerisinde
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols
    """
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["bool", "category",
                                                                     "object"]]  # str içinde vermek gerekiyor kategorik değişkenlerin tiplerini almak istediğimizde.
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()
df["survived"].value_counts()
cat_summary(df, "survived")


####################################
# Hedef Değişkenin Kategorik Değişkenler ile Analizi
#####################################
df.groupby("sex")["survived"].mean()  # Veri setini cinsiyete göre gruplayıp bunların survived değişkeninin ortalamasını bul.


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))


target_summary_with_cat(df, "survived", "sex")
target_summary_with_cat(df, "survived", "pclass")

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)


####################################
# Hedef Değişkenin Sayısal Değişkenler ile Analizi
#####################################
df.groupby("survived")["age"].mean()  # Veri setini, sayısal değişkenlerle analiz edeceksek, hedef değişkene göre gruplayıp devam ediyoruz.
df.groupby("survived").agg({"age": "mean"})  # Üstteki kullanımla aynı sonucu verir.


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")


target_summary_with_num(df, "survived", "age")

for col in num_cols:
    target_summary_with_num(df, "survived", col)



####################################
# 4. Korelasyon Analizi (Analysis of Correlation)
#####################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 2:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]
corr = df[num_cols].corr()
# Korelasyon: -1 ile 1 arasında değişen, iki değişkenin birbirine olan etkisini gösteren bir istatistiksel kavramdır.
# Korelasyon değeri ne kadar 1'e veya -1'e yakınsa , söz konusu olan 2 değişken birbirine o kadar bağlıdır.
# Veri analizi yaparken, bu değerin 1'e yakın olduğu kolonlar göz ardı edilir.

# Isı haritası oluşturma
sns.set(rc={"figure.figsize": (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()


####################################
# Yüksek Korelasyonlu Değikenlerin Silinmesi
#####################################
cor_matrix = df.corr().abs()

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]
cor_matrix[drop_list]
df.drop(drop_list, axis=1)


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib
        matplotlib.use('tkagg')
        from matplotlib import pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)


# Yaklaşık 60 mb'lık 300'den fazla değişkenin olduğu bir veri setinde deneyelim.
# https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_transaction.csv

df = pd.read_csv("datasets/train_transaction.csv")
len(df.columns)
df.head()

drop_list = high_correlated_cols(df, plot=True)
len(df.drop(drop_list, axis=1).columns)