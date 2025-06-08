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
df = pd.read_csv("diabetes.csv")
df.head()
df.shape
df.describe().T


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

# Tek bir kategorik değişken var ve bu da hedef değişkenimiz. Dolayısıyla kategorik değişkenlere göre hedef değişkenin ortalaması işlemi yapılamaz.
cat_cols
target = cat_cols[0]


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")


for col in num_cols:
    target_summary_with_num(df, target, col)


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
# Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb. değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir.
# Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik değerlere işlemleri uygulayabilirsiniz.

# Hedef değişkenin kolonu çıkarılıp geri kalan kolonları içeren dataframe parçası alınıyor.
df_not_targets = df.drop(["Outcome"], axis=1)
# 0'lar NULL ile değiştiriliyor.
df_not_targets.replace(0, np.nan, inplace=True)

# Değeri 0 olabilecek kolonların NaN olan değerleri 0 ile dolduruluyor.
num_cols_can_be_0 = ["DiabetesPedigreeFunction", "Pregnancies"]
for col in num_cols_can_be_0:
    df_not_targets[col] = df_not_targets[col].fillna(0)

# Hedef değişkenin kolonu çıkarılmış data frame ile hedef değişkeni içeren dataframe parçası birleştiriliyor.
df = pd.concat([df_not_targets, df["Outcome"]], axis=1)

# Sayısal değerleri median değeri ile dolduruluyor.
df = df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)


# Aykırı değerler tespit edilip limit değerlerle değiştiriliyor.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

# Adım 2: Yeni değişkenler oluşturunuz.
df.columns = [col.upper() for col in df.columns]
df.describe().T

# Hamileliğin diabet hastalığında bir etmen olduğunu düşünmediğim için, PREGNANCIES için değişken üretmedim.

# Yaşa göre değişken üretimi:
df.loc[df["AGE"] < 18, "NEW_AGE_CAT"] = "YOUNG"
df.loc[((df["AGE"] >= 18) & (df["AGE"] < 56)), "NEW_AGE_CAT"] = "MATURE"
df.loc[df["AGE"] >= 56, "NEW_AGE_CAT"] = "SENIOR"


# Kandaki glukoz değerine göre değişken üretimi
# Mayo Clinic'te araştırdığıma göre:
# A fasting blood sugar level less than 100 mg/dL (5.6 mmol/L) is normal.
# A fasting blood sugar level from 100 to 125 mg/dL (5.6 to 6.9 mmol/L) is considered prediabetes.
# If it's 126 mg/dL (7 mmol/L) or higher on two separate tests, you have diabetes.
df.loc[df["GLUCOSE"] < 100, "NEW_GLUCOSE_CAT"] = "NORMAL"
df.loc[((df["GLUCOSE"] >= 100) & (df["GLUCOSE"] < 125)), "NEW_GLUCOSE_CAT"] = "PREDIABETES"
df.loc[df["GLUCOSE"] >= 125, "NEW_GLUCOSE_CAT"] = "DIABETES"


# Kan basıncına göre değişken üretimi
# https://www.bloodpressureuk.org/your-blood-pressure/understanding-your-blood-pressure/what-do-the-numbers-mean/ web sitesinden araştırdığıma göre:
df.loc[df["BLOODPRESSURE"] < 60, "NEW_BLOODPRESSURE_CAT"] = "LOW"
df.loc[((df["BLOODPRESSURE"] >= 60) & (df["BLOODPRESSURE"] < 80)), "NEW_BLOODPRESSURE_CAT"] = "IDEAL"
df.loc[((df["BLOODPRESSURE"] >= 80) & (df["BLOODPRESSURE"] < 90)), "NEW_BLOODPRESSURE_CAT"] = "PREHIGH"
df.loc[df["BLOODPRESSURE"] >= 90, "NEW_BLOODPRESSURE_CAT"] = "HIGH"



# INSULIN değerleri için araştırmalarımda çok küçük sayılarla karşılaştım. Verilerle örtüştüremediğim şiçin kendim kategorileştireceğim.
df.loc[df["INSULIN"] < 121, "NEW_INSULIN_CAT"] = "NORMAL"
df.loc[((df["INSULIN"] >= 121) & (df["INSULIN"] < 127)), "NEW_INSULIN_CAT"] = "HIGH"
df.loc[df["INSULIN"] >= 127, "NEW_INSULIN_CAT"] = "VERYHIGH"

# BMI kategorileri için Mayo Clinic'ten yararlandım. https://www.mayoclinic.org/diseases-conditions/obesity/symptoms-causes/syc-20375742
df.loc[df["BMI"] < 18.5, "NEW_BMI_CAT"] = "UNDERWEIGHT"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25.0)), "NEW_BMI_CAT"] = "HEALTHY"
df.loc[((df["BMI"] >= 25.0) & (df["BMI"] < 30.0)), "NEW_BMI_CAT"] = "OVERWEIGHT"
df.loc[df["BMI"] >= 30.0, "NEW_BMI_CAT"] = "OBESITY"


# DIABETESPEDIGREEFUNCTION: Bu değişken için kategorileştirmek zor. Birkaç etmene bağlı bir değerin sonucunda bulunan değerler var.
# Bu yüzden DIABETESPEDIGREEFUNCTION için değişken üretmedim.


# Adım 3: Encoding işlemlerini gerçekleştiriniz.
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# cat_cols'tan hedef değişkeni çıkarttım.
cat_cols = [col for col in cat_cols if "OUTCOME" not in col]
df = one_hot_encoder(df, cat_cols)


# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
# num_cols'tan PREGNANCIES ve SKINTHICKNESS'ı çıkarttım
num_cols_to_exclude = ["PREGNANCIES", "SKINTHICKNESS"]
num_cols = [col for col in num_cols if col not in num_cols_to_exclude]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


# Adım 5: Model oluşturunuz.
df.columns
y = df["OUTCOME"]
X = df.drop(["OUTCOME", "SKINTHICKNESS", "BMI", "INSULIN", "BLOODPRESSURE", "GLUCOSE", "AGE", "PREGNANCIES"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=90)  # 1990'lı olduğum için 90 :)
rf_model = RandomForestClassifier(random_state=90).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)  # accuracy_score = 0.7083333333333334