import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage, dendrogram



##########################################
# Görev 1: Veriyi Hazırlama
##########################################

# Adım 1: flo_data_20K.csv verisini okutunuz.
df = pd.read_csv("flo_customer_segmentation/flo_data_20k.csv")
df.head()
df.shape
df.info()
df.describe().T


# Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["tenure"] = (date.today().year - df["first_order_date"].dt.year) * 365 \
                + (date.today().month - df["first_order_date"].dt.month) * 30 \
                + (date.today().day - df["first_order_date"].dt.day)

df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["recency"] = (date.today().year - df["last_order_date"].dt.year) * 365 \
                + (date.today().month - df["last_order_date"].dt.month) * 30 \
                + (date.today().day - df["last_order_date"].dt.day)

df["total_spent"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

df["online_buyer"] = df.apply(lambda x: 1 if x['order_num_total_ever_online'] >= x['order_num_total_ever_offline'] else 0, axis=1)

# Çocuklar mağazalara kendi kendine gidip alışveriş yapabilir, fakat online alışverişi ebeveyn gözetiminde yapabilir.
df["is_parent"] = df.apply(lambda x: 1 if (x["order_num_total_ever_online"] > 0) & ("COCUK" in x["interested_in_categories_12"]) else 0, axis=1)

df["is_sportive"] = df.apply(lambda x: 1 if "SPOR" in x["interested_in_categories_12"] else 0, axis=1)
df["brand_lover"] = df.apply(lambda x: 1 if len(x["interested_in_categories_12"].strip("[]").split(',')) > 2 else 0, axis=1)


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
    return cat_cols, num_cols, cat_but_car, num_but_cat


cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

# Seçilen değişkenler
num_cols = [col for col in num_cols if col not in ['first_order_date', 'last_order_date']]
df_to_segment = df[num_cols]

################################################
# Görev 2: K-Means ile Müşteri Segmentasyonu
################################################

# Adım 1: Değişkenleri standartlaştırınız.
df_to_segment.dropna(inplace=True)
df_to_segment = MinMaxScaler().fit_transform(df_to_segment)

# Adım 2: Optimum küme sayısını belirleyiniz.
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df_to_segment)
elbow.show()
elbow.elbow_value_  # 6

# Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df_to_segment)
kmeans.n_clusters
kmeans.cluster_centers_
df["segment"] = kmeans.labels_


# Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz.
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("############################################################")


cat_summary(df, "segment")


#############################################################
# Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
#############################################################

# Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
hc_average = linkage(df_to_segment, "average")
plt.figure(figsize=(20, 20))
plt.title("Dendrograms")
dend = dendrogram(hc_average, truncate_mode="lastp", p=50, show_contracted=True, leaf_font_size=10)
plt.axhline(y=0.6, color="r", linestyle="--")
plt.show()

# Adım 2: Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
cluster = AgglomerativeClustering(n_clusters=9, linkage="average")
clusters = cluster.fit_predict(df_to_segment)
df["hi_segment"] = clusters

# Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz.
cat_summary(df, "hi_segment")


