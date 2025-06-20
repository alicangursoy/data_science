############################################
# Sorting Products
############################################


############################################
# Uygulama: Kurs Sıralama
############################################
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/product_sorting.csv")
print(df.shape)
df.head(10)

############################################
# Sorting by Rating
############################################
df.sort_values("rating", ascending=False).head(20)

############################################
# Sorting by Comment Count or Purchase Count
############################################
df.sort_values("purchase_count", ascending=False).head(20)
df.sort_values("comment_count", ascending=False).head(20)

############################################
# Sorting by Rating, Comment, and Purchase
############################################
from sklearn.preprocessing import MinMaxScaler

# Bu sıralama probleminde bizim için önemli olan 3 faktör var: Rating, Comment, Purchase
# Rating 1-5 arasında değerler alabiliyor.
# Comment ve Purchase'ın Rating'in değerini ezmemesi için, bu 2 faktörü de Rating gibi 1-5 arasında değişecek şekilde scale ediyoruz.
df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)).fit(df[["purchase_count"]]).transform(
    df[["purchase_count"]])

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

# Yukarıdaki iki kod da aynı şey, \'ı alt satıra geçmek için kullanıyoruz.

df.describe().T

df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["comment_count"]]). \
    transform(df[["comment_count"]])

(df["comment_count_scaled"] * 32 / 100 +
 df["purchase_count_scaled"] * 26 / 100 +
 df["rating"] * 42 / 100)


def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["comment_count_scaled"] * w1 / 100
            + dataframe["purchase_count_scaled"] * w2 / 100
            + dataframe["rating"] * w3 / 100)


df["weighted_sorting_score"] = weighted_sorting_score(df)

df.sort_values("weighted_sorting_score", ascending=False).head(20)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)


################################################
# Bayesian Average Rating Score (BAR score)
################################################


# Sorting Products with 5 Star Rated
# Sorting Products According to Distribution of 5 Star Rating

import math
import scipy.stats as st


# Puan dağılımlarının üzerinden ağırlıklı bir şekilde olasılıksal ortalama hesabı yapar.
def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


df.head()

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)

df.sort_values("weighted_sorting_score", ascending=False).head()
df.sort_values("bar_score", ascending=False).head()

df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False)


########################################################################
# Hybrid Sorting: BAR Score + Diğer Faktörler
########################################################################

def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return bar_score * bar_w / 100 + wss_score * wss_w / 100


df["hybrid_sorting_score"] = hybrid_sorting_score(df)

# BAR score, potansiyelli verileri üst sıralara taşımada önemlidir.

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)


########################################################################
# Uygulama: IMDB Movie Scoring & Sorting
########################################################################
import pandas as pd
import math
import scipy.stats as stat
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/movies_metadata.csv", low_memory=False)  # DtypeWarning kapatmak için
df = df[["title", "vote_average", "vote_count"]]
df.head()
df.shape


########################################################################
# Vote Average'a Göre Sıralama
########################################################################

df.sort_values("vote_average", ascending=False).head(20)
df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T
df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head(20)

from sklearn.preprocessing import MinMaxScaler

df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)). \
    fit(df[["vote_count"]]).transform(df[["vote_count"]])

df["average_vote_count_score"] = df["vote_count_score"] * df["vote_average"]

df.sort_values("average_vote_count_score", ascending=False).head(20)


########################################################################
# IMDB Weighted Rating
########################################################################

# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)
# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)

# Film 1:
# r = 8
# M = 500
# v = 1000

# (1000 / (1000+500)) * 8 = 5.33


# Film 2:
# r = 8
# M = 500
# v = 3000

# (3000 / (3000+500)) * 8 = 6.85


# Film 1:
# r = 8
# M = 500
# v = 1000

# Birinci bölüm:
# (1000 / (1000+500)) * 8 = 5.33

# İkinci bölüm:
# (500 / (1000+500)) * 7 = 2.33

# Toplam = 5.33 + 2.33 = 7.66


# Film 2:
# r = 8
# M = 500
# v = 3000

# Birinci bölüm:
# (3000 / (3000+500)) * 8 = 6.85

# İkinci bölüm:
# (500 / (3000+500)) * 7 = 1

# Toplam = 5.33 + 2.33 = 7.85


M = 2500
C = df["vote_average"].mean()


def weighted_rating(r, v, M, c):
    return (v/(v+M) * r) + (M/(v+M) * C)


df.sort_values("average_vote_count_score", ascending=False).head(20)

weighted_rating(7.40000, 11444.00000, M, C)  # Deadpool filmi
weighted_rating(8.10000, 14075.00000, M, C)  # Inception filmi
weighted_rating(8.50000, 8358.00000, M, C)  # The Shawshank Redemption filmi

df["weighted_rating"] = weighted_rating(df["vote_average"], df["vote_count"], M, C)
df.sort_values("weighted_rating", ascending=False).head(20)


########################################################################
# Bayesian Average Rating Score
########################################################################
# The Dark Knight
# The Shawshank Redemption
# Fight Club
# Inception
# Pulp Fiction

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])  # The Shawshank Redeption filminin Bayesian Rating Score'u
bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905])  # The Godfather filminin Bayesian Rating Score'u

df = pd.read_csv("datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:]

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five", "six", "seven", \
                                                                "eight", "nine", "ten"]]), axis=1)

df.sort_values("bar_score", ascending=False).head(20)

