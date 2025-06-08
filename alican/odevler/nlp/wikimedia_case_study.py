from warnings import filterwarnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud
import nltk
import re

filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

# Bağlaçları içeren kütüphane indiriliyor.
nltk.download("stopwords")

# Metin tamamen temizlendikten sonra, tokenizasyon işlemi için gerekli kütüphane indiriliyor.
nltk.download("punkt")
nltk.download('punkt_tab')

# Lemmatization işlemi için gerekli kütüphane indiriliyor.
nltk.download("wordnet")

#######################################################
# Görev 1: Metin Ön İşleme İşlemlerini Gerçekleştiriniz
#######################################################


# Adım 1: Metin ön işleme için clean_text adında fonksiyon oluşturunuz.
def clean_text(text, lower=True):
    if lower:
        text = text.lower()
    else:
        text = text.str.upper()

    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d", "", text)
    text = text.replace("\n", " ")
    return text


df = pd.read_csv("wiki_data.csv", sep=',')

# Adım 2: Yazdığınız fonksiyonu veri seti içerisindeki tüm metinlere uygulayınız.
df["text"] = df.apply(lambda x: clean_text(x["text"]), axis=1)


# Adım 3: Metin içinde öznitelik çıkarımı yaparken önemli olmayan kelimeleri çıkaracak remove_stopwords adında fonksiyon yazınız.
def remove_stopwords(text):
    sw = stopwords.words("english")
    text = " ".join(word for word in text.split() if word not in sw)
    return text


# Adım 4: Yazdığınız fonksiyonu veri seti içerisindeki tüm metinlere uygulayınız.
df["text"] = df.apply(lambda x: remove_stopwords(x["text"]), axis=1)


# Adım 5: Metinde az geçen (1000'den az, 2000'den az gibi) kelimeleri bulunuz. Ve bu kelimeleri metin içerisinden çıkartınız.
temp_df = pd.Series(" ".join(df["text"]).split()).value_counts()
drops = temp_df[temp_df <= 2000]
df["text"] = df["text"].apply(lambda x: " ".join(x for x in x.split() if x not in drops))


# Adım 6: Metinleri tokenize edip sonuçları gözlemleyiniz.
df["text"].apply(lambda x: TextBlob(x).words).head()


# Adım 7: Lemmatization işlemi yapınız.
df["text"] = df["text"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))



#######################################################
# Görev 2: Metin Ön İşleme İşlemlerini Gerçekleştiriniz
#######################################################

# Adım 1: Metindeki terimlerin frekanslarını hesaplayınız.
term_freq = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
term_freq.columns = ["words", "term_freq"]
term_freq.sort_values("term_freq", ascending=False)

# Adım 2: Bir önceki adımda bulduğunuz terim frekanslarının Barplot grafiğini oluşturunuz.
term_freq.plot.bar(x="words", y="term_freq")
plt.show()

# Görsel açıdan birçok kelimeyi göstermek karmaşık bir görsele neden oldu.
# O yüzden belli frekansın üstündeki kelimeleri gösteriyoruz.
term_freq[term_freq["term_freq"] > 7000].shape
term_freq[term_freq["term_freq"] > 7000].plot.bar(x="words", y="term_freq")
plt.show()

# Adım 3: Kelimeleri WordCloud ile görselleştiriniz.
text = " ".join(i for i in df["text"])
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#########################################################
# Görev 3: Tüm Aşamaları Tek Bir Fonksiyon Olarak Yazınız
#########################################################
def get_term_frequency(dataframe, colname_to_process):
    term_freq = dataframe[colname_to_process].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
    term_freq.columns = ["words", "term_freq"]
    term_freq = term_freq.sort_values("term_freq", ascending=False)
    print(term_freq)
    return term_freq


def extract_texts_with_more_frequency(dataframe, colname_to_process, term_freq_th=2000):
    temp_df = pd.Series(" ".join(dataframe[colname_to_process]).split()).value_counts()
    drops = temp_df[temp_df <= 2000]
    return dataframe[colname_to_process].apply(lambda x: " ".join(x for x in x.split() if x not in drops))


def visualize_with_barplot(term_freq, term_freq_visual_th):
    term_freq[term_freq["term_freq"] > term_freq_visual_th].shape
    term_freq[term_freq["term_freq"] > term_freq_visual_th].plot.bar(x="words", y="term_freq")
    plt.show(block=True)


def visualize_with_wordcloud(dataframe, colname_to_process):
    text = " ".join(i for i in dataframe[colname_to_process])
    wordcloud = WordCloud().generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def process_wiki_data(dataframe, colname_to_process="text", lower=True, term_freq_th=2000, visualize=True, term_freq_visual_th=7000):
    # Adım 3: Fonksiyonu açıklayan 'docstring' yazınız.
    """
    Verilen metin listesinde, veri temizliği yapıp eğer istenirse bu metin listesi belirli eşik değerlerince
    görselleştiren fonksiyondur.
    :param dataframe: Metin listesi. Tipi: pandas.DataFrame
    :param colname_to_process: Metin listesindeki metinlerin bulunduğu değişkenin adı.
                  Tipi: string. Varsayılan değeri "text"
    :param lower: Metin listesindeki metinlerin metin temizliği adımında küçük harf mi yoksa büyük harf mi olacağının
                  belirtildiği değişken. Tipi: boolean. Varsayılan değeri: True
    :param term_freq_th: Metin listesindeki kelimelerden en az kaç kere geçenlerin göz önünde bulundurulacağının
                  belirtildiği değişken. Tipi: integer. Varsayılan değeri 2000.
    :param visualize: Metin listesi işlendikten sonra, eğer istenirse barplot ve wordcloud yöntemleriyle
                  metnin görselleştirileceğini belirten değişken. Tipi: boolean. Varsayılan değeri: True
    :param term_freq_visual_th: Metin listesindeki kelimelerden en az kaç kere geçenlerin görselleştirileceğinin
                  belirtildiği değişken. Tipi: integer. Varsayılan değeri 7000.
    """
    print("Verilen metin listesinin işlenmesine başladı...")
    # Adım 1: Metin ön işleme işlemlerini gerçekleştiriniz.
    dataframe[colname_to_process] = df.apply(lambda x: clean_text(x[colname_to_process], lower=lower), axis=1)
    dataframe[colname_to_process] = df.apply(lambda x: remove_stopwords(x[colname_to_process]), axis=1)
    dataframe[colname_to_process] = extract_texts_with_more_frequency(dataframe, colname_to_process, term_freq_th)
    dataframe[colname_to_process].apply(lambda x: TextBlob(x).words).head()
    dataframe[colname_to_process] = dataframe[colname_to_process].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    term_freq = get_term_frequency(dataframe, colname_to_process)

    # Adım 2: Görselleştirme işlemlerini fonksiyona argüman olarak ekleyiniz.
    if visualize:
        visualize_with_barplot(term_freq, term_freq_visual_th)
        visualize_with_wordcloud(dataframe, colname_to_process)

    print("Verilen metin listesinin işlenmesi işlemi bitti...")


process_wiki_data(df)
