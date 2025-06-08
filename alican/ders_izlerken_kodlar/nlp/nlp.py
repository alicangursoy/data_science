#############################################################
# Introduction to Text Mining and Natural Language Processing
#############################################################


##############################################################
# Sentiment Analysis and Sentiment Modeling for Amazon Reviews
##############################################################

# 1. Text Preprocessing
# 2. Text Visualization
# 3. Sentiment Analysis
# 4. Feature Engineering
# 5. Sentiment Modeling


# !pip install nltk
# !pip install textblob
# !pip install wordcloud

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

filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: "%.2f" % x)


########################################################################
# 1. Text Preprocessing
########################################################################

df = pd.read_csv("datasets/amazon_reviews.csv", sep=",")
df.head()


##########################################
# Normalizing Case Folding
##########################################

df["reviewText"] = df["reviewText"].str.lower()


##########################################
# Punctuations: Noktalama işaretlerinin silinmesi
##########################################

df["reviewText"] = df["reviewText"].str.replace("[^\w\s]", "", regex=True)
# regular expression


##########################################
# Numbers: Sayıların silinmesi
##########################################
df["reviewText"] = df["reviewText"].str.replace("\d", "", regex=True)


##########################################
# Stopwords: Bağlaçlar vb. kelimelerin silinmesi
##########################################
import nltk
# nltk.download("stopwords")

sw = stopwords.words("english")
df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))


##########################################
# Rarewords: Az geçen kelimelerin silinmesi
##########################################

temp_df = pd.Series(" ".join(df["reviewText"]).split()).value_counts()
drops = temp_df[temp_df <= 1]

df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x for x in x.split() if x not in drops))


##########################################
# Tokenization: Cümleleri kelimelerine bölmek
##########################################
# nltk.download("punkt")
# nltk.download('punkt_tab')

df["reviewText"].apply(lambda x: TextBlob(x).words).head()


##########################################
# Lemmatization: Kelimeleri sade hallerine çekirmek.
##########################################

# nltk.download("wordnet")

df["reviewText"] = df["reviewText"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


##############################################
# 2. Text Visualization
##############################################


####################################
# Terim Frekanslarının Hesaplanması
####################################

tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending=False)

####################################
# Barplot
####################################

# Görsel açıdan bir çok kelimeyi göstermek zor olabilir. Bunları indirgemek için frekansı 500'den büyükleri alıyoruz.
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()


####################################
# Wordcloud
####################################

text = " ".join(i for i in df["reviewText"])

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud.png")


####################################
# Şablonlara Göre Wordcloud
####################################

tr_mask = np.array(Image.open("tr.png"))
wc = WordCloud(max_words=1000,
               background_color="white",
               mask=tr_mask,
               contour_width=3,
               contour_color="firebrick").generate(text)
plt.figure(figsize=[10, 10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()


##############################################
# 3. Sentiment Analysis
##############################################

df["reviewText"].head()

# nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()
sia.polarity_scores("The film was awesome")  # compound: 0.5859 => compound >= 0 -> Pozitif else Negatif

sia.polarity_scores("I liked this music but it is not good as the other one")  # compound: -0.298 => Olumsuz bir cümle.

df["polarity_score"] = df["reviewText"].apply(lambda x: sia.polarity_scores(x)["compound"])


##############################################
# 3. Sentiment Modeling
##############################################

##############################################
# Feature Engineering
##############################################

df["reviewText"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
df["sentiment_label"] = df["reviewText"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"].value_counts()

df.groupby("sentiment_label")["overall"].mean()

df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

y = df["sentiment_label"]
X = df["reviewText"]



##################################
# Count Vectors
##################################

# Count Vectors: frekans temsiller
# TF-IDF Vectors: normalize edilmiş frekans temsiller
# Word Embeddings (Word2Vec, GloVe, BERT vs)


# words
# kelimelerin nümerik temsilleri


# ngram
a = """Bu örneği anlaşılabilmesi için daha uzun metin üzerinden göstereceğim.
N-gram'lar birlikte kullanılan kelimelerin kombinasyonlarını gösterir ve feature üretmek için kullanılır"""

TextBlob(a).ngrams(3)


##################################
# Count Vectors
##################################
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["This is the first document.",
          "This document is the second document.",
          "And this one is the third one.",
          "Is this the first document?"]

# word frekans
vectorizer = CountVectorizer()
X_c = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names_out()
X_c.toarray()

# n-gram frekans
vectorizer2 = CountVectorizer(analyzer="word", ngram_range=(2, 2))
X_n = vectorizer2.fit_transform(corpus)
vectorizer2.get_feature_names_out()
X_n.toarray()


vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(X)
vectorizer.get_feature_names_out()[10:15]
X_count.toarray()


###################################
# TF-IDF
###################################

from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_vectorizer.fit_transform(X)

tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
X_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(X)


################################################
# 5. Sentiment Modeling
################################################

################################
# Logistic Regression
################################

log_model = LogisticRegression().fit(X_tf_idf_word, y)
cross_val_score(log_model, X_tf_idf_word, y, scoring="accuracy", cv=5).mean()  # 0.830111902339776

new_review = pd.Series("this product is great")
new_review = TfidfVectorizer().fit(X).transform(new_review)

log_model.predict(new_review)  # 1 => positive


new_review2 = pd.Series("look at that shit very bad")
new_review2 = TfidfVectorizer().fit(X).transform(new_review2)

log_model.predict(new_review2)  # 0 => negative

new_review3 = pd.Series("it was good but I am sure that it fits me")
new_review3 = TfidfVectorizer().fit(X).transform(new_review3)

log_model.predict(new_review3)  # 1 => positive

random_review = pd.Series(df["reviewText"].sample(1).values)
new_review4 = TfidfVectorizer().fit(X).transform(random_review)
log_model.predict(new_review4)  # 1 => positive


#############################################
# Random Forests
#############################################

# Count Vectors
rf_model = RandomForestClassifier().fit(X_count, y)
cross_val_score(rf_model, X_count, y, cv=5, n_jobs=-1).mean()  # 0.8406917599186166

# TF-IDF Word-Level
rf_model = RandomForestClassifier().fit(X_tf_idf_word, y)
cross_val_score(rf_model, X_tf_idf_word, y, cv=5, n_jobs=-1).mean()  # 0.824618514750763


# TF-IDF N-GRAM
rf_model = RandomForestClassifier().fit(X_tf_idf_ngram, y)
cross_val_score(rf_model, X_tf_idf_ngram, y, cv=5, n_jobs=-1).mean()  # 0.7849440488301119



################################################
# Hyperparametre Optimizasyonu
################################################

rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [8, None],
             "max_features": [7, "auto"],
             "min_samples_split": [2, 5, 8],
             "n_estimators": [100, 200]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X_count, y)
rf_best_grid.best_params_
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_count, y)

cross_val_score(rf_final, X_count, y, cv=5, n_jobs=-1).mean()  # 0.8128179043743643

