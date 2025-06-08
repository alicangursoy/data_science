#####################################################
# AB Testi ile Bidding Yöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi ve averagebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchase metriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır. Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleri ab_testing.xlsx excel’inin ayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna Average Bidding uygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.




#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.
import pandas as pd
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
import scipy.stats as st
import math

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


df_control_group = pd.read_excel("ab_testing.xlsx", "Control Group")
df_test_group = pd.read_excel("ab_testing.xlsx", "Test Group")



# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.
df_control_group.describe().T
df_test_group.describe().T




# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.
df = pd.concat([df_control_group, df_test_group], axis=1)
df.describe().T

# Kolon isimleri duplike oluyordu. Tekilleştiriliyor.
cols = pd.Series(df.columns)
for dup in cols[cols.duplicated()].unique():
    cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
df.columns = cols

#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.
# H0: M1 = M2  (Maximum Bidding ve Average Bidding Yöntemlerinin Uygulanması Sonucu Purchase Sayısı Ortalamaları Arasında İstatistiksel Olarak Anlamlı Bir Fark Yoktur)
# H1: M1 != M2 (Maximum Bidding ve Average Bidding Yöntemlerinin Uygulanması Sonucu Purchase Sayısı Ortalamaları Arasında İstatistiksel Olarak Anlamlı Bir Fark Vardır)


# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz
df["Purchase"].describe().T
df["Purchase.1"].describe().T


#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################


######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız. Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz

# Normallik varsayımı
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.

test_stat, pvalue = shapiro(df.loc[:, "Purchase"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.
# Kontrol grubunun "Purchase" değeri için (p-value = 0.5891) H0 reddedilemez.

test_stat, pvalue = shapiro(df.loc[:, "Purchase.1"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.
# Test grubunun "Purchase" (biz "Purchase.1" olarak isimlendirdik bu alanı) değeri için (p-value = 0.1541) H0 reddedilemez.


# Varyans homojenliği
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[:, "Purchase"].dropna(),
                           df.loc[:, "Purchase.1"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < 0.05 OLMADIĞI için (p-value = 0.1083) H0 REDDEDILEMEZ.




# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz
# Varsayımlar sağlandığı için parametrik test uygulanmalı.
# Uygulayacağımız test: Bağımsız iki örneklem t testi
test_stat, p_value = ttest_ind(df.loc[:, "Purchase"],
                               df.loc[:, "Purchase.1"], equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, p_value))


# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

# Bağımsız iki örneklem t testi sonucunda p-value = 0.3493 çıktı. Yani p-value değeri 0.05'ten küçük değil.
# Dolayısıyla, kontrol ve test grubu satın alma ortalamaları arasında istatistiki olarak anlamlı bir fark YOKTUR.


##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.
# Bağımsız iki örneklem t testi kullandım. Bunun nedenleri:
# 1. Purchase değişkeni her iki grup için de NORMAL DAĞILIM'a sahipti.
# 2. Purchase değişkeni varyansları homojendi.



# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.
# Elde ettiğimiz test sonuçlarına göre Maximum Bidding yöntemi ile Average Bidding yöntemi arasında istatistiki olarak
# anlamlı bir fark yoktur. Dolayısıyla, Average Bidding yöntemini uygularsanız istediğiniz verimi alamayabilirsiniz.
# Maximum Bidding yöntemi ile devam edebilirsiniz.


up = [1115, 454, 258, 253, 220]
down = [143, 35, 26, 19, 9]
comments = pd.DataFrame({f"up": up, "down": down})
comments["score_up_down_diff"] = comments.up - comments.down
comments["score_average_rating"] = comments.up / (comments.up + comments.down)
wilson_scores = [0.95036, 0.90208, 0.86924, 0.89349, 0.92701]
comments["wilson_scores"] = wilson_scores

comments.sort_values("wilson_scores", ascending=False)
comments.sort_values("score_up_down_diff", ascending=False)
comments.sort_values("score_average_rating", ascending=False)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


wilson_lower_bound(600, 400)  # 0.5693094295142663
wilson_lower_bound(2, 0)  # 0.3423802275066531
wilson_lower_bound(100, 1)  # 0.9460328420055449
