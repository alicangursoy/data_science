##################################################################
# VERİ YAPILARI (DATA STRUCTURES)
##################################################################
# - Veri Yapılarına Giriş ve Hızlı Özet
# - Sayılar (Numbers): int, float, complex
# - Karakter Dizileri (Strings): str
# - Boolean (TRUE-FALSE): bool
# - Liste (List)
# - Sözlük (Dictionary)
# - Demet (Tuple)
# - Set

##################################################################
# Veri Yapılarına Giriş ve Hızlı Özet
##################################################################

# Sayılar: integer
x = 46
type(x)

# Sayılar: float
x = 10.3
type(x)

# Sayılar: complex
x = 2j + 1
type(x)

# String
x = "Hello ai era"
type(x)

# Boolean
True
False
type(True)
5 == 4
3 == 2
1 == 1
type(3 == 2)

# Liste
x = ["btc", "eth", "xrp"]
type(x)

# Sözlük (dictionary)
x = {"name": "Peter", "Age": 36}
type(x)

# Tuple
x = ("python", "ml", "ds")
type(x)

# Set
x = {"python", "ml", "ds"}
type(x)

# Not: Liste, tuple, set ve dictionary veri yapıları aynı zamanda Python Collections (Arrays) olarak geçmektedir.

##################################################################
# Sayılar (Numbers): int, float, complex
##################################################################
a = 5
b = 10.5
a * 3
a / 7
a * b / 10
a ** 2

##################################################################
# Tipleri değiştirmek
##################################################################
int(b)
float(a)

int(a * b / 10)

c = a * b / 10
int(c)

##################################################################
# Karakter Dizileri (Strings)
##################################################################
print("John")
print('John')
"John"
name = "John"
name = 'John'

##################################################################
# Çok Satırlı Karakter Dizileri
##################################################################
"""
Karakter Dizileri (Strings): str,
List, Dictionary, Tuple, Set
Boolean (TRUE-FALSE): bool"""

long_str = """Veri Yapıları: Hızlı Özet,
Sayılar (Numbers): int, float, complex,
Karakter Dizileri (Strings): str,
List, Dictionary, Tuple, Set,
Boolean (TRUE-FALSE): bool"""

##################################################################
# Karakter Dizilerinin Elemanlarına Erişmek
##################################################################
name[0]
name[3]

##################################################################
# Karakter Dizilerinde Slice İşlemi
##################################################################
name[0:2]
long_str[0:10]

##################################################################
# String İçerisinde Karakter Sorgulamak
##################################################################
"veri" in long_str
"Veri" in long_str
"bool" in long_str

##################################################################
# String (Karakter Dizisi) Metotları
##################################################################
dir(int)  # int veri yapısına ait metotları döner.
dir(str)

##################################################################
# len
##################################################################
name = "john"
type(name)
type(len)  # builtin_function_or_method
len(name)
len("alicangürsoy")
len("softtech")

##################################################################
# upper() & lower(): küçük-büyük dönüşümleri
##################################################################
"miuul".upper()
"MIUUL".lower()

# type(upper)
# type(upper())


##################################################################
# replace: karakter değiştirir
##################################################################
hi = "hello ai era"
hi.replace("l", "p")


##################################################################
# split: böler
##################################################################
"Hello AI Era".split()


##################################################################
# strip: kırpar
##################################################################
" ofofo ".strip()
"ofofo".strip("o")

##################################################################
# capitalize: ilk harfi büyütür
##################################################################
"foo".capitalize()

dir("foo")
"foo".startswith("f")


##################################################################
# Liste (List)
##################################################################
# - Değiştirilebilir
# - Sıralıdır. Index işlemleri yapılabilir.
# - Kapsayıcıdır. Birden fazla veri yapısı tipinde eleman içerebilir.

notes = [1, 2, 3, 4]
type(notes)

names = ["a", "b", "v", "d"]

not_nam = [1, 2, 3, "a", "b", True, [1, 2, 3]]

not_nam[0]
not_nam[5]
not_nam[6]
not_nam[6][1]
type(not_nam[6])
type(not_nam[6][1])

notes[0] = 99
notes

##################################################################
# Liste Metotları (List Methods)
##################################################################
dir(notes)
len(notes)
len(not_nam)
notes.append(100)
notes

notes.pop(0)
notes.insert(2, 99)

##################################################################
# Sözlük (Dictionary)
##################################################################
# - Değiştirilebilir.
# - Sırasız. (3.7 sonra sıralı.)
# - Kapsayıcı

# key-value

dictionary = {"REG": "Regression",
              "LOG": "Logistic Regression",
              "CART": "Classification and Reg"}
dictionary["REG"]

dictionary = {"REG": ["RMSE", 10],
              "LOG": ["MSE", 20],
              "CART": ["SSE", 30]}

dictionary["CART"][1]

##################################################################
# Key Sorgulama
##################################################################
"YSA" in dictionary

##################################################################
# Key'e Göre Value'ya Erişmek
##################################################################
dictionary["REG"]
dictionary.get("REG")

##################################################################
# Value Değiştirmek
##################################################################
dictionary["REG"] = ["YSA",10]

##################################################################
# Tüm Key'lere Erişmek
##################################################################
dictionary.keys()

##################################################################
# Tüm Value'lara Erişmek
##################################################################
dictionary.values()

##################################################################
# Tüm Çiftleri Tuple Halinde Listeye Çevirme
##################################################################
dictionary.items()

##################################################################
# Varolan bir key-value çiftini günceller
##################################################################
dictionary.update({"REG": 11})

##################################################################
# Key yoksa, dictionary'ye ekler
##################################################################
dictionary.update({"RF": 12})
dictionary

##################################################################
# Demet (Tuple)
##################################################################
# - Değiştirilemez.
# - Sıralıdır.
# - Kapsayıcıdır.

t = ("john", "mark", 1, 2)
type(t)
t[0]
t[0:3]
t[0] = 99  # TypeError: 'tuple' object does not support item assignment
t1 = tuple(t)


##################################################################
# Set
##################################################################
# - Değiştirilebilir.
# - Sırasız + Eşsizdir.
# - Kapsayıcıdır.

##################################################################
# difference(): İki kümenin farkı
##################################################################

set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

set1.difference(set2)
set1 - set2

set2.difference(set1)
set2 - set1

##################################################################
# symmetric_difference(): İki kümede de birbirlerine göre olmayanlar
##################################################################
set1.symmetric_difference(set2)
set2.symmetric_difference(set1)

##################################################################
# intersection(): İki kümenin kesişimi
##################################################################
set1.intersection(set2)
set2.intersection(set1)
set1 & set2

##################################################################
# union(): İki kümenin bileşimi
##################################################################
set1.union(set2)
set2.union(set1)


##################################################################
# isdisjoint(): İki kümenin kesişimi boş mu?
##################################################################
set1 = set([7, 8, 9])
set2 = set([5, 6, 7, 8, 9, 10])
set1.isdisjoint(set2)
set2.isdisjoint(set1)

##################################################################
# issubset(): Bir küme diğerinin alt kümesi mi
##################################################################
set1.issubset(set2)
set2.issubset(set1)

##################################################################
# issubset(): Bir küme diğerini kapsıyor mu
##################################################################
set1.issuperset(set2)
set2.issuperset(set1)