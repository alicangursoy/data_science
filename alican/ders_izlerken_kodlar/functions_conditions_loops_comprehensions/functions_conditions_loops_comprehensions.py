##################################################################
# FONKSİYONLAR, KOŞULLAR, DÖNGÜLER, COMPREHENSIONS
##################################################################

# - Fonksiyonlar (Functions)
# - Koşullar (Conditions)
# - Döngüler (Loops)
# - Comprehensions

##################################################################
# FONKSİYONLAR (FUNCTIONS)
##################################################################

##################################################################
# Fonksiyon Okuryazarlığı
##################################################################

print("a")
# ?print : ?function_name: Fonksiyon hakkında bbilgi verir, fakat bunu kodda kullanmamalıyız, kodlamada hatalı bir kullanım oluyor. Terminalde yapabiliriz bunu.
print("a", "b")  # a ve b'yi araya bir boşluk koyarak yazar.
print("a", "b", sep="__")  # a ve b'yi araya sep değişkeninde verilen string'i koyarak yazdırır.
help(print)  # ? ne göre daha kapsamlı bir dökümantasyon sunar.

##################################################################
# Fonksiyon Tanımlama
##################################################################


# def calculate(x):
#    print(x * 2)


# calculate(5)


# İki parametreli/argümanlı fonksiyon tanımlayalım.
def summer(arg1, arg2):
    print(arg1 + arg2)


summer(7, 8)


# #################################################################
# Docstring: Yazdığımız fonksyionu/metodu dökümante
# etmek için """""" 3.çift tırnaktan sonra Enter'a basınca otomatik dolduruyor PyCharm.
# Dökümantasyon stilini değiştirmek için Preferences -> doctring diye aratıyoruz.
# Sol taraftan Tools -> Python Integrated Tools. Sağ tarafta Docstring Format'tan istediğimiz formatı seçiyoruz.
# Aşağıdaki stil, NumPy stili.
# #################################################################
def summer(arg1, arg2):
    """
    Sum of two numbers
    Parameters
    ----------
    arg1: int, float
    arg2: int, float

    Returns
    -------
    int, float
    """
    print(arg1 + arg2)


# ?summer
help(summer)

##################################################################
# Fonksiyonların Statement/Body Bölümü
##################################################################
# def function_name(parameters/arguments):
#     statements (function body)


def say_hi(string):
    print(string)
    print("Hello")
    print("Hi")


say_hi("miuul")


def multiplication(a, b):
    c = a * b
    print(c)


multiplication(10, 9)


# girilen değerleri bir liste içinde saklayacak fonksiyon
list_store = []


def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


add_element(1, 8)
add_element(18, 8)
add_element(180, 10)


##################################################################
# Ön Tanımlı Argümanlar/Parametreler (Default Parameters/Arguments)
##################################################################

def divide(a, b=1):
    print(a / b)


divide(1, 2)

divide(1)


def calculate(warm, moisture, charge):
    warm = warm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (warm + moisture) / charge
    return warm, moisture, charge, output


result = calculate(98, 12, 78)
type(result)
warm, moisture, charge, output = calculate(98, 12, 78)