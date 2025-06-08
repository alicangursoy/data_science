# Görev 1:
x = 8
type(x)
y = 3.2
type(y)
z = 8j + 18
type(z)
a = "Hello world"
type(a)
b = True
type(b)
c = 23 < 22
type(c)
l = [1, 2, 3, 4]
type(l)
d = {"Name": "Jake",
     "Age": 27,
     "Adress": "Downtown"}
type(d)
t = ("Machine Learning", "Data Science")
type(t)
s = {"Python", "Machine Learning", "Data Science"}
type(s)

# Görev 2:
text = "The goal is to turn data into information, and information into insight"
result = text.upper().replace(',', ' ').replace('.', ' ').replace('  ', ' ').split(' ')

# Görev 3:
lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]
len(lst)
lst[0]
lst[10]
[ch for ch in lst if ch in ['D', 'A', 'T', 'A']]
lst.pop(8)
lst.append('E')
lst.insert(8, 'N')

# Görev 4:
dict = {'Christian': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}
dict.keys()
dict.values()
dict['Daisy'][1] = 13
dict['Ahmet'] = ["Turkey", 24]
dict.pop('Antonio')

# Görev 5:
l = [2, 13, 18, 93, 22]


def func(lst):
    even_list = []
    odd_list = []
    for num in lst:
        if num % 2 == 0:
            even_list.append(num)
        else:
            odd_list.append(num)
    return even_list, odd_list


even_list, odd_list = func(l)

# Görev 6:
ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]
for ind, ogr in enumerate(ogrenciler, 1):
    if ind <= 3:
        print("Mühendislik Fakültesi " + str(ind) + " . öğrenci: " + ogr)
    else:
        print("Tıp Fakültesi " + str(ind - 3) + " . öğrenci: " + ogr)

# Görev 7:
ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]
for ders_kodu, kredi, kontenjan in zip(ders_kodu, kredi, kontenjan):
    print(f"Kredisi {kredi} olan {ders_kodu} kodlu dersin kontenjanı {kontenjan} kişidir.")

# Görev 8:
kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])


def set_operation(set1, set2):
    if set1.issuperset(set2):
        print(set1.intersection(set2))
    else:
        print(set2.difference(set1))

set_operation(kume1, kume2)