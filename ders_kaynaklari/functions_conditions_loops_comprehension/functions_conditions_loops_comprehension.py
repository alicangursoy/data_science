##################################################################
# KOŞULLAR (CONDITIONS)
##################################################################
# True-False'u hatırlayalım.
1 == 1
1 == 2

# if
if 1 == 1:
    print("something")

if 1 == 2:
    print("something")


number = 10

if number == 10:
    print("Number is 10")


def number_check(number):
    if number == 10:
        print("Number is 10")


number_check(12)
number_check(10)


def number_check(number):
    if number == 10:
        print("Number is 10")
    else:
        print("Number is not 10")


number_check(12)


def number_check(number):
    if number > 10:
        print("Number is greater than 10")
    elif number < 10:
        print("Number is smaller than 10")
    else:
        print("Number is 10")


number_check(20)
number_check(9)
number_check(10)


##################################################################
# DÖNGÜLER (LOOPS)
##################################################################
# for loop

students = ["John", "Mark", "Venessa", "Mariam"]

students[0]
students[1]
students[2]

for student in students:
    print(student)

for student in students:
    print(student.upper())

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(salary)

for salary in salaries:
    print(int((salary * 20 / 100) + salary))


def new_salary(salary, rate=20):
    return int((salary * rate / 100) + salary)


for salary in salaries:
    print(new_salary(salary, 10))


for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 20))


##################################################################
# Uygulama - Mülakat Sorusu
##################################################################

# Amaç: Aşağıdaki şekilde string değiştiren fonksiyon yazmak istiyoruz.

# before: "hi my name is john and i am learning python"
# after: "Hi My NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

def upper_even_ordered_letter(string):
    tmp = ""
    for i in range(len(string)):
        if i % 2 == 0:
            tmp += string[i].upper()
        else:
            tmp += string[i].lower()
    return tmp


input_str = "hi my name is john and i am learning python"
output_str = upper_even_ordered_letter(input_str)
print(output_str)


##################################################################
# break & continue & while
##################################################################

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    if salary == 3000:
        break
    print(salary)

for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

number = 1
while number < 5:
    print(number)
    number += 1


##################################################################
# Enumerate: Otomatik Counter/Indexer ile for loop
##################################################################
students = ["John", "Mark", "Venessa", "Mariam"]

for student in students:
    print(student)

for index, student in enumerate(students):
    print(index, student)

for index, student in enumerate(students, 1):
    print(index, student)

A = []
B = []
for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)


##################################################################
# Uygulama  - Mülakat Sorusu
##################################################################
# divide_students fonksiyonu yazınız.
# Çift indexte yer alan öğrencileri bir listeye alınız.
# Tek indexte yer alan öğrencileri başka bir listeye alınız.
# Fakat bu iki liste tek bir liste olarak return olsun.

def divide_students(list_of_students):
    groups = [[], []]
    for index, student in enumerate(list_of_students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    return groups


students = ["John", "Mark", "Venessa", "Mariam"]
print(divide_students(students))


##################################################################
# Uygulama - Mülakat Sorusu - Enumarete ile
##################################################################
def upper_even_ordered_letter(string):
    tmp = ""
    for index, letter in enumerate(string):
        if index % 2 == 0:
            tmp += string[index].upper()
        else:
            tmp += string[index].lower()
    return tmp


input_str = "hi my name is john and i am learning python"
output_str = upper_even_ordered_letter(input_str)
print(output_str)


##################################################################
# Zip
##################################################################

students = ["John", "Mark", "Venessa", "Mariam"]

departments = ["mathematics", "statistics", "physics", "astronomy"]

ages = [23, 30, 26, 22]

list(zip(students, departments, ages))


##################################################################
# lambda, map, filter, reduce
##################################################################


new_sum = lambda a, b: a + b
new_sum(4, 5)

# map

salaries = [1000, 2000, 3000, 4000, 5000]


def new_salary(wage):
    return wage * 20 / 100 + wage


list(map(new_salary, salaries))

list(map(lambda x: x * 20 / 100 + x, salaries))

list(map(lambda x: x ** 2, salaries))


# FILTER


list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 == 0, list_store))


# REDUCE
from functools import reduce

list_store = [1, 2, 3, 4]
reduce(lambda a, b: a + b, list_store)


##################################################################
# COMPREHENSIONS
##################################################################

##################################################################
# List Comprehension
##################################################################

salaries = [1000, 2000, 3000, 4000, 5000]


def new_salary(x):
    return x * 20 / 100 + x


null_list = []
for salary in salaries:
    null_list.append(new_salary(salary))

null_list = []
for salary in salaries:
    if salary > 3000:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary * 2))


# Eğer comprehension'ın içinde sadece if kullanılacaksa for'un sağına yazılır.
[salary * 2 for salary in salaries if salary < 3000]

# Eğer comprehension'ın içinde if-else kullanılacaksa for'un soluna yazılır.
[new_salary(salary * 2) if salary < 3000 else new_salary(salary) for salary in salaries]


students = ["John", "Mark", "Vanessa", "Mariam"]

students_no = ["John", "Vanessa"]

[student.lower() if student in students_no else student.upper() for student in students]


##################################################################
# Dict Comprehension
##################################################################

dictionary = {"a": 1,
              "b": 2,
              "c": 3,
              "d": 4}

dictionary.keys()
dictionary.values()
dictionary.items()

{k: v ** 2 for (k, v) in dictionary.items()}

{k.upper(): v  for (k, v) in dictionary.items()}

{k.upper(): v * 2 for (k, v) in dictionary.items()}


##################################################################
# Uygulama - Mülakat Sorusu
##################################################################

# Amaç: çift sayıların karesi alınarak bir sözlüğe eklenmek istenmektedir.

numbers = range(10)
new_dict = {}
for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2

{number: number ** 2 for number in numbers if number % 2 == 0}


##################################################################
# List & Dictionary Comprehension Uygulamaları
##################################################################

##################################################################
# Bir Veri Setindeki Değişken İsimlerini Değiştirmek
##################################################################

# before:
# ['total', 'speeding', 'alcohol', 'not_distracted', 'no_previous', 'ins_premium', 'ins_loses', 'abbrev']

# after:
# ['TOTAL', 'SPEEDING', 'ALCOHOL', 'NOT_DISTRACTED', 'NO_PREVIOUS', 'INS_PREMIUM', 'INS_LOSES', 'ABBREV']

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns = [inp.upper() for inp in df.columns]

# İsminde "INS" olan değişkenlerin başına FLAG diğerlerine NO_FLAG eklemek istiyoruz.
df.columns = ["FLAG_" + inp.upper() if "INS" in inp.upper() else "NO_FLAG_" + inp.upper() for inp in df.columns]


# Amaç: key'i string, value'su aşağıdaki gibi bir liste olan sözlük oluşturmak.
# Sadece sayısal değişkenler için

# Output:
# {'total' : ['mean', 'min', 'max', 'var'],
# 'speeding' : ['mean', 'min', 'max', 'var']
# 'alcohol' : ['mean', 'min', 'max', 'var']
# 'not_distracted' : ['mean', 'min', 'max', 'var']
# 'no_previous' : ['mean', 'min', 'max', 'var']
# 'ins_premium' : ['mean', 'min', 'max', 'var']
# 'ins_loses' : ['mean', 'min', 'max', 'var']}

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

num_cols = [col for col in df.columns if df[col].dtype != "O"]
dictionary = {}
agg_list = ['mean', 'min', 'max', 'sum']

for col in num_cols:
    dictionary[col] = agg_list

new_dict = {col: agg_list for col in df.columns if df[col].dtype != "O"}

df[num_cols].head()
df[num_cols].agg(new_dict)