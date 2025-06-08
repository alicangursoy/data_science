##################################################################
# Sayılar (Numbers) ve Karakter Dizileri (Strings)
##################################################################


print("Hello world")

print("Hello AI Era")

type(9)
type(9.2)
type("Hello AI Era")

##################################################################
# Atamalar ve Değişkenler (Assignments & Variables)
##################################################################
a = 9
a
b = "hello ai era"
b
c = 10
a * c

a * 10

d = a - c

##################################################################
# Virtual Environment (Sanal Ortam) ve (Package Management) Paket Yönetimi
##################################################################

# Sanal ortamların listelenmesi:
# conda env list

# Sanal ortam oluşturma:
# conda create -n myenv

# Sanal ortamı aktif etme:
# conda activate myenv

# Aktif olan sanal ortamı deaktif etme:
# conda deactivate

# Aktif olan sanal ortamda kurulu olan paketlerin listelenmesi:
# conda list

# Bir paket yüklemek istediğimizde:
# conda install package_name (ör: conda install numpy: Numpy paketini ve bunun bapımlı olduğu paketleri indirir.)

# Birden fazla paket yüklemek istediğimizde:
# conda install package_name1 package_name2 package_name3 ...

# Paket silme:
# conda remove package_name

# Belirli bir versiyona göre paket yükleme:
# conda install numpy=2.2.1

# Paket yükseltme:
# conda upgrade numpy

# Tüm paketlerin yükseltilmesi:
# conda upgrade -all

# pip: pypi (python package index) paket yönetim aracı

# Paket yükleme:
# pip install package_name

# Paket yükleme versiyona göre:
# pip install pandass==1.2.1

# Sanal ortamdaki paket bilgilerinin export edilmesi:
# conda env export > environment.yaml

# Sanal ortam silme
# conda env remove -n myenv

# Ortam dosyasından sanal ortam oluşturma:
# conda env create -f environment.yaml
