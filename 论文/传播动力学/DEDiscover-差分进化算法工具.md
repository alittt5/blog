## 1.资源链接

https://dediscover.org/intro/
https://github.com/aleblanc39/dediscover

## 2. 安装步骤

**安装版本为8的gcc、g++、gfortran**
```
sudo apt-get install gcc-8
sudo apt-get install g++-8
sudo apt-get install gfortran-8

下载后只有 gcc-8才能运行，使用下面的语法将gcc指向gcc-8

sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-8

检查gcc、g++、gfortran的版本
gcc -v
```

**安装cmake**
```
sudo apt install cmake
最低版本是 3.1
```

**安装boost**
```
sudo apt install libboost-dev-all
最低版本是1.62
```

**安装BLAS和LAPACK**
这是两个C语言库
```
sudo apt install libblas3  
sudo apt install libblas-dev  
sudo apt install liblapack3
官方的教程
```
或参考 https://zhuanlan.zhihu.com/p/520848641
**记得在~/.bashrc里添加**

**安装sdkman**
```
curl -s "https://get.sdkman.io" | bash

安装正确版本的java和gradle
sdk install gradle 6.8.2
sdk install java 8.0.282.fx-zulu 

export JAVA_HOME=/home/_my-user-name_/.sdkman/candidates/java/current

使用最新的 8.0.xxx.fx-zulu 格式
```

**安装maven**
```
sudo apt install maven
```

**git拉取**

https://github.com/aleblanc39/gradle-cmake-plugin
https://github.com/aleblanc39/dediscover
```
在gradle-cmake-plugin文件夹内
gradle publishToMavenLocal
在dediscover文件夹内
gradle runDED
```

当出现下图时，安装完成。
![[Pasted image 20221127123344.png]]