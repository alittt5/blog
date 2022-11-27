# 深度学习入门
## 第二章 感知机
### 2.1 感知机是什么
**感知机**接收多个输入信号，输出一个信号。这里所说的“信号”可以想 象成电流或河流那样具备“流动性”的东西。像电流流过导线，向前方输送 电子一样，感知机的信号也会形成流，向前方输送信息。但是，和实际的电 流不同的是，感知机的信号只有“流/不流”（1/0）两种取值。在本书中，0 对应“不传递信号”，1对应“传递信号”。
$${y_i} = {w_i}{x_i} + ... + {w_n}{x_n} + {b_i}$$
![[pic/2C`5]_W1_@Q})QR[~L8Z7ZJ.png]]
![[pic/%0PZ9YG`%24]J84}QT%RV8R.png]]
### 2.2 感知机的实现
使用使用权重和偏置实现**与门**。
``` python
def AND(x1, x2):  
    x = np.array([x1, x2])  
    w = np.array([0.5, 0.5])  
    b = -0.7  
    tmp = np.sum(w*x) + b  
    if tmp <= 0:  
        return 0  
    else:  
        return 1
```
这里把−θ命名为偏置b，但是请注意，偏置和权重w1、w2的作用是不 一样的。具体地说，w1和w2是控制输入信号的重要性的参数，而偏置是调 整神经元被激活的容易程度（输出信号为1的程度）的参数。比如，若b为 −0.1，则只要输入信号的加权总和超过0.1，神经元就会被激活。但是如果b 为−20.0，则输入信号的加权总和必须超过20.0，神经元才会被激活。像这样， 偏置的值决定了神经元被激活的容易程度。另外，这里我们将w1和w2称为权重， 将b称为偏置，但是根据上下文，有时也会将b、w1、w2这些参数统称为权重。

**非门**
``` python
def NAND(x1, x2):  
    x = np.array([x1, x2])  
    w = np.array([-0.5, -0.5])  
    b = 0.7  
    tmp = np.sum(w*x) + b  
    if tmp <= 0:  
        return 0  
    else:  
        return 1
```
**或门**
``` python
def OR(x1, x2):  
    x = np.array([x1, x2])  
    w = np.array([0.5, 0.5])  
    b = -0.2  
    tmp = np.sum(w*x) + b  
    if tmp <= 0:  
        return 0  
    else:  
        return 1
```
### 2.3 线性和非线性
图2-7中的○和△无法用一条直线分开，但是如果将“直线”这个限制条件去掉，就可以实现了。比如，我们可以像图2-8那样，作出分开○和△的空间。 感知机的局限性就在于它只能表示由一条直线分割的空间。图2-8这样弯 曲的曲线无法用感知机表示。另外，由图2-8这样的曲线分割而成的空间称为 非线性空间，由直线分割而成的空间称为**线性空间**。线性、非线性这两个术语在机器学习领域很常见，可以将其想象成图2-6和图2-8所示的直线和曲线。
![pic/[I1OPN4D]3H}G}ZU3[B2D$JW.png]]
![[Z6VK0}OH%_H1)K@_@RD@2ZV.png]]

## 第三章 神经网络
### 3.1 从感知机到神经网络
用图来表示神经网络的话，如图3-1所示。我们把最左边的一列称为**输入层**，最右边的一列称为**输出层**，中间的一列称为中间层。中间层有时也称为**隐藏层**。“隐藏”一词的意思是，隐藏层的神经元（和输入层、输出 层不同）肉眼看不见。另外，本书中把输入层到输出层依次称为第0层、第 1层、第2层（层号之所以从0开始，是为了方便后面基于Python进行实现）。 图3-1中，第0层对应输入层，第1层对应中间层，第2层对应输出层。
![[pic/MI[D@_RGD[TP()9}Q9(6VYT.png]]
### 3.2 激活函数
**激活函数的作用在于决定如何来激活输入信号的总和。** 所谓激活函数（Activation Function），就是在人工神经网络的神经元上运行的函数，负责将神经元的输入映射到输出端。然后这里也给出维基百科的定义：在计算网络中， 一个节点的激活函数(Activation Function)定义了该节点在给定的输入或输入的集合下的输出。标准的计算机芯片电路可以看作是根据输入得到开（1）或关（0）输出的数字电路激活函数。这与神经网络中的线性感知机的行为类似。然而，只有非线性激活函数才允许这种网络仅使用少量节点来计算非平凡问题。 在人工神经网络中，这个功能也被称为传递函数。 。
![[pic/Pasted image 20220920151028.png]]
![[pic/Pasted image 20220920151115.png]]
使用非线性激活函数是为了**增加神经网络模型的非线性因素**，以便使网络更加强大，增加它的能力，使它可以学习复杂的事物，复杂的表单数据，以及表示输入输出之间非线性的复杂的任意函数映射。
#### 3.2.1 sigmoid函数
sigmoid函数又称 Logistic函数，用于隐层神经元输出，取值范围为(0,1)，可以用来做二分类。
sigmoid函数表达式：$$\sigma (x) = \frac{1}{{1 + {e^{ - x}}}}$$
![[pic/C0H4)GZTMVQH4Q_Y4CKD3FQ.png]]
**优点：**
1.  Sigmoid函数的输出在(0,1)之间，输出范围有限，优化稳定，可以用作输出层。
2.  连续函数，便于求导。

**缺点：**
1.  sigmoid函数在变量取绝对值非常大的正值或负值时会**出现饱和现象**，意味着函数会变得很平，并且对输入的微小改变会变得不敏感。在反向传播时，当梯度接近于0，权重基本不会更新，很容易就会**出现梯度消失**的情况，从而无法完成深层网络的训练。  
2.  **sigmoid函数的输出不是0均值的**，会导致后层的神经元的输入是非0均值的信号，这会对梯度产生影响。  
3.  **计算复杂度高**，因为sigmoid函数是指数形式。


#### 3.2.2 Tanh函数
Tanh函数也称为双曲正切函数，取值范围为[-1,1]。
Tanh函数定义如下：$$\tanh (x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$

![[pic/[WRR))LW3QTU%EZQBY_5A8K.png]]
Tanh函数是 0 均值的，因此实际应用中 Tanh 会比 sigmoid 更好。但是仍然存在梯度饱和与exp计算的问题。

#### 3.2.3 ReLU函数
ReLU函数定义如下：$$f(x)=\max (0, x)$$
![[pic/]({8_NFQ_CWH7)6K0F}66X8.png]]
**优点：**
1.  使用ReLU的SGD算法的收敛速度比 sigmoid 和 tanh 快。
2.  在x>0区域上，不会出现梯度饱和、梯度消失的问题。
3.  计算复杂度低，不需要进行指数运算，只要一个阈值就可以得到激活值。

**缺点：**
1.  ReLU的输出**不是0均值**的。
2.  **Dead ReLU Problem(神经元坏死现象)**：ReLU在负数区域被kill的现象叫做dead relu。ReLU在训练的时很“脆弱”。在x<0时，梯度为0。这个神经元及之后的神经元梯度永远为0，不再对任何数据有所响应，导致相应参数永远不会被更新。
产生这种现象的两个原因：参数初始化问题；learning rate太高导致在训练过程中参数更新太大。
**解决方法**：采用Xavier初始化方法，以及避免将learning rate设置太大或使用adagrad等自动调节learning rate的算法。

参考  [知乎](https://zhuanlan.zhihu.com/p/337902763)


### 3.3 神经网络信号传递
![[pic/U8Z$S9YVP@~H$_A7(3%)AZU.png]]

信号传递计算方法：$$a_1^{(1)}=w_{11}^{(1)} x_1+w_{12}^{(1)}x_2+b_1^{(1)}$$

矩阵实现方法：$$\boldsymbol{A}^{(1)}=\boldsymbol{X}\boldsymbol{W}^{(1)}+\boldsymbol{B}^{(1)}$$

``` python 
import sys, os  
sys.path.append(os.pardir)    
import numpy as np  
import pickle  
from dataset.mnist import load_mnist  
from common.functions import sigmoid, softmax  
  
  
def get_data():  
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)  
    return x_test, t_test  
  
  
def init_network():  
    with open("sample_weight.pkl", 'rb') as f:  
        network = pickle.load(f)  
    return network  
  
  
def predict(network, x):  
    W1, W2, W3 = network['W1'], network['W2'], network['W3']  
    b1, b2, b3 = network['b1'], network['b2'], network['b3']  
  
    a1 = np.dot(x, W1) + b1  
    z1 = sigmoid(a1)  
    a2 = np.dot(z1, W2) + b2  
    z2 = sigmoid(a2)  
    a3 = np.dot(z2, W3) + b3  
    y = softmax(a3)  
    return y  
  
  
x, t = get_data()  
print(x.shape,t.shape)  
network = init_network()  
accuracy_cnt = 0  
for i in range(len(x)):  
    y = predict(network, x[i])  
  
    p= np.argmax(y)  
    if p == t[i]:  
        accuracy_cnt += 1  
  
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
```

### 3.4 输出层激活函数
输出层所用的激活函数，要根据求解问题的性质决定。一般地，回归问题可以使用恒等函数，二元分类问题可以使用 sigmoid函数，多元分类问题可以使用 softmax函数。

**softmax函数**
$$
y_k=\frac{\exp \left(a_k\right)}{\sum_{i=1}^n \exp \left(a_i\right)}
$$
上面的softmax函数在计算机的运算上有一定的缺陷。这个缺陷就是溢出问题。softmax函数的实现中要进行指数函数的运算，但是此时指数函数的值很容易变得非常大。比如，$e^{100}$的值 会超过20000，$e^{100}$ 会变成一个后面有40多个0的超大值，$e^{1000}$的结果会返回 一个表示无穷大的inf。如果在这些超大值之间进行除法运算，结果会出现“不 确定”的情况.在进行softmax的指数函数的运算时，加上（或者减去）某个常数并不会改变运算的结果。这里的$C^{}$可以使用任何值，但是为了防止溢出，一般会使用输入信号中的最大值。softmax函数的输出是0.0到1.0之间的实数。并且，softmax 函数的输出值的总和是1。

一般而言，神经网络只把输出值最大的神经元所对应的类别作为识别结果。 并且，即便使用softmax函数，输出值最大的神经元的位置也不会变。因此， 神经网络在进行分类时，输出层的softmax函数可以省略。在实际的问题中， 由于指数函数的运算需要一定的计算机运算量，因此输出层的softmax函数 一般会被省略。
### 3.5 神经网络层数、数量的确定
输入层和输出层的层数、大小是最容易确定的。每个网络都有一个输入层，一个输出层。输入层的神经元数目等于将要处理的数据的变量数。输出层的神经元数目等于每个输入对应的输出数。不过，确定隐藏层的层数和大小却是一项挑战。
下面是在分类问题中确定隐藏层的层数，以及每个隐藏层的神经元数目的一些原则：

-   在数据上画出分隔分类的期望边界。
-   将期望边界表示为一组线段。
-   线段数等于第一个隐藏层的隐藏层神经元数。
-   将其中部分线段连接起来（每次选择哪些线段连接取决于设计者），并增加一个新隐藏层。也就是说，每连接一些线段，就新增一个隐藏层。
-   每次连接的连接数等于新增隐藏层的神经元数目。
>_在神经网络中，当且仅当数据必须以非线性的方式分割时，才需要隐藏层。

隐藏层的层数与神经网络的效果/用途，可以用如下表格概括：![[pic/Pasted image 20220920155108.png]]
-   **没有隐藏层**：仅能够表示线性可分函数或决策
-   **隐藏层数=1**：可以拟合任何“包含从一个有限空间到另一个有限空间的连续映射”的函数
-   **隐藏层数=2**：搭配适当的激活函数可以表示任意精度的任意决策边界，并且可以拟合任何精度的任何平滑映射
-   **隐藏层数>2**：多出来的隐藏层可以学习复杂的描述（某种自动特征工程）

层数越深，理论上拟合函数的能力增强，效果按理说会更好，但是实际上更深的层数可能会带来过拟合的问题，同时也会增加训练难度，使模型难以收敛。因此我的经验是，在使用BP神经网络时，最好可以参照已有的表现优异的模型，如果实在没有，则根据上面的表格，从一两层开始尝试，尽量不要使用太多的层数。在CV、NLP等特殊领域，可以使用CNN、RNN、attention等特殊模型，不能不考虑实际而直接无脑堆砌多层神经网络。**尝试迁移和微调已有的预训练模型，能取得事半功倍的效果**。

在隐藏层中使用太少的神经元将导致欠拟合(underfitting)。相反，使用过多的神经元同样会导致一些问题。首先，隐藏层中的神经元过多可能会导致**过拟合(overfitting)**。当神经网络具有过多的节点（过多的信息处理能力）时，训练集中包含的有限信息量不足以训练隐藏层中的所有神经元，因此就会导致过拟合。即使训练数据包含的信息量足够，隐藏层中过多的神经元会增加训练时间，从而难以达到预期的效果。显然，选择一个合适的隐藏层神经元数量是至关重要的。

**经验公式**
$$
N_h=\frac{N_s}{\left(\alpha *\left(N_i+N_o\right)\right)}
$$
其中： Ni 是输入层神经元个数；  
No是输出层神经元个数；  
Ns是训练集的样本数；  
α 是可以自取的任意值变量，通常范围可取 2-10。

总而言之，隐藏层神经元是最佳数量需要**自己通过不断试验获得**，建议从一个较小数值比如1到5层和1到100个神经元开始，如果欠拟合然后慢慢添加更多的层和神经元，如果过拟合就减小层数和神经元。此外，在实际过程中还可以考虑引入**Batch Normalization, Dropout, 正则化**等降低过拟合的方法。

通常，**对所有隐藏层使用相同数量的神经元就足够了**。对于某些数据集，拥有较大的第一层并在其后跟随较小的层将导致更好的性能，因为第一层可以学习很多低阶的特征，这些较低层的特征可以馈入后续层中，提取出较高阶特征。

[Hornik et al., 1989] 证明，只需二个包含足够多神经元的隐层，多层前馈网络就能以任意精度逼近任意复杂度的连续函数.然而，如何设置隐层神经元的个数仍是未解决的问题，实际应用中通常靠"试错法" (trial-by-error) 调整.

## 第四章 神经网络的学习
### 4.1 数据驱动
![[pic/1663687138590.png]]
人们以自己的经验和直觉为线索，通过反复试验推进工作。而机器学习的方法则极力避免人为介入，尝试从收集到的数据中发现答案（模式）。神经网络或深度学习则比以往的机器学习方法更能避免人为介入。
机器学习的方法中，由机器从收集到的数据中找出规律性。
深 度 学 习 有 时 也 称 为 端到端机器学习（end-to-end machine learning）。这里所说的端到端是指从一端到另一端的意思，也就是从原始数据（输入）中获得目标结果（输出）的意思。

机器学习中，一般将数据分为训练数据和测试数据两部分来进行学习和实验等。首先，使用训练数据进行学习，寻找最优的参数；然后，使用测试数据评价训练得到的模型的实际能力。为什么需要将数据分为**训练数据**和**测试数据**呢？因为我们追求的是模型的泛化能力。为了正确评价模型的泛化能力，就必须划分训练数据和测试数据。另外，训练数据也可以称为**监督数据**。

**泛化能力**是指处理未被观察过的数据（不包含在训练数据中的数据）的能力。获得泛化能力是机器学习的最终目标。

只对某个数据集过度拟合的状态称为**过拟合（over fitting）**。
### 4.2 损失函数
神经网络以某个指标为线索寻找最优权重参数。神经网络的学习中所用的指标称为损失函数（loss function）。这个损失函数可以使用任意函数， 但一般用均方误差和交叉熵误差等。

损失函数是表示神经网络性能的“恶劣程度”的指标，即当前的神经网络对监督数据在多大程度上不拟合，在多大程度上不一致。

**深度学习为什么使用 cross entropy loss**

分类问题，都用 onehot + cross entropy

training 过程中，分类问题用 cross entropy，回归问题用 mean squared error。

training 之后，validation / testing 时，使用 classification error，更直观，而且是我们最关注的指标。


### 4.3 欠拟合、过拟合
**过拟合：** 一个假设在训练数据上能够获得比其他假设更好的拟合， 但是在测试数据集上却不能很好地拟合数据，此时认为这个假设出现了过拟合的现象。(模型过于复杂)（高方差）

**欠拟合：** 一个假设在训练数据上不能获得更好的拟合，并且在测试数据集上也不能很好地拟合数据，此时认为这个假设出现了欠拟合的现象。(模型过于简单)（高偏差）

**通过Loss判断**
训练集loss 不断下降，验证集loss不断下降：网络正常，仍在学习。
训练集loss 不断下降，验证集loss趋于不变，可能出现**过拟合**，数据分布不均匀。
训练集loss 不断下降，验证集loss不断上升，可能出现**过拟合**。
训练集loss 趋于不变，验证集loss不断下降，**数据集有问题**。
训练集loss 趋于不变，验证集loss趋于不变，学习过程中遇到瓶颈，可以减小学习率或批量数目和更换梯度优化算法，也有可能网络设计问题。
训练集loss 不断上升，验证集loss不断上升，可能网络结构有问题，超参数设置不正确。

**通过Accuracy判断**
验证集的作用是在训练的过程对对比训练数据与测试数据的准确率，便于判断模型的训练效果是过拟合还是欠拟合 。
过拟合：**训练数据的准确率较高而测试数据的准确率较低**
欠拟合：**训练数据的准确率和测试数据的准确率均较低**

![[Pasted image 20220924101002.png]]
**解决过拟合可能的方法**
方法1：改进模型，有些神经网络的层由于其理论性质本身可能导致结果函数的震荡从而导致过拟合。比如BatchNorm, self-Attention. 去掉这样的层会改善过拟合现象。关于这个问题有很多深刻的理论分析。还有一些手段比如dropout，或者降低神经网络层数等等，相当于降低神经网络的解空间。

方法2：加大训练的batch_size. 使得训练的过程更加关注大量数的平均误差值。但是batch_size不能随意加大，因为batch_size太大会导致mini_batch太大，导致随机梯度下降的时候梯度的随机性下降从而导致训练结果不容易逃逸出局部极小值。

方法3：改进数据设计，使得数据设计更加合理。比如如果特征1对应标签1.特征2对应标签2.当我们发现标签1和标签2的区别可能是比较大的时候，那么设计特征1和特征2的参数化的时候就可以把数据设计得使得其区别更大。这样相当于把样本（x,y）的x值整体拉伸，在y值不变的基础上就可以使得数据的震荡性下降。

方法4： 增加数据量，改善采样方法。 数据量的增加可以多样化数据。采样方法更加均匀可以避免数据在少数类型的样本上权重过大。可以的办法比如Gibbs采样，或者数据分类，每一类选取一些样本，或者直接分类学习，每一类一个神经网络。

方法5：换优化器。如果使用了Adam这类优化器，那么就会提高过拟合风险。把Adam换成SGD可以提升模型的泛化性能，但是会增加训练时间。对于这个问题，也有很多理论解释。但是这也取决于数据的形式。如果特征到标签这个映射是平缓的，那么adam是有不错的效果的而SGD会特别慢。但是如果特征到标签这个映射是很震荡的，那么adam优化器的结果会过拟合，SGD就会更好。

方法6：如果有办法把标签的数据变动变得平缓的话，那么对于神经网络的学习是非常有利的。因为神经网络的学习有一个原理叫频率原理，也就是说神经网络喜欢先学习比较平缓的函数，这样得出的模型可能会偏向于平缓。这样的话就不容易过拟合。

## 第五章 误差反向传播--高效计算权重参数的梯度

![[Pasted image 20220922150541.png]]

$$
\begin{aligned}
&X=Z_0=\left[\begin{array}{c}
0.35 \\
0.9
\end{array}\right] \\
&\mathrm{y}_{\text {out }}=0.5 \\
&W 0=\left[\begin{array}{ll}
w_{31} & w_{32} \\
w_{41} & w_{42}
\end{array}\right]=\left[\begin{array}{ll}
0.1 & 0.8 \\
0.4 & 0.6
\end{array}\right] \\
&W 1=\left[\begin{array}{ll}
w_{53} & w_{54}
\end{array}\right]=\left[\begin{array}{ll}
0.3 & 0.9
\end{array}\right]
\end{aligned}
$$

$$
\begin{aligned}
&z 1=\left[\begin{array}{c}
z_3 \\
z_4
\end{array}\right]=w_0 * X=\left[\begin{array}{ll}
w_{31} & w_{32} \\
w_{41} & w_{42}
\end{array}\right] *\left[\begin{array}{l}
x_1 \\
x_2
\end{array}\right] \\
&=\left[\begin{array}{c}
w_{31} * x_1+w_{32} * x_2 \\
w_{41} * x_1+w_{42} * x_2
\end{array}\right] \\
&=\left[\begin{array}{l}
0.1 * 0.35+0.8 * 0.9 \\
0.4 * 0.35+0.6 * 0.9
\end{array}\right] \\
&=\left[\begin{array}{c}
0.755 \\
0.68
\end{array}\right]
\end{aligned}
$$
$$
\begin{aligned}
&y 1=\left[\begin{array}{l}
y_3 \\
y_4
\end{array}\right]=f(w 0 * X)=f\left(\left[\begin{array}{ll}
w_{31} & w_{32} \\
w_{41} & w_{42}
\end{array}\right] *\left[\begin{array}{l}
x_1 \\
x_2
\end{array}\right]\right) \\
&=f\left(\left[\begin{array}{l}
w_{31} * x_1+w_{32} * x_2 \\
w_{41} * x_1+w_{42} * x_2
\end{array}\right]\right) \\
&=f\left(\left[\begin{array}{l}
0.755 \\
0.68
\end{array}\right]\right) \\
&=\left[\begin{array}{l}
0.680 \\
0.663
\end{array}\right]
\end{aligned}
$$
$$
\begin{aligned}
&z 2=w 1 * y 1=\left[\begin{array}{ll}
w_{53} & w_{54}
\end{array}\right] *\left[\begin{array}{l}
y_3 \\
y_4
\end{array}\right] \\
&=\left[w_{53} * y_3+w_{54} * y_4\right] \\
&=[0.801]
\end{aligned}
$$
$$
\begin{aligned}
& y 2=f(z 2)=f(w 1 * y 1)=f\left(\left[\begin{array}{ll}
w_{53} & w_{54}
\end{array}\right] *\left[\begin{array}{l}
y_3 \\
y_4
\end{array}\right]\right) \\
  &=f\left(\left[w_{53} * y_3+w_{54} * y_4\right]\right) \\
&=f([0.801]) \\
&=[0.690]
\end{aligned}
$$
Error 0.19
$$
C=\frac{1}{2}(0.690-0.5)^2=0.01805
$$

$$
\left\{\begin{array}{c}
C=\frac{1}{2}\left(y_5-y_{\text {out }}\right)^2 \\
y_5=f\left(z_5\right) \\
z_5=\left(w_{53} * y_3+w_{54} * y_4\right)
\end{array}\right.
$$
$$
\begin{aligned}
&\frac{\partial C}{\partial w_{53}}=\frac{\partial C}{\partial y_5} * \frac{\partial y_5}{\partial z_5} * \frac{\partial z_5}{\partial w_{53}} \\
&=\left(y_5-y_{\text {out }}\right) * f\left(z_5\right) *\left(1-f\left(z_5\right)\right) * y_3 \\
&=(0.69-0.5) *(0.69) *(1-0.69) * 0.68 \\
&=0.02763
\end{aligned}
$$
$$
\left\{\begin{array}{l}
w_{31}=w_{31}-\eta\frac{\partial C}{\partial w_{31}}=0.09661944 \\
w_{32}=w_{32}-\eta\frac{\partial C}{\partial w_{32}}=0.78985831 \\
w_{41}=w_{41}-\eta\frac{\partial C}{\partial w_{41}}=0.39661944 \\
w_{42}=w_{42}-\eta\frac{\partial C}{\partial w_{42}}=0.58985831
\end{array}\right.
$$

再按照这个权重参数进行一遍正向传播得出来的Error为0.165,继续迭代，不断修正权值，使得损失函数越来越小，预测值不断逼近0.5.

## 第六章 神经网络学习重要观点

### 6.1 参数更新 

神经网络的学习的目的是找到使损失函数的值尽可能小的参数。这是寻找最优参数的问题，解决这个问题的过程称为最优化（optimization）。参数的梯度（导数）作为了线索。

使用参数的梯度，沿梯度方向更新参数，并重复这个步骤多次，从而逐渐靠近最优参数，这个过程称为随机梯度下降法（stochastic gradient descent）， 简称SGD。

### 6.2 优化算法

#### 6.2.1 SGD
随机梯度下降法
$$\boldsymbol{W} \leftarrow \boldsymbol{W}-\eta \frac{\partial L}{\partial \boldsymbol{W}}$$
损失函数关于W的梯度记为$\frac{\partial L}{\partial \boldsymbol{W}}$,η表示学习率，实际上会取0.01或0.001这些事先决定好的值。

**SGD的缺点**
如果函数的形状非均向（anisotropic），比如呈延伸状，搜索的路径就会非常低效。因此，我们需要比单纯朝梯度方向前进的SGD更聪明的方法。SGD低效的根本原因是，梯度的方向并没有指向最小值的方向。
解决某些问题时可能没有效率。SGD低效的根本原因是，梯度的方向并没有指向最小值的方向。

#### 6.2.2 Momentum
动量
$$\begin{gathered}
\boldsymbol{v} \leftarrow \alpha \boldsymbol{v}-\eta \frac{\partial L}{\partial \boldsymbol{W}} \\
\boldsymbol{W} \leftarrow \boldsymbol{W}+\boldsymbol{v}
\end{gathered}$$

新出现了一个变量v，对应物理上的速度。表示了物体在梯度方向上受力，在这个力的作用下，物体的速度增加这一物理法则。

如果上一次的momentum（即v）与这一次的负梯度方向是相同的，那这次下降的幅度就会加大，所以这样做能够达到加速收敛的过程。

#### 6.2.3 AdaGrad

**学习率衰减**随着学习的进行，使学习率逐渐减小。
$$\begin{aligned}
&\boldsymbol{h} \leftarrow \boldsymbol{h}+\frac{\partial L}{\partial \boldsymbol{W}} \odot \frac{\partial L}{\partial \boldsymbol{W}} \\
&\boldsymbol{W} \leftarrow \boldsymbol{W}-\eta \frac{1}{\sqrt{\boldsymbol{h}}} \frac{\partial L}{\partial \boldsymbol{W}}
\end{aligned}$$

AdaGrad会为参数的每个元素适当地调整学习率，在更新参数时，通过乘以$\frac{1}{\sqrt{\boldsymbol{h}}}$ ，就可以调整学习的尺度。可以按参数的元素进行学习率衰减，使变动大的参数的学习率逐渐减小。参数的元素中变动较大（被大幅更新）的元素的学习率将变小。变动较大$\boldsymbol{h}$较大，$\frac{\eta}{\sqrt{\boldsymbol{h}}}$会变小。

 **AdaGrad的主要优点之一是它消除了手动调整学习率的需要**。AdaGrad在迭代过程中不断调整学习率，并让目标函数中的每个参数都分别拥有自己的学习率。
 
 AdaGrad的主要弱点是它在分母中累积平方梯度：由于每个添加项都是正数，因此在训练过程中累积和不断增长。这反过来又导致学习率不断变小并最终变得无限小，此时算法不再能够获得额外的知识即导致模型不会再次学习。Adadelta算法旨在解决此缺陷。
#### Adam
就是融合了Momentum和AdaGrad的方法。通过组合前面两个方法的优点，有望 实现参数空间的高效搜索。

```
alpha：同样也称为学习率或步长因子，它控制了权重的更新比率（如 0.001）。
较大的值（如 0.3）在学习率更新前会有更快的初始学习，而较小的值（如 1.0E-5）会令训练收敛到更好的性能。
beta1：一阶矩估计的指数衰减率（如 0.9）。
beta2：二阶矩估计的指数衰减率（如 0.999）。该超参数在稀疏梯度（如在 NLP 或计算机视觉任务中）中应该设置为接近 1 的数。
epsilon：该参数是非常小的数，其为了防止在实现中除以零（如 10E-8）。

```

``` python 
不同框架下默认参数
TensorFlow：learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08.
Keras：lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0.
Blocks：learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-08, decay_factor=1.
Lasagne：learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08
Caffe：learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08
MxNet：learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
Torch：learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8

```

``` python
# coding: utf-8  
import numpy as np  
  
class SGD:  
  
    """確率的勾配降下法（Stochastic Gradient Descent）"""  
  
    def __init__(self, lr=0.01):  
        self.lr = lr  
          
    def update(self, params, grads):  
        for key in params.keys():  
            params[key] -= self.lr * grads[key]   
  
  
class Momentum:  
  
    """Momentum SGD"""  
  
    def __init__(self, lr=0.01, momentum=0.9):  
        self.lr = lr  
        self.momentum = momentum  
        self.v = None  
        def update(self, params, grads):  
        if self.v is None:  
            self.v = {}  
            for key, val in params.items():                                  
                self.v[key] = np.zeros_like(val)  
                  
        for key in params.keys():  
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]   
            params[key] += self.v[key]  
  
  
class Nesterov:  
  
    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""  
  
    def __init__(self, lr=0.01, momentum=0.9):  
        self.lr = lr  
        self.momentum = momentum  
        self.v = None  
        def update(self, params, grads):  
        if self.v is None:  
            self.v = {}  
            for key, val in params.items():  
                self.v[key] = np.zeros_like(val)  
              
        for key in params.keys():  
            params[key] += self.momentum * self.momentum * self.v[key]  
            params[key] -= (1 + self.momentum) * self.lr * grads[key]  
            self.v[key] *= self.momentum  
            self.v[key] -= self.lr * grads[key]  
  
  
class AdaGrad:  
  
    """AdaGrad"""  
  
    def __init__(self, lr=0.01):  
        self.lr = lr  
        self.h = None  
        def update(self, params, grads):  
        if self.h is None:  
            self.h = {}  
            for key, val in params.items():  
                self.h[key] = np.zeros_like(val)  
              
        for key in params.keys():  
            self.h[key] += grads[key] * grads[key]  
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)  
  
  
class RMSprop:  
  
    """RMSprop"""  
  
    def __init__(self, lr=0.01, decay_rate = 0.99):  
        self.lr = lr  
        self.decay_rate = decay_rate  
        self.h = None  
        def update(self, params, grads):  
        if self.h is None:  
            self.h = {}  
            for key, val in params.items():  
                self.h[key] = np.zeros_like(val)  
              
        for key in params.keys():  
            self.h[key] *= self.decay_rate  
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]  
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)  
  
  
class Adam:  
  
    """Adam (http://arxiv.org/abs/1412.6980v8)"""  
  
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):  
        self.lr = lr  
        self.beta1 = beta1  
        self.beta2 = beta2  
        self.iter = 0  
        self.m = None  
        self.v = None  
        def update(self, params, grads):  
        if self.m is None:  
            self.m, self.v = {}, {}  
            for key, val in params.items():  
                self.m[key] = np.zeros_like(val)  
                self.v[key] = np.zeros_like(val)  
          
        self.iter += 1  
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)           
          
        for key in params.keys():  
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]  
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])  
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])  
              
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)  
              
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias  
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
```

### 6.3 权重初始值
将权重初始值设为0不是一个好主意，是因为在误差反向传播法中，所有的权重值都会进行相同的更新。比如，在2层神经网络中，假设第1层和第2层的权重为0。这 样一来，正向传播时，因为输入层的权重为0，所以第2层的神经元全部会被传递相同的值。第2层的神经元中全部输入相同的值，这意味着反向传播 时第2层的权重全部都会进行相同的更新。因此，权重被更新为相同的值，并拥有了对称的值（重复的值）。这使得神经网络拥有许多不同的权重的意义丧失了。为了防止“权重均一化” （严格地讲，是为了瓦解权重的对称结构），必须随机生成初始值。(梯度消失训练极度缓慢)

如果前一层的节点数为n，则Xavier 初始值使用标准差为$\frac{1}{\sqrt{\boldsymbol{n}}}$的分布。线性

``` python
w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
```

当激活函数使用ReLU时，，He初始值使用标准差为$\sqrt\frac{2}{{\boldsymbol{n}}}$的高斯分布。
``` python
w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)
```


### 6.4 batch normalization

Batch Normalization (BN) 就被添加在每一个全连接和激励函数之间。Batch Norm的思路是调整各层的激活值分布使其拥有适当的广度。就是进行使数据分布的均值为0、方差为1的正规化。

$$\begin{aligned}
&\mu_B \leftarrow \frac{1}{m} \sum_{i=1}^m x_i \\
&\sigma_B^2 \leftarrow \frac{1}{m} \sum_{i=1}^m\left(x_i-\mu_B\right)^2 \\
&\hat{x}_i \leftarrow \frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}}
\end{aligned}$$

batch normalization的是指在神经网络中激活函数的前面，将wx+b按照特征进行normalization，这样做的好处有三点：

1、提高梯度在网络中的流动。Normalization能够使特征全部缩放到[0,1]，这样在反向传播时候的梯度都是在1左右，避免了梯度消失现象。
2、提升学习速率。归一化后的数据能够快速的达到收敛。
3、减少模型训练对初始化的依赖。

没有BN的时候，模型初始权重值的变化会非常影响梯度下降的结果，因为在前向传播的过程中，激活值的分布会发生变化，由一开始的标准正态分布逐渐发生偏移，也就是internal covariate shift内部协方差移位，均值不再是0，方差也不再是1。而BN就是对于每一层的输入值都进行normalization，通过将每个值减去当前分布的均值，再除以标准差，重新得到标准正态分布，在经过一次仿射变换，既保留了上一层学习的结果，又让分布与一开始的分布偏差没有那么大，因此不用那么小心的选择初始值了，模型的健壮性变得更强。

Why does batch normalization work?

(1) We know that normalizing input features can speed up learning, one intuition is that doing same thing for hidden layers should also work.

(2)solve the problem of covariance shift

Suppose you have trained your cat-recognizing network use black cat, but evaluate on colored cats, you will see data distribution changing(called covariance shift). Even there exist a true boundary separate cat and non-cat, you can't expect learn that boundary only with black cat. So you may need to retrain the network.

For a neural network, suppose input distribution is constant, so output distribution of a certain hidden layer should have been constant. But as the weights of that layer and previous layers changing in the training phase, the output distribution will change, this cause covariance shift from the perspective of layer after it. Just like cat-recognizing network, the following need to re-train. To recover this problem, we use batch normal to force a zero-mean and one-variance distribution. It allow layer after it to learn independently from previous layers, and more concentrate on its own task, and so as to speed up the training process.

(3)Batch normal as regularization(slightly)

In batch normal, mean and variance is computed on mini-batch, which consist not too much samples. So the mean and variance contains noise. Just like dropout, it adds some noise to hidden layer's activation(dropout randomly multiply activation by 0 or 1).

This is an extra and slight effect, don't rely on it as a regularizer.

为什么批处理规范化有效?

(1)我们知道规范化输入特征可以加快学习速度，直觉告诉我们，对隐藏层做同样的事情也应该有效。

(2)解决协方差移位问题

假设你已经训练了你的猫识别网络使用黑猫，但评估彩色猫，你会看到数据分布变化(称为协方差偏移)。即使猫和非猫之间有一个真正的界限，你也不能指望只有黑猫才能知道这个界限。所以你可能需要重新训练这个网络。

对于一个神经网络，假设输入分布是恒定的，那么某个隐层的输出分布应该是恒定的。但在训练阶段，随着该层和前一层权重的变化，输出分布将发生变化，这导致后一层的协方差发生偏移。就像猫识别网络一样，下面这些需要重新训练。为了恢复这个问题，我们使用批正态来强制一个零均值和一方差分布。它使后一层可以独立地从前一层学习，更专注于自己的任务，从而加快训练过程。

(3)批处理正常为正则化(略)

在批正态法中，均值和方差是在样本数量不多的小批上计算的。均值和方差包含噪声。就像dropout，它为隐藏层的激活添加了一些噪音(dropout将激活随机乘以0或1)。

这是一个额外的和轻微的影响，不要依赖它作为正则化。

### 6.5 正则化

权值衰减（weight decay）是一直以来经常被使用的一种抑制过拟合的方法。该方法通过 在学习的过程中对大的权重进行惩罚，来抑制过拟合。

不是权值衰减可以防止过拟，而是权值后面跟的正则项可以防止。权值的作用是为了调节正则项在损失函数中占的权重。权重越大，则说明正则项对损失函数的影响越大。

在损失函数中，weight decay是放在正则项（regularization）前面的一个系数，正则项一般指示模型的复杂度，所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。

### Dropout

Dropout可以作为训练深度神经网络的一种trick供选择。在每个训练批次中，通过忽略一半的特征检测器（让一半的隐层节点值为0），可以明显地减少过拟合现象。这种方式可以减少特征检测器（隐层节点）间的相互作用，检测器相互作用是指某些检测器依赖其他检测器才能发挥作用。

Dropout说的简单一点就是：我们在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征。保证网络的每一层在训练阶段和测试阶段数据分布相同。当丢失率为0.5 时，Dropout会有最强的正则化效果。

首先假设一层神经网络中有n个神经元，其中一个神经元的输出是x，输出期望也是x。加上dropout后，有p的概率这个神经元失活，那么这个神经元的输出期望就变成了(1-p)*x+p*0=(1-p)x，我们需要保证这个神经元在训练和测试阶段的输出期望基本不变。那么就有两种方式来解决：  
第一种在训练的时候，让这个神经元的输出缩放1/(1-p)倍，那么它的输出期望就变成(1-p)x/(1-p)+p*=x，和不dropout的输出期望一致；  
第二种方式是在测试的时候，让神经元的输出缩放(1-p)倍，那么它的输出期望就变成了(1-p)x，和训练时的期望是一致的。

假设我们要训练这样一个神经网络，如图2所示。

![](https://pic3.zhimg.com/80/v2-a7b5591feb14da95d29103913b61265a_1440w.jpg)
（1）首先随机（临时）删掉网络中一半的隐藏神经元，输入输出神经元保持不变（图3中虚线为部分临时被删除的神经元）
  
![](https://pic3.zhimg.com/80/v2-24f1ffc4ef118948501eb713685c068a_1440w.jpg)
（2） 然后把输入x通过修改后的网络前向传播，然后把得到的损失结果通过修改的网络反向传播。一小批训练样本执行完这个过程后，在没有被删除的神经元上按照随机梯度下降法更新对应的参数（w，b）。

（3）然后继续重复这一过程：

-   恢复被删掉的神经元（此时被删除的神经元保持原样，而没有被删除的神经元已经有所更新）
-   从隐藏层神经元中随机选择一个一半大小的子集临时删除掉（备份被删除神经元的参数）。
-   对一小批训练样本，先前向传播然后反向传播损失并根据随机梯度下降法更新参数（w，b） （没有被删除的那一部分参数得到更新，删除的神经元参数保持被删除前的结果）。

### 6.6 超参数的验证
根据不同的数据集，有的会事先分成训练数据、验证数据、测试数据三部分，有的只分成训练数据和测试数据两部分，有的则不进行分割。在这种情况下，用户需要自行进行分割。用于调整超参数的数据，一般称为验证数据（validation data）。

超参数的最优化的内容，简单归纳一下，如下所示。 
步骤0 设定超参数的范围。 
步骤1 从设定的超参数范围中随机采样。 
步骤2 使用步骤1中采样到的超参数的值进行学习，通过验证数据评估识别精度（但是要将epoch设置得很小）。 
步骤3 重复步骤1和步骤2（100次等），根据它们的识别精度的结果，缩小超参数的范围。

## 第七章 卷积神经网络

### 7.1 卷积
![[1664066340212.png]]
卷积的作用：**滤波/特征提取**

有时要向输入数据的周围填入固定的数据（比如0等），这称为填充（padding）
![[1664066503365.png]]
在输入特征图的每一边添加一定数目的行列，使得输出的特征图的长、宽 = 输入的特征图的长、宽使用，填充主要是为了调整输出的大小。比如，对大小为(4, 4)的输入数据应用(3, 3)的滤波器时，输出大小变为(2, 2)，相当于输出大小比输入大小缩小了 2个元素。这在反复进行多次卷积运算的深度网络中会成为问题。为什么呢？因为如果每次进行卷积运算都会缩小 空间，那么在某个时刻输出大小就有可能变为 1，导致无法再应用卷积运算。为了避免出现这样的情况，就要使用填充。在刚才的例子中，将填充的幅度设为 1，那么相对于输入大小(4, 4)，输出大小也保持为原来的(4, 4)。因此，卷积运算就可以在保持空间大小不变的情况下将数据传给下一层。
>设置填充的目的：希望每个输入方块都能作为卷积窗口的中心。

应用滤波器的位置间隔称为步幅（stride）

问题：一个尺寸 a a 的特征图，经过 b b 的卷积层，步幅（stride）=c，填充（padding）=d，
      请计算出输出的特征图尺寸？
答：若d等于0，也就是不填充，输出的特征图的尺寸=（a-b）/c+1
    若d不等于0，也就是填充，输出的特征图的尺寸=（a+2d-b）/c+1

#### 7.2 池化层

在卷积神经网络中通常会在相邻的卷积层之间加入一个池化层，池化层可以有效的缩小参数矩阵的尺寸，从而减少最后连接层的中的参数数量。所以加入池化层可以加快计算速度和防止过拟合的作用。

池化的原理或者是过程：pooling是在不同的通道上分开执行的（就是池化操作不改变通道数），且不需要参数控制。然后根据窗口大小进行相应的操作。 一般有max pooling、average pooling等。

首要作用，下采样（downsamping）
降维、去除冗余信息、对特征进行压缩、简化网络复杂度、减小计算量、减小内存消耗等等。
可以扩大感知野。
可以实现不变性，其中不变形性包括，平移不变性、旋转不变性和尺度不变性。

**感受野** :卷积神经网络每一层输出的特征图(feature map)上的像素点映射回输入图像上的区域大小
**上采样**：实际上就是放大图像，指的是任何可以让图像变成更高分辨率的技术。它有反卷积(Deconvolution，也称转置卷积)、上池化(UnPooling)方法、双线性插值（各种插值算法）。
**下采样**：实际上就是缩小图像，主要目的是为了使得图像符合显示区域的大小，生成对应图像的缩略图。比如说在CNN中的池化层或卷积层就是下采样。不过卷积过程导致的图像变小是为了提取特征，而池化下采样是为了降低特征的维度。
