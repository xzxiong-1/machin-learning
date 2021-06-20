"""
说明：有一堆已经清理好的留言单词及它的所属类，
现在根据已有的数据求一条新的留言所属分类。


回忆讲义中，多变量的贝叶斯公式，
假设某个体有n项特征（Feature），分别为F1、F2、...、Fn。
现有m个类别（Category），分别为C1、C2、...、Cm。
贝叶斯分类器就是计算出概率最大的那个分类，也就是求下面这个算式的最大值：
P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C) / P(F1F2...Fn)

由于 P(F1F2...Fn) 对于所有的类别都是相同的，可以省略，问题就变成了求

　P(F1F2...Fn|C)P(C)

的最大值。

朴素贝叶斯分类器则是更进一步，假设所有特征都彼此独立，因此

　P(F1F2...Fn|C)P(C) = P(F1|C)P(F2|C) ... P(Fn|C)P(C)

上式等号右边的每一项，都可以从统计资料中得到，
由此就可以计算出每个类别对应的概率，从而找出最大概率的那个类。
"""
# In[]:
from numpy import *

#加载数据
postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
classVec = [0, 1, 0, 1, 0, 1]
 
# In[]:
#合并所有单词，利用set来去重，得到所有单词的唯一列表，即创建词汇表
dataSet = postingList
vocabSet = set([])
for document in dataSet:
    # | 操作在python中表示集合的合集/并集
    vocabSet = vocabSet | set(document)

vocabSet = list(vocabSet)
vocabSet

# In[]:
# 将单词列表变为数字向量列表,将每个单词组成的样本转为用数字表示的特征向量
# 输入词汇表vocabList，和要转为特征向量的单词集合inputSet
# 这是一种词袋模型（bagofwords）的编码方式，具体如下描述
def bagOfWords2VecMN(vocabList, inputSet):
    # 特征向量的长度就是词汇表vocabList的长度，初始值全是0
    returnVec = [0] * len(vocabList)   
    for word in inputSet:
        if word in vocabList:
            # 对于输入的词汇集合样本inputSet，如果单词出现一次
            # 则对应位置的0加1,看下面例子
            returnVec[vocabList.index(word)] += 1   #对应单词位置加1
    return returnVec

# In[]:
# 这测试的目的，是可以将所有单词和对应的索引值一起显示出来
from pandas import Series
Series(vocabSet)

# In[]:
"""
['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']
对应的特征向量如下
[0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]


看下面的词汇表：列出了部分出现的词汇，出现词汇所在索引位置
在特征向量对应位置上进行加1操作，这里都是出现1次，所以都是1
0          take
1          help  出现
2            my  出现
3            is
4            so
5         steak
6           how
7           has  出现
8     dalmation
9           dog  出现
10      garbage
11         park
12            I
13    worthless
14         stop
15        licks
16         food
17         flea  出现
18     problems
19          not
20      posting
21           to
22         cute
23       stupid
24          him
25          ate
26       please
27           mr
28       buying
29         love
30         quit
31        maybe

"""

# In[]:
# 将数据集转为词袋模型下的特征向量，在讲自然语言处理时候也说过该方式
trainMat = []
#print(len(vocabSet))
for postinDoc in postingList:
    res = bagOfWords2VecMN(vocabSet, postinDoc)
    print(res)
    #print(len(res),res)
    trainMat.append(res)


# In[]:
# 将训练数据与对应的label（真实类别）转为数组array结构
trainMatrix = array(trainMat)
trainCategory = array(classVec)
trainMatrix

# In[]:
numTrainDocs = len(trainMatrix)  # 样本数
numWords = len(trainMatrix[0])  # 每个样本特征数
numTrainDocs,numWords

# In[]:
# 求类别的先验概率P(C),只有两类0类和1类，
# 通过统计，按1类所占比例求出P(1)，因为只有两类P(0)=1-P(1)
# label只有0和1两个值，所以sum(trainCategory)就是1类的数量
# 通过已有样本中的出现次数/频率来计算先验概率P(C)
pAbusive = sum(trainCategory) / float(numTrainDocs)  
# 1类的先验概率
pAbusive

# In[]:
"""
关键是求每个类别下每个特征的概率值，p(F1|C)p(F2|C)...p(Fn|C)，统计频率即可
该例子的类别与特征如下：
类别      特征
0        [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
1        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
0        [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0]
1        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
0        [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]
1        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0]

针对类别1，需要求出p(F1|1)p(F2|1)...p(F32|1)这32个值
如何求呢？
这个概率值就是该特征对应单词出现的次数/所有单词个数
比如，举个例子，对于类别是1的样本，F1只是出现1次，
样本特征值保存在 trainMatrix 中，一个二维数组，
单词总的个数为n1 = sum(trainMatrix[1])+sum(trainMatrix[3])+sum(trainMatrix[5])
所以p(F1|1) = 1/n1
但是这样会出现一个问题，就是F1-F32对应的一些单词可能没有出现一次，导致p为0值，
那么p(F1|1)*p(F2|1)*...*p(F32|1)的乘积也会是0，不管其他的概率如何，
为了避免这种情况出现，
初始化单词列表和总单词数时，初始化为1和2，这是一种处理技巧。

通过观察，还会发现一个问题，就是p(F1|1)p(F2|1)...p(F32|1)中的一些值可能会非常小。
那么计算乘积p(F1|C)p(F2|C)...p(Fn|C)会更小，在计算机会导致下溢，
就是值太小的时候，会四舍五入到0.

一种解决下溢的办法是对乘积取自然对数。
ln(a*b) = ln(a) + ln(b)，
对数处理后将乘转化为加，来避免下溢出问题

对于贝叶斯问题，就是
求P(F1|C)P(F2|C) ... P(Fn|C)P(C)转为求
ln(P(F1|C)P(F2|C) ... P(Fn|C)P(C)) = ln(P(F1|C))+ln(P(F2|C)) + ... + ln(P(C))
这都是常用的技巧。
"""
# In[]:
#初始化所有单词的基本数量为1
p0Num = ones(numWords)
p1Num = ones(numWords)
p0Num

# In[]:
#初始化总单词数为2 
p0Denom = 2.0
p1Denom = 2.0   

# In[]:
# 根据上面讲解求p(F1|C)p(F2|C)...p(Fn|C)
# 对6个样本进行统计
for i in range(numTrainDocs):
    # 1类进行统计
    if trainCategory[i] == 1:
        # 累计该类中每个特征对应单词的数量
        p1Num += trainMatrix[i]
        # 累计该类中所有单词的数量
        p1Denom += sum(trainMatrix[i])
    else: # 0类进行统计
        p0Num += trainMatrix[i]     
        p0Denom += sum(trainMatrix[i])

# In[]:
print(p1Num)
print(p1Denom)

# In[]:
print(p0Num)
print(p0Denom)

# In[]:
# In[]:
# 返回的是0、1各自两个分类中每个单词数量除以该分类单词总量再取对数ln
# 即求出了p(F1|C)p(F2|C)...p(Fn|C)
p1Vect = log(p1Num / p1Denom) 
p0Vect = log(p0Num / p0Denom) 

# 求出了0、1两个类中各单词所占该类所有单词的比例，以及0、1的类别比例
p0V, p1V, pAb = p0Vect, p1Vect, pAbusive

p0V, p1V, pAb 

# In[]:
# 根据公式求后验概率
# vec2Classify是输入的特征向量
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    print(vec2Classify)
    print(p1Vec)
    print("vec2Classify * p1Vec:",vec2Classify * p1Vec)
    print(sum(vec2Classify * p1Vec))
    print("pClass1:",pClass1)
    print(log(pClass1))
    
    # 将特征向量的每个特征值乘以已经计算出的该特征在1类和0类的概率值
    p1 = sum(vec2Classify * p1Vec) + log(pClass1) 
     
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1) # 由于使用的是ln，这里其实都是对数相加

    print("p1:",p1)
    print(p0)
    if p1 > p0:
        return 1
    else:
        return 0

# In[]:
#下面是预测两条样本数据的类别
testEntry = ['love', 'my', 'dalmation']
thisDoc = array(bagOfWords2VecMN(vocabSet, testEntry)) #先将测试数据转为numpy的词袋模型 [0 2 0 5 1 0 0 3 ...]

print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)) #传值判断

testEntry = ['stupid', 'garbage']
thisDoc = array(bagOfWords2VecMN(vocabSet, testEntry))
print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))



 

 


