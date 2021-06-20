# _*_ coding:utf-8_*_
# 开发人员：**
# 开发时间：2019/12/4 11:12
# 文件名称：text1.py 
# 开发工具：PyCharm
import math
from iteration import combination

def L(x,y,p=2):
    if len(x)==len(y)and len(x)>1:
        sum=0
        for i in range(len(x)):
            sum+=math.pow(abs(x[i] - y[i]), p)
            return math.pow(sum,1/p)
        else:
            return 0

x1 = [1, 1]
x2 = [5, 1]
x3 = [4, 4]
for i in range(1,5):
    r={}






