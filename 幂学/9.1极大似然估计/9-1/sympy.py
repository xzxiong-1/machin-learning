# In[]:
# 求根号2的平方，sympy库计算的结果最准确
import sympy,math
import numpy as np
print(math.sqrt(2)**2)
print(np.sqrt(2)**2)
print(sympy.sqrt(2)**2)

# In[]:
#simplify 表达式化简
from sympy import *
x,y,z,a,b,c = symbols('x,y,z,a,b,c')
f = (2/3)*x**2 + (1/3)*x**2 + x + x + 1
print(simplify(f))
print(f.simplify())

# In[]:
#expand 表达式展开
f = (x+1)**2
expand(f)
#f.expand()

# In[]:
# solve 方程自动求解
f1 = 2*x - y + z - 10
f2 = 3*x + 2*y - z - 16
f3 = x + 6*y - z - 28
solve([f1,f2,f3])

# In[]:
#diff 求导
#diff(你的函数，自变量，求导的次数），默认1次
print(diff(sin(2*x),x,1))
print(sin(2*x).diff(x))
print(diff(sin(2*x),x))

# In[]:
#定积分
integrate(exp(x),(x,-oo,0))

# In[]:
#不定积分
f = 3*x**2 + 1
integrate(f,x)