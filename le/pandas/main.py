import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 用列表生成Series序列
s=pd.Series([1,2,3,4,5,pd.NA,np.nan,np.nan])

# 用Series、字典、对象生成DataFrame
s=pd.DataFrame({
    'a':[1,2,3,4],
    'b':pd.Timestamp('20230601'),
    'c':pd.Series([1,2,3,4],index=['a1','b1','c1','d1'],dtype='float32'),
    'd':pd.Categorical(['test','train','test','train']),
    'e':'foo',
    'f':pd.date_range('20230601',periods=4,freq='2D')
    })
print(s)


s1=s.sort_index(axis=1,ascending=False)
print(s1)

s2=s.sort_values(by='f',ascending=False)
print(s2)

print('*'*20)
s3=s.set_index('f')
print(s3.loc['20230605'])

print('*'*20)
s4=s.iloc[0:2,-3:]
print(s4)

# 按位置赋值
print('*'*20)
s.iat[0,2]=100
print(s)

# 比较
print('*'*20)
print("s1==s2:\n",s1.iloc[:, 1].eq(s2.iloc[:, 1]))
print("s1>s2:\n",s1.iloc[:, 1].gt(s1.iloc[:, 1]))
print("s1<s1:\n",s1.iloc[:, 1].lt(s1.iloc[:, 1]))
print("s1>=s1:\n",s1.iloc[:, 1].ge(s1.iloc[:, 1]))
print("s1<=s1:\n",s1.iloc[:, 1].le(s1.iloc[:, 1]))
print("s1!=s1:\n",s1.iloc[:, 1].ne(s1.iloc[:, 1]))

# 统计
print('*'*20)
print('s.mean():\n',s.iloc[:, 0].mean())
print('s.cumsum():\n',s.iloc[:, 0].cumsum())

# concat

# join
print('*'*20)
left=pd.DataFrame({'key':['K0','K1','K2','K3'],
                   'A':['A0','A1','A2','A3'],
                   'B':['B0','B1','B2','B3']})
right=pd.DataFrame({'key':['K0','K1','K2','K3'],
                   'C':['C0','C1','C2','C3'],
                   'D':['D0','D1','D2','D3']})
result=pd.merge(left,right,on='key')
print(result)

# 追加Append

# 每天的累计值 = 昨天累计值 + 今天随机值
ts=pd.Series(np.random.randn(1000),index=pd.date_range('20230601',periods=1000,freq='D'))
print(ts.head(5))
# ts.cumsum().plot()
# plt.show()

df=pd.DataFrame(np.random.randn(1000,4),index=pd.date_range('20230601',periods=1000,freq='D'),columns=['A','B','C','D'])
# df.cumsum().plot()
# plt.legend(loc='best')
# plt.show()

# csv输入输出
df.to_csv('data.csv')
pd.read_csv('data.csv')

# HDF5
# !pip install tables
df.to_hdf('data.h5',key='df')
pd.read_hdf('data.h5',key='df')

# Excel
df.to_excel('data.xlsx',sheet_name='Sheet1')
pd.read_excel('data.xlsx','Sheet1',na_values=['NA'])