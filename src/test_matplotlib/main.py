import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei']


xp=np.array([1,2,3])
yp=np.array([2,4,5])
yp2=np.array([3,4,7])

plt.subplot(2,3,1)
plt.plot(xp,yp,'o-.b', markersize=20,mec='r',mfc='pink')
plt.plot(xp,yp2,'o:b', markersize=20,mec='blue',mfc='pink')

plt.title('数据可视化1')
plt.xlabel('1号数据')
plt.ylabel('2号数据')

# 网格
plt.grid()


xp3=np.array([1,2,3])
yp3=np.array([3,3,5])
yp4=np.array([7,5,5])
plt.subplot(2,3,2)
plt.plot(xp3,yp3,'o-.b', markersize=20,mec='r',mfc='pink')
plt.plot(xp3,yp4,'o:b', markersize=20,mec='blue',mfc='pink')
plt.title('数据可视化2')
plt.xlabel('3号数据')
plt.ylabel('4号数据')

plt.suptitle('总数据可视化')

# 散点图
xp=np.array([1,2,3,2,4,5])
yp=np.array([2,4,5,3,4,7])
yp2=np.array([3,4,7,5,6,8])
plt.subplot(2,3,3)
plt.scatter(xp,yp,color='r',cmap='rainbow')
plt.scatter(xp,yp2,color='blue',cmap='rainbow')
plt.colorbar()
plt.title('散点图')

# 柱状图
xp=np.array([1,2,3,4,5])
yp=np.array([2,4,5,3,7])
plt.subplot(2,3,4)
plt.bar(xp,yp)
plt.title('柱状图')

# 直方图
xp=np.random.randn(1000)
plt.subplot(2,3,5)
plt.hist(xp)

# 饼图
xp=np.array([1,2,3,4,5])
mylabels=['1号数据','2号数据','3号数据','4号数据','5号数据']
myexplode=[0,0,0,0,0.1]
plt.subplot(2,3,6)
plt.pie(xp, labels=mylabels, explode=myexplode, startangle=90,counterclock=False,shadow=True)
plt.legend(title='数据来源',bbox_to_anchor=(1.1,1.1),loc='upper right',fontsize=10)
plt.title('饼图')

plt.show()

