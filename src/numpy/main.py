import numpy as np

arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([1, 2, 3, 4, 5])

# add
newarr = np.add(arr1, arr2)
print("arr1:", arr1)
print("arr2:", arr2)
print("Sum:", newarr)
# subtract
newarr = np.subtract(arr1, arr2)
print("Subtract:", newarr)
# multiply
newarr = np.multiply(arr1, arr2)
print("Multiply:", newarr)
# divide整除
newarr = np.divide(arr1, arr2)
print("Divide:", newarr)
# power幂
newarr = np.power(arr1, arr2)
print("Power:", newarr)
# mod余数
newarr = np.mod(arr1, arr2)
print("Mod:", newarr)
# absolute
newarr = np.absolute(arr1)
print("Absolute:", newarr) 
