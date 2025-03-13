import numpy as np

# 创建一个一维数组
arr = np.array([1, 2, 3, 4])
print("创建一维数组:", arr)

# 创建一个全为零的数组
zeros_array = np.zeros((3, 4))
print("创建全零数组:", zeros_array)

# 创建一个全为一的数组
ones_array = np.ones((2, 3))
print("创建全一数组:", ones_array)

# 创建一个整数序列
arange_array = np.arange(0, 10, 2)
print("创建整数序列:", arange_array)

# 生成线性空间中的数字序列
linspace_array = np.linspace(0, 1, 5)
print("生成线性空间序列:", linspace_array)

# 计算数组的平均值
mean_value = np.mean(arr)
print("计算平均值:", mean_value)

# 计算数组的总和
sum_value = np.sum(arr)
print("计算总和:", sum_value)

# 找到数组中的最大值
max_value = np.max(arr)
print("找到最大值:", max_value)

# 找到数组中的最小值
min_value = np.min(arr)
print("找到最小值:", min_value)

# 计算数组的标准差
std_value = np.std(arr)
print("计算标准差:", std_value)

# 计算两个数组的点积
dot_product = np.dot(arr, arr)
print("计算点积:", dot_product)

# 计算数组元素之间的高阶乘积
einsum_result = np.einsum('ij,jk->ik', arr, arr)
print("计算高阶乘积:", einsum_result)

# 对数组进行排序
sorted_arr = np.sort(arr)
print("对数组排序:", sorted_arr)

# 获取数组排序索引
sorted_indices = np.argsort(arr)
print("获取排序索引:", sorted_indices)

# 生成一个指定形状和范围的随机数组
random_array = np.random.rand(3, 4)
print("生成随机数组:", random_array)

# 生成一个具有正态分布的随机数组
normal_array = np.random.randn(3, 4)
print("生成正态分布随机数组:", normal_array)

# 将NumPy数组保存到.npy文件中
np.save('my_array.npy', arr)
print("数组已保存到文件")

# 从文件加载数组
loaded_array = np.load('my_array.npy')
print("从文件加载数组:", loaded_array)

# 创建一个具有相同形状和类型的数组，但值为0
zeros_like_array = np.zeros_like(arr)
print("创建与原数组形状和类型相同但值为0的数组:", zeros_like_array)

# 创建一个具有相同形状和类型的数组，但值为1
ones_like_array = np.ones_like(arr)
print("创建与原数组形状和类型相同但值为1的数组:", ones_like_array)

# 创建一个具有指定形状和值的数组
full_array = np.full((3, 4), 7)
print("创建指定形状和值的数组:", full_array)

# 创建一个指定形状和类型的未初始化数组
empty_array = np.empty((3, 4))
print("创建未初始化数组:", empty_array)

# 用指定的值填充数组
np.fill(empty_array, 1)
print("用值1填充数组:", empty_array)

# 重新塑造数组
reshaped_array = np.reshape(arr, (2, 2))
print("重新塑造数组:", reshaped_array)

# 将数组调整大小
resized_array = np.resize(arr, (2, 2))
print("调整数组大小:", resized_array)

# 删除数组的指定元素
deleted_array = np.delete(arr, [1, 2], axis=0)
print("删除数组指定元素:", deleted_array)

# 向数组的末尾追加元素
appended_array = np.append(arr, [5, 6])
print("向数组末尾追加元素:", appended_array)

# 沿着指定轴连接数组
concatenated_array = np.concatenate((arr, arr), axis=0)
print("沿着指定轴连接数组:", concatenated_array)

# 按索引将数组分割成多个子数组
split_arrays = np.split(arr, 2)
print("按索引分割数组:", split_arrays)

# 计算数组沿指定轴的平均值
mean_value_axis0 = np.mean(arr, axis=0)
print("沿axis=0轴计算平均值:", mean_value_axis0)

# 计算数组沿指定轴的总和
sum_value_axis1 = np.sum(arr, axis=1)
print("沿axis=1轴计算总和:", sum_value_axis1)

# 找到数组沿指定轴的最大值
max_value_axis1 = np.max(arr, axis=1)
print("沿axis=1轴找到最大值:", max_value_axis1)

# 找到数组沿指定轴的最小值
min_value_axis1 = np.min(arr, axis=1)
print("沿axis=1轴找到最小值:", min_value_axis1)

# 计算数组沿指定轴的标准差
std_value_axis0 = np.std(arr, axis=0)
print("沿axis=0轴计算标准差:", std_value_axis0)

# 计算数组沿指定轴的方差
var_value_axis0 = np.var(arr, axis=0)
print("沿axis=0轴计算方差:", var_value_axis0)

# 计算数组元素的绝对值
abs_array = np.abs(arr)
print("计算数组元素绝对值:", abs_array)

# 计算数组元素的指数
exp_array = np.exp(arr)
print("计算数组元素指数:", exp_array)

# 计算数组元素的自然对数
log_array = np.log(arr)
print("计算数组元素自然对数:", log_array)

# 计算数组元素的平方根
sqrt_array = np.sqrt(arr)
print("计算数组元素平方根:", sqrt_array)

# 计算数组元素以10为底的对数
log10_array = np.log10(arr)
print("计算数组元素以10为底的对数:", log10_array)

# 计算数组元素的正弦值
sin_array = np.sin(arr)
print("计算数组元素正弦值:", sin_array)

# 计算数组元素的余弦值
cos_array = np.cos(arr)
print("计算数组元素余弦值:", cos_array)

# 计算数组元素的正切值
tan_array = np.tan(arr)
print("计算数组元素正切值:", tan_array)

# 计算数组元素逻辑斯蒂函数值
logistic_array = np.logistic(arr)
print("计算数组元素逻辑斯蒂函数值:", logistic_array)

# 计算数组元素logit函数值
logit_array = np.logit(arr)
print("计算数组元素logit函数值:", logit_array)

# 计算数组元素Sigmoid函数值
sigmoid_array = np.sigmoid(arr)
print("计算数组元素Sigmoid函数值:", sigmoid_array)

# 计算数组元素符号值
sign_array = np.sign(arr)
print("计算数组元素符号值:", sign_array)

# 返回大于或等于输入元素的值的最小整数
ceil_array = np.ceil(arr)
print("返回元素的上一个整数:", ceil_array)

# 返回小于或等于输入元素的最大整数值
floor_array = np.floor(arr)
print("返回元素的下一个整数:", floor_array)

# 返回四舍五入到最接近的整数值
rounded_array = np.round(arr)
print("返回四舍五入值:", rounded_array)