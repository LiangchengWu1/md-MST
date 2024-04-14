import numpy as np

# 从文件读取上三角矩阵的点距离数据（从第三行开始）
data = []
with open("E:\workspace\HABC3\datas\SYM\sym300", "r") as file:
    num_points = int(file.readline())  # 读取第一行的数值
    lines = file.readlines()[1:]
    for line in lines:
        data.extend(map(float, line.strip().split()))



print(num_points)
# 初始化完整的距离矩阵
distance_matrix = np.full((num_points, num_points), 9999)
print(len(data))

# 将上三角矩阵的数据填入完整的距离矩阵
index = 0
for i in range(num_points):
    for j in range(i + 1, num_points):
        distance_matrix[i][j] = data[index]
        distance_matrix[j][i] = data[index]  # 距离矩阵是对称的
        index += 1

# 打印完整的距离矩阵
print(distance_matrix)
