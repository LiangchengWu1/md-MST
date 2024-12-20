import matplotlib.pyplot as plt
import networkx as nx

# 提供的 100 个节点的坐标
x_coordinates = [509, 960, 809, 811, 205, 326, 493, 864, 643, 683, 404, 756, 484, 532, 670, 990, 165, 489, 540,
                  414, 277, 295, 355, 815, 515, 844, 741, 607, 410, 743, 119, 693, 792, 670, 555, 861, 18, 752, 365,
                  485, 636, 312, 327, 932, 837, 956, 979, 550, 372, 795, 714, 343, 982, 239, 9, 461, 660, 704, 990,
                  733, 436, 648, 436, 261, 408, 420, 711, 565, 590, 806, 572, 88, 95, 923, 4, 890, 741, 816, 904,
                  779, 1, 241, 693, 64, 340, 845, 981, 611, 578, 525, 182, 250, 503, 662, 727, 948, 220, 362, 442,
                  642, 577, 455, 411, 120, 353, 571, 860, 300, 33, 675, 563, 266, 393, 238, 118, 728, 835, 403, 814,
                  156, 994, 347, 999, 906, 790, 348, 911, 713, 140, 239, 794, 143, 501, 641, 533, 478, 910, 602, 453,
                  44, 676, 664, 18, 76, 34, 552, 340, 634, 516, 913, 769, 772, 434, 321, 816, 189, 248, 533, 962,
                  794, 665, 707, 531, 637, 780, 352, 401, 541, 815, 369, 17, 148, 828, 756, 707, 229, 827, 528, 315,
                  454, 591, 676, 255, 699, 112, 128, 603, 459, 506, 65, 95, 432, 538, 912, 191, 20, 626, 799, 743]

y_coordinates = [78, 809, 811, 205, 326, 493, 864, 643, 683, 404, 756, 484, 532, 670, 990, 165, 489, 540, 414, 277,
                  295, 355, 815, 515, 844, 741, 607, 410, 743, 119, 693, 792, 670, 555, 861, 18, 752, 365, 485, 636,
                  312, 327, 932, 837, 956, 979, 550, 372, 795, 714, 343, 982, 239, 9, 461, 660, 704, 990, 733, 436,
                  648, 436, 261, 408, 420, 711, 565, 590, 806, 572, 88, 95, 923, 4, 890, 741, 816, 904, 779, 1, 241,
                  693, 64, 340, 845, 981, 611, 578, 525, 182, 250, 503, 662, 727, 948, 220, 362, 442, 642, 577, 455,
                  411, 120, 353, 571, 860, 300, 33, 675, 563, 266, 393, 238, 118, 728, 835, 403, 814, 156, 994, 347,
                  999, 906, 790, 348, 911, 713, 140, 239, 794, 143, 501, 641, 533, 478, 910, 602, 453, 44, 676, 664,
                  18, 76, 34, 552, 340, 634, 516, 913, 769, 772, 434, 321, 816, 189, 248, 533, 962, 794, 665, 707,
                  531, 637, 780, 352, 401, 541, 815, 369, 17, 148, 828, 756, 707, 229, 827, 528, 315, 454, 591, 676,
                  255, 699, 112, 128, 603, 459, 506, 65, 95, 432, 538, 912, 191, 20, 626, 799, 743]


# 提供的簇信息
clusters_dict = {"req_c_a": [0, 1, 2, 3, 1, 4, 4, 5, 2, 3, 3, 3, 5, 1, 4, 2, 6, 5, 2, 4, 5, 3, 1, 1, 4, 1, 7, 8, 5, 1, 4, 4, 3, 5, 5, 5, 9, 2, 1, 1, 2, 7, 3, 1, 5, 0, 3, 6, 2, 3, 5, 3, 3, 6, 8, 5, 3, 8, 6, 7, 7, 7, 1, 7, 2, 7, 10, 5, 11, 4, 12, 2, 9, 4, 5, 1, 4, 7, 3, 11, 1, 5, 6, 3, 6, 8, 7, 1, 7, 4, 5, 2, 2, 10, 3, 9, 3, 2, 12, 1],
                 "orig_dist": [8, 6, 0, 10, 5, 4, 11, 11, 9, 8, 12, 6, 10]}

# 创建一个图形对象
G = nx.Graph()

# 添加节点
for i in range(100):
    G.add_node(i, pos=(x_coordinates[i], y_coordinates[i]))

# 添加边（这里没有提供边信息，你可以根据具体情况添加边）

# 提供不同簇的颜色映射
color_sequence = ['#2878b5', '#9ac9db', '#f8ac8c', '#c82423', '#ff8884', '#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2',
                  '#BEB8DC', '#E7DAD2', '#F27970', '#BB9727', '#54B345', '#32B897', '#05B9E2', '#8983BF', '#C76DA2',
                  '#A1A9D0', '#F0988C', '#B883D4', '#9E9E9E', '#CFEAF1', '#C4A5DE', '#F6CAE5', '#96CCCB', '#228B22',
                  '#218868', '#212121', '#20B2AA', '#1F1F1F', '#1E90FF', '#1E1E1E', '#1C86EE', '#1C1C1C', '#1A1A1A',
                  '#191970', '#1874CD', '#171717', '#141414', '#121212', '#104E8B', '#0F0F0F', '#0D0D0D', '#0A0A0A',
                  '#080808', '#050505', '#030303', '#00FFFF', '#00FF7F', '#00FF00', '#00FA9A', '#00F5FF', '#00EEEE',
                  '#00EE76', '#00EE00', '#00E5EE', '#00CED1', '#00CDCD', '#00CD66', '#00CD00', '#00C5CD', '#00BFFF',
                  '#00B2EE', '#009ACD', '#008B8B', '#008B45', '#008B00', '#00868B', '#00688B', '#006400', '#0000FF',
                  '#0000EE', '#0000CD', '#0000AA', '#00008B']

color_map = {i: color_sequence[i] for i in range(len(color_sequence))}

# 根据簇信息对节点进行着色
node_colors = [color_map[cluster] for cluster in clusters_dict["req_c_a"]]

# 绘制图形
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=50)

# 显示图形
plt.show()
