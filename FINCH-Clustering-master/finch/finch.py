import json
import time
import argparse
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
import warnings
import sys
from sklearn.manifold import MDS

try:
    from pynndescent import NNDescent

    pynndescent_available = True
except Exception as e:
    warnings.warn('pynndescent not installed: {}'.format(e))
    pynndescent_available = False
    pass


def clust_rank(
        mat,
        use_ann_above_samples,
        initial_rank=None,
        distance='cosine',
        verbose=False):
    s = mat.shape[0]
    if initial_rank is not None:
        orig_dist = np.empty(shape=(1, 1))
    elif s <= use_ann_above_samples:
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        if not pynndescent_available:
            raise MemoryError(
                "You should use pynndescent for inputs larger than {} samples.".format(use_ann_above_samples))
        if verbose:
            print('Using PyNNDescent to compute 1st-neighbours at this step ...')

        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=distance,
        )

        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T

    A = A.tolil()
    A.setdiag(0)
    return A, orig_dist


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    return u, num_clust


def cool_mean(M, u):
    s = M.shape[0]
    un, nf = np.unique(u, return_counts=True)
    umat = sp.csr_matrix((np.ones(s, dtype='float32'), (np.arange(0, s), u)), shape=(s, len(un)))
    return (umat.T @ M) / nf[..., np.newaxis]


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c)
    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()
    v = np.argsort(d[idx])
    v = v[:2]
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    a = sp.lil_matrix(adj.get_shape())
    a[x, y] = 1
    return a


def req_numclust(c, data, req_clust, distance, use_ann_above_samples, verbose):
    iter_ = len(np.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)
    for i in range(iter_):
        adj, orig_dist = clust_rank(mat, use_ann_above_samples, initial_rank=None, distance=distance, verbose=verbose)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_


def FINCH(
        data,
        initial_rank=None,
        req_clust=None,
        distance='cosine',
        ensure_early_exit=True,
        verbose=True,
        use_ann_above_samples=70000):
    """ FINCH clustering algorithm.
    :param data: Input matrix with features in rows.
    :param initial_rank: Nx1 first integer neighbor indices (optional).
    :param req_clust: Set output number of clusters (optional). Not recommended.
    :param distance: One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended 'cosine'.
    :param ensure_early_exit: [Optional flag] may help in large, high dim datasets, ensure purity of merges and helps early exit
    :param verbose: Print verbose output.
    :param use_ann_above_samples: Above this data size (number of samples) approximate nearest neighbors will be used to speed up neighbor
        discovery. For large scale data where exact distances are not feasible to compute, set this. [default = 70000]
    :return:
            c: NxP matrix where P is the partition. Cluster label for every partition.
            num_clust: Number of clusters.
            req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.

    The code implements the FINCH algorithm described in our CVPR 2019 paper
        Sarfraz et al. "Efficient Parameter-free Clustering Using First Neighbor Relations", CVPR2019
         https://arxiv.org/abs/1902.11266
    For academic purpose only. The code or its re-implementation should not be used for commercial use.
    Please contact the author below for licensing information.
    Copyright
    M. Saquib Sarfraz (saquib.sarfraz@kit.edu)
    Karlsruhe Institute of Technology (KIT)
    """
    # Cast input data to float32
    data = data.astype(np.float32)
    min_sim = None
    adj, orig_dist = clust_rank(data,
                                use_ann_above_samples,
                                initial_rank,
                                distance,
                                verbose)
    initial_rank = None
    group, num_clust = get_clust(adj, [], min_sim)
    c, mat = get_merge([], group, data)

    if verbose:
        print('Partition 0: {} clusters'.format(num_clust))

    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = np.max(orig_dist * adj.toarray())

    exit_clust = 2
    c_ = c
    k = 1
    num_clust = [num_clust]

    while exit_clust > 1:
        adj, orig_dist = clust_rank(mat, use_ann_above_samples, initial_rank, distance, verbose)
        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)
        c_, mat = get_merge(c_, u, data)

        num_clust.append(num_clust_curr)
        c = np.column_stack((c, c_))
        exit_clust = num_clust[-2] - num_clust_curr

        if num_clust_curr == 1 or exit_clust < 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break

        if verbose:
            print('Partition {}: {} clusters'.format(k, num_clust[k]))
        k += 1
    req_clust = num_clust[-1]
    # if req_clust is not None:
    #     if req_clust not in num_clust:
    #         ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
    #         req_c = req_numclust(c[:, ind[-1]], data, req_clust, distance, use_ann_above_samples, verbose)
    #     else:
    #         req_c = c[:, num_clust.index(req_clust)]
    # else:
    #     req_c = None
    ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
    req_c = req_numclust(c[:, ind[-1]], data, req_clust, distance, use_ann_above_samples, verbose)
    adj, orig_dist = clust_rank(mat, use_ann_above_samples, initial_rank, distance, verbose)
    return orig_dist, num_clust, req_c
    # return num_clust ,req_c, orig_dist


def main(arg1, arg2):
    # arg1 = "0.5751478,0.4280818,;0.1783997,0.1631058,;0.9975356,0.1036414,;0.7548277,0.466578,;0.9733804,0.9470604,;0.8747535,0.5077925,;0.9874244,0.9775977,;0.1879515,0.8796882,;0.8597863,0.5423701,;0.3107023,0.4890885,;0.2183955,0.0347802,;0.174873,0.195589,;0.1356338,0.0390406,;0.1503235,0.0720445,;0.2759982,0.0469186,;0.3084879,0.4973043,;0.5830073,0.7469067,;0.2043888,0.9910128,;0.2625951,0.4976722,;0.1367608,0.1277423,;0.1945188,0.8926375,;0.3177737,0.0429666,;0.2080078,0.6641732,;0.248973,0.9234697,;0.8656634,0.3649241,;0.5499011,0.3542275,;0.1706268,0.5385256,;0.3993296,0.3046448,;0.2939352,0.1281759,;0.6925292,0.648799,;0.0469297,0.7824946,;0.2454369,0.1340739,;0.1202258,0.0545996,;0.3293529,0.2847847,;0.4862586,0.2406566,;0.1268235,0.4341361,;0.9144353,0.1940729,;0.1770415,0.438654,;0.0616125,0.7443099,;0.5313316,0.101102,;0.5486286,0.6457522,;0.3574949,0.3053949,;0.9761198,0.0204752,;0.4948607,0.6821552,;0.8174804,0.2148862,;0.5362514,0.545879,;0.0887579,0.4773508,;0.1526059,0.0812047,;0.8571549,0.932838,;0.5221443,0.4510417,;0.9217967,0.4696139,;0.0075835,0.7353583,;0.9801746,0.2878505,;0.9584522,0.405765,;0.5663486,0.3539941,;0.2048992,0.3547086,;0.9852381,0.6586196,;0.1459514,0.2236282,;0.0313114,0.2275745,;0.197865,0.7438644,;0.8173528,0.9622735,;0.8234293,0.7869758,;0.5848993,0.6983079,;0.6647302,0.5244404,;0.6450552,0.9228839,;0.8913792,0.1650324,;0.0168629,0.3692667,;0.3824479,0.7535022,;0.9878973,0.6361699,;0.5997486,0.9960733,;0.503404,0.9251767,;0.9865536,0.4837406,;0.4687849,0.2505844,;0.6307713,0.4079459,;0.8857003,0.7918145,;0.1438891,0.2369346,;0.0821574,0.604505,;0.7930718,0.1145126,;0.676856,0.2958829,;0.6375621,0.1630416,;0.5604081,0.5537817,;0.7033821,0.5638705,;0.9910079,0.0850055,;0.2111866,0.3330342,;0.4873479,0.4934262,;0.8673715,0.1274842,;0.8422153,0.2745175,;0.7421067,0.1643857,;0.2489229,0.2217581,;0.9812451,0.3076479,;0.8896741,0.3697573,;0.4109626,0.4038867,;0.7278011,0.8148547,;0.0609209,0.8039754,;0.7096981,0.4695964,;0.0076118,0.3932568,;0.8398521,0.5736814,;0.1657547,0.0178978,;0.0538196,0.5889771,;0.3610789,0.5187958,;0.5769472,0.3758141,;0.0189357,0.1874285,;0.3062636,0.1452562,;0.1038587,0.7971023,;0.1619157,0.9081915,;0.7368346,0.8327394,;0.4829617,0.7351879,;0.578558,0.4746652,;0.2378982,0.2768705,;0.1647393,0.5563692,;0.6181728,0.2864057,;0.667514,0.3199192,;0.7057447,0.3842871,;0.9765707,0.4801707,;0.2636871,0.4992986,;0.6736866,0.9621323,;0.8895104,0.2340579,;0.7313301,0.7143808,;0.0794953,0.2022068,;0.1037764,0.1822791,;0.1515109,0.9678853,;0.4891803,0.2424813,;0.1237103,0.8248905,;0.3021756,0.4868292,;0.8948004,0.9586663,;0.3752235,0.4687991,;0.4534374,0.6522617,;0.3570186,0.5961686,;0.0823482,0.6784302,;0.6391951,0.4140733,;0.0271912,0.8523967,;0.5472513,0.6551453,;0.9685897,0.3292653,;0.2944439,0.157503,;0.6533683,0.4798782,;0.8222973,0.0240186,;0.6804389,0.3699258,;0.3297325,0.6120651,;0.8581876,0.7375496,;0.4035538,0.4338673,;0.6012523,0.090339,;0.7725426,0.0831762,;0.050418,0.7367923,;0.5696077,0.0912979,;0.2026421,0.2807672,;0.5427283,0.2990542,;0.2391182,0.8284706,;0.0347166,0.9930707,;0.035442,0.8119076,;0.0705163,0.7144134,;0.8145227,0.283719,;0.0527578,0.2620469,;0.0325371,0.5867459,;0.0630089,0.8774122,;0.5311089,0.6857769,;0.4765034,0.7416385,;0.0926716,0.8445103,;0.5885004,0.5130489,;0.8357926,0.8511095,;0.4747359,0.3867249,;0.1057589,0.1916037,;0.8080754,0.7144889,;0.0631003,0.4300819,;0.4573758,0.1647817,;0.1466219,0.0039239,;0.219113,0.2470795,;0.7730766,0.5444877,;0.9814476,0.2317476,;0.5054987,0.9895107,;0.185476,0.458099,;0.6365891,0.9012622,;0.0083682,0.0729372,;0.540054,0.7199937,;0.8143242,0.6485588,;0.7765998,0.7787422,;0.8556972,0.1000888,;0.0250461,0.8338419,;0.899641,0.0016672,;0.3528797,0.1138678,;0.5341443,0.8714168,;0.7421994,0.3839502,;0.687726,0.2758371,;0.4315474,0.6257481,;0.8832877,0.6523211,;0.4942469,0.446628,;0.630056,0.4584067,;0.8097202,0.4930498,;0.0171098,0.4601841,;0.5435044,0.6285869,;0.0740775,0.4395216,;0.469472,0.5596939,;0.0631158,0.0374719,;0.6667524,0.5030544,;0.0026479,0.5039056,;0.7141985,0.3753803,;0.0264572,0.3731326,;0.5846026,0.2756747,;0.468,0.0515334,;0.7943971,0.584489,;0.8331347,0.4430684,;0.464934,0.1253846,;0.2762587,0.4464963,;0.9587731,0.2589402,;0.2040171,0.7662412,;0.5163911,0.6849881,;0.7636984,0.2566335,;0.3704938,0.7276251,;0.9840626,0.1737467,;0.3793396,0.0206821,;0.166082,0.3847842,;0.2073134,0.2870498,;0.8987914,0.8639816,;0.0880853,0.2393167,;0.1175821,0.7057962,;0.5182037,0.8848694,;0.4327091,0.367801,;0.8195652,0.4967065,;0.2259382,0.811708,;0.0380251,0.4500162,;0.6220421,0.4733524,;0.4601268,0.6998388,;0.4556623,0.330802,;0.1656854,1.61E-4,;0.0267964,0.4610469,;0.2134186,0.5903576,;0.2112613,0.0245094,;0.6901352,0.9892743,;0.9948134,0.1749154,;0.0388203,0.1662312,;0.5475982,0.4794268,;0.6024569,0.5650889,;0.5531492,0.5073743,;0.6512478,0.5430447,;0.8446749,0.699986,;0.4612117,0.4882943,;0.8597591,0.800209,;0.1295736,0.37589,;0.5692704,0.2706019,;0.2821113,0.4223242,;0.1211296,0.4653646,;0.8723114,0.8573534,;0.4060302,0.9944012,;0.8019933,0.1346329,;0.2291519,0.8049128,;0.7056167,0.0766383,;0.075753,0.028919,;0.220979,0.8573827,;0.0931897,0.5603473,;0.390065,0.1693866,;0.5687823,0.5274289,;"
    # arg1 = "693.0,78.0,;88.0,609.0,;781.0,0.0,;436.0,917.0,;276.0,757.0,;358.0,856.0,;67.0,963.0,;299.0,151.0,;59.0,205.0,;91.0,991.0,;595.0,35.0,;669.0,656.0,;557.0,131.0,;729.0,569.0,;918.0,154.0,;602.0,312.0,;120.0,154.0,;625.0,180.0,;669.0,712.0,;40.0,284.0,;328.0,859.0,;92.0,831.0,;586.0,701.0,;918.0,664.0,;478.0,827.0,;95.0,613.0,;817.0,122.0,;737.0,242.0,;539.0,39.0,;249.0,173.0,;274.0,577.0,;520.0,83.0,;707.0,288.0,;236.0,107.0,;228.0,120.0,;263.0,78.0,;313.0,589.0,;103.0,771.0,;341.0,209.0,;748.0,758.0,;907.0,451.0,;438.0,836.0,;730.0,154.0,;374.0,607.0,;608.0,573.0,;282.0,281.0,;870.0,663.0,;275.0,47.0,;522.0,929.0,;100.0,866.0,;853.0,413.0,;60.0,996.0,;265.0,737.0,;868.0,81.0,;786.0,843.0,;409.0,161.0,;282.0,384.0,;817.0,741.0,;643.0,273.0,;577.0,247.0,;627.0,813.0,;76.0,26.0,;294.0,32.0,;507.0,352.0,;571.0,768.0,;107.0,777.0,;276.0,854.0,;131.0,5.0,;805.0,408.0,;212.0,738.0,;642.0,987.0,;997.0,23.0,;755.0,884.0,;636.0,616.0,;31.0,785.0,;562.0,946.0,;570.0,568.0,;984.0,689.0,;50.0,389.0,;761.0,557.0,;82.0,896.0,;536.0,35.0,;752.0,36.0,;631.0,239.0,;233.0,343.0,;340.0,197.0,;608.0,658.0,;44.0,346.0,;445.0,104.0,;626.0,363.0,;525.0,737.0,;61.0,535.0,;636.0,977.0,;308.0,720.0,;207.0,757.0,;618.0,336.0,;790.0,941.0,;72.0,994.0,;714.0,188.0,;221.0,697.0,;"

    data_a = []
    if arg2 == 'alm':
        file_path = arg1
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines[2:]:
                clean_line = line.strip()  # 删除行首和行尾的空格、换行符等
                row = clean_line.split()  # 分割每行的数字
                row = [int(float(num)) for num in row]  # 转换为整数
                data_a.append(row)  # 将每行添加到矩阵中

        mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress='auto')
        coordinates = mds.fit_transform(data_a)
        data_a = np.array(coordinates)
    elif arg2 == 'sym':
        file_path = arg1
        with open(file_path, 'r') as file:
            num_points = int(file.readline())
            lines = file.readlines()[1:]  # 从第三行开始读取数据
            for line in lines:
                data_a.extend(map(float, line.strip().split()))

        # 计算点的数量

        # 初始化完整的距离矩阵
        distance_matrix = np.full((num_points, num_points), 9999)

        # 将上三角矩阵的数据填入完整的距离矩阵
        index = 0
        for i in range(num_points):
            for j in range(i + 1, num_points):
                distance_matrix[i][j] = data_a[index]
                distance_matrix[j][i] = data_a[index]  # 距离矩阵是对称的
                index += 1
        mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress='auto')
        coordinates = mds.fit_transform(distance_matrix)
        data_a = np.array(coordinates)
    else:
        array_rows = arg1.strip(";").split(";")
        for row in array_rows:
            values = row.split(",")
            row_values = [float(value) for value in values if value != '']
            data_a.append(row_values)
        data_a = np.array(data_a)
    start = time.perf_counter()
    orig_dist, num_clust, req_c_a = FINCH(data_a, distance='euclidean', verbose=False)
    end = time.perf_counter()
    elapsed = end - start

    num_clusters = orig_dist.shape[0]
    nearest_clusters = []
    for cluster_idx in range(num_clusters):
        min_distance = float('inf')
        nearest_cluster_idx = None
        for other_cluster_idx in range(num_clusters):
            if other_cluster_idx != cluster_idx:
                distance = orig_dist[cluster_idx, other_cluster_idx]
                if distance < min_distance:
                    min_distance = distance
                    nearest_cluster_idx = other_cluster_idx
        nearest_clusters.append(nearest_cluster_idx)
    result = {
        "req_c_a": req_c_a.tolist(),
        "orig_dist": nearest_clusters  # It's safe to call tolist() now
    }
    # file_path = r'C:\Users\Administrator\Desktop\FINCH-Clustering-master\FINCH-Clustering-master\finch\output.txt'  # 使用原始字符串
    # with open(file_path, 'a') as file:
    #     file.write(str(elapsed) + '\n')  # 将列表中的每个元素写入文件并换行

    print(json.dumps(result))


if __name__ == '__main__':
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]

    main(arg1, arg2)
    # if len(sys.argv) > 0:
    #     arg1 = sys.argv[1]
    #
    # else:
    #     print("Please provide a value as a command-line argument.")
