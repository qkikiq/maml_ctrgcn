import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def visualize_heatmap_from_csv(csv_path):
    # 读取 CSV 文件（无列标题）
    df = pd.read_csv(csv_path, header=None)

    # 如果第一列是索引（如 0,1,2,...24），去掉它
    if df.shape[1] == 26:
        df = df.iloc[:, 1:]

    matrix = df.values

    # 确保是 25x25
    if matrix.shape != (25, 25):
        raise ValueError(f"Matrix shape must be (25,25), but got {matrix.shape}")

    # 画热力图
    plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(heatmap)
    plt.title("Adjacency Matrix Heatmap (from CSV)")
    plt.xlabel("Node")
    plt.ylabel("Node")
    plt.xticks(ticks=np.arange(25), labels=np.arange(25))
    plt.yticks(ticks=np.arange(25), labels=np.arange(25))
    plt.tight_layout()
    plt.show()


# 用法：传入你的 CSV 路径
visualize_heatmap_from_csv("/dadaY/xinyu/pycharmproject/CTR-GCN-main/visal/__py_debug_temp_var_1605584073.csv")
