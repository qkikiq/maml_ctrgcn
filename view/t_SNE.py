from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


#可视化每帧的特征
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
logits_for_tsne = logits.cpu().detach().numpy()

# 获取当前任务的支持集标签，并移到CPU，转换为NumPy数组
sample_labels_for_tsne = y_spt[i].cpu()
instance_labels = sample_labels_for_tsne.repeat_interleave(m)  # m 是 x_spt.size(5)
# 2. 扩展到帧级别: (N*M*T,)
frame_labels_for_tsne = instance_labels.repeat_interleave(t).numpy()  # t 是 x_spt.size(3)

X_tsne = tsne.fit_transform(logits_for_tsne)

plt.figure(figsize=(8, 6))
# 使用 labels_for_tsne 作为颜色参数 c
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=frame_labels_for_tsne, s=5, cmap='tab10', alpha=0.6)

# 创建图例
# 首先获取唯一的类别标签和它们在 cmap 中的对应颜色
unique_labels = np.unique(frame_labels_for_tsne)
colors = [scatter.cmap(scatter.norm(label)) for label in unique_labels]
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Class {label}',
                              markerfacecolor=color, markersize=5) for label, color in zip(unique_labels, colors)]
plt.legend(handles=legend_elements, title="Classes")

plt.title(f"t-SNE result for Task {i} (2D)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.show()

#可视化类中心
# 1. 确定独特类别
    C_lda = logits.size(1)  # 类别数
    num_total_frames = setsz * m * t
    x_lda = np.random.rand(num_total_frames, C_lda)

    unique_classes = np.unique(y_spt[i].cpu().numpy())  # 得到 ['跑步', '走路']

    # 2. 为 x_lda 中的每一帧分配正确的类别标签
    all_frame_labels = np.empty(num_total_frames, dtype=object)
    for j in range(setsz):
        start_index = j * m * t
        end_index = (j + 1) * m * t
        all_frame_labels[start_index:end_index] = y_spt[i][j].cpu().numpy()

    # 3. 计算每个独特动作类别的中心
    class_centroids = {}
    for cls in unique_classes:
        # 找到属于当前类别的所有帧的索引
        indices_of_frames_in_class = np.where(all_frame_labels == cls)[0]

        # 提取这些帧的特征数据
        frames_of_this_class = x_lda[indices_of_frames_in_class]

        # 计算均值，得到类别中心
        centroids_for_cls = np.mean(frames_of_this_class, axis=0)
        class_centroids[cls] = centroids_for_cls

    # 5. 准备t-SNE可视化
    # 合并所有特征点和中心点
    all_features = x_lda  # 所有LDA特征点
    all_labels = all_frame_labels  # 对应的标签

    # 添加中心点到可视化
    centers_features = np.array([centroid for centroid in class_centroids.values()])
    centers_labels = np.array(list(class_centroids.keys()))

    # 创建标记中心点的标签
    center_markers = np.array(['中心: ' + str(label) for label in centers_labels])

    # 合并特征和标签
    X_combined = np.vstack([all_features, centers_features])
    y_combined = np.hstack([all_labels, centers_labels])
    markers = np.array(['特征点'] * len(all_features) + ['中心点'] * len(centers_features))

    # 6. 应用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_combined) - 1))
    X_tsne = tsne.fit_transform(X_combined)

    # 7. 绘制t-SNE图
    plt.figure(figsize=(10, 8))

    # 为不同类别分配不同颜色
    unique_combined_classes = np.unique(y_combined)
    colors = sns.color_palette("husl", len(unique_combined_classes))
    color_map = {cls: color for cls, color in zip(unique_combined_classes, colors)}

    # 绘制特征点
    for i, cls in enumerate(unique_combined_classes):
        # 特征点
        feature_mask = (y_combined == cls) & (markers == '特征点')
        plt.scatter(
            X_tsne[feature_mask, 0],
            X_tsne[feature_mask, 1],
            c=[color_map[cls]],
            marker='o',
            alpha=0.6,
            s=50,
            label=f'类别 {cls}'
        )

    # 绘制中心点
    for i, cls in enumerate(unique_combined_classes):
        # 中心点
        center_mask = (y_combined == cls) & (markers == '中心点')
        if np.any(center_mask):  # 如果存在该类别的中心
            plt.scatter(
                X_tsne[center_mask, 0],
                X_tsne[center_mask, 1],
                c=[color_map[cls]],
                marker='*',
                s=200,
                edgecolor='black',
                linewidth=1.5,
                label=f'中心 {cls}'
            )

    plt.title('LDA特征与类别中心的t-SNE可视化')
    plt.xlabel('t-SNE维度1')
    plt.ylabel('t-SNE维度2')

    # 添加图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')

    plt.tight_layout()
    plt.savefig('lda_features_and_centroids_tsne.png')
    plt.show()








    # 输出结果
    for cls, centroid_vector in class_centroids.items():
        print(f"类别 '{cls}' 的中心向量: {centroid_vector}")