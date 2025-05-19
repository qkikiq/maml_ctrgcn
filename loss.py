import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDASupervisionSignalGenerator:
    def __init__(self):
        pass

    def forward(self, features, labels):
        """
        生成LDA监督信号
        :param features: torch.Tensor，形状为 (batch_size, feature_dim) 的特征数据
        :param labels: torch.Tensor，形状为 (batch_size,) 的标签数据
        :return: torch.Tensor，形状为 (batch_size, new_feature_dim) 的LDA监督信号
        """
        # 将PyTorch张量转换为numpy数组，因为sklearn的LDA接受numpy数组作为输入
        features_numpy = features.detach().cpu().numpy()
        labels_numpy = labels.detach().cpu().numpy()

        # 初始化并拟合LDA模型
        lda = LinearDiscriminantAnalysis()
        lda.fit(features_numpy, labels_numpy)

        # 对特征数据进行LDA变换
        lda_result_numpy = lda.transform(features_numpy)

        # 将结果转换回PyTorch张量，并移动到与输入特征相同的设备上（如果输入特征在GPU上）
        lda_result = torch.tensor(lda_result_numpy, dtype=torch.float32).to(features.device)

        return lda_result