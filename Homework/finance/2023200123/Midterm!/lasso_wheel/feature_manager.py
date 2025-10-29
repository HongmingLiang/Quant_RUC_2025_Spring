import numpy as np

class FeatureManager:
    """
    特征管理器，负责特征名称管理、特征增删、特征重要性统计与活跃特征追踪。
    """
    def __init__(self):
        """
        初始化特征管理器。
        """
        self.feature_names = []
        self.feature_stats = {}
        self.coef_history = []

    def set_features(self, feature_names, coefs):
        """
        设置特征名称及其初始系数。
        参数：
            feature_names: 特征名称列表
            coefs: 对应的特征系数
        """
        self.feature_names = feature_names
        self.feature_stats = {name: {'coef': coef, 'count': 0} for name, coef in zip(feature_names, coefs)}
        self.coef_history = [coefs.copy()]

    def add_features(self, new_features, initial_coefs=None):
        """
        动态添加新特征。
        参数：
            new_features: 新特征名称列表
            initial_coefs: 新特征的初始系数（可选）
        """
        for i, name in enumerate(new_features):
            self.feature_names.append(name)
            self.feature_stats[name] = {'coef': 0.0 if initial_coefs is None else initial_coefs[i], 'count': 0}

    def remove_features(self, features_to_remove):
        """
        移除指定特征。
        参数：
            features_to_remove: 需要移除的特征名称列表
        """
        for name in features_to_remove:
            if name in self.feature_names:
                self.feature_names.remove(name)
                self.feature_stats.pop(name, None)

    def update_stats(self, coefs):
        """
        更新特征统计信息。
        参数：
            coefs: 当前特征系数
        """
        self.coef_history.append(coefs.copy())
        for name, coef in zip(self.feature_names, coefs):
            if name in self.feature_stats:
                self.feature_stats[name]['coef'] = coef
                if abs(coef) > 1e-4:
                    self.feature_stats[name]['count'] += 1

    def get_importance_ranking(self):
        """
        获取特征重要性排序。
        返回：
            按绝对值降序排列的(特征名, 重要性)列表
        """
        return sorted([(name, abs(stat['coef'])) for name, stat in self.feature_stats.items()], key=lambda x: -x[1])

    def get_active_features(self, threshold=1e-4):
        """
        获取当前活跃特征。
        参数：
            threshold: 判定为活跃特征的系数阈值
        返回：
            活跃特征名称列表
        """
        return [name for name, stat in self.feature_stats.items() if abs(stat['coef']) > threshold] 