import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from feature_manager import FeatureManager
from drift_detector import DriftDetector
import pickle

class StatisticsTracker:
    """
    特征统计跟踪器，记录特征被选中次数、系数累计、活跃时间等。
    """
    def __init__(self):
        self.selection_count = {}
        self.coef_sum = {}
        self.coef_abs_sum = {}
        self.last_active = {}
        self.update_count = 0
        self.stability_history = []

    def update(self, feature_names, coef):
        self.update_count += 1
        for name, c in zip(feature_names, coef):
            if name not in self.selection_count:
                self.selection_count[name] = 0
                self.coef_sum[name] = 0
                self.coef_abs_sum[name] = 0
            if abs(c) > 1e-5:
                self.selection_count[name] += 1
                self.coef_sum[name] += c
                self.coef_abs_sum[name] += abs(c)
                self.last_active[name] = self.update_count

    def get_statistics(self, feature_name):
        return {
            'selection_frequency': self.selection_count.get(feature_name, 0) / max(1, self.update_count),
            'mean_coef': self.coef_sum.get(feature_name, 0) / max(1, self.selection_count.get(feature_name, 1)),
            'mean_abs_coef': self.coef_abs_sum.get(feature_name, 0) / max(1, self.selection_count.get(feature_name, 1)),
            'last_active': self.last_active.get(feature_name, -1)
        }

class OnlineLasso:
    """
    在线Lasso回归模型，支持增量学习、特征动态管理、漂移检测、滑动窗口、正则化动态调整、特征统计跟踪。
    """
    def __init__(self, alpha=0.1, max_iter=1000, tol=1e-4, fit_intercept=True,
                 random_state=None, warm_start=True, memory_factor=0.9,
                 update_strategy='incremental', incremental_method='weighted',
                 history_weight=0.95, new_data_weight=0.05,
                 drift_threshold=0.05, drift_window=50,
                 window_size=None, store_full_data=True, auto_adjust_alpha=False):
        """
        初始化在线Lasso模型。
        参数同readme，新增：
            window_size: 滑动窗口大小（None为无限制）
            store_full_data: 是否存储完整历史数据，否则仅存统计量（预留）
            auto_adjust_alpha: 是否动态调整正则化参数
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.warm_start = warm_start
        self.memory_factor = memory_factor
        self.update_strategy = update_strategy
        self.incremental_method = incremental_method
        self.history_weight = history_weight
        self.new_data_weight = new_data_weight
        self.lasso = Lasso(alpha=alpha, max_iter=max_iter, tol=tol, fit_intercept=fit_intercept, random_state=random_state, warm_start=warm_start)
        self.is_fitted = False
        self.loss_history = []
        self.weight_history = []
        self.history_X = []
        self.history_y = []
        self.max_history_size = 1000000
        self.model_ensemble = []
        self.max_ensemble_size = 5
        self.total_samples_seen = 0
        self.feature_manager = FeatureManager()
        self.drift_detector = DriftDetector(threshold=drift_threshold, window_size=drift_window)
        self.online_update_count = 0
        self.window_size = window_size
        self.store_full_data = store_full_data
        self.auto_adjust_alpha = auto_adjust_alpha
        self.statistics_tracker = StatisticsTracker()
        self.feature_stability_history = []

    def fit(self, X, y, feature_names=None):
        """
        初始训练模型。
        参数：
            X: 特征矩阵
            y: 标签
            feature_names: 特征名称列表
        """
        self.lasso.set_params(alpha=self.alpha)
        self.lasso.fit(X, y)
        self.is_fitted = True
        self.feature_manager.set_features(feature_names if feature_names is not None else [f'f{i}' for i in range(X.shape[1])], self.lasso.coef_)
        self._add_to_history(X, y)
        y_pred = self.lasso.predict(X)
        self.loss_history.append(mean_squared_error(y, y_pred))
        self.weight_history.append(self.lasso.coef_.copy())
        self.statistics_tracker.update(self.feature_manager.feature_names, self.lasso.coef_)

    def partial_fit(self, X_new, y_new, new_features=None, removed_features=None):
        """
        增量训练，支持特征动态增删。
        参数：
            X_new: 新特征数据
            y_new: 新标签
            new_features: 新增特征名称列表
            removed_features: 需要移除的特征名称列表
        """
        if not self.is_fitted:
            raise ValueError("模型尚未进行初始训练")
        # 动态添加新特征
        if new_features is not None and len(new_features) > 0:
            self._add_features(new_features, X_new)
        # 动态移除特征
        if removed_features is not None and len(removed_features) > 0:
            self._remove_features(removed_features)
        self._add_to_history(X_new, y_new)
        # 滑动窗口处理
        if self.window_size is not None and len(self.history_X) > self.window_size:
            self.history_X = self.history_X[-self.window_size:]
            self.history_y = self.history_y[-self.window_size:]
        # 动态调整正则化参数
        if self.auto_adjust_alpha:
            self._adjust_alpha(len(X_new), sum([x.shape[0] for x in self.history_X]))
        # 增量更新策略
        if self.update_strategy == 'incremental':
            self.lasso.set_params(alpha=self.alpha)
            self.lasso.fit(X_new, y_new)
        elif self.update_strategy == 'retrain':
            X_hist = np.vstack(self.history_X)
            y_hist = np.concatenate(self.history_y)
            self.lasso.set_params(alpha=self.alpha)
            self.lasso.fit(X_hist, y_hist)
        elif self.update_strategy == 'ensemble':
            model = Lasso(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol, fit_intercept=self.fit_intercept, random_state=self.random_state, warm_start=self.warm_start)
            model.fit(X_new, y_new)
            self.model_ensemble.append(model)
            if len(self.model_ensemble) > self.max_ensemble_size:
                self.model_ensemble.pop(0)
        self.online_update_count += 1
        y_pred = self.lasso.predict(X_new)
        self.loss_history.append(mean_squared_error(y_new, y_pred))
        self.weight_history.append(self.lasso.coef_.copy())
        self.feature_manager.update_stats(self.lasso.coef_)
        self.statistics_tracker.update(self.feature_manager.feature_names, self.lasso.coef_)
        self._update_feature_stability()
        self.drift_detector.update(y_new, y_pred)
        # 漂移检测自适应重置
        if self.drift_detector.drift_detected:
            self.reset_on_drift()

    def _add_features(self, new_features, X_new):
        """
        内部方法：扩展特征空间，模型系数、特征管理、历史数据同步扩展。
        """
        n_new = len(new_features)
        # 扩展模型系数
        new_coef = np.zeros(n_new)
        self.lasso.coef_ = np.concatenate([self.lasso.coef_, new_coef])
        # 扩展特征管理
        self.feature_manager.add_features(new_features)
        # 扩展历史数据（假设新特征历史为0）
        for i in range(len(self.history_X)):
            n_samples = self.history_X[i].shape[0]
            self.history_X[i] = np.hstack([self.history_X[i], np.zeros((n_samples, n_new))])

    def _remove_features(self, removed_features):
        """
        内部方法：缩减特征空间，模型系数、特征管理、历史数据同步缩减。
        """
        idxs = [self.feature_manager.feature_names.index(f) for f in removed_features if f in self.feature_manager.feature_names]
        keep = [i for i in range(len(self.feature_manager.feature_names)) if i not in idxs]
        self.lasso.coef_ = self.lasso.coef_[keep]
        self.feature_manager.remove_features(removed_features)
        for i in range(len(self.history_X)):
            self.history_X[i] = self.history_X[i][:, keep]

    def _add_to_history(self, X, y):
        """
        内部方法：将数据添加到历史缓存。
        """
        self.history_X.append(X)
        self.history_y.append(y)
        if len(self.history_X) > self.max_history_size:
            self.history_X = self.history_X[-self.max_history_size:]
            self.history_y = self.history_y[-self.max_history_size:]

    def _adjust_alpha(self, n_new, n_total):
        """
        动态调整正则化参数alpha，按新旧数据量比例自适应。
        """
        ratio = n_new / max(1, n_total)
        self.alpha = self.alpha * (1 + 0.5 * ratio)
    
    def _update_feature_stability(self):
        if len(self.feature_manager.coef_history) < 2:
            return  # 如果历史系数不足两次，无法计算稳定性
    
        # 获取前一次和当前的系数
        prev_coef = self.feature_manager.coef_history[-2]
        curr_coef = self.feature_manager.coef_history[-1]
    
        # 对齐特征顺序
        prev_features = self.feature_manager.feature_names[-2]
        curr_features = self.feature_manager.feature_names[-1]
    
        # 创建特征名到索引的映射
        prev_feature_map = {name: i for i, name in enumerate(prev_features)}
        curr_feature_map = {name: i for i, name in enumerate(curr_features)}
    
        # 找到共同的特征
        common_features = list(set(prev_features) & set(curr_features))
    
        # 提取共同特征对应的系数
        aligned_prev_coef = np.array([prev_coef[prev_feature_map[f]] for f in common_features])
        aligned_curr_coef = np.array([curr_coef[curr_feature_map[f]] for f in common_features])
    
        # 如果共同特征不足，直接跳过稳定性计算
        if len(aligned_prev_coef) == 0:
            self.feature_stability_history.append(None)
            return
    
        # 计算稳定性（相关系数）
        if np.std(aligned_prev_coef) > 0 and np.std(aligned_curr_coef) > 0:
            stability = np.corrcoef(aligned_prev_coef, aligned_curr_coef)[0, 1]
        else:
            stability = 0  # 如果标准差为0，相关性定义为0
    
        self.feature_stability_history.append(stability)

    def predict(self, X):
        """
        预测新样本。
        参数：X 特征矩阵
        返回：预测值
        """
        if self.update_strategy == 'ensemble' and self.model_ensemble:
            preds = [m.predict(X) for m in self.model_ensemble]
            return np.mean(preds, axis=0)
        return self.lasso.predict(X)

    def score(self, X, y):
        """
        模型评估，返回R²分数。
        参数：X, y
        返回：R²分数
        """
        return r2_score(y, self.predict(X))

    def get_model_statistics(self):
        """
        获取模型统计信息，包括损失、稀疏度、特征重要性、漂移状态等。
        返回：dict
        """
        return {
            'loss_history': self.loss_history,
            'weight_history': self.weight_history,
            'nonzero_features': np.sum(self.lasso.coef_ != 0),
            'sparsity': 1 - np.sum(self.lasso.coef_ != 0) / len(self.lasso.coef_),
            'feature_importance': self.feature_manager.get_importance_ranking(),
            'drift_detected': self.drift_detector.drift_detected,
            'feature_stability': self.feature_stability_history
        }

    def get_feature_stats(self):
        """
        获取所有特征的统计信息。
        返回：dict，key为特征名，value为统计量
        """
        return {name: self.statistics_tracker.get_statistics(name) for name in self.feature_manager.feature_names}

    def reset_on_drift(self):
        """
        漂移时重置模型。
        """
        self.lasso = Lasso(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol, fit_intercept=self.fit_intercept, random_state=self.random_state, warm_start=self.warm_start)
        self.is_fitted = False
        self.loss_history = []
        self.weight_history = []
        self.history_X = []
        self.history_y = []
        self.feature_manager = FeatureManager()
        self.drift_detector.reset()
        self.statistics_tracker = StatisticsTracker()
        self.feature_stability_history = []
        self.online_update_count = 0
    

    
    def save_model(self, filepath, scaler, feature_names):
        """
        保存模型、Scaler和特征列到文件。
        参数：
            filepath: 保存文件的路径。
            scaler: 数据预处理的Scaler对象。
            feature_names: 训练时使用的特征列名称（列表）。
        """
        with open(filepath, 'wb') as f:
            # 将模型、scaler和特征列打包成一个字典保存
            pickle.dump({'model': self, 'scaler': scaler, 'feature_names': feature_names}, f)
    
    @staticmethod
    def load_model(filepath):
        """
        从文件加载模型、Scaler和特征列。
        参数：
            filepath: 保存文件的路径。
        返回：
            model: 加载的模型。
            scaler: 加载的Scaler对象。
            feature_names: 加载的特征列名称。
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)  # 加载保存的字典
            return data['model'], data['scaler'], data['feature_names']  # 分别返回模型、Scaler和特征列
