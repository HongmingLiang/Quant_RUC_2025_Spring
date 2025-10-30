import numpy as np

class DriftDetector:
    """
    漂移检测器，基于滑动窗口的损失变化检测模型性能漂移。
    """
    def __init__(self, threshold=0.05, window_size=50):
        """
        初始化漂移检测器。
        参数：
            threshold: 漂移检测阈值（相对损失变化比例）
            window_size: 检测滑动窗口大小
        """
        self.threshold = threshold
        self.window_size = window_size
        self.loss_history = []
        self.drift_detected = False

    def update(self, y_true, y_pred):
        """
        更新检测器状态，输入新一批预测结果，判断是否发生漂移。
        参数：
            y_true: 真实标签
            y_pred: 预测标签
        """
        loss = np.mean((y_true - y_pred) ** 2)
        self.loss_history.append(loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
        if len(self.loss_history) == self.window_size:
            recent_loss = np.mean(self.loss_history[-self.window_size//2:])
            past_loss = np.mean(self.loss_history[:self.window_size//2])
            if abs(recent_loss - past_loss) > self.threshold * past_loss:
                self.drift_detected = True
            else:
                self.drift_detected = False

    def reset(self):
        """
        重置检测器状态。
        """
        self.loss_history = []
        self.drift_detected = False 