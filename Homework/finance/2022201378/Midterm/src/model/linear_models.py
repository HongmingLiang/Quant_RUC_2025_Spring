from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

class LinearModel:
    def __init__(self, model_type='linear', use_log=True):
        """
        model_type: 'linear', 'lasso', 'ridge', 'elasticnet'
        use_log: 是否对目标值做 log1p 变换（只影响拟合，不影响 MAE 计算）
        """
        self.model_type = model_type
        self.use_log = use_log

        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'lasso':
            self.model = Lasso(max_iter=10000)
        elif model_type == 'ridge':
            self.model = Ridge(max_iter=10000)
        elif model_type == 'elasticnet':
            self.model = ElasticNet(max_iter=10000, random_state=42)
        else:
            raise ValueError("model_type 必须是 'linear'、'lasso'、'ridge' 或 'elasticnet'")

    def fit(self, train_data, target_column='Price', test_size=0.2, random_state=111):
        """训练模型并划分验证集"""
        X = train_data.drop(columns=[target_column])
        y = train_data[target_column]

        # 如果使用 log 变换，仅用于拟合
        if self.use_log:
            y_fit = np.log1p(y)
        else:
            y_fit = y

        X_train, X_val, y_train_fit, y_val_fit = train_test_split(
            X, y_fit, test_size=test_size, random_state=random_state
        )

        # 保存原始 y，用于 MAE 计算
        self.X_train, self.X_val = X_train, X_val
        self.y_train_orig = y.loc[X_train.index]
        self.y_val_orig = y.loc[X_val.index]

        # GridSearchCV 调参
        if self.model_type in ['lasso', 'ridge', 'elasticnet']:
            if self.model_type == 'lasso':
                param_grid = {'alpha': [0.01, 0.1, 1, 10, 50]}
                base_model = Lasso(max_iter=10000)
            elif self.model_type == 'ridge':
                param_grid = {'alpha': [0.01, 0.1, 1, 10, 50]}
                base_model = Ridge(max_iter=10000)
            elif self.model_type == 'elasticnet':
                param_grid = {
                    'alpha': [0.01, 0.1, 1, 10],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }
                base_model = ElasticNet(max_iter=10000, random_state=42)

            grid = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring='neg_mean_absolute_error',
                cv=5
            )
            grid.fit(X_train, y_train_fit)
            self.model = grid.best_estimator_
        else:
            self.model.fit(X_train, y_train_fit)

        return self.model

    def evaluate(self):
        """在验证集上计算原始 MAE"""
        y_val_pred = self.model.predict(self.X_val)
        if self.use_log:
            y_val_pred = np.expm1(y_val_pred)

        val_mae = mean_absolute_error(self.y_val_orig, y_val_pred)
        print(f"[{self.model_type.upper()}] 验证集 MAE (原始房价/租金): {val_mae:.2f}")
        return val_mae

    def cross_val_mae(self, cv=6):
        """手动计算 CV MAE，保证在原始房价/租金水平"""
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        maes = []
        X = self.X_train
        y_orig = self.y_train_orig
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr_orig, y_val_orig = y_orig.iloc[train_idx], y_orig.iloc[val_idx]

            # 拟合
            if self.use_log:
                y_tr_fit = np.log1p(y_tr_orig)
            else:
                y_tr_fit = y_tr_orig

            model_clone = self._clone_model()
            model_clone.fit(X_tr, y_tr_fit)

            y_val_pred = model_clone.predict(X_val)
            if self.use_log:
                y_val_pred = np.expm1(y_val_pred)

            maes.append(mean_absolute_error(y_val_orig, y_val_pred))
        return np.mean(maes)

    def report(self, cv=6, print_report=True):
        """报告训练集、验证集和 CV MAE"""
        # 训练集预测
        y_train_pred = self.model.predict(self.X_train)
        if self.use_log:
            y_train_pred = np.expm1(y_train_pred)
        train_mae = mean_absolute_error(self.y_train_orig, y_train_pred)

        # 验证集预测
        y_val_pred = self.model.predict(self.X_val)
        if self.use_log:
            y_val_pred = np.expm1(y_val_pred)
        val_mae = mean_absolute_error(self.y_val_orig, y_val_pred)

        # CV MAE
        cv_mae = self.cross_val_mae(cv=cv)

        if print_report:
            print(f"{self.model_type.upper():<10} | "
                  f"In sample MAE: {train_mae:.2f} | "
                  f"Out of sample MAE: {val_mae:.2f} | "
                  f"{cv}-fold CV MAE: {cv_mae:.2f}")

        return train_mae, val_mae, cv_mae

    def predict(self, X):
        """预测"""
        y_pred = self.model.predict(X)
        if self.use_log:
            y_pred = np.expm1(y_pred)
        return y_pred

    def get_predict_results(self, X):
        """返回带 ID 的预测结果"""
        ids = X['ID']
        X_features = X.drop(columns=['ID'], errors='ignore')
        y_pred = self.predict(X_features)
        return pd.DataFrame({'ID': ids, 'Price': y_pred})

    def _clone_model(self):
        """复制当前模型结构，用于 CV"""
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'lasso':
            return Lasso(max_iter=10000)
        elif self.model_type == 'ridge':
            return Ridge(max_iter=10000)
        elif self.model_type == 'elasticnet':
            return ElasticNet(max_iter=10000, random_state=42)
