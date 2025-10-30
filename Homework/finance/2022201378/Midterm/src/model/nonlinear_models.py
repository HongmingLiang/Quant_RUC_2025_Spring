import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np

class NonLinearModel:
    """非线性回归模型（LightGBM）"""

    def __init__(self, model_type='lgb', params=None):
        self.model_type = model_type
        self.model = None
        self.params = params or {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'random_state': 111,
            'verbose': -1
        }

    def fit(self, train_data: pd.DataFrame, target_column: str, test_size=0.2, random_state=111):
        X = train_data.drop(columns=[target_column], errors='ignore')
        y = train_data[target_column]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model = LGBMRegressor(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            callbacks=[early_stopping(stopping_rounds=50)]
        )

        # 保存训练集和验证集用于评估
        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def get_predict_results(self, X):
        # 1️⃣ 保存 ID
        ids = X['ID']

        # 2️⃣ 去掉 ID 列，只保留训练时使用的特征
        X_features = X.drop(columns=['ID'], errors='ignore')

        # 3️⃣ 预测
        y_pred = self.model.predict(X_features)

        # 4️⃣ 拼接 ID 和预测值
        pred_df = pd.DataFrame({
            'ID': ids,
            'Price': y_pred
        })

        return pred_df

    def evaluate(self):
        y_pred = self.model.predict(self.X_val)
        mae = mean_absolute_error(self.y_val, y_pred)
        r2 = r2_score(self.y_val, y_pred)
        print(f"[{self.model_type.upper()}] 验证集 MAE: {mae:.4f}")
        print(f"[{self.model_type.upper()}] 验证集 R²: {r2:.4f}")
        # return mae, r2

    def feature_importance(self, top_k=20):
        """输出特征重要性"""
        importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        # print("Top feature importances:")
        # print(importance.head(top_k))
        return importance
