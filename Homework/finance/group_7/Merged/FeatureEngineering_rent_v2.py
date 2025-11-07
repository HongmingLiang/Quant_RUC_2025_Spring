"""
租金数据特征工程与特征选择 - V2（支持Train/Test一致性）
================================================================================
重要改进：
1. 使用类的方式实现 fit/transform 模式
2. 保证训练集和测试集特征工程完全一致
3. 保存所有转换参数供测试集使用
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


class RentFeatureEngineer:
    """
    租金数据特征工程类
    支持fit/transform模式，保证训练集和测试集特征一致
    """
    
    def __init__(self):
        """初始化特征工程器"""
        self.fitted = False
        self.feature_info = {
            'original_features': [],
            'log_features': [],
            'sqrt_features': [],
            'polynomial_features': [],
            'interaction_features': [],
            'binned_features': [],
            'ratio_features': [],
            'statistical_features': []
        }
        
        # 存储转换参数
        self.params = {
            'skewed_features': [],
            'sqrt_features': [],
            'ratio_pairs': [],
            'interaction_pairs': [],
            'binning_rules': {},
            'poly_features': [],
            'group_stats': {},  # 存储分组统计的映射
            'fill_values': {}   # 存储缺失值填充值
        }
    
    def fit(self, df, target='Price'):
        """
        在训练集上拟合特征工程
        
        参数:
            df: 训练集数据框
            target: 目标变量名称
        
        返回:
            self
        """
        print("\n" + "=" * 80)
        print("在训练集上拟合特征工程")
        print("=" * 80)
        
        # 分离特征和目标
        if target in df.columns:
            X = df.drop(columns=[target])
            self.target = target
        else:
            X = df.copy()
            self.target = None
        
        self.feature_info['original_features'] = X.columns.tolist()
        print(f"\n原始特征数: {len(self.feature_info['original_features'])}")
        
        # ====================================================================
        # 1. 确定需要对数变换的特征
        # ====================================================================
        print("\n【步骤1】识别需要对数变换的特征...")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if X[col].min() > 0:
                skewness = X[col].skew()
                if abs(skewness) > 1.0:
                    self.params['skewed_features'].append(col)
        
        # 限制数量
        self.params['skewed_features'] = self.params['skewed_features'][:30]
        print(f"发现 {len(self.params['skewed_features'])} 个偏态特征")
        
        # ====================================================================
        # 2. 确定需要平方根变换的特征
        # ====================================================================
        print("\n【步骤2】识别需要平方根变换的特征...")
        
        sqrt_keywords = ['面积', '建筑面积', '使用面积', '套内面积']
        
        for col in X.columns:
            for keyword in sqrt_keywords:
                if keyword in col and col in numeric_cols and X[col].min() >= 0:
                    self.params['sqrt_features'].append(col)
                    break
        
        print(f"发现 {len(self.params['sqrt_features'])} 个需要平方根变换的特征")
        
        # ====================================================================
        # 3. 确定比例特征对
        # ====================================================================
        print("\n【步骤3】识别比例特征对...")
        
        area_cols = [col for col in X.columns if '面积' in col]
        
        for i, col1 in enumerate(area_cols[:5]):
            for col2 in area_cols[i+1:6]:
                if col2 in numeric_cols and X[col2].abs().max() > 0:
                    self.params['ratio_pairs'].append((col1, col2))
        
        print(f"发现 {len(self.params['ratio_pairs'])} 个比例特征对")
        
        # ====================================================================
        # 4. 确定交互特征对
        # ====================================================================
        print("\n【步骤4】识别交互特征对...")
        
        keywords = ['面积', '室', '厅', '卫', '楼层', '建筑年代', '总价', '单价', 
                    '距离', '人口', '房价', '均价', 'CBD', '地铁', '学校']
        
        important_features = []
        for col in X.columns:
            if any(keyword in col for keyword in keywords):
                important_features.append(col)
                if len(important_features) >= 20:
                    break
        
        for i, col1 in enumerate(important_features[:10]):
            for col2 in important_features[i+1:11]:
                self.params['interaction_pairs'].append((col1, col2))
        
        print(f"发现 {len(self.params['interaction_pairs'])} 个交互特征对")
        
        # ====================================================================
        # 5. 确定分箱规则
        # ====================================================================
        print("\n【步骤5】确定分箱规则...")
        
        binning_candidates = {
            '面积': [0, 50, 80, 100, 150, 200, np.inf],
            '楼层': [0, 5, 10, 20, 30, np.inf],
            '建筑年代': [0, 1980, 1990, 2000, 2010, 2020, np.inf]
        }
        
        for keyword, bins in binning_candidates.items():
            matching_cols = [col for col in X.columns if keyword in col]
            
            for col in matching_cols[:2]:
                try:
                    # 验证分箱是否可行
                    _ = pd.cut(X[col], bins=bins, labels=False, duplicates='drop')
                    self.params['binning_rules'][col] = bins
                except:
                    continue
        
        print(f"确定了 {len(self.params['binning_rules'])} 个分箱规则")
        
        # ====================================================================
        # 6. 确定多项式特征
        # ====================================================================
        print("\n【步骤6】识别多项式特征...")
        
        for keyword in ['面积', '总价', '单价']:
            matching = [col for col in X.columns if keyword in col and col in numeric_cols]
            if matching:
                self.params['poly_features'].append(matching[0])
                if len(self.params['poly_features']) >= 3:
                    break
        
        print(f"发现 {len(self.params['poly_features'])} 个多项式特征")
        
        # ====================================================================
        # 7. 计算分组统计（重要！）
        # ====================================================================
        print("\n【步骤7】计算分组统计...")
        
        group_cols = [col for col in X.columns if any(k in col for k in ['区', '街道', '商圈', '行政'])]
        
        if group_cols:
            group_col = group_cols[0]
            value_cols = [col for col in numeric_cols if col != group_col][:5]
            
            for val_col in value_cols:
                try:
                    # 计算并存储分组均值映射
                    group_mean_map = X.groupby(group_col)[val_col].mean().to_dict()
                    self.params['group_stats'][f'{val_col}_group_mean'] = {
                        'group_col': group_col,
                        'value_col': val_col,
                        'mapping': group_mean_map,
                        'default': X[val_col].mean()  # 默认值（用于未见过的组）
                    }
                except:
                    continue
        
        print(f"计算了 {len(self.params['group_stats'])} 个分组统计")
        
        # ====================================================================
        # 8. 计算缺失值填充值
        # ====================================================================
        print("\n【步骤8】计算缺失值填充值...")
        
        for col in X.columns:
            if col in numeric_cols:
                self.params['fill_values'][col] = X[col].median()
        
        print(f"计算了 {len(self.params['fill_values'])} 个填充值")
        
        self.fitted = True
        
        print("\n" + "=" * 80)
        print("✓ 特征工程参数拟合完成！")
        print("=" * 80)
        
        return self
    
    def transform(self, df, target='Price'):
        """
        使用已拟合的参数转换数据
        
        参数:
            df: 要转换的数据框（训练集或测试集）
            target: 目标变量名称
        
        返回:
            df_transformed: 转换后的数据框
        """
        if not self.fitted:
            raise ValueError("必须先调用 fit() 方法！")
        
        print("\n" + "=" * 80)
        print("应用特征工程转换")
        print("=" * 80)
        
        df_new = df.copy()
        
        # 分离目标变量
        if target in df_new.columns:
            y = df_new[target].copy()
            X = df_new.drop(columns=[target])
        else:
            y = None
            X = df_new.copy()
        
        print(f"\n输入特征数: {X.shape[1]}")
        
        # ====================================================================
        # 1. 对数变换
        # ====================================================================
        print("\n【步骤1】应用对数变换...")
        
        for col in self.params['skewed_features']:
            if col in X.columns and X[col].min() > 0:
                new_col_name = f'{col}_log'
                X[new_col_name] = np.log1p(X[col])
                self.feature_info['log_features'].append(new_col_name)
        
        print(f"创建了 {len(self.params['skewed_features'])} 个对数变换特征")
        
        # ====================================================================
        # 2. 平方根变换
        # ====================================================================
        print("\n【步骤2】应用平方根变换...")
        
        for col in self.params['sqrt_features']:
            if col in X.columns and X[col].min() >= 0:
                new_col_name = f'{col}_sqrt'
                X[new_col_name] = np.sqrt(X[col])
                self.feature_info['sqrt_features'].append(new_col_name)
        
        print(f"创建了 {len(self.params['sqrt_features'])} 个平方根变换特征")
        
        # ====================================================================
        # 3. 比例特征
        # ====================================================================
        print("\n【步骤3】应用比例特征...")
        
        for col1, col2 in self.params['ratio_pairs']:
            if col1 in X.columns and col2 in X.columns:
                new_col_name = f'ratio_{col1}_{col2}'
                X[new_col_name] = X[col1] / (X[col2] + 1e-6)
                self.feature_info['ratio_features'].append(new_col_name)
        
        print(f"创建了 {len(self.params['ratio_pairs'])} 个比例特征")
        
        # ====================================================================
        # 4. 交互特征
        # ====================================================================
        print("\n【步骤4】应用交互特征...")
        
        for col1, col2 in self.params['interaction_pairs']:
            if col1 in X.columns and col2 in X.columns:
                new_col_name = f'interact_{col1}_x_{col2}'
                X[new_col_name] = X[col1] * X[col2]
                self.feature_info['interaction_features'].append(new_col_name)
        
        print(f"创建了 {len(self.params['interaction_pairs'])} 个交互特征")
        
        # ====================================================================
        # 5. 分箱特征
        # ====================================================================
        print("\n【步骤5】应用分箱特征...")
        
        for col, bins in self.params['binning_rules'].items():
            if col in X.columns:
                try:
                    new_col_name = f'{col}_binned'
                    X[new_col_name] = pd.cut(X[col], bins=bins, labels=False, duplicates='drop')
                    X[new_col_name] = X[new_col_name].fillna(0)
                    self.feature_info['binned_features'].append(new_col_name)
                except:
                    continue
        
        print(f"创建了 {len(self.params['binning_rules'])} 个分箱特征")
        
        # ====================================================================
        # 6. 多项式特征
        # ====================================================================
        print("\n【步骤6】应用多项式特征...")
        
        for col in self.params['poly_features']:
            if col in X.columns:
                new_col_name = f'{col}_squared'
                X[new_col_name] = X[col] ** 2
                self.feature_info['polynomial_features'].append(new_col_name)
        
        print(f"创建了 {len(self.params['poly_features'])} 个多项式特征")
        
        # ====================================================================
        # 7. 统计特征（使用训练集计算的映射）
        # ====================================================================
        print("\n【步骤7】应用统计特征...")
        
        created_stats = 0
        for stat_name, stat_info in self.params['group_stats'].items():
            group_col = stat_info['group_col']
            value_col = stat_info['value_col']
            mapping = stat_info['mapping']
            default = stat_info['default']
            
            if group_col in X.columns and value_col in X.columns:
                # 使用训练集的映射
                X[stat_name] = X[group_col].map(mapping).fillna(default)
                
                # 创建与均值的差异
                diff_name = f'{value_col}_diff_from_mean'
                X[diff_name] = X[value_col] - X[stat_name]
                
                self.feature_info['statistical_features'].append(stat_name)
                self.feature_info['statistical_features'].append(diff_name)
                created_stats += 2
        
        print(f"创建了 {created_stats} 个统计特征")
        
        # ====================================================================
        # 8. 处理异常值和缺失值
        # ====================================================================
        print("\n【步骤8】处理异常值和缺失值...")
        
        # 替换无穷值
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 使用训练集计算的中位数填充
        for col in X.columns:
            if X[col].isnull().any():
                fill_value = self.params['fill_values'].get(col, X[col].median())
                X[col].fillna(fill_value, inplace=True)
        
        # ====================================================================
        # 组合数据
        # ====================================================================
        if y is not None:
            df_transformed = pd.concat([X, y], axis=1)
        else:
            df_transformed = X.copy()
        
        print("\n" + "=" * 80)
        print("✓ 特征工程转换完成！")
        print("=" * 80)
        print(f"输入特征数: {len(self.feature_info['original_features'])}")
        print(f"输出特征数: {df_transformed.shape[1] - (1 if y is not None else 0)}")
        print(f"新增特征数: {df_transformed.shape[1] - len(self.feature_info['original_features']) - (1 if y is not None else 0)}")
        print("=" * 80)
        
        return df_transformed
    
    def fit_transform(self, df, target='Price'):
        """
        拟合并转换训练集
        
        参数:
            df: 训练集数据框
            target: 目标变量名称
        
        返回:
            df_transformed: 转换后的数据框
        """
        self.fit(df, target)
        return self.transform(df, target)
    
    def save(self, filepath):
        """保存特征工程器到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"\n✓ 特征工程器已保存至: {filepath}")
    
    @staticmethod
    def load(filepath):
        """从文件加载特征工程器"""
        with open(filepath, 'rb') as f:
            engineer = pickle.load(f)
        print(f"\n✓ 特征工程器已从 {filepath} 加载")
        return engineer


# ============================================================================
# 特征选择函数（保持不变）
# ============================================================================

def feature_selection(df, target='Price', n_features=50):
    """
    特征选择主函数
    
    参数:
        df: 特征工程后的数据框
        target: 目标变量名称
        n_features: 最终保留的特征数量
    
    返回:
        selected_features: 选中的特征列表
        selection_info: 特征选择信息
    """
    
    print("\n" + "=" * 80)
    print("开始特征选择")
    print("=" * 80)
    
    if target not in df.columns:
        print(f"错误：目标变量 '{target}' 不在数据中")
        return df.columns.tolist(), {}
    
    X = df.drop(columns=[target])
    y = df[target]
    
    feature_scores = {}
    selection_info = {}
    
    print(f"\n初始特征数: {X.shape[1]}")
    print(f"目标特征数: {n_features}")
    
    # ============================================================================
    # 1. 移除低方差特征
    # ============================================================================
    print("\n【方法1】移除低方差特征...")
    
    variances = X.var()
    low_variance_cols = variances[variances < 0.001].index.tolist()
    
    print(f"发现 {len(low_variance_cols)} 个低方差特征，将被移除")
    X = X.drop(columns=low_variance_cols)
    selection_info['low_variance_removed'] = low_variance_cols
    
    # ============================================================================
    # 2. 相关性分析
    # ============================================================================
    print("\n【方法2】相关性分析...")
    
    # 与目标变量的相关性
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    
    print("\n前20个与目标变量最相关的特征:")
    print(correlations.head(20))
    
    # 保存相关性分数
    for feature in X.columns:
        if feature not in feature_scores:
            feature_scores[feature] = {}
        feature_scores[feature]['correlation'] = correlations.get(feature, 0)
    
    selection_info['correlations'] = correlations
    
    # 移除与目标变量相关性极低的特征
    low_corr_threshold = 0.01
    low_corr_features = correlations[correlations < low_corr_threshold].index.tolist()
    
    print(f"\n移除 {len(low_corr_features)} 个与目标变量相关性 < {low_corr_threshold} 的特征")
    X = X.drop(columns=low_corr_features)
    selection_info['low_correlation_removed'] = low_corr_features
    
    # ============================================================================
    # 3. 移除高度相关的特征（多重共线性）
    # ============================================================================
    print("\n【方法3】移除高度相关的特征对...")
    
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_threshold = 0.8
    to_drop = []
    
    for column in upper_triangle.columns:
        if any(upper_triangle[column] > high_corr_threshold):
            to_drop.append(column)
    
    print(f"移除 {len(to_drop)} 个与其他特征高度相关 (> {high_corr_threshold}) 的特征")
    X = X.drop(columns=to_drop)
    selection_info['high_correlation_removed'] = to_drop
    
    print(f"剩余特征数: {X.shape[1]}")
    
    # ============================================================================
    # 4. Lasso特征选择
    # ============================================================================
    print("\n【方法4】Lasso特征选择...")
    
    try:
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Lasso CV
        lasso = LassoCV(cv=5, random_state=42, max_iter=5000, n_jobs=-1)
        lasso.fit(X_scaled, y)
        
        # 获取特征重要性
        lasso_coefs = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)
        
        print(f"\nLasso最优alpha: {lasso.alpha_:.4f}")
        print(f"Lasso选择的非零特征数: {(lasso_coefs > 0).sum()}")
        
        print("\n前20个Lasso系数最大的特征:")
        print(lasso_coefs.head(20))
        
        # 保存Lasso分数
        for feature in X.columns:
            if feature in feature_scores:
                feature_scores[feature]['lasso'] = lasso_coefs.get(feature, 0)
        
        selection_info['lasso_coefs'] = lasso_coefs
        
    except Exception as e:
        print(f"Lasso计算出错: {e}")
        lasso_coefs = pd.Series(1, index=X.columns)
        selection_info['lasso_coefs'] = lasso_coefs
    
    # ============================================================================
    # 5. Random Forest特征重要性
    # ============================================================================
#    print("\n【方法5】Random Forest特征重要性...")
    
#    try:
#        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
#        rf.fit(X, y)
        
#        rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        
#        print("\n前20个Random Forest重要性最高的特征:")
#        print(rf_importance.head(20))
        
        # 保存RF分数
#        for feature in X.columns:
#            if feature in feature_scores:
#                feature_scores[feature]['rf_importance'] = rf_importance.get(feature, 0)
        
#        selection_info['rf_importance'] = rf_importance
        
#    except Exception as e:
#        print(f"Random Forest计算出错: {e}")
#        rf_importance = pd.Series(1/X.shape[1], index=X.columns)
#        selection_info['rf_importance'] = rf_importance
    
    # ============================================================================
    # 6. 互信息
    # ============================================================================
    print("\n【方法6】互信息分析...")
    
    try:
        mi_scores = mutual_info_regression(X, y, random_state=42, n_jobs=-1)
        mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        print("\n前20个互信息最高的特征:")
        print(mi_scores.head(20))
        
        # 保存MI分数
        for feature in X.columns:
            if feature in feature_scores:
                feature_scores[feature]['mutual_info'] = mi_scores.get(feature, 0)
        
        selection_info['mutual_info'] = mi_scores
        
    except Exception as e:
        print(f"互信息计算出错: {e}")
        mi_scores = pd.Series(1/X.shape[1], index=X.columns)
        selection_info['mutual_info'] = mi_scores
    
    # ============================================================================
    # 7. 综合评分选择特征
    # ============================================================================
    print("\n【方法7】综合评分...")
    
    # 归一化各个分数
    final_scores = pd.DataFrame(feature_scores).T
    
    # 对每个评分指标进行归一化
    for col in final_scores.columns:
        min_val = final_scores[col].min()
        max_val = final_scores[col].max()
        if max_val > min_val:
            final_scores[col] = (final_scores[col] - min_val) / (max_val - min_val)
        else:
            final_scores[col] = 0
    
    # 计算加权平均（可以调整权重）
    weights = {
        'correlation': 0.25,
        'lasso': 0.35,
        'rf_importance': 0.25,
        'mutual_info': 0.15
    }
    
    # 只使用存在的列
    available_cols = [col for col in weights.keys() if col in final_scores.columns]
    
    final_scores['final_score'] = 0
    for col in available_cols:
        final_scores['final_score'] += final_scores[col] * weights[col]
    
    # 按最终得分排序
    final_scores = final_scores.sort_values('final_score', ascending=False)
    
    print("\n前30个综合得分最高的特征:")
    print(final_scores.head(30))
    
    # 选择前N个特征
    selected_features = final_scores.head(n_features).index.tolist()
    
    selection_info['final_scores'] = final_scores
    selection_info['selected_features'] = selected_features
    
    print("\n" + "=" * 80)
    print("特征选择完成！")
    print("=" * 80)
    print(f"最终选择特征数: {len(selected_features)}")
    print("=" * 80)
    
    return selected_features, selection_info


def visualize_feature_importance(selection_info, top_n=30, save_path=None):
    """
    可视化特征重要性
    
    参数:
        selection_info: 特征选择信息字典
        top_n: 显示前N个特征
        save_path: 保存路径
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 相关性
    if 'correlations' in selection_info:
        corr = selection_info['correlations'].head(top_n)
        axes[0, 0].barh(range(len(corr)), corr.values)
        axes[0, 0].set_yticks(range(len(corr)))
        axes[0, 0].set_yticklabels(corr.index, fontsize=8)
        axes[0, 0].set_xlabel('绝对相关系数')
        axes[0, 0].set_title(f'前{top_n}个特征与目标变量的相关性')
        axes[0, 0].invert_yaxis()
    
    # 2. Lasso系数
    if 'lasso_coefs' in selection_info:
        lasso = selection_info['lasso_coefs'].head(top_n)
        axes[0, 1].barh(range(len(lasso)), lasso.values)
        axes[0, 1].set_yticks(range(len(lasso)))
        axes[0, 1].set_yticklabels(lasso.index, fontsize=8)
        axes[0, 1].set_xlabel('Lasso系数绝对值')
        axes[0, 1].set_title(f'前{top_n}个Lasso系数最大的特征')
        axes[0, 1].invert_yaxis()
    
    # 3. Random Forest重要性
#    if 'rf_importance' in selection_info:
#        rf = selection_info['rf_importance'].head(top_n)
#        axes[1, 0].barh(range(len(rf)), rf.values)
#        axes[1, 0].set_yticks(range(len(rf)))
#        axes[1, 0].set_yticklabels(rf.index, fontsize=8)
#        axes[1, 0].set_xlabel('特征重要性')
#        axes[1, 0].set_title(f'前{top_n}个Random Forest重要性最高的特征')
#        axes[1, 0].invert_yaxis()
    
    # 4. 最终得分
    if 'final_scores' in selection_info:
        final = selection_info['final_scores']['final_score'].head(top_n)
        axes[1, 1].barh(range(len(final)), final.values)
        axes[1, 1].set_yticks(range(len(final)))
        axes[1, 1].set_yticklabels(final.index, fontsize=8)
        axes[1, 1].set_xlabel('综合得分')
        axes[1, 1].set_title(f'前{top_n}个综合得分最高的特征')
        axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存至: {save_path}")
    
    plt.show()


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("租金数据特征工程与特征选择流程 V2")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n【步骤1】加载数据...")
    train_file = "C:/Users/lenovo/Desktop/code/租金数据_训练集_已清洗.csv"
    test_file = "C:/Users/lenovo/Desktop/code/租金数据_测试集_已清洗.csv"  # 如果有
    
    df_train = pd.read_csv(train_file)
    print(f"训练集形状: {df_train.shape}")
    
    # 2. 创建特征工程器并在训练集上拟合
    print("\n【步骤2】在训练集上执行特征工程...")
    engineer = RentFeatureEngineer()
    df_train_engineered = engineer.fit_transform(df_train, target='Price')
    
    # 3. 保存特征工程器
    engineer_file = "C:/Users/lenovo/Desktop/code/feature_engineer.pkl"
    engineer.save(engineer_file)
    
    # 4. 保存特征工程后的训练集
    train_engineered_file = "C:/Users/lenovo/Desktop/code/租金数据_训练集_特征工程_v2.csv"
    df_train_engineered.to_csv(train_engineered_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 特征工程后的训练集已保存至: {train_engineered_file}")
    
    # 5. 如果有测试集，使用相同的参数转换
    try:
        df_test = pd.read_csv(test_file)
        print(f"\n【步骤3】在测试集上应用相同的特征工程...")
        print(f"测试集形状: {df_test.shape}")
        
        df_test_engineered = engineer.transform(df_test, target='Price')
        
        test_engineered_file = "C:/Users/lenovo/Desktop/code/租金数据_测试集_特征工程_v2.csv"
        df_test_engineered.to_csv(test_engineered_file, index=False, encoding='utf-8-sig')
        print(f"\n✓ 特征工程后的测试集已保存至: {test_engineered_file}")
        
        # 验证特征一致性
        train_features = set(df_train_engineered.columns) - {'Price'}
        test_features = set(df_test_engineered.columns) - {'Price'}
        
        print("\n" + "=" * 80)
        print("【特征一致性检查】")
        print("=" * 80)
        print(f"训练集特征数: {len(train_features)}")
        print(f"测试集特征数: {len(test_features)}")
        print(f"是否完全一致: {train_features == test_features}")
        
        if train_features != test_features:
            print(f"\n训练集独有特征: {train_features - test_features}")
            print(f"测试集独有特征: {test_features - train_features}")
        else:
            print("\n✓ 训练集和测试集特征完全一致！")
        print("=" * 80)
        
    except FileNotFoundError:
        print("\n(未找到测试集文件，跳过测试集转换)")
    
    # 6. 特征选择（只在训练集上进行）
    print("\n【步骤4】执行特征选择...")
    selected_features, selection_info = feature_selection(df_train_engineered, target='Price', n_features=50)
    
    # 7. 创建最终数据集
    print("\n【步骤5】创建最终数据集...")
    df_train_final = df_train_engineered[selected_features + ['Price']]
    
    final_train_file = "C:/Users/lenovo/Desktop/code/租金数据_训练集_最终_v2.csv"
    df_train_final.to_csv(final_train_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 最终训练集已保存至: {final_train_file}")
    
    # 如果有测试集，也应用相同的特征选择
    try:
        df_test_final = df_test_engineered[selected_features + (['Price'] if 'Price' in df_test_engineered.columns else [])]
        
        final_test_file = "C:/Users/lenovo/Desktop/code/租金数据_测试集_最终_v2.csv"
        df_test_final.to_csv(final_test_file, index=False, encoding='utf-8-sig')
        print(f"✓ 最终测试集已保存至: {final_test_file}")
    except:
        pass
    
    # 8. 保存特征列表
    features_file = "C:/Users/lenovo/Desktop/code/selected_features_v2.txt"
    with open(features_file, 'w', encoding='utf-8') as f:
        f.write("选中的特征列表:\n")
        f.write("=" * 80 + "\n")
        for i, feat in enumerate(selected_features, 1):
            f.write(f"{i}. {feat}\n")
    
    print(f"\n✓ 特征列表已保存至: {features_file}")
    
    # 9. 可视化
    print("\n【步骤6】生成特征重要性可视化...")
    viz_file = "C:/Users/lenovo/Desktop/code/feature_importance_v2.png"
    visualize_feature_importance(selection_info, top_n=30, save_path=viz_file)
    
    print("\n" + "=" * 80)
    print("✓ 所有流程已完成！")
    print("=" * 80)

