import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入 tqdm 进度条
import os  # 导入 os 模块用来处理文件夹路径
import joblib  # 导入 joblib 用于保存模型

# 假设你有数据
data = pd.read_csv('./data/raw_data/ruc_Class25Q2_train_price.csv', low_memory=False)  # 替换为你自己的数据路径

# 选择 'lon' 和 'lat' 作为特征
X = data[['lon', 'lat']]

# 评估不同的 K 值使用 KMeans 聚类并计算 Silhouette Score
def evaluate_kmeans_with_silhouette(X, max_k=30):
    silhouette_scores = []
    best_score = -1
    best_k = 1

    # 使用 tqdm 包裹 for 循环，显示进度条
    for k in tqdm(range(12, max_k + 1), desc="Evaluating KMeans", unit="K"):  # 从 12 到 max_k
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)  # 使用 KMeans 聚类进行训练

        # 获取聚类结果
        predictions = kmeans.predict(X)

        # 计算 Silhouette Score
        score = silhouette_score(X, predictions)
        silhouette_scores.append(score)

        if score > best_score:
            best_score = score
            best_k = k

        print(f"n_clusters={k}, Silhouette Score={score}")

    # 绘制轮廓系数图
    plt.plot(range(12, max_k + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. K (Number of Clusters)')
    plt.show()

    print(f"Best K: {best_k} with Silhouette Score: {best_score}")
    return best_k

# 评估不同的K值
best_k = evaluate_kmeans_with_silhouette(X, max_k=30)

# 使用最佳 K 值的 KMeans 模型
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(X)

# 将聚类标签添加到原始数据中
data['cluster_label'] = kmeans.labels_

# 查看带有聚类标签的数据
print(data[['lon', 'lat', 'cluster_label']].head())

# 创建文件夹 './model' 如果它不存在
output_dir = './src/model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存 KMeans 模型到 './model/kmeans_model.pkl'
joblib.dump(kmeans, os.path.join(output_dir, 'kmeans_model.pkl'))

# 保存带有聚类标签的数据到 './model/data_with_cluster_labels.csv'
data.to_csv(os.path.join(output_dir, 'data_with_cluster_labels.csv'), index=False)

print("Model and data saved successfully!")
