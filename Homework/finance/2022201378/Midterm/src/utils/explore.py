import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = ['STFangsong']
plt.rcParams['axes.unicode_minus'] = False 
from sklearn.cluster import KMeans
import numpy as np
import geopandas as gpd
import contextily as ctx
import os
import coordTransform as ct

gaode_tiles = "http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}"

osm_tiles = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"


def split_by_city_coords(df, city_col='城市', coord_cols=['coord_x', 'coord_y']):
    """
    将 DataFrame 按城市分组，并提取坐标字段

    参数：
    df : pd.DataFrame
        原始数据
    city_col : str
        城市列名
    coord_cols : list
        坐标列名列表，默认 ['coord_x','coord_y']

    返回：
    city_dict : dict
        key: 城市名
        value: DataFrame，只包含坐标列
    """
    city_dict = {}
    for city, group in df.groupby(city_col):
        city_dict[city] = group[coord_cols].dropna().reset_index(drop=True)
    return city_dict

def plot_elbow_and_best_k(city_coords, k_max=10, plot=True):
    """
    对每个城市使用肘部法（Elbow Method）画聚类 SSE 图，并尝试返回最佳聚类数

    参数：
    city_coords : dict
        key: 城市名
        value: pd.DataFrame，只包含 coord_x 和 coord_y
    k_max : int
        最大聚类数（默认 10）
    plot : bool
        是否画图（默认 True）

    返回：
    best_k_dict : dict
        key: 城市名
        value: 推荐的聚类数 K
    """
    best_k_dict = {}

    for city, df in city_coords.items():
        if df.shape[0] == 0:
            print(f"跳过城市 {city}，数据为空 (0 行)")
            continue  # 跳过空数据
        coords = df[['coord_x','coord_y']].values
        sse = []
        K_range = range(1, k_max+1)
        
        # 计算 SSE
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42).fit(coords)
            sse.append(kmeans.inertia_)
        
        # 简单拐点检测：差分最大下降点
        sse_diff = np.diff(sse)
        # 取差值绝对值最大的点的索引 + 1 作为最佳K（排除K=1）
        best_k = int(np.argmin(sse_diff[1:]) + 2)
        best_k_dict[city] = best_k

        # 画图
        if plot:
            plt.figure(figsize=(6,4))
            plt.plot(K_range, sse, marker='o')
            plt.axvline(best_k, color='r', linestyle='--', label=f'Best K={best_k}')
            plt.xlabel("Number of clusters K")
            plt.ylabel("SSE (Inertia)")
            plt.title(f"Elbow Method - {city}")
            plt.xticks(K_range)
            plt.legend()
            plt.grid(True)
            plt.show()
    
    return best_k_dict

def plot_elbow_and_clusters(city_coords, k_max=10, plot_elbow=True, plot_clusters=True):
    """
    对每个城市画肘部法 SSE 图，并同时绘制聚类散点图（以中位数为中心）。
    
    参数：
    city_coords : dict
        key: 城市名
        value: pd.DataFrame，只包含 coord_x 和 coord_y
    k_max : int
        最大聚类数
    plot_elbow : bool
        是否画肘部法图
    plot_clusters : bool
        是否画聚类散点图
    
    返回：
    best_k_dict : dict
        key: 城市名
        value: 推荐的聚类数 K
    """
    best_k_dict = {}
    
    for city, df in city_coords.items():
        if df.shape[0] == 0:
            print(f"跳过城市 {city}，数据为空 (0 行)")
            continue
        
        coords = df[['coord_x','coord_y']].values
        sse = []
        K_range = range(1, k_max+1)
        
        # 计算 SSE
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42).fit(coords)
            sse.append(kmeans.inertia_)
        
        # 简单拐点检测：差分最大下降点
        sse_diff = np.diff(sse)
        best_k = int(np.argmin(sse_diff[1:]) + 2)
        best_k_dict[city] = best_k
        
        # 绘制肘部法
        if plot_elbow:
            plt.figure(figsize=(6,4))
            plt.plot(K_range, sse, marker='o')
            plt.axvline(best_k, color='r', linestyle='--', label=f'Best K={best_k}')
            plt.xlabel("Number of clusters K")
            plt.ylabel("SSE (Inertia)")
            plt.title(f"Elbow Method - {city}")
            plt.xticks(K_range)
            plt.legend()
            plt.grid(True)
            plt.show()
        
        # 聚类散点图
        if plot_clusters:
            kmeans = KMeans(n_clusters=best_k, random_state=42).fit(coords)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            
            plt.figure(figsize=(6,6))
            for i in range(best_k):
                cluster_points = coords[labels == i]
                plt.scatter(cluster_points[:,0], cluster_points[:,1], label=f'Cluster {i+1}', alpha=0.6)
                
                # 标出中位数点
                median_x, median_y = np.median(cluster_points, axis=0)
                plt.scatter(median_x, median_y, marker='X', color='k', s=100)
            
            plt.scatter(centers[:,0], centers[:,1], marker='*', color='red', s=150, label='Centroids')
            plt.xlabel("coord_x")
            plt.ylabel("coord_y")
            plt.title(f"Clusters - {city}")
            plt.legend()
            plt.grid(True)
            plt.show()
    
    return best_k_dict

def plot_elbow_and_clusters_side_by_side(city_coords, k_max=10):
    """
    对每个城市在同一张图中左右显示：
    左侧：肘部法 SSE 曲线
    右侧：聚类散点图（以中位数为中心，标出聚类中心）

    参数：
    city_coords : dict
        key: 城市名
        value: pd.DataFrame，只包含 coord_x 和 coord_y
    k_max : int
        最大聚类数
    
    返回：
    best_k_dict : dict
        key: 城市名
        value: 推荐的聚类数 K
    """
    best_k_dict = {}

    for city, df in city_coords.items():
        if df.shape[0] == 0:
            print(f"跳过城市 {city}，数据为空 (0 行)")
            continue

        coords = df[['coord_x','coord_y']].values

        # ------------------
        # 计算 SSE
        # ------------------
        sse = []
        K_range = range(1, k_max+1)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42).fit(coords)
            sse.append(kmeans.inertia_)

        # 简单拐点检测
        sse_diff = np.diff(sse)
        best_k = int(np.argmin(sse_diff[1:]) + 2)
        best_k_dict[city] = best_k

        # ------------------
        # 创建左右两个子图
        # ------------------
        fig, axes = plt.subplots(1, 2, figsize=(14,5))

        # 左侧：肘部法
        axes[0].plot(K_range, sse, marker='o')
        axes[0].axvline(best_k, color='r', linestyle='--', label=f'Best K={best_k}')
        axes[0].set_xlabel("Number of clusters K")
        axes[0].set_ylabel("SSE (Inertia)")
        axes[0].set_title(f"Elbow Method - {city}")
        axes[0].set_xticks(K_range)
        axes[0].legend()
        axes[0].grid(True)

        # 右侧：聚类散点图
        kmeans = KMeans(n_clusters=best_k, random_state=42).fit(coords)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        for i in range(best_k):
            cluster_points = coords[labels == i]
            axes[1].scatter(cluster_points[:,0], cluster_points[:,1], label=f'Cluster {i+1}', alpha=0.6)
            median_x, median_y = np.median(cluster_points, axis=0)
            axes[1].scatter(median_x, median_y, marker='X', color='k', s=80)  # 中位数点

        axes[1].scatter(centers[:,0], centers[:,1], marker='*', color='red', s=120, label='Centroids')
        axes[1].set_xlabel("coord_x")
        axes[1].set_ylabel("coord_y")
        axes[1].set_title(f"Clusters - {city}")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    return best_k_dict

def plot_city_clusters_on_map_separately(city_coords, k_dict=None, zoom_factor=0.01):
    """
    每个城市单独画聚类散点图（经纬度坐标），并加底图。

    参数：
    city_coords : dict
        key: 城市名
        value: pd.DataFrame，只包含 'coord_x' 和 'coord_y'
    k_dict : dict, optional
        key: 城市名
        value: 聚类数 K，若为 None，则默认 3
    zoom_factor : float
        经纬度扩展范围，用于截取底图
    
    返回：
    None
    """
    for city, df in city_coords.items():
        if df.shape[0] == 0:
            print(f"跳过城市 {city}，数据为空 (0 行)")
            continue

        coords = df[['coord_x','coord_y']].values
        K = k_dict.get(city, 3) if k_dict else 3

        # 聚类
        kmeans = KMeans(n_clusters=K, random_state=42).fit(coords)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {'label': labels},
            geometry=gpd.points_from_xy(coords[:,0], coords[:,1]),
            crs="EPSG:4326"
        )

        # 绘制聚类散点图
        fig, ax = plt.subplots(figsize=(8,8))
        for cluster_id in range(K):
            gdf[gdf['label']==cluster_id].plot(ax=ax, markersize=20, label=f'Cluster {cluster_id+1}', alpha=0.6)
            cluster_points = coords[labels==cluster_id]
            median_x, median_y = np.median(cluster_points, axis=0)
            ax.scatter(median_x, median_y, marker='X', color='k', s=100)  # 中位数

        # 标出中心点
        ax.scatter(centers[:,0], centers[:,1], marker='*', color='red', s=150, label='Centroids')

        # 设置范围
        min_coord_x, max_coord_x = coords[:,0].min()-zoom_factor, coords[:,0].max()+zoom_factor
        min_coord_y, max_coord_y = coords[:,1].min()-zoom_factor, coords[:,1].max()+zoom_factor
        ax.set_xlim(min_coord_x, max_coord_x)
        ax.set_ylim(min_coord_y, max_coord_y)

        # 加高德底图
        ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=gaode_tiles)

        ax.set_title(city)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()
        plt.show()

def fix_city_coords(city_coords):
    """
    将 city_coords 中所有 coord_x 和 coord_y 加 1，修复误减问题。
    """
    fixed_coords = {}
    for city, df in city_coords.items():
        df = df.copy()
        if 'coord_x' in df.columns and 'coord_y' in df.columns:
            df['coord_x'] = df['coord_x'] - 1.0417
            df['coord_y'] = df['coord_y'] - 1.0127
        fixed_coords[city] = df
    return fixed_coords


def plot_city_clusters_and_save_pdf(city_coords, k_dict=None, zoom_factor=0.01, save_dir="./results/map_figures"):
    """
    每个城市单独画聚类散点图，先将火星坐标(GCJ02)转换为WGS84坐标，
    使用高德底图绘制并保存为 PDF 矢量图。

    参数：
    city_coords : dict
        key: 城市名
        value: pd.DataFrame，只包含 'coord_x' 和 'coord_y'
    k_dict : dict, optional
        key: 城市名
        value: 聚类数 K，若为 None，则默认 3
    zoom_factor : float
        经纬度扩展范围，用于截取底图
    save_dir : str
        保存路径
    
    返回：
    None
    """
    os.makedirs(save_dir, exist_ok=True)

    for city, df in city_coords.items():
        # if str(city) != "5":  
        #     print(f"跳过城市 {city}，仅处理城市 5")
        #     continue
        if df.shape[0] == 0:
            print(f"跳过城市 {city}，数据为空 (0 行)")
            continue

        coords = df[['coord_x', 'coord_y']].dropna().values
        if coords.shape[0] == 0:
            print(f"跳过城市 {city}，经纬度全为空")
            continue

        # ✅ 坐标转换：GCJ02 -> WGS84
        coords = np.array([ct.gcj02_to_wgs84(lon, lat) for lon, lat in coords])

        K = k_dict.get(city, 3) if k_dict else 3

        # 聚类
        kmeans = KMeans(n_clusters=K, random_state=42).fit(coords)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {'label': labels},
            geometry=gpd.points_from_xy(coords[:,0], coords[:,1]),
            crs="EPSG:4326"
        )

        # 绘图
        fig, ax = plt.subplots(figsize=(8,8))
        for cluster_id in range(K):
            gdf[gdf['label'] == cluster_id].plot(ax=ax, markersize=20,
                                                 label=f'Cluster {cluster_id+1}', alpha=0.6)
            cluster_points = coords[labels == cluster_id]
            median_x, median_y = np.median(cluster_points, axis=0)
            ax.scatter(median_x, median_y, marker='X', color='k', s=100)  # 中位点

        ax.scatter(centers[:,0], centers[:,1], marker='*', color='red', s=150, label='Centroids')

        # 设置显示范围
        min_x, max_x = coords[:,0].min() - zoom_factor, coords[:,0].max() + zoom_factor
        min_y, max_y = coords[:,1].min() - zoom_factor, coords[:,1].max() + zoom_factor
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        # ✅ 使用 OSM 瓦片（WGS84 下对齐）
        ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

        ax.set_title(f"{city} 聚类图（WGS84坐标）", fontsize=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()

        # 保存为 PDF
        save_path = os.path.join(save_dir, f"{city}_clusters.pdf")
        fig.savefig(save_path, format="pdf", bbox_inches='tight', dpi=600)
        plt.close(fig)

        print(f"✅ {city} 聚类图已保存至 {save_path}")