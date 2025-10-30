from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import json
import re
import joblib
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from tqdm import tqdm  
tqdm.pandas()

class PriceDataProcessor:
    """房价数据处理"""
    
    def __init__(self, target='Price', columns_json='./data/data_info/price_info.json'):
        self.target = target
        self.train_data = None
        self.test_data = None
        # 加载字段信息
        with open(columns_json, 'r', encoding='utf-8') as f:
            self.columns_info = json.load(f)

        # 加载BERT情感分析模型和分词器
        self.model = AutoModelForSequenceClassification.from_pretrained("jackietung/bert-base-chinese-sentiment-finetuned")
        self.tokenizer = AutoTokenizer.from_pretrained("jackietung/bert-base-chinese-sentiment-finetuned")
        # 加载 KMeans 聚类模型
        self.kmeans = joblib.load('./src/model/kmeans_model.pkl')
    
    def load_raw_data(self, train_path='./data/raw_data/ruc_Class25Q2_train_price.csv', test_path='./data/raw_data/ruc_Class25Q2_test_price.csv'):
        with open(train_path, 'r', encoding='utf-8') as f:
            self.train_data = pd.read_csv(f, low_memory=False)
        with open(test_path, 'r', encoding='utf-8') as f:
            self.test_data = pd.read_csv(f, low_memory=False)
        return self.train_data, self.test_data
    
    def load_processed_data(self, train_path='./data/processed/processed_train_price.csv', test_path='./data/processed/processed_test_price.csv'):
        with open(train_path, 'r', encoding='utf-8') as f:
            self.train_data = pd.read_csv(f, low_memory=False)
        with open(test_path, 'r', encoding='utf-8') as f:
            self.test_data = pd.read_csv(f, low_memory=False)
        return self.train_data, self.test_data

    def fill_missing_values(self):
        def fill_inner_area(df):
            df = df.copy()

            # 计算每个 cluster_label 的平均 建筑面积/套内面积 比例
            df['area_ratio'] = df['建筑面积'] / df['套内面积'].replace(0, np.nan)
            ratio_map = (
                df.replace([np.inf, -np.inf], np.nan)
                .dropna(subset=['area_ratio'])
                .groupby('cluster_label')['area_ratio']
                .mean()
                .to_dict()
            )

            # 按比例填充缺失的套内面积
            def fill_row(row):
                if pd.isna(row['套内面积']) or row['套内面积'] == 0:
                    ratio = ratio_map.get(row['cluster_label'])
                    if ratio and ratio > 0:
                        return row['建筑面积'] / ratio
                return row['套内面积']


            df['套内面积'] = df.apply(fill_row, axis=1)
            df.drop(columns=['area_ratio'], inplace=True)
            return df

        # 按比例填充套内面积
        self.train_data = fill_inner_area(self.train_data)
        self.test_data = fill_inner_area(self.test_data)

        # 其他缺失值使用前向 + 后向填充
        self.train_data = self.train_data.ffill().bfill()
        self.test_data = self.test_data.ffill().bfill()


    def extract_rooms(self, house_type):
        """提取房屋户型中的室、厅、厨、卫数量"""

        # 如果 house_type 为 NaN 或为空字符串，返回默认值
        if pd.isna(house_type) or house_type == '':
            return {'室': 0, '厅': 0, '厨': 0, '卫': 0}  # 默认返回零
        # 先检查是否包含 "房间"，如果有，按默认值处理
        if "房间" in house_type:
            house_type = "1室1厅1厨1卫"  # 默认填充为 1室1厅1厨1卫

        # 用正则表达式提取数字
        room_info = re.findall(r'(\d+)室|(\d+)厅|(\d+)厨|(\d+)卫', house_type)
        
        # 返回提取的室、厅、厨、卫数量
        room_dict = {
            '室': 0, '厅': 0, '厨': 0, '卫': 0
        }
        for match in room_info:
            if match[0]: room_dict['室'] = int(match[0])
            if match[1]: room_dict['厅'] = int(match[1])
            if match[2]: room_dict['厨'] = int(match[2])
            if match[3]: room_dict['卫'] = int(match[3])
        
        return room_dict

    def calculate_orientation_score(self, orientation):
        """计算房屋朝向得分"""
        score = 0
        if pd.isna(orientation) or orientation == '':
            return 0

        # 检查单字朝向
        single_directions = ['东', '南', '西', '北']
        for dir in single_directions:
            if dir in orientation:
                score += 1

        # 检查双字朝向
        double_directions = ['东南', '西南', '东北', '西北', '东南西', '南北', '东南北', '南北东']
        for dir in double_directions:
            if dir in orientation:
                score += 0.5

        # 检查是否包含“南北”，再加 2 分
        if '南 北' in orientation:
            score += 2

        return score

    def add_features(self):
        # 处理房屋朝向信息
        self.train_data['朝向得分'] = self.train_data['房屋朝向'].apply(self.calculate_orientation_score)
        self.test_data['朝向得分'] = self.test_data['房屋朝向'].apply(self.calculate_orientation_score)
        # # 批量处理提取情感得分
        # self.train_data['客户反馈得分'] = self.batch_get_sentiment_score(self.train_data['客户反馈'])
        # self.test_data['客户反馈得分'] = self.batch_get_sentiment_score(self.test_data['客户反馈'])
        # 针对位置信息 使用 KNN 模型提取
        self.train_data['cluster_label'] = self.kmeans.predict(self.train_data[['lon', 'lat']])
        self.test_data['cluster_label'] = self.kmeans.predict(self.test_data[['lon', 'lat']])
        # 处理年份信息
        for df in [self.train_data, self.test_data]:
            df['交易时间_dt'] = pd.to_datetime(df['交易时间'], errors='coerce')
            df['上次交易_dt'] = pd.to_datetime(df['上次交易'], errors='coerce')

            # 计算天数差
            df['交易间隔_天'] = (df['交易时间_dt'] - df['上次交易_dt']).dt.days
            # 删除临时列（可选）
            df.drop(columns=['交易时间_dt', '上次交易_dt'], inplace=True)
        # 处理房屋房型信息
        for df_name in ['train_data', 'test_data']:
            df = getattr(self, df_name) 
            # 提取 "房屋户型" 中的室、厅、厨、卫数量
            room_features = df['房屋户型'].apply(self.extract_rooms)
            room_df = pd.DataFrame(room_features.tolist())
            updated_df = pd.concat([df, room_df], axis=1)
            setattr(self, df_name, updated_df)

        

    def batch_get_sentiment_score(self, texts, batch_size=128):
        """ 使用BERT模型对文本进行批量情感分析，返回情感得分 """
        # 初始化进度条
        total_batches = (len(texts) + batch_size - 1) // batch_size  # 计算总的批次数
        sentiment_scores = []
        
        # 批量处理
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Sentiment", total=total_batches, unit="batch"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts.tolist(), return_tensors="pt", padding=True, truncation=True, max_length=512)

            # 禁用梯度计算
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # 获取每个文本的正面情感得分并添加到列表中
            sentiment_scores.extend(predictions[:, 1].tolist())  # 获取正面情感的得分
        
        return sentiment_scores

    # 删除标注 ignore 的列
    def drop_ignored_columns(self):
        ignore_cols = [col for col, col_type in self.columns_info.items() if col_type == 'ignore']
        ignore_cols.append('lon')  # 删除经度列
        ignore_cols.append('lat')  # 删除纬度列
        self.train_data = self.train_data.drop(columns=ignore_cols, errors='ignore')
        self.test_data = self.test_data.drop(columns=ignore_cols, errors='ignore')
        print(f"删除 ignore 的列:{ignore_cols}")

    # 处理文本类信息
    def process_text(self):
        text_cols = [col for col, col_type in self.columns_info.items() if col_type == 'Text']
        self.train_data = self.train_data.drop(columns=text_cols, errors='ignore')
        self.test_data = self.test_data.drop(columns=text_cols, errors='ignore')
        print(f"删除文本类列:{text_cols}")
    
    # 处理数值类信息
    def process_numeric(self):
        num_cols = [col for col, col_type in self.columns_info.items() if col_type == 'Numeric']
        print(f"处理数值类列:{num_cols}")

        for col in num_cols:
            # 强制转换为字符串
            self.train_data[col] = self.train_data[col].astype(str)
            self.test_data[col] = self.test_data[col].astype(str)

            # 提取浮点数，如果无法提取则填0.0
            self.train_data[col] = self.train_data[col].apply(
                lambda x: float(re.search(r'\d+(\.\d+)?', x).group()) if re.search(r'\d+(\.\d+)?', x) else 0.0
            )
            self.test_data[col] = self.test_data[col].apply(
                lambda x: float(re.search(r'\d+(\.\d+)?', x).group()) if re.search(r'\d+(\.\d+)?', x) else 0.0
            )
    
    # 处理时间类信息
    def process_time(self):
        text_cols = [col for col, col_type in self.columns_info.items() if col_type == 'Time']
        self.train_data = self.train_data.drop(columns=text_cols, errors='ignore')
        self.test_data = self.test_data.drop(columns=text_cols, errors='ignore')
        print(f"删除时间类列:{text_cols}")

    # 处理独热编码类信息
    def process_categorical(self):
        """对类别型特征进行独热编码"""
        # 找出类别型列
        cat_cols = [col for col, col_type in self.columns_info.items() if col_type == 'Categorical']
        cat_cols.append('cluster_label')  # 添加聚类标签列
        print(f"处理类别型列:{cat_cols}")

        # 初始化 OneHotEncoder
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # 训练集独热编码
        cat_train = self.ohe.fit_transform(self.train_data[cat_cols])
        cat_train_df = pd.DataFrame(cat_train, columns=self.ohe.get_feature_names_out(cat_cols), index=self.train_data.index)

        # 测试集独热编码（使用训练集的编码器）
        cat_test = self.ohe.transform(self.test_data[cat_cols])
        cat_test_df = pd.DataFrame(cat_test, columns=self.ohe.get_feature_names_out(cat_cols), index=self.test_data.index)

        # 删除原来的类别列
        self.train_data = self.train_data.drop(columns=cat_cols)
        self.test_data = self.test_data.drop(columns=cat_cols)

        # 拼接独热编码后的列
        self.train_data = pd.concat([self.train_data, cat_train_df], axis=1)
        self.test_data = pd.concat([self.test_data, cat_test_df], axis=1)

    def prepare_data(self): 
        self.process_numeric()      # 处理数值类信息
        self.add_features()        # 添加特征
        self.fill_missing_values() # 填充缺失值
        print("处理标注为 ignore 的列...")
        self.drop_ignored_columns() # 删除标注 ignore 的列
        print("处理文本类信息...")
        self.process_text()         # 处理文本类信息
        print("处理时间类信息...")
        self.process_time()         # 处理时间类信息
        print("处理类别型信息...")
        self.process_categorical()  # 处理独热编码类信息

        return self.train_data, self.test_data
    
    def split_data(self, test_size=0.2, random_state=111):
        X = self.train_data.drop(columns=[self.target], errors='ignore')
        y = self.train_data[self.target]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        return X_train_scaled, X_val_scaled, y_train.values, y_val.values
    
    def save_data(self, train_path='./data/processed/processed_train_price.csv', test_path='./data/processed/processed_test_price.csv'):
        self.train_data.to_csv(train_path, index=False)
        self.test_data.to_csv(test_path, index=False)
        print(f"Processed data saved to {train_path} and {test_path}")

class RentDataProcessor:
    """租房数据处理"""

    def __init__(self, target='Price', columns_json='./data/data_info/rent_info.json'):
        self.target = target
        self.train_data = None
        self.test_data = None
        # 加载字段信息
        with open(columns_json, 'r', encoding='utf-8') as f:
            self.columns_info = json.load(f)
        # 可选：如果租房数据也使用聚类
        self.kmeans = joblib.load('./src/model/kmeans_model.pkl')
        # 加载BERT情感分析模型和分词器
        self.model = AutoModelForSequenceClassification.from_pretrained("jackietung/bert-base-chinese-sentiment-finetuned")
        self.tokenizer = AutoTokenizer.from_pretrained("jackietung/bert-base-chinese-sentiment-finetuned")

    def load_raw_data(self, train_path='./data/raw_data/ruc_Class25Q2_train_rent.csv',
                      test_path='./data/raw_data/ruc_Class25Q2_test_rent.csv'):
        self.train_data = pd.read_csv(train_path, low_memory=False)
        self.test_data = pd.read_csv(test_path, low_memory=False)
        return self.train_data, self.test_data
    
    def load_processed_data(self, train_path='./data/processed/processed_train_rent.csv',
                            test_path='./data/processed/processed_test_rent.csv'):
        self.train_data = pd.read_csv(train_path, low_memory=False)
        self.test_data = pd.read_csv(test_path, low_memory=False)
        return self.train_data, self.test_data

    def batch_get_sentiment_score(self, texts, batch_size=128):
        """ 使用BERT模型对文本进行批量情感分析，返回情感得分 """
        # 初始化进度条
        total_batches = (len(texts) + batch_size - 1) // batch_size  # 计算总的批次数
        sentiment_scores = []
        
        # 批量处理
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Sentiment", total=total_batches, unit="batch"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts.tolist(), return_tensors="pt", padding=True, truncation=True, max_length=512)

            # 禁用梯度计算
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # 获取每个文本的正面情感得分并添加到列表中
            sentiment_scores.extend(predictions[:, 1].tolist())  # 获取正面情感的得分
        
        return sentiment_scores

    def fill_missing_values(self):
        """填充缺失值，可以按前向+后向填充"""
        self.train_data = self.train_data.ffill().bfill()
        self.test_data = self.test_data.ffill().bfill()
        print("Filled missing values using forward and backward fill.")

    def add_features(self):
        # # # 批量处理提取情感得分
        self.train_data['客户反馈得分'] = self.batch_get_sentiment_score(self.train_data['客户反馈'])
        self.test_data['客户反馈得分'] = self.batch_get_sentiment_score(self.test_data['客户反馈'])

        # 如果有聚类模型，可提取 cluster_label
        self.train_data['cluster_label'] = self.kmeans.predict(self.train_data[['lon', 'lat']])
        self.test_data['cluster_label'] = self.kmeans.predict(self.test_data[['lon', 'lat']])
        
        # 如果有交易时间 / 上次交易
        for df in [self.train_data, self.test_data]:
            if '交易时间' in df.columns and '上次交易' in df.columns:
                df['交易时间_dt'] = pd.to_datetime(df['交易时间'], errors='coerce')
                df['上次交易_dt'] = pd.to_datetime(df['上次交易'], errors='coerce')
                df['交易间隔_天'] = (df['交易时间_dt'] - df['上次交易_dt']).dt.days
                df.drop(columns=['交易时间_dt', '上次交易_dt'], inplace=True)

    def process_numeric(self):
        num_cols = [col for col, col_type in self.columns_info.items() if col_type == 'Numeric']
        print(f"处理数值类列:{num_cols}")

        for col in num_cols:
            self.train_data[col] = self.train_data[col].astype(str)
            self.test_data[col] = self.test_data[col].astype(str)

            self.train_data[col] = self.train_data[col].apply(
                lambda x: float(re.search(r'\d+(\.\d+)?', x).group()) if re.search(r'\d+(\.\d+)?', x) else 0.0
            )
            self.test_data[col] = self.test_data[col].apply(
                lambda x: float(re.search(r'\d+(\.\d+)?', x).group()) if re.search(r'\d+(\.\d+)?', x) else 0.0
            )

            mean_value = self.train_data[col].mean()
            self.train_data[col] = self.train_data[col].fillna(mean_value)
            self.test_data[col] = self.test_data[col].fillna(mean_value)

    def process_text(self):
        text_cols = [col for col, col_type in self.columns_info.items() if col_type == 'Text']
        self.train_data = self.train_data.drop(columns=text_cols, errors='ignore')
        self.test_data = self.test_data.drop(columns=text_cols, errors='ignore')
        print(f"删除文本类列:{text_cols}")

    def process_time(self):
        time_cols = [col for col, col_type in self.columns_info.items() if col_type == 'Time']
        self.train_data = self.train_data.drop(columns=time_cols, errors='ignore')
        self.test_data = self.test_data.drop(columns=time_cols, errors='ignore')
        print(f"删除时间类列:{time_cols}")

    # 删除标注 ignore 的列
    def drop_ignored_columns(self):
        ignore_cols = [col for col, col_type in self.columns_info.items() if col_type == 'ignore']
        ignore_cols.append('lon')  # 删除经度列
        ignore_cols.append('lat')  # 删除纬度列
        self.train_data = self.train_data.drop(columns=ignore_cols, errors='ignore')
        self.test_data = self.test_data.drop(columns=ignore_cols, errors='ignore')
        print(f"删除 ignore 的列:{ignore_cols}")
        return self.train_data, self.test_data

    def process_categorical(self):
        cat_cols = [col for col, col_type in self.columns_info.items() if col_type == 'Categorical']
        cat_cols.append('cluster_label')  # 添加聚类标签列
        print(f"处理类别型列:{cat_cols}")

        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_train = self.ohe.fit_transform(self.train_data[cat_cols])
        cat_test = self.ohe.transform(self.test_data[cat_cols])

        cat_train_df = pd.DataFrame(cat_train, columns=self.ohe.get_feature_names_out(cat_cols), index=self.train_data.index)
        cat_test_df = pd.DataFrame(cat_test, columns=self.ohe.get_feature_names_out(cat_cols), index=self.test_data.index)

        self.train_data = self.train_data.drop(columns=cat_cols)
        self.test_data = self.test_data.drop(columns=cat_cols)

        self.train_data = pd.concat([self.train_data, cat_train_df], axis=1)
        self.test_data = pd.concat([self.test_data, cat_test_df], axis=1)

    def prepare_data(self): 
        self.process_numeric()      # 处理数值类信息
        self.add_features()        # 添加特征
        self.fill_missing_values() # 填充缺失值
        print("处理标注为 ignore 的列...")
        self.drop_ignored_columns() # 删除标注 ignore 的列
        print("处理文本类信息...")
        self.process_text()         # 处理文本类信息
        print("处理时间类信息...")
        self.process_time()         # 处理时间类信息
        print("处理类别型信息...")
        self.process_categorical()  # 处理独热编码类信息
        return self.train_data, self.test_data

    def split_data(self, test_size=0.2, random_state=111):
        X = self.train_data.drop(columns=[self.target], errors='ignore')
        y = self.train_data[self.target]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        return X_train_scaled, X_val_scaled, y_train.values, y_val.values

    def save_data(self, train_path='./data/processed/processed_train_rent.csv',
                  test_path='./data/processed/processed_test_rent.csv'):
        self.train_data.to_csv(train_path, index=False)
        self.test_data.to_csv(test_path, index=False)
        print(f"Processed data saved to {train_path} and {test_path}")
