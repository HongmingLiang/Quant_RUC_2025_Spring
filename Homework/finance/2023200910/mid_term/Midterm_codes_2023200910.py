#!/usr/bin/env python
# coding: utf-8

# 1.对数据预处理，调动大模型对客户反馈进行情感评分

# In[4]:


import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# In[ ]:


class SentimentAnalyzer:
    def __init__(self, model_path):
        """初始化情感分析器：加载模型和分词器（仅加载一次，节省时间）"""
        print("=" * 50)
        print("📦 正在初始化情感分析模型...")
        self.device = self._get_device()
        self.tokenizer = self._load_tokenizer(model_path)
        self.model = self._load_model(model_path)
        self.batch_size = 64 if self.device.type == "cuda" else 16  # 自动适配设备
        print(f"✅ 模型初始化完成！使用设备：{self.device} | 批量处理大小：{self.batch_size}")
        print("=" * 50 + "\n")

    def _get_device(self):
        """获取运行设备（GPU优先）"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("⚠️ 未检测到GPU，将使用CPU（处理速度可能较慢）")
            return torch.device("cpu")

    def _load_tokenizer(self, model_path):
        """加载分词器（含路径验证）"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在：{model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True
            )
            return tokenizer
        except Exception as e:
            raise RuntimeError(f"分词器加载失败：{e}")

    def _load_model(self, model_path):
        """加载情感分类模型（含异常处理）"""
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, local_files_only=True
            ).to(self.device)
            model.eval()  # 切换到评估模式
            return model
        except Exception as e:
            raise RuntimeError(f"模型加载失败：{e}")

    def _read_input_file(self, input_path):
        """读取输入CSV文件（支持多编码，处理空值）"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在：{input_path}")
        if not input_path.endswith(".csv"):
            raise ValueError("仅支持CSV格式文件，请检查文件后缀")

        # 尝试常见编码，解决中文乱码
        encodings = ["utf-8-sig", "gbk", "utf-8"]
        for encoding in encodings:
            try:
                df = pd.read_csv(input_path, encoding=encoding)
                print(f"✅ 成功读取文件（编码：{encoding}），共{len(df)}行数据")
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError("无法解析文件编码，请检查文件格式或尝试转换编码")

    def _validate_output_path(self, output_input):
        """验证并处理输出路径（自动补全CSV后缀，创建目录）"""
        # 处理用户输入：若只给目录，默认生成文件名；若给文件名，补全CSV后缀
        if os.path.isdir(output_input):
            # 输入是目录，默认文件名
            default_name = "情感分析结果_" + pd.Timestamp.now().strftime("%Y%m%d%H%M%S") + ".csv"
            output_path = os.path.join(output_input, default_name)
        else:
            # 输入是文件路径，补全CSV后缀
            output_path = output_input if output_input.endswith(".csv") else output_input + ".csv"

        # 创建输出目录（若不存在）
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✅ 自动创建输出目录：{output_dir}")

        # 检查文件是否已存在
        if os.path.exists(output_path):
            overwrite = input(f"⚠️ 文件 {output_path} 已存在，是否覆盖？（y/n）").strip().lower()
            if overwrite != "y":
                new_name = input("请输入新的文件名（无需后缀，默认CSV）：").strip()
                output_path = os.path.join(output_dir, new_name + ".csv")
        return output_path

    def _batch_analyze(self, texts):
        """批量情感分析（带进度条）"""
        all_probs = []
        total_batches = len(texts) // self.batch_size + (1 if len(texts) % self.batch_size != 0 else 0)

        print(f"\n🚀 开始情感分析（共{len(texts)}条文本，{total_batches}批）")
        with tqdm(total=total_batches, desc="分析进度") as pbar:
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i+self.batch_size]
                # 强制转为字符串，避免非文本数据报错
                batch_texts = [str(text) for text in batch_texts]

                # 分词并推理
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                all_probs.extend(probs.cpu().numpy())

                pbar.update(1)
        return all_probs

    def analyze_single_file(self):
        """单次文件分析流程：输入路径→验证→分析→保存"""
        # 1. 询问输入文件路径
        while True:
            input_path = input("\n请输入需要分析的CSV文件路径（示例：C:\\data\\input.csv）：").strip()
            try:
                df = self._read_input_file(input_path)
                break
            except Exception as e:
                print(f"❌ 输入错误：{e}，请重新输入")

        # 2. 确认待分析列（默认最后一列，支持用户选择）
        feedback_col = df.columns[-1]
        change_col = input(f"\n默认分析最后一列【{feedback_col}】，是否需要更换列？（y/n）").strip().lower()
        if change_col == "y":
            print(f"可选列名：{list(df.columns)}")
            while True:
                new_col = input("请输入要分析的列名：").strip()
                if new_col in df.columns:
                    feedback_col = new_col
                    break
                else:
                    print(f"❌ 列名【{new_col}】不存在，请重新输入")
        print(f"📝 确认分析列：【{feedback_col}】")

        # 3. 询问输出路径和文件名
        while True:
            output_input = input("\n请输入结果保存路径（示例：C:\\data\\result 或 C:\\data\\my_result.csv）：").strip()
            try:
                output_path = self._validate_output_path(output_input)
                break
            except Exception as e:
                print(f"❌ 输出路径错误：{e}，请重新输入")

        # 4. 提取文本并分析
        texts = df[feedback_col].fillna("").tolist()
        all_probs = self._batch_analyze(texts)

        # 5. 解析结果并保存
        df["情感_负面概率"] = [round(p[0], 3) for p in all_probs]
        df["情感_正面概率"] = [round(p[1], 3) for p in all_probs]
        df["情感标签"] = ["正面" if p[1] > 0.5 else "负面" for p in all_probs]
        df["情感得分"] = [round(p[1], 3) for p in all_probs]

        # 保存文件
        try:
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"\n💾 结果已保存至：{output_path}")

            # 显示情感分布统计
            label_count = df["情感标签"].value_counts()
            print(f"\n📊 情感分布统计：")
            for label, count in label_count.items():
                print(f"- {label}：{count}条（占比{count/len(df):.2%}）")
        except Exception as e:
            print(f"\n❌ 保存失败：{e}，请检查文件是否被占用")

    def run(self):
        """主程序入口：循环处理文件，支持继续/退出"""
        print("🎉 欢迎使用交互式情感分析程序！")
        while True:
            self.analyze_single_file()

            # 询问是否继续
            continue_flag = input("\n是否需要分析其他文件？（y/n）").strip().lower()
            if continue_flag != "y":
                print("\n👋 程序结束，感谢使用！")
                break


# --------------------------
# 程序启动入口（需配置模型路径）
# --------------------------
if __name__ == "__main__":
    # 配置你的本地模型路径（固定一次，无需每次输入）
    MODEL_PATH = "C:\\Users\\Administrator\\Desktop\\work\\lecture-python-programming.myst-main\\lectures\\model\\Erlangshen-RoBERTa-330M-Sentiment"

    try:
        # 初始化分析器并启动
        analyzer = SentimentAnalyzer(MODEL_PATH)
        analyzer.run()
    except Exception as e:
        print(f"\n❌ 程序初始化失败：{e}")
        print("请检查：1. 模型路径是否正确；2. 模型文件是否完整；3. 依赖库是否安装")


# 2.正式进行数据处理，加载库和设置日志方便排查bug

# In[5]:


import re
import warnings
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# -------------------------- 日志配置 --------------------------
import os
if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('predictions'):
    os.makedirs('predictions')

log_filename = f"logs/prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')



# 3.定义特征处理的函数

# In[6]:


# -------------------------- 工具函数 --------------------------
def extract_numeric(text, default=np.nan):
    if pd.isna(text) or str(text).strip() in ["", "无"]:
        return default
    numeric_match = re.search(r'(\d+\.?\d*)', str(text))
    return float(numeric_match.group(1)) if numeric_match else default


def extract_area(area_str):
    if pd.isna(area_str):
        return np.nan
    try:
        return float(str(area_str).split('㎡')[0])
    except:
        return extract_numeric(area_str)


def parse_room_info(room_str, is_price=True):
    room, hall, bath = pd.NA, pd.NA, pd.NA
    kitchen = pd.NA if is_price else None
    if pd.isna(room_str) or str(room_str).strip() in ['', '·']:
        result = [room, hall, bath]
        return result + [kitchen] if is_price else result
    s = str(room_str)
    room_match = re.search(r'(\d+)(室|房间|居室)', s)
    hall_match = re.search(r'(\d+)厅', s)
    bath_match = re.search(r'(\d+)卫', s)
    room = int(room_match.group(1)) if room_match else 0
    hall = int(hall_match.group(1)) if hall_match else 0
    bath = int(bath_match.group(1)) if bath_match else 0
    if is_price:
        kitchen_match = re.search(r'(\d+)厨', s)
        kitchen = int(kitchen_match.group(1)) if kitchen_match else 0
        return [room, hall, kitchen, bath]
    return [room, hall, bath]


def extract_floor(floor_str):
    if pd.isna(floor_str):
        return np.nan, np.nan
    try:
        floor_str = str(floor_str).strip()
        if '共' in floor_str:
            total_match = re.search(r'共(\d+)层', floor_str)
            total_num = int(total_match.group(1)) if total_match else np.nan
            if '底' in floor_str:
                current_num = 1
            elif '顶' in floor_str:
                current_num = total_num if not np.isnan(total_num) else np.nan
            elif '高' in floor_str and not np.isnan(total_num):
                current_num = int(total_num * 0.8)
            elif '中' in floor_str and not np.isnan(total_num):
                current_num = int(total_num * 0.5)
            elif '低' in floor_str and not np.isnan(total_num):
                current_num = int(total_num * 0.2)
            else:
                current_match = re.search(r'(\d+)', floor_str)
                current_num = int(current_match.group(1)) if current_match else np.nan
            return current_num, total_num
        elif '/' in floor_str and '层' in floor_str:
            parts = floor_str.split('/')
            current_num = int(parts[0]) if parts[0].isdigit() else np.nan
            total_num = int(parts[1].replace('层', '')) if parts[1].replace('层', '').isdigit() else np.nan
            return current_num, total_num
    except Exception as e:
        logger.warning(f"楼层提取失败: {floor_str}, 错误: {str(e)}")
        return np.nan, np.nan


def process_orientation(orientation_str):
    if pd.isna(orientation_str):
        return 0, 0, 0, 0
    return (
        1 if '东' in orientation_str else 0,
        1 if '南' in orientation_str else 0,
        1 if '西' in orientation_str else 0,
        1 if '北' in orientation_str else 0
    )


def process_decoration(decoration_str):
    if pd.isna(decoration_str):
        return 0, 0, 0, 0, 0
    return (
        1 if '毛坯' in decoration_str else 0,
        1 if '简装' in decoration_str else 0,
        1 if '中装' in decoration_str else 0,
        1 if '精装' in decoration_str else 0,
        1 if '豪装' in decoration_str else 0
    )


def relative_mae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mean_abs = np.mean(np.abs(y_true))
    if mean_abs == 0:
        logger.warning("目标变量均值为0，无法计算相对MAE")
        return 0
    return mae / mean_abs


# 房价配套设施和交通特征处理
def process_price_features(df):
    price_surrounding_categories = {
        '医药类': ['医院', '药店', '诊所', '中医院', '三甲', '药房'],
        '商超类': ['广场', '商圈', '万达', '华润', '永辉', '物美', '百货', '大润发', 
                  '商场', '商城', '沃尔玛', '家乐福', '盒马', '生鲜', '超市'],
        '教育类': ['幼儿园', '大学城', '图书馆'],
        '生活服务类': ['市场', '菜场', '便利店', '邮局', '邮政', '饭店', '餐馆', '美食街', '理发店'],
        '娱乐类': ['影院', '影城', 'ktv', '酒吧', '火锅', '麦当劳', '肯德基', '星巴克'],
        '休闲类': ['花园', '篮球场', '健身房', '游泳池', '体育馆', '博物馆', '樱花', '步行街'],
        '商业类': ['金融街', '商铺']
    }

    traffic_keywords = ['公交', '地铁', '共享单车']

    if '周边配套' in df.columns:
        facility_raw = df['周边配套'].astype(str)
        for cat_name, keywords in price_surrounding_categories.items():
            df[f'surrounding_{cat_name}'] = facility_raw.apply(
                lambda x: 1 if any(kw in x for kw in keywords) else 0
            )
        logger.info(f"房价周边配套特征处理完成（{len(price_surrounding_categories)}个大类）")
    else:
        logger.warning("房价数据中未找到'周边配套'列，跳过该特征处理")

    if '交通出行' in df.columns:
        traffic_raw = df['交通出行'].astype(str)
        for keyword in traffic_keywords:
            df[f'traffic_{keyword}'] = traffic_raw.apply(lambda x: 1 if keyword in x else 0)
        logger.info(f"房价交通特征处理完成（保留：{traffic_keywords}）")
    else:
        logger.warning("房价数据中未找到'交通出行'列，跳过该特征处理")

    return df


# 房租配套设施处理（生成虚拟变量）
def process_rent_facilities(df):
    rent_facility_keywords = [
        '家具', '家电', '床', '衣柜', '沙发', '桌子',  # 家具类
        '空调', '冰箱', '洗衣机', '电视', '热水器',     # 家电类
        '宽带', 'WiFi', '网络',                         # 网络类
        '暖气', '地暖', '壁挂炉', '空调制热',           # 供暖类
        '电梯', '停车位', '阳台', '独立卫生间'          # 其他便利设施
    ]

    if '配套设施' in df.columns:
        facility_raw = df['配套设施'].astype(str)
        for keyword in rent_facility_keywords:
            df[f'facility_{keyword}'] = facility_raw.apply(
                lambda x: 1 if keyword in x else 0
            )
        logger.info(f"房租配套设施处理完成（生成{len(rent_facility_keywords)}个虚拟变量）")
    else:
        logger.warning("房租数据中未找到'配套设施'列，跳过该特征处理")
    return df




# In[7]:


# -------------------------- 数据预处理（引入对数转换处理右偏价格） --------------------------
def preprocess_data(df, target_col, is_train=True, is_price=True, fill_values=None, log_params=None):
    try:
        df_clean = df.copy()
        logger.info(f"开始预处理{'训练' if is_train else '测试'}{'房价' if is_price else '房租'}数据，原始样本数: {len(df_clean)}")

        # 检查目标列是否存在且为正数（对数转换前提）
        if is_train and target_col in df_clean.columns:
            # 过滤价格<=0的异常值（无法进行对数转换）
            invalid_mask = (df_clean[target_col] <= 0)
            if invalid_mask.sum() > 0:
                logger.warning(f"发现{invalid_mask.sum()}条价格<=0的记录，已过滤（对数转换需正值）")
                df_clean = df_clean[~invalid_mask].copy()

            # 训练集：计算对数转换参数并处理异常值
            if log_params is None:
                log_params = {}
            # 对价格进行对数转换（+1避免log(0)，对结果影响可忽略）
            df_clean[f'{target_col}_log'] = np.log1p(df_clean[target_col])
            log_params['used_log'] = True
            logger.info(f"对目标变量{target_col}进行对数转换（log1p），缓解右偏分布")

            # 基于对数转换后的值处理异常值（核心修改：替代原始价格的IQR）
            Q1_log = df_clean[f'{target_col}_log'].quantile(0.25)
            Q3_log = df_clean[f'{target_col}_log'].quantile(0.75)
            IQR_log = Q3_log - Q1_log
            df_clean = df_clean[
                (df_clean[f'{target_col}_log'] >= Q1_log - 1.5*IQR_log) & 
                (df_clean[f'{target_col}_log'] <= Q3_log + 1.5*IQR_log)
            ]
            logger.info(f"基于对数转换值处理异常值后样本数: {len(df_clean)}")
            log_params['iqr_log'] = (Q1_log, Q3_log, IQR_log)  # 保存对数转换后的IQR参数

        # 处理情感得分
        sentiment_cols = [col for col in df_clean.columns if '情感得分' in col or 'sentiment' in col.lower()]
        if sentiment_cols:
            if len(sentiment_cols) > 1:
                df_clean['情感得分'] = df_clean[sentiment_cols].mean(axis=1)
                logger.info(f"合并多个情感得分列: {sentiment_cols} → 情感得分")
            else:
                df_clean = df_clean.rename(columns={sentiment_cols[0]: '情感得分'})
        else:
            logger.warning("未检测到情感得分列")

        # 面积特征
        area_cols = [col for col in df_clean.columns if '面积' in col]
        for col in area_cols:
            df_clean[f'{col}_数值'] = df_clean[col].apply(extract_area)
        logger.info(f"处理面积特征: {area_cols}")

        # 户型特征
        layout_col = '房屋户型' if is_price and '房屋户型' in df_clean.columns else '户型' if '户型' in df_clean.columns else None
        if layout_col:
            room_features = df_clean[layout_col].apply(lambda x: parse_room_info(x, is_price))
            if is_price:
                df_clean[['室', '厅', '厨', '卫']] = pd.DataFrame(room_features.tolist(), index=df_clean.index)
            else:
                df_clean[['室', '厅', '卫']] = pd.DataFrame(room_features.tolist(), index=df_clean.index)
            for col in ['室', '厅', '卫'] + (['厨'] if is_price else []):
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median() if is_train else 0)
            logger.info(f"处理户型特征: {layout_col}")
        else:
            logger.warning(f"{'房价' if is_price else '房租'}数据中未找到户型列，跳过该特征处理")

        # 楼层特征
        floor_col = '所在楼层' if '所在楼层' in df_clean.columns else '楼层' if '楼层' in df_clean.columns else None
        if floor_col:
            df_clean[['当前楼层', '总楼层']] = pd.DataFrame(
                df_clean[floor_col].apply(extract_floor).tolist(), index=df_clean.index
            )
            df_clean['楼层比例'] = df_clean.apply(
                lambda row: row['当前楼层'] / row['总楼层'] 
                if row['总楼层'] != 0 and not pd.isna(row['总楼层']) and not pd.isna(row['当前楼层']) 
                else np.nan, axis=1
            )
            logger.info(f"处理楼层特征: {floor_col}")
        else:
            logger.warning(f"{'房价' if is_price else '房租'}数据中未找到楼层列，跳过该特征处理")

        # 朝向特征（区分列名）
        orient_col = '房屋朝向' if is_price else '朝向'
        if orient_col in df_clean.columns:
            df_clean[['朝东', '朝南', '朝西', '朝北']] = pd.DataFrame(
                df_clean[orient_col].apply(process_orientation).tolist(), index=df_clean.index
            )
            logger.info(f"处理{orient_col}特征")
        else:
            logger.warning(f"未找到'{orient_col}'列，跳过朝向处理")

        # 装修特征（区分列名）
        deco_col = '装修情况' if is_price else '装修'
        if deco_col in df_clean.columns:
            df_clean[['毛坯', '简装', '中装', '精装', '豪装']] = pd.DataFrame(
                df_clean[deco_col].apply(process_decoration).tolist(), index=df_clean.index
            )
            logger.info(f"处理{deco_col}特征")
        else:
            logger.warning(f"未找到'{deco_col}'列，跳过装修处理")

        # 特有特征处理
        if is_price:
            df_clean = process_price_features(df_clean)
        else:
            df_clean = process_rent_facilities(df_clean)

        # 提取数值列（缺失值填充严格区分训练/测试集）
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if '情感得分' in df_clean.columns and '情感得分' not in numeric_cols:
            try:
                df_clean['情感得分'] = pd.to_numeric(df_clean['情感得分'])
                numeric_cols.append('情感得分')
                logger.info("情感得分转换为数值型")
            except:
                logger.warning("情感得分无法转换为数值型，已排除")

        # 训练集：计算填充值
        if is_train:
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            if f'{target_col}_log' in numeric_cols:
                numeric_cols.remove(f'{target_col}_log')  # 排除对数转换列
            fill_values = {}
            for col in numeric_cols:
                if col == '情感得分':
                    fill_val = df_clean[col].mean() if not np.isnan(df_clean[col].mean()) else 0
                else:
                    fill_val = df_clean[col].median() if not np.isnan(df_clean[col].median()) else 0
                fill_values[col] = fill_val
                df_clean[col].fillna(fill_val, inplace=True)
            logger.info(f"训练集计算填充值完成（{len(fill_values)}个特征）")
        # 测试集：使用训练集填充值
        else:
            if fill_values is None:
                raise ValueError("测试集预处理必须传入训练集的fill_values")
            for col in numeric_cols:
                if col in fill_values:
                    df_clean[col].fillna(fill_values[col], inplace=True)
                else:
                    df_clean[col].fillna(0, inplace=True)
            logger.info(f"测试集使用训练集填充值完成（{len(numeric_cols)}个特征）")

        # 最终缺失值兜底
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)

        # 提取ID
        if 'ID' not in df_clean.columns:
            logger.warning("未找到ID列，自动生成ID")
            ids = pd.Series(range(len(df_clean)), name='ID')
        else:
            ids = df_clean['ID']

        # 分离特征和目标（训练集返回对数转换后的目标）
        if is_train:
            X = df_clean.select_dtypes(include=['int64', 'float64']).drop(
                columns=[target_col, f'{target_col}_log'], errors='ignore'
            )
            y = df_clean[f'{target_col}_log'] if f'{target_col}_log' in df_clean.columns else df_clean[target_col]
            logger.info(f"\n===== 训练集处理后特征（共{len(X.columns)}个） =====")
            logger.info(f"特征列表: {list(X.columns)}")
            print(f"\n===== 训练集处理后特征 =====")
            print(list(X.columns))
            return X, y, ids, fill_values, log_params  # 新增返回log参数
        else:
            X = df_clean.select_dtypes(include=['int64', 'float64'])
            logger.info(f"\n===== 测试集处理后特征（共{len(X.columns)}个） =====")
            logger.info(f"特征列表: {list(X.columns)}")
            print(f"\n===== 测试集处理后特征 =====")
            print(list(X.columns))
            return X, None, ids

    except Exception as e:
        logger.error(f"预处理失败: {str(e)}", exc_info=True)
        raise




# In[8]:


# -------------------------- 模型训练与预测（适配对数转换） --------------------------
def predict_and_evaluate(train_path, test_path, target_name, target_col, is_price=True):
    try:
        logger.info(f"\n===== 开始预测{target_name} =====")

        # 读取数据
        def read_csv(path):
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin-1', 'iso-8859-1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(path, encoding=encoding)
                    logger.info(f"成功读取{path}，编码: {encoding}，样本数: {len(df)}")
                    logger.info(f"数据列名: {df.columns.tolist()}")
                    return df
                except:
                    continue
            raise Exception(f"无法读取文件: {path}（已尝试编码: {encodings}）")

        train_df = read_csv(train_path)
        test_df = read_csv(test_path)

        # 预处理（训练集获取填充值和对数参数）
        X_train, y_train_log, _, fill_values, log_params = preprocess_data(
            train_df, target_col, is_train=True, is_price=is_price
        )
        X_test, _, test_ids = preprocess_data(
            test_df, target_col, is_train=False, is_price=is_price, fill_values=fill_values
        )

        # 特征对齐
        common_feats = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_feats]
        X_test = X_test[common_feats]
        logger.info(f"\n===== 共同特征（共{len(common_feats)}个） =====")
        logger.info(f"共同特征列表: {list(common_feats)}")
        print(f"\n===== 共同特征 =====")
        print(list(common_feats))

        # 模型定义
        models = {
            'OLS': LinearRegression(),
            '套索': Lasso(alpha=0.1, random_state=42),
            '岭回归': Ridge(alpha=1.0, random_state=42),
            'XGBoost': xgb.XGBRegressor(
                objective='reg:squarederror', n_estimators=100, 
                learning_rate=0.1, max_depth=5, random_state=42, verbosity=0
            )
        }

        # 全局特征选择和标准化
        k = min(30, len(common_feats))
        global_selector = SelectKBest(f_regression, k=k)
        X_train_selected = global_selector.fit_transform(X_train, y_train_log)
        X_test_selected = global_selector.transform(X_test)
        selected_feats = [common_feats[i] for i in range(len(common_feats)) if global_selector.get_support()[i]]
        logger.info(f"\n===== 全局选择特征（共{len(selected_feats)}个） =====")
        logger.info(f"特征列表: {selected_feats}")
        print(f"\n===== 全局选择特征 =====")
        print(selected_feats)

        global_scaler = StandardScaler()
        X_train_scaled = global_scaler.fit_transform(X_train_selected)
        X_test_scaled = global_scaler.transform(X_test_selected)

        # 分割验证集（基于对数转换后的目标）
        X_tr, X_val, y_tr_log, y_val_log = train_test_split(
            X_train_scaled, y_train_log, test_size=0.2, random_state=42
        )
        # 还原验证集原始价格（用于评估）
        y_val_original = np.expm1(y_val_log)  # expm1 = exp(x) - 1，对应log1p的逆操作

        # 训练模型并评估
        metrics = {name: {'样本内': None, '样本外': None, '6倍交叉验证': None} for name in models}
        predictions = {}

        for name, model in models.items():
            logger.info(f"\n----- 训练{name}模型 -----")
            # 初步训练（基于对数目标）
            model.fit(X_tr, y_tr_log)
            # 样本内评估（还原为原始价格）
            y_tr_pred_log = model.predict(X_tr)
            y_tr_pred_original = np.expm1(y_tr_pred_log)
            y_tr_original = np.expm1(y_tr_log)
            metrics[name]['样本内'] = round(relative_mae(y_tr_original, y_tr_pred_original), 4)
            # 样本外评估（还原为原始价格）
            y_val_pred_log = model.predict(X_val)
            y_val_pred_original = np.expm1(y_val_pred_log)
            metrics[name]['样本外'] = round(relative_mae(y_val_original, y_val_pred_original), 4)

            # 交叉验证（基于对数目标，评估时还原）
            kf = KFold(n_splits=6, shuffle=True, random_state=42)
            cv_rmae = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                logger.info(f"交叉验证fold {fold+1}/6")
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train_log = y_train_log.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val_log = y_train_log.iloc[val_idx]
                y_fold_val_original = np.expm1(y_fold_val_log)  # 还原真实值

                # 特征选择和标准化（fold内）
                fold_selector = SelectKBest(f_regression, k=k)
                X_fold_train_selected = fold_selector.fit_transform(X_fold_train, y_fold_train_log)
                X_fold_val_selected = fold_selector.transform(X_fold_val)
                fold_scaler = StandardScaler()
                X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train_selected)
                X_fold_val_scaled = fold_scaler.transform(X_fold_val_selected)

                # 训练和预测（对数空间）
                model.fit(X_fold_train_scaled, y_fold_train_log)
                y_fold_pred_log = model.predict(X_fold_val_scaled)
                y_fold_pred_original = np.expm1(y_fold_pred_log)  # 还原预测值

                cv_rmae.append(relative_mae(y_fold_val_original, y_fold_pred_original))

            metrics[name]['6倍交叉验证'] = round(np.mean(cv_rmae), 4)
            logger.info(f"{name}指标: 样本内{metrics[name]['样本内']}, 样本外{metrics[name]['样本外']}, 交叉验证{metrics[name]['6倍交叉验证']}")

            # 测试集预测（对数空间→还原为原始价格）
            model.fit(X_train_scaled, y_train_log)
            test_pred_log = model.predict(X_test_scaled)
            test_pred_original = np.expm1(test_pred_log)  # 核心：将对数预测转换回原始价格
            test_pred_original = np.maximum(test_pred_original, 0)  # 确保非负
            predictions[name] = pd.DataFrame({
                'ID': test_ids,
                'price': test_pred_original
            })
            logger.info(f"{name}测试集预测完成（已还原为原始价格尺度）")

        metrics_df = pd.DataFrame(metrics).T
        logger.info(f"\n{target_name}模型指标汇总:\n{metrics_df}")
        return metrics_df, predictions

    except Exception as e:
        logger.error(f"{target_name}处理失败: {str(e)}", exc_info=True)
        raise




# In[9]:


# -------------------------- 主程序 --------------------------
def main():
    try:
        logger.info("===== 程序启动 =====")
        config = {
            '房价': {
                'train_path': 'train_price.csv',
                'test_path': 'test_price.csv',
                'target_col': 'Price',
                'target_name': '房价',
                'is_price': True
            },
            '房租': {
                'train_path': 'train_rent.csv',
                'test_path': 'test_rent.csv',
                'target_col': 'Price',
                'target_name': '房租',
                'is_price': False
            }
        }

        # 处理房价和房租
        print("\n===== 开始处理房价数据 =====")
        price_metrics, price_preds = predict_and_evaluate(**config['房价'])
        print("\n===== 开始处理房租数据 =====")
        rent_metrics, rent_preds = predict_and_evaluate(** config['房租'])

        # 合并同模型预测结果
        if price_preds and rent_preds:
            for model_name in price_preds.keys():
                if model_name in rent_preds:
                    merged_df = pd.concat([price_preds[model_name], rent_preds[model_name]], ignore_index=True)
                    save_path = f"predictions/{model_name}_合并预测.csv"
                    merged_df.to_csv(save_path, index=False, encoding='utf-8-sig')
                    logger.info(f"{model_name}合并预测保存: {save_path}（共{len(merged_df)}条）")
                    print(f"{model_name}合并预测保存: {save_path}")

        # 保存指标
        price_metrics.to_csv('房价模型指标.csv', encoding='utf-8-sig')
        rent_metrics.to_csv('房租模型指标.csv', encoding='utf-8-sig')
        print("\n===== 房价模型RMAE指标 =====")
        print(price_metrics)
        print("\n===== 房租模型RMAE指标 =====")
        print(rent_metrics)
        logger.info("所有指标保存完成")

    except Exception as e:
        logger.critical(f"程序运行失败: {str(e)}", exc_info=True)
        print(f"错误: {str(e)}, 日志请查看: {log_filename}")


if __name__ == "__main__":
    main()


# In[ ]:




