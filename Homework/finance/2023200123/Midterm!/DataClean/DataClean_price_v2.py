"""
房价数据清洗脚本
基于租金数据清洗框架重构，保留房价数据特有的处理逻辑
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
import warnings

warnings.filterwarnings('ignore')

# 设置pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 50)


# ========= 1) 专用解析函数（先于通用清洗执行） =========

def extract_floor_info_v3(s):
    """
    解析楼层信息：'中楼层 (共23层)' / '15层(共23层)' / '高楼层(共33层)' 等
    返回 (当前楼层估计, 总楼层)
    规则：低=0.2*总；中=0.5*总；高=0.8*总；顶=总；底=1；地下=-1。
    """
    if pd.isna(s):
        return (np.nan, np.nan)
    s = str(s).strip()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", "", s)
    total_match = re.search(r"共(\d+)层", s)
    total_floor = float(total_match.group(1)) if total_match else np.nan
    current = np.nan
    if "低" in s and not np.isnan(total_floor): current = total_floor * 0.2
    elif "中" in s and not np.isnan(total_floor): current = total_floor * 0.5
    elif "高" in s and not np.isnan(total_floor): current = total_floor * 0.8
    elif "顶" in s and not np.isnan(total_floor): current = total_floor
    elif "底" in s: current = 1.0
    elif "地下" in s: current = -1.0
    if np.isnan(current):
        cur_match = re.search(r"(\d+)层", s)
        if cur_match: current = float(cur_match.group(1))
        else:
            nums = re.findall(r"\d+", s)
            if nums and "共" not in s: current = float(nums[0])
    current = float(np.round(current, 2)) if not np.isnan(current) else np.nan
    total_floor = float(total_floor) if not np.isnan(total_floor) else np.nan
    return (current, total_floor)

def parse_ladder_ratio(s):
    """ 解析梯户比例：'一梯三户' -> 1/3；'2梯4户' -> 0.5 """
    if pd.isna(s): return np.nan
    s = str(s)
    cn2num = {'一':1, '二':2, '两':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9, '十':10}
    digits = [cn2num.get(ch) for ch in s if ch in cn2num]
    if len(digits) >= 2 and digits[1] != 0: return digits[0] / digits[1]
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2 and float(nums[1]) != 0: return float(nums[0]) / float(nums[1])
    return np.nan

def parse_house_type(s):
    """ 解析户型：'3室2厅1厨2卫' -> (3, 2, 1, 2) """
    if pd.isna(s): return (np.nan, np.nan, np.nan, np.nan)
    s = str(s)
    get_first = lambda pat: float(re.findall(pat, s)[0]) if re.findall(pat, s) else np.nan
    return (get_first(r"(\d+)室"), get_first(r"(\d+)厅"), get_first(r"(\d+)厨"), get_first(r"(\d+)卫"))

def extract_numeric_or_range(val):
    """ 提取数值或区间的平均值 (%, -, 暂无, 单位) """
    if pd.isna(val): return np.nan
    s = str(val).replace(",", "").strip()
    if s == "": return np.nan
    if "%" in s:
        nums = re.findall(r"\d+\.?\d*", s)
        return float(nums[0])/100 if nums else np.nan
    if '暂无' in s: return 0.0
    nums = re.findall(r"\d+\.?\d*", s)
    if not nums: return np.nan
    nums = [float(x) for x in nums]
    if "-" in s and len(nums) == 2: return float(np.mean(nums))
    return float(np.mean(nums))

# 修正：增强 split_labels 以处理更多分隔符（包括空格）
def split_labels(x):
    """ 分割多标签字符串 (处理 / , ， 、 空格) """
    if pd.isna(x): return []
    # 将所有常见分隔符统一替换为 /
    x = str(x).replace("、", "/").replace(" ", "/").replace(",", "/").replace("，", "/")
    # 按 / 分割并去除空字符串
    return [t.strip() for t in x.split('/') if t.strip()]


# ========= 2) 应用专用解析函数的函数 =========
def apply_special_text_parsers(df):
    """ 对特殊文本列进行解析 """
    if "所在楼层" in df.columns:
        df[["当前楼层", "总楼层"]] = df["所在楼层"].apply(lambda x: pd.Series(extract_floor_info_v3(x)))
    if "梯户比例" in df.columns:
        df["梯户比"] = df["梯户比例"].apply(parse_ladder_ratio)
    if "房屋户型" in df.columns:
        df[["卧室数", "客厅数", "厨房数", "卫生间数"]] = df["房屋户型"].apply(lambda x: pd.Series(parse_house_type(x)))
    return df


# ========= 3) 通用"带单位数值"清洗 =========
UNIT_EXCLUDE_COLS_PRICE = {
    "所在楼层", "梯户比例", "房屋户型", # 已被上面特殊处理
    "房屋优势", "核心卖点", "户型介绍", "周边配套", "交通出行", "客户反馈", # 纯文本
    "物业类别", "房屋年限", # 类别文本
    '抵押信息', '交易权属', '产权所属', # 类别文本
    '供暖', '供水', '供电', '建筑结构_comm', '产权描述', '房屋朝向', '环线' # 类别或多标签
}

def apply_unit_numeric_clean(df, exclude_cols=None):
    """ 通用"带单位数值"清洗（排除特殊文本列） """
    if exclude_cols is None:
        exclude_cols = UNIT_EXCLUDE_COLS_PRICE

    for col in df.select_dtypes(include="object").columns:
        if col in exclude_cols:
            continue
        # 增加判断，避免对纯数字列执行 apply
        if pd.api.types.is_numeric_dtype(df[col]):
             continue
        sample = df[col].dropna().astype(str).head(40)
        # 稍微放宽判断条件，因为有些列可能数字不多但仍需处理（如物业费）
        if len(sample) > 0 and (sample.str.contains(r"\d").mean() > 0.1 or
            sample.str.contains(r"[元㎡m²%层户年米公里]").mean() > 0.1):
            try: # 增加错误处理
                df[col] = df[col].apply(extract_numeric_or_range)
            except Exception as e:
                print(f"  警告: 清洗列 '{col}' 时出错: {e}. 跳过此列.")
    return df


# ========= 4) 日期处理函数 =========
def process_transaction_dates(df, current_year=2025):
    """ 将交易时间转化为"距今年数"和"距上次交易年数" """
    for col in ["交易时间", "上次交易"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    if "交易时间" in df.columns:
        df["交易距今年数"] = current_year - df["交易时间"].dt.year
        # 使用中位数填充交易距今年数本身的缺失
        if df["交易距今年数"].isna().any():
             median_trade_age = df["交易距今年数"].median()
             df["交易距今年数"] = df["交易距今年数"].fillna(median_trade_age)
    if "交易时间" in df.columns and "上次交易" in df.columns:
        df["距上次交易年数"] = (df["交易时间"] - df["上次交易"]).dt.days / 365
        # 优先使用交易距今年数的中位数填充
        median_trade_age_fill = df["交易距今年数"].median() if "交易距今年数" in df.columns else 5 # 兜底值5年
        df["距上次交易年数"] = df["距上次交易年数"].fillna(median_trade_age_fill)
    return df


# ========= 5) 缺失值填充函数 =========
def fill_by_group_mean(df, col, group):
    """ 按分组填充均值 """
    if col in df.columns and group in df.columns:
        # 计算分组均值，处理可能的分组全NA情况
        group_mean = df.groupby(group)[col].transform('mean')
        df[col] = df[col].fillna(group_mean)
    return df

def fill_by_group_mode(df, col, group):
    """ 按分组填充众数 """
    if col in df.columns and group in df.columns:
        # 计算分组众数，处理可能的分组全NA或无众数情况
        # transform 对每个组应用函数，如果 mode() 为空则无法填充
        # 改用更健壮的方式：先计算好每组的众数
        mode_map = df.groupby(group)[col].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
        fill_values = df[group].map(mode_map)
        df[col] = df[col].fillna(fill_values)
    return df

def fill_missing_values(train, test):
    """ 
    填充缺失值（层级填充：板块 -> 区域 -> 城市 -> 全局）
    重构：先层级填充所有列（数值/单标签/多标签），再进行编码
    """
    print("  开始层级填充...")
    
    # 1. 套内面积 = 建筑面积 × 0.85 (优先使用计算规则)
    if '套内面积' in train.columns and '建筑面积' in train.columns:
        mask_train = train['套内面积'].isna() & ~train['建筑面积'].isna()
        train.loc[mask_train, '套内面积'] = train.loc[mask_train, '建筑面积'] * 0.85
        mask_test = test['套内面积'].isna() & ~test['建筑面积'].isna()
        test.loc[mask_test, '套内面积'] = test.loc[mask_test, '建筑面积'] * 0.85
        print(f"    ✓ 套内面积基于建筑面积填充: 训练集{mask_train.sum()}条, 测试集{mask_test.sum()}条")

    # 2. 数值型列层级填充 (均值)
    numeric_cols_basic = ['物 业 费', '停车费用', '停车位', '容 积 率', '绿 化 率',
                          '楼栋总数', '房屋总数', '建筑年代', '燃气费']
    numeric_cols_parsed = ['梯户比', '卧室数', '客厅数', '厨房数', '卫生间数', 
                           '当前楼层', '总楼层', '建筑面积', '套内面积']
    numeric_cols = numeric_cols_basic + numeric_cols_parsed
    numeric_cols = [c for c in numeric_cols if c in train.columns]
    
    print(f"    正在层级填充 {len(numeric_cols)} 个数值列...")
    for col in numeric_cols:
        missing_before = train[col].isna().sum()
        if missing_before == 0:
            continue
        # 层级填充：板块 -> 区域 -> 城市 -> 全局
        if '板块_comm' in train.columns:
            train = fill_by_group_mean(train, col, '板块_comm')
            test = fill_by_group_mean(test, col, '板块_comm')
        if '区域' in train.columns:
            train = fill_by_group_mean(train, col, '区域')
            test = fill_by_group_mean(test, col, '区域')
        if '城市' in train.columns:
            train = fill_by_group_mean(train, col, '城市')
            test = fill_by_group_mean(test, col, '城市')
        # 全局均值兜底
        global_mean = train[col].mean()
        train[col] = train[col].fillna(global_mean)
        test[col] = test[col].fillna(global_mean)
        missing_after = train[col].isna().sum()
        if missing_before > 0:
            print(f"      - {col}: {missing_before} -> {missing_after} (填充{missing_before - missing_after}条)")

    # 3. 单标签类别列层级填充 (众数)
    single_label_cols = ['环线', '建筑结构', '装修情况', '配备电梯', '别墅类型',
                         '交易权属', '房屋年限', '产权所属', '房屋用途']  # 房屋用途是单标签
    single_label_cols = [c for c in single_label_cols if c in train.columns]
    
    # 调试：检查房屋用途是否在列表中
    if '房屋用途' not in train.columns:
        print(f"    ⚠️ 警告：'房屋用途'列不在训练集中！")
        print(f"    当前训练集列名（前20个）: {list(train.columns[:20])}")
    else:
        print(f"    ✓ '房屋用途'列存在，缺失值数: {train['房屋用途'].isna().sum()}")
        if train['房屋用途'].isna().sum() > 0:
            print(f"    '房屋用途'前10个值:\n{train['房屋用途'].head(10)}")
    
    print(f"    正在层级填充 {len(single_label_cols)} 个单标签类别列...")
    for col in single_label_cols:
        missing_before = train[col].isna().sum()
        if missing_before == 0:
            continue
        # 层级填充：板块 -> 区域 -> 城市 -> 全局
        if '板块_comm' in train.columns:
            train = fill_by_group_mode(train, col, '板块_comm')
            test = fill_by_group_mode(test, col, '板块_comm')
        if '区域' in train.columns:
            train = fill_by_group_mode(train, col, '区域')
            test = fill_by_group_mode(test, col, '区域')
        if '城市' in train.columns:
            train = fill_by_group_mode(train, col, '城市')
            test = fill_by_group_mode(test, col, '城市')
        # 全局众数兜底
        if not train[col].mode().empty:
            global_mode = train[col].mode().iloc[0]
            train[col] = train[col].fillna(global_mode)
            test[col] = test[col].fillna(global_mode)
        missing_after = train[col].isna().sum()
        if missing_before > 0:
            print(f"      - {col}: {missing_before} -> {missing_after} (填充{missing_before - missing_after}条)")

    # 4. 多标签类别列层级填充 (众数，针对整个字符串)
    multi_label_cols = ['房屋朝向', '物业类别', '建筑结构_comm', '产权描述',
                        '供水', '供暖', '供电', '房屋优势']  # 房屋用途已移至单标签
    multi_label_cols = [c for c in multi_label_cols if c in train.columns]
    
    print(f"    正在层级填充 {len(multi_label_cols)} 个多标签类别列...")
    for col in multi_label_cols:
        missing_before = train[col].isna().sum()
        if missing_before == 0:
            continue
        # 层级填充：板块 -> 区域 -> 城市 -> 全局
        if '板块_comm' in train.columns:
            train = fill_by_group_mode(train, col, '板块_comm')
            test = fill_by_group_mode(test, col, '板块_comm')
        if '区域' in train.columns:
            train = fill_by_group_mode(train, col, '区域')
            test = fill_by_group_mode(test, col, '区域')
        if '城市' in train.columns:
            train = fill_by_group_mode(train, col, '城市')
            test = fill_by_group_mode(test, col, '城市')
        # 全局众数兜底
        if not train[col].mode().empty:
            global_mode = train[col].mode().iloc[0]
            train[col] = train[col].fillna(global_mode)
            test[col] = test[col].fillna(global_mode)
        missing_after = train[col].isna().sum()
        if missing_before > 0:
            print(f"      - {col}: {missing_before} -> {missing_after} (填充{missing_before - missing_after}条)")

    # 5. 经纬度坐标 (区域均值)
    coord_cols = ['coord_x', 'coord_y', 'lon', 'lat']
    coord_cols = [c for c in coord_cols if c in train.columns]
    if coord_cols:
        print(f"    正在层级填充 {len(coord_cols)} 个坐标列...")
        for coord in coord_cols:
            missing_before = train[coord].isna().sum()
            if missing_before == 0:
                continue
            if '区域' in train.columns:
                train = fill_by_group_mean(train, coord, '区域')
                test = fill_by_group_mean(test, coord, '区域')
            # 全局均值兜底
            global_coord_mean = train[coord].mean()
            train[coord] = train[coord].fillna(global_coord_mean)
            test[coord] = test[coord].fillna(global_coord_mean)
            missing_after = train[coord].isna().sum()
            if missing_before > 0:
                print(f"      - {coord}: {missing_before} -> {missing_after}")

    print("  ✓ 层级填充完成！")
    return train, test


# ========= 6) 编码函数 =========
def encode_categorical_features(train, test):
    """ 
    对类别特征进行编码 (OneHot + MultiLabel)
    注意：调用此函数前，应已完成层级填充，这里只做最后兜底
    """
    # 单标签列
    single_label_cols = [
        "环线", "建筑结构", "装修情况", "配备电梯", "别墅类型",
        "交易权属", "房屋年限", "产权所属", "房屋用途"  # 房屋用途是单标签
    ]
    single_label_cols = [c for c in single_label_cols if c in train.columns]

    # 多标签列
    multi_label_cols = [
        "房屋朝向", "物业类别", "建筑结构_comm",
        "产权描述", "供水", "供暖", "供电", "房屋优势"
    ]
    multi_label_cols = [c for c in multi_label_cols if c in train.columns]

    # OneHot 编码（单标签）
    train_onehot = pd.DataFrame(index=train.index)
    test_onehot = pd.DataFrame(index=test.index)
    ohe = None
    if single_label_cols:
        # 最后兜底：将剩余NaN填充为"Unknown"（理论上不应有NaN）
    #    for col in single_label_cols:
    #        if train[col].isna().any():
    #            print(f"    警告: {col} 仍有缺失值，用'Unknown'填充")
    #            train[col] = train[col].fillna("Unknown")
    #            test[col] = test[col].fillna("Unknown")
        
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(train[single_label_cols])
        
        # 调试：打印编码后的特征名
        if "房屋用途" in single_label_cols:
            house_use_features = [f for f in ohe.get_feature_names_out(single_label_cols) if f.startswith('房屋用途_')]
            print(f"    [调试] 房屋用途编码后特征数: {len(house_use_features)}")
            print(f"    [调试] 房屋用途前5个特征: {house_use_features[:5]}")
        
        train_onehot = pd.DataFrame(ohe.transform(train[single_label_cols]), columns=ohe.get_feature_names_out(single_label_cols), index=train.index)
        test_onehot = pd.DataFrame(ohe.transform(test[single_label_cols]), columns=ohe.get_feature_names_out(single_label_cols), index=test.index)
        
        # 调试：检查编码后是否全为0
        if "房屋用途" in single_label_cols:
            house_use_cols = [c for c in train_onehot.columns if c.startswith('房屋用途_')]
            if house_use_cols:
                sum_check = train_onehot[house_use_cols].sum().sum()
                print(f"    [调试] 训练集房屋用途编码后总和: {sum_check} (应该≈训练集行数)")
                if sum_check == 0:
                    print(f"    ⚠️ 警告：训练集房屋用途编码后全为0！可能存在数据类型问题")

    # MultiLabel 编码（多标签）
    encoded_train_multi = []
    encoded_test_multi = []
    mlb_dict = {}
    for col in multi_label_cols:
        mlb = MultiLabelBinarizer()
        train_parsed = train[col].apply(split_labels)
        test_parsed = test[col].apply(split_labels)
        
        # 检查是否有空列表（理论上层级填充后不应有NaN，但split_labels可能返回空列表）
        empty_count_train = (train_parsed.apply(len) == 0).sum()
        empty_count_test = (test_parsed.apply(len) == 0).sum()
        if empty_count_train > 0 or empty_count_test > 0:
            print(f"    警告: {col} 有空标签列表 (训练集:{empty_count_train}, 测试集:{empty_count_test})")
            # 空列表填充为 ['Unknown']
            train_parsed = train_parsed.apply(lambda x: x if len(x) > 0 else ['Unknown'])
            test_parsed = test_parsed.apply(lambda x: x if len(x) > 0 else ['Unknown'])
        
        # Fit on training data only to avoid data leakage
        mlb.fit(train_parsed)
        mlb_dict[col] = mlb
        # Transform both train and test
        train_encoded = pd.DataFrame(mlb.transform(train_parsed), columns=[f"{col}_{c}" for c in mlb.classes_], index=train.index)
        test_encoded = pd.DataFrame(mlb.transform(test_parsed), columns=[f"{col}_{c}" for c in mlb.classes_], index=test.index)
        encoded_train_multi.append(train_encoded)
        encoded_test_multi.append(test_encoded)
    train_multi = pd.concat(encoded_train_multi, axis=1) if encoded_train_multi else pd.DataFrame(index=train.index)
    test_multi = pd.concat(encoded_test_multi, axis=1) if encoded_test_multi else pd.DataFrame(index=test.index)

    # 合并并删除原始列
    train_encoded = pd.concat([train, train_onehot, train_multi], axis=1)
    test_encoded = pd.concat([test, test_onehot, test_multi], axis=1)
    drop_cols = single_label_cols + multi_label_cols
    train_encoded = train_encoded.drop(columns=[c for c in drop_cols if c in train_encoded.columns])
    test_encoded = test_encoded.drop(columns=[c for c in drop_cols if c in test_encoded.columns])

    return train_encoded, test_encoded


def encode_geographic_features(train, test, target_col='Price/m2'):
    """ 对地理特征进行编码 (城市/区域 OneHot, 区域/板块 Target Mean) """
    # 1. 城市 OneHot
    city_train = pd.DataFrame(index=train.index)
    city_test = pd.DataFrame(index=test.index)
    ohe_city = None # 初始化
    if '城市' in train.columns:
        # 填充城市缺失值（如果存在）
        if train['城市'].mode().empty:
            print("警告：城市列没有众数，无法填充缺失值。")
            train_city_mode = "未知城市" # 使用默认值
        else:
            train_city_mode = train['城市'].mode().iloc[0]
        train['城市'] = train['城市'].fillna(train_city_mode)
        test['城市'] = test['城市'].fillna(train_city_mode) # 用训练集众数填充测试集

        ohe_city = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe_city.fit(train[['城市']])
        city_train = pd.DataFrame(ohe_city.transform(train[['城市']]), columns=ohe_city.get_feature_names_out(['城市']), index=train.index)
        city_test = pd.DataFrame(ohe_city.transform(test[['城市']]), columns=ohe_city.get_feature_names_out(['城市']), index=test.index)




    # 2. 区域/板块均价编码
    # 初始化映射字典和全局均值
    region_mean = {}
    block_mean = {}
    city_mean = {}
    global_mean = np.nan # 稍后计算

    if target_col in train.columns:
        global_mean = train[target_col].mean() # 计算全局均值

        if '区域' in train.columns:
            region_mean = train.groupby('区域')[target_col].mean().to_dict()
            train['区域_均价编码'] = train['区域'].map(region_mean)
            # 填充测试集时，对于训练集中未出现的区域，先尝试用全局均值填充
            test['区域_均价编码'] = test['区域'].map(region_mean).fillna(global_mean)

        if '板块_comm' in train.columns:
            block_mean = train.groupby('板块_comm')[target_col].mean().to_dict()
            train['板块_均价编码'] = train['板块_comm'].map(block_mean)
            # 层级填充测试集
            def hierarchical_fill(row):
                if '板块_comm' in row and row['板块_comm'] in block_mean: return block_mean[row['板块_comm']]
                elif '区域' in row and row['区域'] in region_mean: return region_mean[row['区域']]
                elif '城市' in row and row['城市'] in city_mean: return city_mean[row['城市']]
                else: return global_mean # 使用前面计算的全局均值
            test['板块_均价编码'] = test.apply(hierarchical_fill, axis=1)
            # 确保即使层级填充失败（例如所有地理信息都缺失），也有全局均值作为兜底
            test['板块_均价编码'] = test['板块_均价编码'].fillna(global_mean)


    # 3. 合并
    train_geo = pd.concat([city_train, train], axis=1)
    test_geo = pd.concat([city_test, test], axis=1)

    return train_geo, test_geo


# ========= 7) 主清洗函数 =========
def clean_price_data(train_path, test_path, output_train_path, output_test_path, target_col='Price/m2', current_year=2025):
    """
    房价数据完整清洗流程 (适配 DataClean_Rent.py 框架)

    参数:
        train_path: 训练集路径
        test_path: 测试集路径
        output_train_path: 清洗后训练集输出路径
        output_test_path: 清洗后测试集输出路径
        target_col: 目标列名
        current_year: 当前年份
    """

    print("=" * 60)
    print("开始加载房价数据...")
    print("=" * 60)

    # ---------- 1️⃣ 读取原始数据 ----------
    try:
        price_train = pd.read_csv(train_path, low_memory=False)
        price_test = pd.read_csv(test_path, low_memory=False)
    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e.filename}。请检查路径。")
        return None, None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None, None


    print(f"训练集形状: {price_train.shape}")
    print(f"测试集形状: {price_test.shape}")

    train = price_train.copy()
    test = price_test.copy()

    print("\n" + "=" * 60)
    print("步骤1: 处理交易时间")
    print("=" * 60)
    # ---------- 2️⃣ 处理日期列 ----------
    train = process_transaction_dates(train, current_year)
    test = process_transaction_dates(test, current_year)

    # 【第1.5步】预处理：将所有"伪装成类别的缺失值"转换为真正的NaN
    print("\n【第1.5步】统一缺失值表示...")
    # 定义所有可能表示缺失的值
    missing_indicators = ['未知结构']
    
    # 对所有列进行替换（object类型列）
    replaced_count = 0
    for col in train.columns:
        if train[col].dtype == 'object':  # 只处理字符串类型列
            before_null = train[col].isnull().sum()
            train[col] = train[col].replace(missing_indicators, np.nan)
            after_null = train[col].isnull().sum()
            if after_null > before_null:
                replaced_count += (after_null - before_null)
                print(f"  {col}: 替换了 {after_null - before_null} 个伪缺失值")
    
    print(f"总共替换了 {replaced_count} 个伪缺失值为NaN")

    print("\n" + "=" * 60)
    print("步骤2: 解析特殊文本列（楼层、梯户比、户型）")
    print("=" * 60)
    # ---------- 3️⃣ 解析"特殊文本列" ----------
    train = apply_special_text_parsers(train)
    test = apply_special_text_parsers(test)

    print("\n" + "=" * 60)
    print("步骤3: 通用数值清洗（提取带单位的数值）")
    print("=" * 60)
    # ---------- 4️⃣ "通用单位清洗" ----------
    train = apply_unit_numeric_clean(train)
    test = apply_unit_numeric_clean(test)

    print("\n" + "=" * 60)
    print("步骤4: 删除无用列")
    print("=" * 60)
    # ---------- 5️⃣ 删除已解析的原始列和无用列 ----------
    drop_cols_initial = [
        "开发商", "物业公司", "物业办公电话", "交易时间", "环线位置", "板块", # 通用
        "所在楼层", "梯户比例", "房屋户型", # 已解析
        "抵押信息", "上次交易" # 价格数据特有但通常无用
    ]
    drop_cols_exist = [c for c in drop_cols_initial if c in train.columns]
    train = train.drop(columns=drop_cols_exist, errors='ignore')
    test = test.drop(columns=drop_cols_exist, errors='ignore')
    print(f"已删除列: {drop_cols_exist}")

    print("\n" + "=" * 60)
    print("步骤5: 填充缺失值")
    print("=" * 60)
    # ---------- 6️⃣ 填充缺失值 ----------
    train['Price/m2'] = train['Price'] / train['套内面积']
    train, test = fill_missing_values(train, test)

    print("\n" + "=" * 60)
    print("步骤6 & 7: 编码类别特征 (OneHot + MultiLabel)")
    print("=" * 60)
    # ---------- 7️⃣ 类别特征编码 ----------
    train, test = encode_categorical_features(train, test)
    print(f"  单标签+多标签编码完成")

    print("\n" + "=" * 60)
    print("步骤8 & 9: 编码地理特征 & 拼接")
    print("=" * 60)
    # ---------- 8️⃣ 地理特征编码 ----------
    train, test = encode_geographic_features(train, test, target_col)
    print(f"  地理特征编码完成 (城市OHE, 区域/板块均价)")

    print("\n" + "=" * 60)
    print("步骤10: 删除文本和地理原始列")
    print("=" * 60)
    # ---------- 9️⃣ 删除文本和地理列 ----------
    drop_cols_text = ["核心卖点", "户型介绍", "周边配套", "交通出行", "客户反馈"]
    drop_cols_geo = ["城市", "区县", "板块_comm", "区域"] # 保留坐标
    # drop_cols_geo = ["城市", "区县", "板块_comm", "区域", "coord_x", "coord_y", "lon", "lat"] # 删除坐标
    drop_final = drop_cols_text + drop_cols_geo
    drop_final_exist = [c for c in drop_final if c in train.columns]
    train = train.drop(columns=drop_final_exist, errors='ignore')
    test = test.drop(columns=drop_final_exist, errors='ignore')
    print(f"  已删除文本和地理原始列: {drop_final_exist}")

    print("\n" + "=" * 60)
    print("步骤11: 检查并处理最终缺失值") # 修改标题
    print("=" * 60)
    # ---------- 1️⃣0️⃣ 最终缺失值检查和处理 ----------
    train_missing = train.isna().sum()
    train_missing_ratio = train_missing / len(train)
    high_missing_cols = train_missing_ratio[train_missing_ratio > 0.5].index.tolist()

    # 供热费在价格数据中缺失严重，直接删除
    if '供热费' in train.columns: high_missing_cols.append('供热费')
    high_missing_cols = list(set(high_missing_cols)) #去重

    if high_missing_cols:
        print(f"  发现高缺失率列(>50% 或 供热费): {high_missing_cols}")
        print(f"  删除这些列...")
        train = train.drop(columns=high_missing_cols, errors='ignore')
        test = test.drop(columns=high_missing_cols, errors='ignore')

    # 修正：添加最终数值列缺失值填充 (使用中位数)
    train_missing_final = train.isna().sum()
    test_missing_final = test.isna().sum()

    if train_missing_final.sum() > 0:
        print(f"\n训练集仍有缺失值的列:")
        print(train_missing_final[train_missing_final > 0])
        print("  使用训练集的中位数填充剩余数值列缺失值...")
        num_cols_final = train.drop(columns = ['Price']).select_dtypes(include=np.number).columns
        train_median = train[num_cols_final].median() # 计算训练集的中位数
        train[num_cols_final] = train[num_cols_final].fillna(train_median)
        test[num_cols_final] = test[num_cols_final].fillna(train_median) # 用训练集的中位数填充测试集
        print("✓ 已填充训练集和测试集剩余数值列缺失值")
    else:
        print("\n✓ 训练集无缺失值")

    # 再次检查测试集（理论上应该也被填充了）
    test_missing_after_fill = test.isna().sum()
    if test_missing_after_fill.sum() > 0:
        print(f"\n警告：填充后测试集仍有缺失值的列:\n{test_missing_after_fill[test_missing_after_fill > 0]}")
    else:
        print("\n✓ 测试集无缺失值")


    print("\n" + "=" * 60)
    print("步骤12: 保存清洗后的数据")
    print("=" * 60)
    # ---------- 1️⃣1️⃣ 保存 ----------
    try:
        train.to_csv(output_train_path, index=False, encoding='utf-8-sig')
        test.to_csv(output_test_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 训练集已保存: {output_train_path}")
        print(f"  形状: {train.shape}")
        print(f"\n✓ 测试集已保存: {output_test_path}")
        print(f"  形状: {test.shape}")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")


    print("\n" + "=" * 60)
    print("数据清洗完成！")
    print("=" * 60)

    return train, test


# ========= 主程序入口 =========
if __name__ == "__main__":
    # 设置数据路径 (使用相对路径或确保绝对路径正确)
    # 建议将数据文件放在脚本同目录下或使用配置管理
    TRAIN_PATH = "ruc_Class25Q2_train_price.csv"  # 假设文件在同目录下
    TEST_PATH = "ruc_Class25Q2_test_price.csv"   # 假设文件在同目录下
    OUTPUT_TRAIN_PATH = "Price_train_cleaned_refactored.csv"
    OUTPUT_TEST_PATH = "Price_test_cleaned_refactored.csv"

    # 执行清洗
    train_cleaned, test_cleaned = clean_price_data(
        TRAIN_PATH,
        TEST_PATH,
        OUTPUT_TRAIN_PATH,
        OUTPUT_TEST_PATH,
        target_col='Price/m2' # 明确目标列为 Price
    )

    if train_cleaned is not None and test_cleaned is not None:
        print("\n" + "=" * 60)
        print("清洗后数据预览")
        print("=" * 60)
        print("\n训练集前5行:")
        print(train_cleaned.head())
        # print("\n训练集列名:") # 列太多可能刷屏，可选注释掉
        # print(train_cleaned.columns.tolist())

