"""
租金数据清洗 v2.0 - 解决训练集/测试集特征不一致问题

核心改进：
1. ✅ 保存编码器到文件（pickle）
2. ✅ 测试集使用训练集的编码器
3. ✅ 确保训练集和测试集列完全一致
4. ✅ 自动对齐缺失的列

使用方法：
  # 步骤1：处理训练集（会自动保存编码器）
  train_df = clean_rent_data_v2("train.csv", is_train=True, save_encoders=True)
  
  # 步骤2：处理测试集（会自动加载编码器）
  test_df = clean_rent_data_v2("test.csv", is_train=False, save_encoders=False)
  
  # 步骤3：对齐列（确保完全一致）
  test_df = align_columns(test_df, train_df)

作者: AI Assistant
版本: v2.0
日期: 2025-10-23
"""

import pandas as pd
import numpy as np
import re
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

# ============================================================================
# 辅助函数（与v1相同）
# ============================================================================

def extract_numeric_or_range(val):
    """
    通用数值提取函数
    注意：缺失值已在第1.5步统一处理，这里只处理数值提取
    """
    if pd.isna(val): 
        return np.nan
    s = str(val).replace(",", "").strip()
    
    # 再次检查空字符串（防御性编程）
    if s == "" or s in ['nan', 'None', 'NaN']: 
        return np.nan
    
    # 处理百分比
    if "%" in s:
        nums = re.findall(r"\d+\.?\d*", s)
        return float(nums[0])/100 if nums else np.nan
    
    # 提取所有数字
    nums = re.findall(r"\d+\.?\d*", s)
    if not nums: 
        return np.nan
    nums = [float(x) for x in nums]
    
    # 处理范围值（取平均）
    if "-" in s and len(nums) == 2:
        return float(np.mean(nums))
    
    return float(np.mean(nums))


def extract_floor_info(floor_str):
    """解析楼层信息"""
    if pd.isna(floor_str):
        return pd.Series([np.nan, np.nan])
    
    s = str(floor_str).strip()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", "", s)
    
    total_match = re.search(r"共?[/（(]?(\d+)层", s)
    if not total_match:
        total_match = re.search(r"/(\d+)层", s)
    total_floor = float(total_match.group(1)) if total_match else np.nan
    
    current = np.nan
    
    if "低楼层" in s and not np.isnan(total_floor):
        current = total_floor * 0.2
    elif "中楼层" in s and not np.isnan(total_floor):
        current = total_floor * 0.5
    elif "高楼层" in s and not np.isnan(total_floor):
        current = total_floor * 0.8
    elif "顶层" in s and not np.isnan(total_floor):
        current = total_floor
    elif "底层" in s:
        current = 1.0
    elif "地下" in s:
        current = -1.0
    else:
        cur_match = re.search(r"(\d+)/", s)
        if cur_match:
            current = float(cur_match.group(1))
        else:
            nums = re.findall(r"(\d+)", s)
            if nums and "/" not in s:
                current = float(nums[0])
    
    return pd.Series([current, total_floor])


def parse_house_type(room_str):
    """解析户型"""
    if pd.isna(room_str):
        return pd.Series([np.nan, np.nan, np.nan])
    
    s = str(room_str)
    
    shi_match = re.search(r"(\d+)室", s)
    shi = float(shi_match.group(1)) if shi_match else np.nan
    
    ting_match = re.search(r"(\d+)厅", s)
    ting = float(ting_match.group(1)) if ting_match else np.nan
    
    wei_match = re.search(r"(\d+)卫", s)
    wei = float(wei_match.group(1)) if wei_match else np.nan
    
    return pd.Series([shi, ting, wei])


def parse_transaction_date(date_str):
    """解析交易时间"""
    if pd.isna(date_str):
        return np.nan
    
    try:
        date_str = str(date_str).strip()
        if "-" in date_str or "/" in date_str:
            dt = pd.to_datetime(date_str, errors='coerce')
            return dt.year if pd.notna(dt) else np.nan
        elif "年" in date_str:
            year_match = re.search(r"(\d{4})年", date_str)
            return float(year_match.group(1)) if year_match else np.nan
    except:
        pass
    
    return np.nan


def split_multi_labels(x):
    """分割多标签字符串"""
    if pd.isna(x): 
        return []
    x = str(x).replace("、", "/").replace(" ", "/").replace("，", "/")
    return [t.strip() for t in re.split(r"[\/|,]", x) if t.strip()]


def fill_by_group_mean(df, col, group):
    """按分组填充均值"""
    df[col] = df.groupby(group)[col].transform(lambda x: x.fillna(x.mean()))
    return df


def fill_by_group_mode(df, col, group):
    """按分组填充众数"""
    def fill_func(x):
        mode_vals = x.mode()
        if not mode_vals.empty:
            return x.fillna(mode_vals.iloc[0])
        else:
            return x
    df[col] = df.groupby(group)[col].transform(fill_func)
    return df


# ============================================================================
# 新增：编码器管理函数
# ============================================================================

def save_encoders(encoders, save_dir="C:/Users/lenovo/Desktop/code/encoders/"):
    """保存所有编码器到文件"""
    os.makedirs(save_dir, exist_ok=True)
    
    for name, encoder in encoders.items():
        filepath = os.path.join(save_dir, f"{name}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(encoder, f)
    
    print(f"✓ 编码器已保存到: {save_dir}")


def load_encoders(save_dir="C:/Users/lenovo/Desktop/code/encoders/"):
    """从文件加载所有编码器"""
    encoders = {}
    
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"编码器目录不存在: {save_dir}")
    
    for filename in os.listdir(save_dir):
        if filename.endswith('.pkl'):
            name = filename[:-4]  # 去掉.pkl
            filepath = os.path.join(save_dir, filename)
            with open(filepath, 'rb') as f:
                encoders[name] = pickle.load(f)
    
    print(f"✓ 已加载 {len(encoders)} 个编码器")
    return encoders


def align_columns(test_df, train_df, exclude_cols=['ID', 'Price', '单价']):
    """
    对齐测试集和训练集的列
    
    参数:
        test_df: 测试集DataFrame
        train_df: 训练集DataFrame
        exclude_cols: 不需要对齐的列（如ID、Price等）
    
    返回:
        对齐后的测试集DataFrame
    """
    print("\n" + "=" * 80)
    print("对齐训练集和测试集的列")
    print("=" * 80)
    
    # 获取训练集的列（排除特定列）
    train_cols = [col for col in train_df.columns if col not in exclude_cols]
    test_cols = [col for col in test_df.columns if col not in exclude_cols]
    
    # 找出差异
    missing_in_test = set(train_cols) - set(test_cols)
    extra_in_test = set(test_cols) - set(train_cols)
    
    print(f"\n训练集列数: {len(train_cols)}")
    print(f"测试集列数: {len(test_cols)}")
    print(f"测试集缺少的列: {len(missing_in_test)}")
    print(f"测试集多余的列: {len(extra_in_test)}")
    
    # 添加缺失的列（填充0）
    if missing_in_test:
        print(f"\n添加缺失的列（共{len(missing_in_test)}个）...")
        for col in missing_in_test:
            test_df[col] = 0
            if len(list(missing_in_test)[:5]) and col in list(missing_in_test)[:5]:
                print(f"  + {col}")
        if len(missing_in_test) > 5:
            print(f"  ... 还有 {len(missing_in_test)-5} 个")
    
    # 删除多余的列
    if extra_in_test:
        print(f"\n删除多余的列（共{len(extra_in_test)}个）...")
        for col in extra_in_test:
            if col in test_df.columns:
                test_df = test_df.drop(columns=[col])
                if len(list(extra_in_test)[:5]) and col in list(extra_in_test)[:5]:
                    print(f"  - {col}")
        if len(extra_in_test) > 5:
            print(f"  ... 还有 {len(extra_in_test)-5} 个")
    
    # 重新排列列顺序，使其与训练集一致（保留ID列）
    if 'ID' in test_df.columns:
        final_cols = ['ID'] + train_cols
    else:
        final_cols = train_cols
    
    # 只选择存在的列
    final_cols = [col for col in final_cols if col in test_df.columns]
    test_df = test_df[final_cols]
    
    print(f"\n✓ 对齐完成！测试集最终列数: {test_df.shape[1]}")
    print("=" * 80)
    
    return test_df


# ============================================================================
# 主清洗流程 v2.0
# ============================================================================

def clean_rent_data_v2(file_path, is_train=True, save_encoders_flag=False, encoders=None, return_encoders=True):
    """
    租金数据清洗主函数 v2.0
    
    参数:
        file_path: 数据文件路径
        is_train: 是否为训练集
        save_encoders_flag: 是否保存编码器（仅训练集需要）
        encoders: 预加载的编码器字典（测试集使用）
        return_encoders: 是否返回编码器（True返回元组，False只返回DataFrame）
    
    返回:
        如果 return_encoders=True: (清洗后的DataFrame, 编码器字典)
        如果 return_encoders=False: 只返回清洗后的DataFrame
    """
    
    print("=" * 80)
    print(f"开始清洗{'训练集' if is_train else '测试集'}数据 (v2.0)")
    print("=" * 80)
    
    # 初始化编码器字典
    if encoders is None:
        encoders = {}
    
    # ========== 步骤1-11：基础数据处理 ==========
    print("\n【第1步】加载数据...")
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    
    # 【第1.5步】预处理：将所有"伪装成类别的缺失值"转换为真正的NaN
    print("\n【第1.5步】统一缺失值表示...")
    # 定义所有可能表示缺失的值
    missing_indicators = ['未知', 'https://img.ljcdn.com/usercent', 'https://image1.ljcdn.com/rent-']
    
    # 对所有列进行替换（object类型列）
    replaced_count = 0
    for col in df.columns:
        if df[col].dtype == 'object':  # 只处理字符串类型列
            before_null = df[col].isnull().sum()
            df[col] = df[col].replace(missing_indicators, np.nan)
            after_null = df[col].isnull().sum()
            if after_null > before_null:
                replaced_count += (after_null - before_null)
                print(f"  {col}: 替换了 {after_null - before_null} 个伪缺失值")
    
    print(f"总共替换了 {replaced_count} 个伪缺失值为NaN")
    
    print("\n【第2步】处理Price（目标变量）...")
    if is_train and 'Price' in df.columns:
        print(f"Price缺失数: {df['Price'].isnull().sum()}")
        print(f"Price统计: 均值={df['Price'].mean():.2f}, 中位数={df['Price'].median():.2f}")
    
    print("\n【第3步】清洗面积...")
    df['面积_数值'] = df['面积'].apply(extract_numeric_or_range)
    print(f"面积提取完成，缺失数: {df['面积_数值'].isnull().sum()}")
    
    print("\n【第4步】解析户型...")
    df[['室', '厅', '卫']] = df['户型'].apply(parse_house_type)
    
    print("\n【第5步】解析楼层...")
    df[['当前楼层', '总楼层']] = df['楼层'].apply(extract_floor_info)
    df['楼层比例'] = df['当前楼层'] / df['总楼层']
    df['楼层比例'] = df['楼层比例'].replace([np.inf, -np.inf], np.nan)
    
    print("\n【第6步】解析交易时间...")
    df['交易年份'] = df['交易时间'].apply(parse_transaction_date)
    current_year = 2025
    df['交易距今年数'] = current_year - df['交易年份']
    
    print("\n【第7步】处理建筑年代...")
    df['建筑年代_数值'] = df['建筑年代'].apply(extract_numeric_or_range)
    df['房龄'] = current_year - df['建筑年代_数值']
    
    print("\n【第8步】清洗数值型字段...")
    numeric_cols_to_clean = {
        '绿 化 率': '绿化率_数值',
        '容 积 率': '容积率_数值',
        '物 业 费': '物业费_数值',
        '停车位': '停车位_数值',
        '停车费用': '停车费用_数值',
        '房屋总数': '房屋总数_数值',
        '楼栋总数': '楼栋总数_数值',
        '燃气费': '燃气费_数值',
        '供热费': '供热费_数值'
    }
    
    for old_col, new_col in numeric_cols_to_clean.items():
        if old_col in df.columns:
            df[new_col] = df[old_col].apply(extract_numeric_or_range)
    
    print("\n【第9步】处理二值特征...")
    binary_cols = ['电梯', '车位', '用水', '用电', '燃气', '采暖']
    for col in binary_cols:
        if col in df.columns:
            df[f'{col}_有无'] = df[col].apply(
                lambda x: 0 if pd.isna(x) or str(x).strip() in ['无', '未知', '', 'NaN'] else 1
            )
    
    print("\n【第10步】统计配套设施数量...")
    def count_facilities(facility_str):
        if pd.isna(facility_str):
            return 0
        facility_str = str(facility_str).strip()
        if facility_str in ['', '无', '未知']:
            return 0
        facilities = re.split(r'[,，、；;]', facility_str)
        return len([f for f in facilities if f.strip()])
    
    df['配套设施数量'] = df['配套设施'].apply(count_facilities)
    
    print("\n【第11步】创建衍生特征...")
    # ⚠️ 注意：测试集不创建单价（因为没有Price）
    if is_train and 'Price' in df.columns:
        df['均价'] = df['Price'] / df['面积_数值']
        print("  创建特征: 均价")
    
    df['总房间数'] = df['室'].fillna(0) + df['厅'].fillna(0) + df['卫'].fillna(0)
    df['人均面积'] = df['面积_数值'] / df['室'].replace(0, np.nan)
    
    print("\n【第12步】填充分类变量的缺失值（层级众数）...")
    # 所有需要编码的分类变量
    single_cat_cols = ['装修', '付款方式', '租赁方式', '环线位置', '租期', '车位', '用水', '用电', '燃气', '采暖']
    multi_label_cols = ['物业类别', '朝向', '产权描述', '供水', '供暖', '供电', '建筑结构', '配套设施']
    all_cat_cols = single_cat_cols + multi_label_cols
    
    # 层级众数填充
    for col in all_cat_cols:
        if col in df.columns:
            missing_before = df[col].isnull().sum()
            if missing_before > 0:
                # 板块众数填充
                if '板块' in df.columns:
                    df = fill_by_group_mode(df, col, '板块')
                # 区县众数填充
                if '区县' in df.columns and df[col].isnull().any():
                    df = fill_by_group_mode(df, col, '区县')
                # 城市众数填充
                if '城市' in df.columns and df[col].isnull().any():
                    df = fill_by_group_mode(df, col, '城市')
                # 全局众数填充
                if df[col].isnull().any():
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val.iloc[0])
                    else:
                        df[col] = df[col].fillna('其他')  # 极端情况
                
                missing_after = df[col].isnull().sum()
                print(f"  {col}: {missing_before} -> {missing_after}")
    
    print("\n【第13步】标准化分类变量...")
    # 转换为字符串，并清理无效值（nan、None等）
    for col in single_cat_cols:
        if col in df.columns:
            # 先确保没有NaN（双重保险）
            if df[col].isnull().any():
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else '其他')
            
            # 转换为字符串
            df[f'{col}_标准'] = df[col].astype(str)
            
            # 清理astype(str)产生的 'nan'、'None'、'NaN' 等无效字符串
            # 计算有效值的众数（排除无效字符串）
            valid_values = df[f'{col}_标准'][~df[f'{col}_标准'].isin(['nan', 'None', 'NaN', '', 'null'])]
            if len(valid_values) > 0:
                replacement_value = valid_values.mode().iloc[0]
            else:
                replacement_value = '其他'
            
            df[f'{col}_标准'] = df[f'{col}_标准'].replace(['nan', 'None', 'NaN', '', 'null'], replacement_value)
    
    for col in multi_label_cols:
        if col in df.columns:
            # 先确保没有NaN（双重保险）
            if df[col].isnull().any():
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else '其他')
            
            # 转换为字符串
            df[f'{col}_标准'] = df[col].astype(str)
            
            # 清理astype(str)产生的 'nan'、'None'、'NaN' 等无效字符串
            # 计算有效值的众数（排除无效字符串）
            valid_values = df[f'{col}_标准'][~df[f'{col}_标准'].isin(['nan', 'None', 'NaN', '', 'null'])]
            if len(valid_values) > 0:
                replacement_value = valid_values.mode().iloc[0]
            else:
                replacement_value = '其他'
            
            df[f'{col}_标准'] = df[f'{col}_标准'].replace(['nan', 'None', 'NaN', '', 'null'], replacement_value)
    
    # ========== 步骤14：缺失值填充 ==========
    print("\n【第14步】填充数值型变量的缺失值...")
    
    # Price填充（仅训练集）
    if is_train and 'Price' in df.columns:
        price_missing_before = df['Price'].isnull().sum()
        if price_missing_before > 0:
            print(f"  特殊处理 - Price缺失值填充...")
            if '板块' in df.columns:
                df = fill_by_group_mean(df, 'Price', '板块')
            if '区县' in df.columns and df['Price'].isnull().any():
                df = fill_by_group_mean(df, 'Price', '区县')
            if '城市' in df.columns and df['Price'].isnull().any():
                df = fill_by_group_mean(df, 'Price', '城市')
            if df['Price'].isnull().any():
                df['Price'] = df['Price'].fillna(df['Price'].median())
    
    # 其他数值字段填充
    all_numeric_fill_cols = [
        '面积_数值', '室', '厅', '卫', '当前楼层', '总楼层', '楼层比例',
        '交易年份', '交易距今年数', '建筑年代_数值', '房龄',
        '容积率_数值', '绿化率_数值', '物业费_数值', '停车位_数值', 
        '停车费用_数值', '房屋总数_数值', '楼栋总数_数值', 
        '燃气费_数值', '供热费_数值', '配套设施数量',
        'lon', 'lat', 'coord_x', 'coord_y',
        '总房间数', '人均面积'
    ]
    
    if is_train and 'Price' in df.columns:
        all_numeric_fill_cols.append('单价')
    
    existing_numeric_cols = [col for col in all_numeric_fill_cols if col in df.columns]
    
    # 层级填充
    if '板块' in df.columns:
        for col in existing_numeric_cols:
            if df[col].isnull().any():
                df = fill_by_group_mean(df, col, '板块')
    
    if '区县' in df.columns:
        for col in existing_numeric_cols:
            if df[col].isnull().any():
                df = fill_by_group_mean(df, col, '区县')
    
    if '城市' in df.columns:
        for col in existing_numeric_cols:
            if df[col].isnull().any():
                df = fill_by_group_mean(df, col, '城市')
    
    # 全局中位数填充
    for col in existing_numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            if pd.notna(median_val):
                df[col] = df[col].fillna(median_val)
            else:
                df[col] = df[col].fillna(0)
    
    # ========== 步骤15：目标编码（改进版）==========
    print("\n【第15步】板块和区县的目标编码...")
    
    if is_train and '均价' in df.columns:
        # 训练集：创建并保存编码映射
        if '板块' in df.columns:
            plate_mean = df.groupby('板块')['均价'].mean().to_dict()
            encoders['plate_encoding'] = plate_mean
            df['板块_价格编码'] = df['板块'].map(plate_mean)
            global_mean = df['均价'].mean()
            df['板块_价格编码'] = df['板块_价格编码'].fillna(global_mean)
            print(f"  板块编码完成: {len(plate_mean)} 个板块")
        
        if '区县' in df.columns:
            district_mean = df.groupby('区县')['均价'].mean().to_dict()
            encoders['district_encoding'] = district_mean
            df['区县_价格编码'] = df['区县'].map(district_mean)
            df['区县_价格编码'] = df['区县_价格编码'].fillna(global_mean)
            print(f"  区县编码完成: {len(district_mean)} 个区县")
    else:
        # 测试集：使用训练集的编码映射
        if '板块' in df.columns and 'plate_encoding' in encoders:
            df['板块_价格编码'] = df['板块'].map(encoders['plate_encoding'])
            # 未见过的板块用训练集的全局均值
            default_val = np.mean(list(encoders['plate_encoding'].values()))
            df['板块_价格编码'] = df['板块_价格编码'].fillna(default_val)
            print(f"  ✓ 使用训练集的板块编码")
        
        if '区县' in df.columns and 'district_encoding' in encoders:
            df['区县_价格编码'] = df['区县'].map(encoders['district_encoding'])
            default_val = np.mean(list(encoders['district_encoding'].values()))
            df['区县_价格编码'] = df['区县_价格编码'].fillna(default_val)
            print(f"  ✓ 使用训练集的区县编码")
    
    # ========== 步骤16-17：独热编码（改进版）==========
    from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
    
    print("\n【第16步】独热编码（单值分类变量）...")
    single_encode_cols = [f'{col}_标准' for col in single_cat_cols if f'{col}_标准' in df.columns]
    
    if single_encode_cols:
        if is_train:
            # 训练集：fit并保存编码器
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe.fit(df[single_encode_cols])
            encoders['single_ohe'] = ohe
        else:
            # 测试集：使用训练集的编码器
            ohe = encoders.get('single_ohe')
            if ohe is None:
                raise ValueError("测试集处理需要训练集的编码器！")
        
        encoded_array = ohe.transform(df[single_encode_cols])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=ohe.get_feature_names_out(single_encode_cols),
            index=df.index
        )
        
        df = pd.concat([df, encoded_df], axis=1)
        print(f"  单值分类独热编码完成，新增 {len(encoded_df.columns)} 列")
    
    print("\n【第17步】多标签独热编码...")
    multi_encode_cols = [f'{col}_标准' for col in multi_label_cols if f'{col}_标准' in df.columns]
    
    all_multi_encoded = []
    for col in multi_encode_cols:
        parsed = df[col].apply(split_multi_labels)
        
        if is_train:
            # 训练集：fit并保存
            mlb = MultiLabelBinarizer()
            mlb.fit(parsed)
            encoders[f'mlb_{col}'] = mlb
        else:
            # 测试集：使用训练集的编码器
            mlb = encoders.get(f'mlb_{col}')
            if mlb is None:
                raise ValueError(f"测试集处理需要训练集的 {col} 编码器！")
        
        encoded_array = mlb.transform(parsed)
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=[f"{col}_{label}" for label in mlb.classes_],
            index=df.index
        )
        all_multi_encoded.append(encoded_df)
        
        print(f"  {col}: {len(mlb.classes_)} 个标签")
    
    if all_multi_encoded:
        multi_encoded_df = pd.concat(all_multi_encoded, axis=1)
        df = pd.concat([df, multi_encoded_df], axis=1)
        print(f"  多标签独热编码完成，新增 {multi_encoded_df.shape[1]} 列")
    
    # ========== 步骤18：城市编码 ==========
    print("\n【第18步】城市独热编码...")
    if '城市' in df.columns:
        if is_train:
            ohe_city = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe_city.fit(df[['城市']])
            encoders['city_ohe'] = ohe_city
        else:
            ohe_city = encoders.get('city_ohe')
            if ohe_city is None:
                raise ValueError("测试集处理需要训练集的城市编码器！")
        
        city_encoded = pd.DataFrame(
            ohe_city.transform(df[['城市']]),
            columns=ohe_city.get_feature_names_out(['城市']),
            index=df.index
        )
        df = pd.concat([df, city_encoded], axis=1)
        print(f"  城市编码完成，生成 {len(city_encoded.columns)} 列")
    
    # ========== 步骤19：删除原始文本列 ==========
    print("\n【第19步】删除原始文本列...")
    text_cols_to_drop = [
        '户型', '装修', '楼层', '面积', '朝向', '交易时间', '付款方式', '租赁方式',
        '电梯', '车位', '用水', '用电', '燃气', '采暖', '租期', '配套设施',
        '环线位置', '物业类别', '建筑年代', '开发商', '房屋总数', '楼栋总数',
        '物业公司', '绿 化 率', '容 积 率', '物 业 费', '建筑结构',
        '物业办公电话', '产权描述', '供水', '供暖', '供电', '燃气费', '供热费',
        '停车位', '停车费用', '客户反馈', '板块', '年份', '区县', '城市','建筑年代_数值','均价', 'coord_x', 'coord_y'
    ]
    
    text_cols_to_drop.extend([f'{col}_标准' for col in single_cat_cols + multi_label_cols])
    cols_to_drop = [col for col in text_cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    print(f"  删除了 {len(cols_to_drop)} 个文本列")
    
    # ========== 步骤19.5：测试集对齐列（关键步骤！）==========
    if not is_train and 'final_columns' in encoders:
        print("\n【步骤19.5】对齐测试集与训练集的列...")
        
        target_cols = encoders['final_columns']
        current_cols = [col for col in df.columns if col not in ['ID', 'Price', '单价']]
        
        # 添加缺失的列（用0填充）
        missing = set(target_cols) - set(current_cols)
        if missing:
            print(f"  添加 {len(missing)} 个缺失列（填充0）")
            for col in missing:
                df[col] = 0
        
        # 删除多余的列
        extra = set(current_cols) - set(target_cols)
        if extra:
            print(f"  删除 {len(extra)} 个多余列")
            df = df.drop(columns=list(extra), errors='ignore')
        
        # 重新排列列顺序，使其与训练集一致
        if 'ID' in df.columns:
            df = df[['ID'] + target_cols]
        else:
            df = df[target_cols]
        
        print(f"  ✓ 列对齐完成：共 {df.shape[1]} 列")
    
    # 如果是训练集，确保列顺序一致（ID放最前面）
    if is_train and 'ID' in df.columns:
        other_cols = [col for col in df.columns if col != 'ID']
        df = df[['ID'] + other_cols]
    
    # ========== 步骤20：最终检查 ==========
    print("\n【第20步】最终检查...")
    print(f"最终数据形状: {df.shape}")
    
    object_cols = df.select_dtypes(include='object').columns.tolist()
    if object_cols:
        print(f"⚠️ 仍有非数值列: {object_cols}")
    else:
        print("✓ 所有列均为数值型")
    
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"⚠️ 仍有 {missing_count} 个缺失值")
    else:
        print("✓ 无缺失值")
    
    # 保存编码器和最终列名
    if is_train and save_encoders_flag:
        # ⭐ 关键：保存清洗后的完整列名（排除目标变量）
        final_cols = [col for col in df.columns if col not in ['Price', '单价', 'ID']]
        encoders['final_columns'] = final_cols
        save_encoders(encoders)
    
    print("\n" + "=" * 80)
    print(f"{'训练集' if is_train else '测试集'}清洗完成！")
    print("=" * 80)
    
    if return_encoders:
        return df, encoders
    else:
        return df


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 步骤1：处理训练集
    print("\n" + "="*80)
    print("步骤1：处理训练集")
    print("="*80)
    
    train_file = "C:/Users/lenovo/Desktop/Data/ruc_Class25Q2_train_rent.csv"
    train_cleaned, train_encoders = clean_rent_data_v2(
        train_file, 
        is_train=True, 
        save_encoders_flag=True
    )
    
    # 保存训练集
    train_cleaned.to_csv("C:/Users/lenovo/Desktop/code/租金数据_训练集_已清洗_v2.csv", index=False, encoding='utf-8-sig')
    print(f"\n✓ 训练集已保存，形状: {train_cleaned.shape}")
    
    # 步骤2：处理测试集
    print("\n" + "="*80)
    print("步骤2：处理测试集")
    print("="*80)
    
    test_file = "C:/Users/lenovo/Desktop/Data/ruc_Class25Q2_test_rent.csv"
    
    # 加载训练集的编码器
    loaded_encoders = load_encoders()
    
    test_cleaned, _ = clean_rent_data_v2(
        test_file,
        is_train=False,
        save_encoders_flag=False,
        encoders=loaded_encoders
    )
    
    # 步骤3：对齐列
    print("\n" + "="*80)
    print("步骤3：对齐测试集和训练集的列")
    print("="*80)
    
    test_cleaned_aligned = align_columns(test_cleaned, train_cleaned)
    
    # 保存测试集
    test_cleaned_aligned.to_csv("C:/Users/lenovo/Desktop/code/租金数据_测试集_已清洗_v2.csv", index=False, encoding='utf-8-sig')
    print(f"\n✓ 测试集已保存，形状: {test_cleaned_aligned.shape}")
    
    # 最终验证
    print("\n" + "="*80)
    print("最终验证")
    print("="*80)
    
    train_cols = set(train_cleaned.columns) - {'Price', '单价'}
    test_cols = set(test_cleaned_aligned.columns)
    
    if train_cols == test_cols:
        print("✅ 完美！训练集和测试集列完全一致")
        print(f"   共同列数: {len(train_cols)}")
    else:
        missing = train_cols - test_cols
        extra = test_cols - train_cols
        print(f"❌ 仍有差异:")
        print(f"   测试集缺少: {missing}")
        print(f"   测试集多余: {extra}")