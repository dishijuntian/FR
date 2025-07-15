# 航班排名特征工程指南

## 特征分组与处理方案

### 1. 核心标识符组 (Core Identifiers)
**特征**: `['Id', 'ranker_id', 'profileId', 'requestDate']`

**现实意义**: 
- `Id`: 每个航班选项的唯一标识符
- `ranker_id`: 搜索会话标识符，用于分组排名
- `profileId`: 用户标识符，用于个性化建模
- `requestDate`: 搜索请求时间

**处理方案**:
```python
# 时间特征提取
df['request_hour'] = pd.to_datetime(df['requestDate']).dt.hour
df['request_day_of_week'] = pd.to_datetime(df['requestDate']).dt.dayofweek
df['request_month'] = pd.to_datetime(df['requestDate']).dt.month
df['is_weekend'] = df['request_day_of_week'].isin([5, 6]).astype(int)

# 用户编码
df['profileId_encoded'] = df['profileId'].astype('category').cat.codes
```

### 2. 用户基本信息组 (User Demographics)
**特征**: `['sex', 'nationality', 'frequentFlyer', 'isVip', 'bySelf', 'isAccess3D']`

**现实意义**:
- `sex`: 用户性别
- `nationality`: 用户国籍
- `frequentFlyer`: 常旅客状态
- `isVip`: VIP状态
- `bySelf`: 是否自主预订
- `isAccess3D`: 内部功能标记

**处理方案**:
```python
# 类别编码
categorical_features = ['sex', 'nationality']
for feature in categorical_features:
    df[f'{feature}_encoded'] = df[feature].astype('category').cat.codes

# 二进制特征保持不变
binary_features = ['frequentFlyer', 'isVip', 'bySelf', 'isAccess3D']
for feature in binary_features:
    df[feature] = df[feature].fillna(0).astype(int)
```

### 3. 企业信息组 (Corporate Information)
**特征**: `['companyID', 'corporateTariffCode']`

**现实意义**:
- `companyID`: 企业标识符
- `corporateTariffCode`: 企业差旅政策代码

**处理方案**:
```python
# 企业编码
df['companyID_encoded'] = df['companyID'].astype('category').cat.codes

# 差旅政策特征
df['has_corporate_tariff'] = df['corporateTariffCode'].notna().astype(int)
df['corporateTariffCode_encoded'] = df['corporateTariffCode'].astype('category').cat.codes
```

### 4. 路线信息组 (Route Information)
**特征**: `['searchRoute']`

**现实意义**:
- `searchRoute`: 航线路径，单程无"/"，往返有"/"

**处理方案**:
```python
# 路线类型识别
df['is_roundtrip'] = df['searchRoute'].str.contains('/').astype(int)
df['route_encoded'] = df['searchRoute'].astype('category').cat.codes

# 提取出发地和目的地
df['origin'] = df['searchRoute'].str.split('/').str[0]
df['destination'] = df['searchRoute'].str.split('/').str[1] if df['is_roundtrip'] else df['searchRoute']
```

### 5. 价格信息组 (Pricing Information)
**特征**: `['totalPrice', 'taxes', 'pricingInfo_isAccessTP', 'pricingInfo_passengerCount']`

**现实意义**:
- `totalPrice`: 机票总价
- `taxes`: 税费部分
- `pricingInfo_isAccessTP`: 是否符合企业差旅政策
- `pricingInfo_passengerCount`: 乘客数量

**处理方案**:
```python
# 价格特征工程
df['base_price'] = df['totalPrice'] - df['taxes']
df['tax_ratio'] = df['taxes'] / df['totalPrice']

# 分组内价格排名
df['price_rank_in_group'] = df.groupby('ranker_id')['totalPrice'].rank(method='min')
df['price_percentile_in_group'] = df.groupby('ranker_id')['totalPrice'].rank(pct=True)

# 价格标准化
df['price_normalized'] = df.groupby('ranker_id')['totalPrice'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
)

# 企业政策合规性
df['policy_compliant'] = df['pricingInfo_isAccessTP'].fillna(0).astype(int)
```

### 6. 航班时间信息组 (Flight Timing)
**特征**: `['legs0_departureAt', 'legs0_arrivalAt', 'legs0_duration', 'legs1_departureAt', 'legs1_arrivalAt', 'legs1_duration']`

**现实意义**:
- `legs0_*`: 去程航班时间信息
- `legs1_*`: 返程航班时间信息（如果有）
- `duration`: 飞行时长

**处理方案**:
```python
# 时间特征提取
for leg in ['legs0', 'legs1']:
    if f'{leg}_departureAt' in df.columns:
        df[f'{leg}_departure_hour'] = pd.to_datetime(df[f'{leg}_departureAt']).dt.hour
        df[f'{leg}_arrival_hour'] = pd.to_datetime(df[f'{leg}_arrivalAt']).dt.hour
        df[f'{leg}_departure_day_of_week'] = pd.to_datetime(df[f'{leg}_departureAt']).dt.dayofweek
        
        # 时间段分类
        df[f'{leg}_time_category'] = pd.cut(
            df[f'{leg}_departure_hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['night', 'morning', 'afternoon', 'evening']
        )
        
        # 飞行时长标准化
        df[f'{leg}_duration_normalized'] = df.groupby('ranker_id')[f'{leg}_duration'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
        )

# 总飞行时长
df['total_duration'] = df['legs0_duration'].fillna(0) + df['legs1_duration'].fillna(0)
```

### 7. 航班分段信息组 (Flight Segments)
**特征**: 所有`legs*_segments*_*`特征

**现实意义**:
- 每个航班腿可能包含多个分段（转机）
- 每个分段包含机场、航空公司、机型等信息

**处理方案**:
```python
# 计算连接次数
def count_segments(df, leg):
    segment_count = 0
    for i in range(4):  # segments 0-3
        if f'{leg}_segments{i}_departureFrom_airport_iata' in df.columns:
            segment_count += df[f'{leg}_segments{i}_departureFrom_airport_iata'].notna().sum()
    return segment_count

df['legs0_segment_count'] = df.apply(lambda row: sum([
    1 for i in range(4) 
    if f'legs0_segments{i}_departureFrom_airport_iata' in df.columns and 
    pd.notna(row.get(f'legs0_segments{i}_departureFrom_airport_iata'))
]), axis=1)

# 是否直飞
df['is_direct_flight'] = (df['legs0_segment_count'] == 1).astype(int)

# 主要航空公司
df['primary_carrier'] = df['legs0_segments0_marketingCarrier_code']
df['primary_carrier_encoded'] = df['primary_carrier'].astype('category').cat.codes
```

### 8. 机场与航空公司信息组 (Airport & Airline Information)
**特征**: 所有`*_airport_iata`, `*_marketingCarrier_code`, `*_operatingCarrier_code`

**处理方案**:
```python
# 机场编码
airport_columns = [col for col in df.columns if 'airport_iata' in col]
for col in airport_columns:
    df[f'{col}_encoded'] = df[col].astype('category').cat.codes

# 航空公司编码
carrier_columns = [col for col in df.columns if 'Carrier_code' in col]
for col in carrier_columns:
    df[f'{col}_encoded'] = df[col].astype('category').cat.codes

# 主要出发和到达机场
df['main_departure_airport'] = df['legs0_segments0_departureFrom_airport_iata']
df['main_arrival_airport'] = df['legs0_segments0_arrivalTo_airport_iata']
```

### 9. 服务等级信息组 (Service Class Information)
**特征**: 所有`*_cabinClass`, `*_seatsAvailable`, `*_baggageAllowance_*`

**现实意义**:
- `cabinClass`: 舱位等级 (1.0=经济舱, 2.0=商务舱, 4.0=头等舱)
- `seatsAvailable`: 可用座位数
- `baggageAllowance`: 行李额度

**处理方案**:
```python
# 舱位等级特征
df['primary_cabin_class'] = df['legs0_segments0_cabinClass']
df['is_business_class'] = (df['primary_cabin_class'] == 2.0).astype(int)
df['is_economy_class'] = (df['primary_cabin_class'] == 1.0).astype(int)

# 座位可用性
df['primary_seats_available'] = df['legs0_segments0_seatsAvailable']
df['seat_scarcity'] = df.groupby('ranker_id')['primary_seats_available'].transform(
    lambda x: 1 - (x / x.max()) if x.max() > 0 else 0
)

# 行李额度处理
df['primary_baggage_allowance'] = df['legs0_segments0_baggageAllowance_quantity']
df['baggage_weight_type'] = df['legs0_segments0_baggageAllowance_weightMeasurementType']
```

### 10. 机型信息组 (Aircraft Information)
**特征**: 所有`*_aircraft_code`, `*_flightNumber`

**处理方案**:
```python
# 机型编码
aircraft_columns = [col for col in df.columns if 'aircraft_code' in col]
for col in aircraft_columns:
    df[f'{col}_encoded'] = df[col].astype('category').cat.codes

# 主要机型
df['primary_aircraft'] = df['legs0_segments0_aircraft_code']
df['primary_aircraft_encoded'] = df['primary_aircraft'].astype('category').cat.codes

# 航班号特征
df['primary_flight_number'] = df['legs0_segments0_flightNumber']
```

### 11. 政策规则信息组 (Policy Rules)
**特征**: `['miniRules0_monetaryAmount', 'miniRules0_percentage', 'miniRules0_statusInfos', 'miniRules1_monetaryAmount', 'miniRules1_percentage', 'miniRules1_statusInfos']`

**现实意义**:
- `miniRules0_*`: 取消政策规则
- `miniRules1_*`: 改签政策规则

**处理方案**:
```python
# 取消政策特征
df['cancellation_fee'] = df['miniRules0_monetaryAmount'].fillna(0)
df['cancellation_percentage'] = df['miniRules0_percentage'].fillna(0)
df['can_cancel'] = (df['miniRules0_statusInfos'] != 0).astype(int)

# 改签政策特征
df['exchange_fee'] = df['miniRules1_monetaryAmount'].fillna(0)
df['exchange_percentage'] = df['miniRules1_percentage'].fillna(0)
df['can_exchange'] = (df['miniRules1_statusInfos'] != 0).astype(int)

# 灵活性评分
df['flexibility_score'] = (
    df['can_cancel'] * 0.5 + 
    df['can_exchange'] * 0.3 + 
    (1 - df['cancellation_fee'] / df['totalPrice'].clip(lower=1)) * 0.2
)
```

### 12. 目标变量组 (Target Variable)
**特征**: `['selected']`

**现实意义**: 训练集中的选择标签（0或1）

**处理方案**:
```python
# 验证每个ranker_id只有一个选中的航班
assert df.groupby('ranker_id')['selected'].sum().max() == 1, "每组应该只有一个选中的航班"

# 创建排名目标（用于训练）
df['selection_rank'] = df.groupby('ranker_id')['selected'].transform(
    lambda x: (-x).rank(method='min')
)
```

## 高级特征工程策略

### 1. 组内相对特征
```python
# 相对价格位置
df['price_position'] = df.groupby('ranker_id')['totalPrice'].rank(pct=True)

# 相对时间位置
df['departure_time_position'] = df.groupby('ranker_id')['legs0_departure_hour'].rank(pct=True)

# 相对飞行时长位置
df['duration_position'] = df.groupby('ranker_id')['total_duration'].rank(pct=True)
```

### 2. 交互特征
```python
# 价格-时间交互
df['price_time_interaction'] = df['price_position'] * df['departure_time_position']

# 舱位-价格交互
df['class_price_interaction'] = df['is_business_class'] * df['price_position']

# 政策-价格交互
df['policy_price_interaction'] = df['policy_compliant'] * df['price_position']
```

### 3. 用户偏好特征
```python
# 用户历史选择模式（如果有历史数据）
user_preferences = df.groupby('profileId').agg({
    'is_business_class': 'mean',
    'is_direct_flight': 'mean',
    'legs0_departure_hour': 'mean'
}).add_prefix('user_pref_')

df = df.merge(user_preferences, on='profileId', how='left')
```

### 4. 时间相关特征
```python
# 与理想时间的偏差
df['departure_time_deviation'] = abs(df['legs0_departure_hour'] - 9)  # 假设理想出发时间是9点

# 商务时间偏好
df['business_friendly_time'] = (
    (df['legs0_departure_hour'].between(7, 10)) | 
    (df['legs0_departure_hour'].between(17, 20))
).astype(int)
```

## 特征验证与质量控制

### 1. 数据质量检查
```python
# 检查缺失值
missing_report = df.isnull().sum().sort_values(ascending=False)
print("缺失值报告:")
print(missing_report[missing_report > 0])

# 检查异常值
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# 检查价格异常值
price_outliers = detect_outliers(df, 'totalPrice')
```

### 2. 特征相关性分析
```python
# 计算特征相关性
numeric_features = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_features].corr()

# 识别高度相关的特征
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((
                correlation_matrix.columns[i], 
                correlation_matrix.columns[j], 
                correlation_matrix.iloc[i, j]
            ))
```

### 3. 特征重要性预评估
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 快速特征重要性评估
X = df.select_dtypes(include=[np.number]).drop(['Id', 'selected'], axis=1)
y = df['selected']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

feature_importance = pd.DataFrame({