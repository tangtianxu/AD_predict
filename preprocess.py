import pandas as pd
import numpy as np

csv_path = r"data/TADPOLE_D1_D2.csv"
raw = pd.read_csv(csv_path)
columns_to_extract = ['DX', 'ADAS13', 'Ventricles',
                      'ADAS11', 'MMSE', 'RAVLT_immediate', 'CDRSB',
                      'Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp',
                      # 'FDG','AV45',
                      # 'ABETA_UPENNBIOMK9_04_19_17','TAU_UPENNBIOMK9_04_19_17','PTAU_UPENNBIOMK9_04_19_17',
                      'APOE4', 'AGE', 'PTGENDER', ]

'''
主要预测指标:
DX:诊断结果,MCI:轻度认知障碍;Dementia:痴呆症;NL:正常
ADAS13:阿尔茨海默病评定量表第13个项
Ventricles:脑室体积

认知测试:
ADAS11：阿尔茨海默病评定量表第11项
MMSE：简易精神状态检查
RAVLT_immediate：雷文记忆言语学习测试即时记忆
CDRSB:临床痴呆评分量表，用于评估痴呆症状的严重程度，数值越大越严重

MRI(磁共振成像)指标:
Hippocampus:海马体
WholeBrain:全脑体积
Entorhinal:内嗅皮层
MidTemp:中颞叶

PET(正电子发射断层扫描)指标:
FDG:氟脱氧葡萄糖
AV45:检测脑内淀粉样斑块的PET示踪剂

CSF(脑脊液)指标:
ABETA_UPENNBIOMK9_04_19_17:脑脊液中的β-淀粉样蛋白水平
TAU_UPENNBIOMK9_04_19_17:脑脊液中的总tau蛋白水平
PTAU_UPENNBIOMK9_04_19_17:脑脊液中的磷酸化tau蛋白水平

发现PET、CSF指标缺失较多，删去这两项

风险因素:
APOE4:载脂蛋白E4
AGE:年龄
PTGENDER:性别，男性M，女性F
'''
# 数据提取
data = raw[columns_to_extract]
# 数据清洗及处理
data = data.dropna()  # 删除有缺失值的行
# 将字符串替换为编号
data = data.replace({"Female": 1, "Male": 0,
                     "NL": 4,
                     "NL to MCI": 3, 'MCI to NL': 3,
                     "MCI": 2,
                     "MCI to Dementia": 1, "Dementia to MCI": 1,
                     "Dementia": 0
                     })
# 归一化
exclude_columns = ["DX", "PTGENDER"]
for key in data.keys():
    if key not in exclude_columns:
        try:
            data[key] = data[key] / max(data[key])
        except:
            print(f"Error in column: {key}")
# 查看标签比例决定是否需要解决类别不平衡问题
print(data["DX"].value_counts())

# 分别存为csv文件方便查看和npy文件方便后续调用
csv_path = r"F:\homework\AD\data\preprocessed_data.csv"
data.to_csv(csv_path, index=False)

# npy_path = r"F:\homework\AD\data\preprocessed_data.npy"
# np.save(npy_path, data.values)
