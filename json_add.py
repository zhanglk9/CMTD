import os
import json

with open(os.path.join('data','gta','sample_class_stats_dict_ori.json'), 'r') as of:
    data1 = json.load(of)

with open(os.path.join('data','gta2000_rcs1e-2','sample_class_stats_dict.json'), 'r') as of2:
    data2 = json.load(of2)

# import ipdb;ipdb.set_trace()
# merged_data = {}
# for key in set(data1.keys()).union(data2.keys()):  # 获取所有的 key
#     values1 = data1.get(key, [])  # 从第一个文件获取值，默认为空列表
#     values2 = data2.get(key, [])  # 从第二个文件获取值，默认为空列表
#     merged_data[key] = values1 + values2  # 将值拼接

# merged_data = data1+data2
# 拼接字典
merged_data = data1.copy()  # 复制大字典

for key, value in data2.items():
    if key in merged_data:
        # 如果 key 存在，拼接值
        if isinstance(merged_data[key], list) and isinstance(value, list):
            merged_data[key].extend(value)
        elif isinstance(merged_data[key], str) and isinstance(value, str):
            merged_data[key] += value
        elif isinstance(merged_data[key], dict) and isinstance(value, dict):
            merged_data[key].update(value)
        else:
            raise ValueError(f"无法拼接 key '{key}' 的值，类型不匹配：{type(merged_data[key])} 和 {type(value)}")
    else:
        # 如果 key 不存在，直接添加
        merged_data[key] = value


# 保存合并后的 JSON
with open(os.path.join('data','gta','sample_class_stats_dict.json'), 'w') as f_out:
    json.dump(merged_data, f_out, indent=4)
import ipdb;ipdb.set_trace()