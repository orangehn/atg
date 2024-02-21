lines = open('../input.csv').readlines()
print(lines)
datas = [line.strip('\n').split(',') for line in lines]
data_dict = {}
for data in datas:
    data_dict[data[0]] = [d.strip() for d in data[1:] if len(d.strip()) > 0]  # 删除data[1:]中的空字符串, if是保留条件
print(data_dict.keys())
print()
print(data_dict['ADD'])
inputs = []
for data_input in data_dict.values():  # 是的
    inputs.extend(data_input)
inputs_set = list(set(inputs))  # 除重

inst_names = list(data_dict.keys())
print(inputs_set)
print()
print(inst_names)

# 统计一下每中输入出现的次数
inputs_count = {}
for data_input in data_dict.values():
    for a_input in data_input:
        if a_input in inputs_count:
            inputs_count[a_input] += 1
        else:
            inputs_count[a_input] = 1
inputs_count = list(inputs_count.items())
inputs_count = sorted(inputs_count, key=lambda x: x[1])  # 按count排序
print(inputs_count)
inputs_set = [a_input for a_input, count in inputs_count]

# 好像不需要onehot形式
datas_inst_name = []
datas_input_onehot = []
for inst_name, data_input in data_dict.items():
    onehot = [0] * len(inputs_set)
    for i, a_input in enumerate(inputs_set):  # 按照inputs_set的input顺序
        if a_input in data_input:  # 如果该命令有这个input,这个位置就置1，否则置0
            onehot[i] = 1
    datas_input_onehot.append(onehot)
    datas_inst_name.append(inst_name)

print(datas_inst_name[0])
for a_input, have in zip(inputs_set, datas_input_onehot[0]):
    print(a_input, have)

import matplotlib.pyplot as plt
from collections import defaultdict  # 用于产生一个带有默认值的dict。主要针对key不存在的情况下，也希望有返回值的情况。


def transform_to_poins(inputs_set, inst_names, data_dict):
    # 把属性变成一个个点，如果第x个指令有第y个属性，就添加一个点
    points_x, points_y = [], []
    for y, a_input in enumerate(inputs_set):
        for x, inst_name in enumerate(inst_names):
            if a_input in data_dict[inst_name]:
                points_x.append(x)
                points_y.append(y)
    return points_x, points_y


def group_by_y(points_x, points_y):
    # 把参数名相同的指令分到一起，画一个颜色
    dict_points_x, dict_points_y = defaultdict(list), defaultdict(list)
    for x, y in zip(points_x, points_y):
        dict_points_x[y].append(x)
        dict_points_y[y].append(y)
    return dict_points_x, dict_points_y


def group_by_x(points_x, points_y):
    # 把指令名相同的参数分到一起，画一个颜色
    dict_points_x, dict_points_y = defaultdict(list), defaultdict(list)
    for x, y in zip(points_x, points_y):
        dict_points_x[x].append(x)
        dict_points_y[x].append(y)
    return dict_points_x, dict_points_y


plt.figure(figsize=(14, 7))
points_x, points_y = transform_to_poins(inputs_set, inst_names, data_dict)
dict_points_x, dict_points_y = group_by_y(points_x, points_y)
for key in dict_points_x:
    plt.scatter(dict_points_x[key], dict_points_y[key])

# plt.scatter(points_x, points_y, color = (0.5, 0, 0.5))
plt.xlim(-1, len(datas_inst_name))  # 左右各留一个白
plt.xticks(range(len(datas_inst_name)), inst_names, rotation=80, fontsize=12)  # rotation是文字旋转多少度
plt.ylim(-1, len(inputs_set))
plt.yticks(range(len(inputs_set)), inputs_set, fontsize=12)
plt.show()
