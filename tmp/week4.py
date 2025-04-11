import numpy as np
import matplotlib.pyplot as plt
import math


def get_data():
    x = []
    y = []
    with open("./test.txt", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            words = line.split(',')
            x.append(words[:8])
            if '是' in words[8]:
                y.append(1)
            elif '否' in words[8]:
                y.append(0)
    return np.array(x), np.array(y)


def train_model():
    # 将分类器的概率值存储在model中
    model = {}
    train_data, train_lable = get_data()

    # 首先计算正反类的先验概率
    # model['pri_p_c1']为正类的先验概率，model['pri_p_c0']为反类的先验概率
    positive_cnt = np.sum(train_lable == 1)
    negative_cnt = np.sum(train_lable == 0)
    model['pri_p_c1'] = (positive_cnt + 1) / ((positive_cnt + negative_cnt) + 2)
    model['pri_p_c0'] = (negative_cnt + 1) / ((positive_cnt + negative_cnt) + 2)
    con_p_c1 = []
    con_p_c0 = []

    # 循环计算条件概率，
    # 第一层循环遍历每个属性（不包括连续属性），i表示每种属性
    for i in range(len(train_data[0]) - 2):
        # cnt_in_c1和cnt_in_c0存储，正、反类中，属性i的每种取值对应样例数量
        # 例： cnt_in_c1['青绿'] 表示“好瓜中色泽为青绿的有几个瓜”
        cnt_in_c1 = dict()
        cnt_in_c0 = dict()

        # 第二层循环遍历所有训练数据，j表示每条训练数据
        for j in range(len(train_data)):
            if not train_data[j][i] in cnt_in_c1.keys():
                cnt_in_c1[train_data[j][i]] = 0
            if not train_data[j][i] in cnt_in_c0.keys():
                cnt_in_c0[train_data[j][i]] = 0

            if train_lable[j] == 1:
                cnt_in_c1[train_data[j][i]] += 1
            elif train_lable[j] == 0:
                cnt_in_c0[train_data[j][i]] += 1

        # p_xi_given_c1表示条件概率
        # 例：p_xi_given_c1['青绿'] 表示“好瓜色泽为青绿的概率”
        p_xi_given_c1 = {}
        for key in cnt_in_c1.keys():
            # 拉普拉斯修正要求分子+1，分母+n
            p_xi_given_c1[key] = (cnt_in_c1[key] + 1) / (positive_cnt + len(cnt_in_c1))
        con_p_c1.append(p_xi_given_c1)

        p_xi_given_c0 = {}
        for key in cnt_in_c0.keys():
            p_xi_given_c0[key] = (cnt_in_c0[key] + 1) / (negative_cnt + len(cnt_in_c0))
        con_p_c0.append(p_xi_given_c0)

        # 遍历每个连续属性，i表示每种属性
    for i in range(len(train_data[0]) - 2, len(train_data[0])):
        p_xi_given_c1 = dict()
        p_xi_given_c0 = dict()
        p_xi_given_c1['mu'] = 0
        p_xi_given_c1['sigma'] = 0
        p_xi_given_c0['mu'] = 0
        p_xi_given_c0['sigma'] = 0

        # 计算均值
        for j in range(len(train_data)):
            if train_lable[j] == 1:
                p_xi_given_c1['mu'] += float(train_data[j][i])
            elif train_lable[j] == 0:
                p_xi_given_c0['mu'] += float(train_data[j][i])
        p_xi_given_c1['mu'] = p_xi_given_c1['mu'] / positive_cnt
        p_xi_given_c0['mu'] = p_xi_given_c0['mu'] / negative_cnt

        # 计算方差
        for j in range(len(train_data)):
            if train_lable[j] == 1:
                p_xi_given_c1['sigma'] += (float(train_data[j][i]) - p_xi_given_c1['mu']) ** 2
            elif train_lable[j] == 0:
                p_xi_given_c0['sigma'] += (float(train_data[j][i]) - p_xi_given_c0['mu']) ** 2
        p_xi_given_c1['sigma'] = np.sqrt(p_xi_given_c1['sigma'] / positive_cnt)
        p_xi_given_c0['sigma'] = np.sqrt(p_xi_given_c0['sigma'] / negative_cnt)

        con_p_c1.append(p_xi_given_c1)
        con_p_c0.append(p_xi_given_c0)

    model['con_p_c1'] = con_p_c1
    model['con_p_c0'] = con_p_c0

    return model


# 正态分布公式
def N(mu, sigma, x):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)


def predict(model, x):
    p_c1 = model['pri_p_c1']
    p_c0 = model['pri_p_c0']
    for i in range(len(x) - 2):
        p_c1 *= model['con_p_c1'][i][x[i]]
        p_c0 *= model['con_p_c0'][i][x[i]]
    for i in range(len(x) - 2, len(x)):
        p_c1 *= N(model['con_p_c1'][i]['mu'], model['con_p_c1'][i]['sigma'], float(x[i]))
        p_c0 *= N(model['con_p_c0'][i]['mu'], model['con_p_c0'][i]['sigma'], float(x[i]))
    print("p_bad:", p_c0, "\np_good", p_c1)
    return np.argmax([p_c0, p_c1])


model = train_model()
for key in model.keys():
    print(key, ":\n", model.get(key))
text_x1 = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '0.697', '0.460']
y = predict(model, text_x1)
if y == 1:
    print("好瓜")
else:
    print("坏瓜")
