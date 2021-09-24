# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 16:16:00 2021

@author: zcg54
"""

# def get_amount_0():
#     amount = input('总金额             ')
#     assert 200 >= float(amount) > 0, '单个红包金额不能超过200元'
#     #检查是否为数字，金额是否在(0，200]的金额范围内
#     assert len(amount.split('.')[1]) < 3, '单个红包金额不能超过200元'
#     #检查是否为两位小数

import numpy as np

people_in_chat = 10


def get_amount():
    while True:
        amount = input('总金额         ')
        try:
            float(amount)
        except ValueError:
            print('输入为数字格式             ')
            continue
        try:
            amount.split('.')[1]
        except IndexError:
            if 200 >= int(amount) > 0:
                break
            else:
                print('输入正确的红包金额，单个红包金额不能超过200元')
                continue
        if 200 >= float(amount) > 0 and len(amount.split('.')[1]) < 3:
            break
        else:
            print('输入正确的红包金额，单个红包金额不能超过200元')
    return amount


def get_num_rp(threshold):
    while True:
        num_rp = input('红包个数         ')
        try:
            float(num_rp)
        except ValueError:
            print('输入为数字格式             ')
            continue
        if float(num_rp) > 0:
            try:
                num_rp.index('.')
                print('红包总数为整数')
            except ValueError:
                if int(num_rp) > threshold:
                    print('输入正确的红包数量')
                    continue
                else:
                    break
        else:
            print('红包总数应为正整数')
            continue
    return num_rp


def divide_Redpocket(amount, num_rp):
    amount1 = float(amount) * 100
    total = amount1
    num_rp = int(num_rp)
    m = 0
    Money = np.zeros((1, num_rp))
    for i in range(num_rp - 1):
        m = np.random.randint(1, 2 * amount1 / (10 - i) + 1)
        amount1 -= m
        Money[0, i] = m
    Money[0, num_rp - 1] = total - np.sum(Money, axis=1)
    Money = Money / 100
    return Money


amount = get_amount()
num_rp = get_num_rp(people_in_chat)
Money = divide_Redpocket(amount, num_rp)
np.sum(Money)
