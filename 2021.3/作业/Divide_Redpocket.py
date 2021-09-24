# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 07:46:40 2021

@author: zcg54
"""

'''Class Redpocket'''

import numpy as np

class Redpocket(): 
    
    def __init__(self, people_in_chat):
        self.people_in_chat = people_in_chat
        
    def get_amount(self):
        while True:
            self.amount = input('总金额         ')
            try:
                float(self.amount)
            except ValueError:
                print('输入为数字格式             ')
                continue
            try:
                self.amount.split('.')[1]
            except IndexError:
                if 200 >= int(self.amount) > 0:   
                    break 
                else:
                    print('输入正确的红包金额，单个红包金额不能超过200元')   
                    continue
            if 200 >= float(self.amount) > 0 and len(self.amount.split('.')[1]) < 3:
                break
            else:
                print('输入正确的红包金额，单个红包金额不能超过200元')
        self.amount = float(self.amount)
        # return self.amount
    
    def get_num_rp(self):
        while True:
            self.num_rp = input('红包个数         ')
            try:
                float(self.num_rp)
            except ValueError:
                print('输入为数字格式             ')
                continue
            if float(self.num_rp) > 0:
                try:
                    self.num_rp.index('.')
                    print('红包总数为整数')
                except ValueError:
                    if int(self.num_rp) > self.people_in_chat :
                        print('输入正确的红包数量')
                        continue
                    elif float(self.amount) < float(self.num_rp) * 0.01:
                        print('输入正确的红包数量')
                        continue                        
                    else:
                        break
            else:
                print('红包总数应为正整数')
                continue
        # return self.num_rp   
        # ⬆ self.num_np's type is <class 'str'>

    def divide_Redpocket(self):        
        amount1 = float(self.amount) * 100
        
        if amount1 == int(self.num_rp):            
            print(amount1 == self.num_rp)
            Money = np.ones((1, int(self.num_rp)))
            
        else:
            total = amount1
            self.num_rp = int(self.num_rp)
            #m = 0
            Money = np.zeros((1, self.num_rp))
            for i in range(self.num_rp-1):
                m = np.random.randint(1, 2 * amount1 / (10 - i) + 1)
                amount1 -= m
                Money[0, i] = m
            Money[0, self.num_rp-1] = total - np.sum(Money, axis=1)
            
        Money = Money / 100
        return np.array(Money)    
    
    
    
    
# rp = Redpocket(11)    
# rp.get_amount()
# rp.get_num_rp()
# rp.divide_Redpocket()
    
# Divide_Redpocket CAlling:
# import Divide_Redpocket as rp
# rep = rp.Redpocket(11) 
# rep.get_amount()
# rep.get_num_rp()
# rep.divide_Redpocket()
    
    
    
    
    
    
    
    
    
    