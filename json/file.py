# -*- coding: utf-8 -*-
import json

with open('test.json', 'r') as f:
    a = json.load(f)    #此时a是一个字典对象

b = {
    'ACME': 45.23
}
with open('test2.json', 'w') as f:
    json.dump(b,f)

print(a,b)    
#：https://blog.csdn.net/goodxin_ie/article/details/89387333
