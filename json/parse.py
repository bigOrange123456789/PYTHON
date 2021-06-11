import json
 
prices = {
    'ACME': 145.23,
    'AAPL': 612.78,
    'IBM': 205.55,
    'HPQ': 37.20,
    'FB': 10.75
}
#json对象单引号
#python对象双引号
 
a= json.dumps(prices)    #编码为json   dump倾倒
print(a)

b = json.loads(a)  #解码为python对象
print(b)

#https://blog.csdn.net/goodxin_ie/article/details/89387333
