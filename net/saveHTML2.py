# 爬取一个html并保存
import requests
response = requests.get( "http://www.baidu.com" )#get方法请求
#response.encoding = "utf-8" #设置接收编码格式
open('baidu.html','w',encoding="utf-8").write(response.text)