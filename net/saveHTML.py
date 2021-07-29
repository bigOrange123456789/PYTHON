# 爬取一个html并保存
import requests
response = requests.get( "http://www.baidu.com" )#get方法请求
response.encoding = "utf-8" #设置接收编码格式
file = open("baidu.html","w",encoding="utf-8")  #打开一个文件，w是文件不存在则新建一个文件，这里不用wb是因为不用保存成二进制
file.write( response.text )
file.close()
