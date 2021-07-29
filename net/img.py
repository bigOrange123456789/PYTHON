#保存百度图片到本地

import requests #先导入爬虫的库，不然调用不了爬虫的函数

response = requests.get("https://www.baidu.com/img/baidu_jgylogo3.gif")  #get方法的到图片响应

file = open("baidu_logo.gif","wb") #打开一个文件,wb表示以二进制格式打开一个文件只用于写入

file.write(response.content) #写入文件

file.close()#关闭操作，运行完毕后去你的目录看一眼有没有保存成功