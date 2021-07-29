# 爬取一个视频并保存
import requests
response = requests.get( "https://vd4.bdstatic.com/mda-meixdut18qaai9jf/1080p/cae_h264/1621463318587055548/mda-meixdut18qaai9jf.mp4?v_from_s=hkapp-haokan-nanjing&auth_key=1627290766-0-0-126cdc91d7f9c39af8acd9f43c657662&bcevod_channel=searchbox_feed&pd=1&pt=3&abtest=" )
#保存文件
file = open("test.mp4","wb") #打开一个文件,wb表示以二进制格式打开一个文件只用于写入
file.write(response.content) #写入文件
file.close()#关闭操作，运行完毕后去你的目录看一眼有没有保存成功