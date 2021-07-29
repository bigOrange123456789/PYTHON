# 爬取一个视频并保存
import requests
response = requests.get( "https://vd4.bdstatic.com/mda-meixdut18qaai9jf/1080p/cae_h264/1621463318587055548/mda-meixdut18qaai9jf.mp4?v_from_s=hkapp-haokan-nanjing&auth_key=1627290766-0-0-126cdc91d7f9c39af8acd9f43c657662&bcevod_channel=searchbox_feed&pd=1&pt=3&abtest=" )
print(0)
print(response.headers['Content-Type'])   
print(1)
with open('test.mp4','wb') as f:
            f.write(response.content)
print(2) 