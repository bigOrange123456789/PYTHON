#XHR:XMLHttpRequest
#XML:可扩展标记语言，是一种用于标记电子文件使其具有结构性的标记语言。
#https://blog.csdn.net/onroadliuyaqiong/article/details/106171072
import requests
import json
for page0 in range(2):#重复请求，因为每次请求取得的数据是随机的
    print('=========正在获取第{}页数据========='.format(page0+1))
    #1、分析目标网页，确定爬取的URL路径，header参数
    #这个url是导航页面中SHR文件的地址
    base_url="https://www.baidu.com/sf/vsearch?pd=video&tn=vsearch&lid=fb36e13d00123021&ie=utf-8&wd=%E6%B5%8B%E8%AF%95&rsv_spt=7&rsv_bp=1&f=8&oq=%E6%B5%8B%E8%AF%95&rsv_pq=fb36e13d00123021&rsv_t=37d67auHDtwKNBYexuunr%2BQ4DjrIv4H8Hy09DGoypjneY5XCvlD6BAJNwEjp3yw%2FzMdJ"
    headers={
        'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36',
        'cookie': 'BIDUPSID=5BBA0A144365D9865CBCC21A9CFFC3C1; __yjs_duid=1_83b6e842fb3567dc00fb5598db7426bf1620727598916; PSTM=1620728033; BD_UPN=12314753; BAIDUID=60CE232E501F4901EEE9254DAE0724C2:FG=1; BDUSS=zlQdH5vOU5xUUJMbElqdVVjbkI4N1R-S0NWRU1RZlRGUnZVM09abkRJcFhpUlpoRVFBQUFBJCQAAAAAAAAAAAEAAABYIomLAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFf87mBX~O5gb; BDUSS_BFESS=zlQdH5vOU5xUUJMbElqdVVjbkI4N1R-S0NWRU1RZlRGUnZVM09abkRJcFhpUlpoRVFBQUFBJCQAAAAAAAAAAAEAAABYIomLAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFf87mBX~O5gb; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; H_PS_PSSID=34300_34099_33966_34278_34004_34281_34094_26350_22160; BD_CK_SAM=1; PSINO=7; BAIDUID_BFESS=BEA0005232DAF53BB2C95335A4C03A89:FG=1; BD_HOME=1; BDRCVFR[feWj1Vr5u3D]=I67x6TjHwwYf0; BDRCVFR[fb3VbsUruOn]=ddONZc2bo5mfAF9pywdpAqVuNqsus; __yjs_st=2_MmUyMWU4ZTJmMGVmYjkxMWEwMjZjNGIxMWI3NDQzMWQyNWZiYTg4YjQ3ZDIxZjVmYzM2OWMyYmU2OTk2ZDZiYTNiZjUzZmQ0MzljMzRkOTJmOTQxMmNhYTUzNjgwMWY0NDViZDhkOGUxYmI0Yjc1MmFkZGI1MDRlOGZmYzY1MjFjYjc2NTViZjc1Yjc5ZDU0YzRjOWNiZmY5ZTIzZDY3NjEwYjc3ZTFkMGI0OGE1MjZkMzk3Y2JkNWQwYWNjZjhlMjVmNWZmNWMzM2U4MTNkM2QyMmUwOGRmNzllNGI2ZDJjYWRhOTllZDg1OTc5OTY3NTUyOTM2OTA3NjI1ZTBiYV83XzNiOTZmYjkw; sugstore=0; BA_HECTOR=0h84al8525a18hagd71gfsqns0q; ab_sr=1.0.1_OGJiYjYzYWQ0M2ExMDIwYzFjNWM0MDMyNjIzOWY1ZmY3ODNlNTUxMTUxYzRmNDc1MzlmNzkzYTdiNzViNTBkM2I3YmQyNDYxNzhiZTE0NmYwYTNhZDY5ODIwNWM1ZjFmMWUwYThlYTNiMmM4YzA3OTkzMmY2ODkwMGViNTIxNTA5MTAzYTgxZTVjY2I5ZmVmNjU0YzkxNTJlYmExMTM3OQ==; delPer=1; H_PS_645EC=37d67auHDtwKNBYexuunr%2BQ4DjrIv4H8Hy09DGoypjneY5XCvlD6BAJNwEjp3yw%2FzMdJ; COOKIE_SESSION=2121_0_8_9_23_23_0_0_8_9_1_4_1047_0_0_0_1627284053_0_1627287467%7C9%233137_166_1626793118%7C9'
    }#导航页面SHR文件的用户代理和cookie #可以在SHR文件的headers中查到
    # 2、发送请求-requests 模拟浏览器发送请求，获取响应数据
    response=requests.get(base_url,headers=headers) #xhr文件的response
    data=response.text #获取html页面
    #print(data)
    #break;
    # 3、爬取数据-json模块，把json字符串转化为python可交互的数据类型
    #    3.1数据转换
    json_data=json.loads(data)#将数据转换为json格式
    #    3.2数据解析
    data_list=json_data['data']['response']['videos']#获取页面中的全部视频
    #遍历列表
    for data in data_list:
        video_title=data['title']+'.mp4'  #视频文件名
        video_url=data['play_url']  #视频url
        print(video_title,video_url)

    # 4、保存数据-保存在目标文件夹中
    # 再次发送请求
        print('正在下载：',video_title)
        video_data=requests.get(video_url,headers=headers).content
        break;
        with open('D:\\myEntertainment\\test\\'+video_title,'wb') as f:
            f.write(video_data)
            print('下载完成。。。\n')
#https://blog.csdn.net/onroadliuyaqiong/article/details/106171072        
                