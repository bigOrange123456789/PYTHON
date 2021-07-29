import requests
import json
for page in range(10):
    print('=========正在获取第{}页数据========='.format(page+1))
    #1、分析目标网页，确定爬取的URL路径，header参数
    base_url='https://haokan.baidu.com/videoui/api/videorec?tab=gaoxiao&act=pcFeed&pd=pc&num=20&shuaxin_id=1589679325686'
    headers={
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36',
        'cookie': 'BIDUPSID=6B2C4302696A4AF683B52A44EB718F51; PSTM=1572694630; BAIDUID=6B2C4302696A4AF6F8A61016AEAD93F8:FG=1; BDUSS=J2MXZQNnFDcmFzMW91M3NoRjBiVWZTbWc0ekJoZmRGN1RDbGw0WHFmdGx1ZTFkRVFBQUFBJCQAAAAAAAAAAAEAAAA7B3rQb25yb2FkMTk5MDA5MTkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGUsxl1lLMZdbl; MCITY=-340%3A; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; H_PS_PSSID=1428_31326_21104_31592_31605_31464_31228_30823; BDRCVFR[feWj1Vr5u3D]=I67x6TjHwwYf0; delPer=0; PSINO=1; BCLID=10046807995469437456; BDSFRCVID=WS4OJexroG3HkwRuk3GO8n3LZ3qMFyTTDYLEJs2qYShnrsPVJeC6EG0PtoWQkz--EHtdogKK0gOTH6KF_2uxOjjg8UtVJeC6EG0Ptf8g0M5; H_BDCLCKID_SF=tR333R7oKRu_HRjYbb__-P4DenJgtURZ56bHWh0bWnvbV45cDUJce-6yXbCebxrlMGnnKUT13lc5h4jX0P7_KRtr346-35543bRTLn76LRv0Kj6HybOfhP-UyN3MWh37Je3lMKoaMp78jR093JO4y4Ldj4oxJpOJ5JbMopCafJOKHICmDTKaDMK; PC_TAB_LOG=haokan_website_page; H_BDCLCKID_SF_BFESS=tR333R7oKRu_HRjYbb__-P4DenJgtURZ56bHWh0bWnvbV45cDUJce-6yXbCebxrlMGnnKUT13lc5h4jX0P7_KRtr346-35543bRTLn76LRv0Kj6HybOfhP-UyN3MWh37Je3lMKoaMp78jR093JO4y4Ldj4oxJpOJ5JbMopCafJOKHICmDTKaDMK; Hm_lvt_4aadd610dfd2f5972f1efee2653a2bc5=1589679071; reptileData=%7B%22data%22%3A%22ed1df19db330fd17687ee861d580d290d3c690944b83598af98e54596d2f3cca8aa2f6cef5ffb32a312f11a98855f991871d93bf532cb8ee87352081ec080d8d76fd4985a1cc82c493e1adccdec4885d30359f827350ce303946eb9d581f4c445567b37ac1fa56a910c43fbf96138af77ddf3d42ddd45fac9f1a0b99b33522dc%22%2C%22key_id%22%3A%2230%22%2C%22sign%22%3A%22ec7da06b%22%7D; Hm_lpvt_4aadd610dfd2f5972f1efee2653a2bc5=1589679187'
    }
    # 2、发送请求-requests 模拟浏览器发送请求，获取响应数据
    response=requests.get(base_url,headers=headers)
    data=response.text
    # print(data)
    # 3、爬取数据-json模块，把json字符串转化为python可交互的数据类型
    #    3.1数据转换
    json_data=json.loads(data)
    # print(json_data)
    #    3.2数据解析
    data_list=json_data['data']['response']['videos']
    #遍历列表
    for data in data_list:
        video_title=data['title']+'.mp4'  #视频文件名
        video_url=data['play_url']  #视频url
    #     print(video_title,video_url)

    # 4、保存数据-保存在目标文件夹中
    # 再次发送请求
        print('正在下载：',video_title)
        video_data=requests.get(video_url,headers=headers).content
    #     with open(r'D:\爬虫测试\好看视频\%s'%video_title,'wb') as f:
        with open('D:\\myEntertainment\\test\\'+video_title,'wb') as f:
            f.write(video_data)
            print('下载完成。。。\n')
#https://blog.csdn.net/onroadliuyaqiong/article/details/106171072        
                