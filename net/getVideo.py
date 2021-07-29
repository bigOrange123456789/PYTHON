import requests
import lxml.html
import os
def getUrls(tree):
    def getByTag(tree,tag):
      for tree0 in tree.getchildren():
        if tree0.tag==tag:
            try:
                url=tree0.attrib["href"]
            except Exception as e:
                return;
            if not(url =="/"):
               urls.append(url)
        getByTag(tree0,tag)
    urls=[]
    getByTag(tree,"a")
    return urls;
def getVideoUrls(tree):
    def getByTag(tree,tag):
      for tree0 in tree.getchildren():
        if tree0.tag==tag:
            src=tree0.attrib["src"]#blob
            arr=src.split("blob:")
            url=arr[len(arr)-1]
            if not(url =="/"):
               urls.append(url)
        getByTag(tree0,tag)
    urls=[]
    getByTag(tree,"video")
    return urls;

def save(url):
    name=os.path.basename(url)
    response = requests.get(url)
    print("获取视频："+name)
    #open(name,'wb').write(response.content)  
    
def find(url,n):#n是最大跳转次数
    for h in urlHistory:
        if h==url:
            print("重复地址")
            return;
    urlHistory.append(url)
    
    if url[0:2]=="//":
        arr=url.split("//")
        url=arr[1]
    print(url)
    
    
    response = requests.get(url)
    tree = lxml.html.fromstring(response.text);
    urls1=getVideoUrls(tree)
    urls2=[]
    if len(urls1)==0:
        urls2=getUrls(tree)

    print(urls1,urls2)
    
    htmlUrl=[];
    for url in urls1+urls2:
        if url[0:2]=="//":
            #arr=url.split("//")
            #url=arr[1]
            url="http:"+url
        try:
            response = requests.get( url )
            type0=response.headers['Content-Type'].split(";")[0]
            print(type0)
            if type0=="text/html":
                htmlUrl.append(url)
            if type0=="video/mp4":
                save(url)
        except Exception as e:
            print('无效地址'+url)
        
    #print(urls) 
    if n>0:
      for url in htmlUrl:
        find(url,n-1)
        
url="https://www.baidu.com/sf/vsearch?pd=video&tn=vsearch&lid=afbb881200170410&ie=utf-8&wd=python%E8%8E%B7%E5%8F%96%E5%AD%97%E7%AC%A6%E4%B8%B2%E7%9A%84%E7%AC%AC%E4%B8%80%E4%B8%AA%E5%AD%97%E7%AC%A6&rsv_spt=7&rsv_bp=1&f=8&oq=python%E8%8E%B7%E5%8F%96%E5%AD%97%E7%AC%A6%E4%B8%B2%E7%9A%84%E7%AC%AC%E4%B8%80%E4%B8%AA%E5%AD%97%E7%AC%A6&rsv_pq=afbb881200170410&rsv_t=1874pDT8FHKBPSM0ZYnP5ZCEUN3WxFyG3EnezM9%2BZd3Jatf5BxVOBcOKygKWL5HHxqNj"
urlHistory=[]
find(url,2)  