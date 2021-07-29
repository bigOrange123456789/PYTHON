import requests
import lxml.html
import os

def getVideoUrls(tree):
    def getByTag(tree,tag):
      for tree0 in tree.getchildren():
        print(tree0.tag)
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
    open(name,'wb').write(response.content)  
    
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
    print(urls1)
    htmlUrl=[];
    for url in urls1:
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
        
        
url="http://100.68.223.24:8080/index.html"
urlHistory=[]
find(url,2)  