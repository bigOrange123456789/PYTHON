# 爬取一个html并保存
import requests
response = requests.get( "https://www.baidu.com/sf/vsearch?pd=video&tn=vsearch&lid=d45a24b000026cee&ie=utf-8&wd=%E6%B5%8B%E8%AF%95&rsv_spt=7&rsv_bp=1&f=8&oq=%E6%B5%8B%E8%AF%95&rsv_pq=d45a24b000026cee&rsv_t=38deu%2BC%2FOTJoANeiUAXgG0D4UeoXIdkbzHEB1MVaiv%2Bvu6PiILoaOraVD1gAG%2BK1I3BU" )#get方法请求

import lxml.html;
tree = lxml.html.fromstring(response.text);

treeo=tree.getchildren()
e=tree.get_element_by_id("container")
e=tree.find_class("small_img_con border-radius")[0]
tag=e.tag
att=e.attrib


def getByTag(tree,tag):
    for tree0 in tree.getchildren():
        if tree0.tag==tag:
            print(tree0.attrib["href"])
        getByTag(tree0,tag)
        
getByTag(tree,"a")     
print(response.headers['Content-Type'])    