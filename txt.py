f = open('txt/1.txt',"r",encoding="utf-8")  #可读可写二进制，文件若不存在就创建
data = f.readlines()
data.append("te\nst测试")
f = open('txt/2.txt',"w",encoding="utf-8")    #可读可写二进制，文件若不存在就创建
f.write( ''.join(data))
f.close() #关闭文件
#content=f.read()
#w= content.decode('gbk').encode('utf-8')
#data2= f.read().decode('gbk').encode('utf-8')
#print(data2)
'''
f = open('txt/2.txt',"w")    #可读可写二进制，文件若不存在就创建
f.write( ''.join(data))
f.close() #关闭文件
'''
'''
import codecs
f1 = codecs.open("./txt/1.txt","r","utf-8")
print(f1)
data = f.readlines()
print(data)
data.append("te\nst测试")
print(data)

f = codecs.open("./txt/2.txt","w","utf-8")
f.write( ''.join(data))
f.close() #关闭文件


import codecs
str = "中文输出"
f = codecs.open("./txt/2.txt","w","utf-8")
f.write(str)
f.close()
'''
