
import json
import numpy as np

inpath="in/"
outpath="out/"

with open(inpath+'/face_mesh3.json') as jsonfile:
     data = json.load(jsonfile)    
facelines=data["FACEMESH_TESSELATION"]
landmarks=data["landmarks"]

print("1.生成特征三角形")
triangles0={}
for i in range(len(facelines)):
    for j in range(len(facelines)):
        if not i==j:
            line1=facelines[i]
            line2=facelines[j]
            if line1[1]==line2[0] and not line1[0]==line2[1]:
                c=line2[1]
            elif line1[1]==line2[1] and not line1[0]==line2[0]:
                c=line2[0]
            else:
                continue
            a=line1[0]
            b=line1[1]
            if a==b or a==c or b==c:
                print([a,b,c],i,line1,j,line2)
            a,b,c=np.sort([a,b,c]).tolist()
            triangles0[
                str(a)+","+str(b)+","+str(c)
            ]=1
triangles=[]
for i in triangles0:
    a,b,c=i.split(",")
    a=int(a)
    b=int(b)
    c=int(c)
    triangles.append([a,b,c])
print("triangles.min",np.min(np.array(triangles)))
print("triangles.max",np.max(np.array(triangles)))
print("len(triangles)",len(triangles))

print("2.规范特征点坐标格式")
vertices=[]
for i in landmarks:
    vertices.append([i["x"],i["y"],i["z"]])
print("len(vertices)",len(vertices))

print("保存处理后的三角形和顶点")
def save_obj(vs,fs,path):
    fs0=[]
    for i in fs:
        j=[i[0]+1,i[1]+1,i[2]+1]
        fs0.append(j)
    m={"v":vs,"f":fs0}
    with open(path, 'w') as f:
        for key in m:
            for i in m[key]:
                f.write(key+" ")
                f.write(str(i[0]))
                f.write(" " )
                f.write(str(i[1]))
                f.write(" " )
                f.write(str(i[2]))
                f.write("\n")
save_obj(vertices,triangles,outpath+"test.obj")

print("3.提取特征点的主方向")
def pca(dataMat,drict_mode):  
    def getDirect0(v):#通过3次方判断朝向
        mean = np.mean(v, axis=0).tolist()[0]  # 这个均值理论上为[0,0,0]
        v = v.tolist()
        a = 1
        b = 1
        c = 1
        sum = [0, 0, 0]
        for vi in v:
            sum[0] = sum[0] + (vi[0] - mean[0]) ** 3
            sum[1] = sum[1] + (vi[1] - mean[1]) ** 3
            sum[2] = sum[2] + (vi[2] - mean[2]) ** 3
        if sum[0] == 0 or sum[1] == 0 or sum[2] == 0:
            print("检测到对称性:", sum)
        if sum[0] < 0:
            a = -1
        if sum[1] < 0:
            b = -1
        if sum[2] < 0:
            c = -1
        return np.array([
            [a, 0, 0],
            [0, b, 0],
            [0, 0, c]
        ])
    def getDirect1(v):#对朝向进行处理模型
        return np.array([
            [0,1,0],
            [1,0,0],
            [0,0,1],
        ])
    def getDirect2(v):#通过标记判断朝向
        v=np.mat(v.tolist())
        
        result=np.eye(3)
        #右x 上y 前z
        #v=np.array(v)
        def xRepeat(x,n): return x.repeat(n).reshape(x.shape[0],n).T
        def maxI(x):return np.where(x==np.max(x))[0][0]
        
        def right(x):   return np.array(v)[356,:]
        def left(x):    return np.array(v)[234,:]
        def up(x):      return np.array(v)[10,:]
        def down(x):    return np.array(v)[152,:]#np.array(v)[175,:]
        def forword(x): return np.array(v)[4,:]
        def back(x):    return ( right(x)+left(x)+up(x)+down(x) )/4

        s=np.array( np.max(v, axis=0)-np.min(v, axis=0) )[0]
        s=xRepeat(s,v.shape[0])
        v=v/s

        # center=forword(v)
        # center=xRepeat(center,v.shape[0])
        # v=v-center

        #计算x轴
        maxindex=maxI( np.absolute(right(v)-left(v)) )
        if  maxindex==0:
            m=np.eye(3)
        elif  maxindex==1:
            m=np.array([
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1]
            ])
        else:
            m=np.array([
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]
            ])
        print("m",m)
        print("v0",v)
        print("result0",result)
        result=np.dot(result,m)
        print("result",result)
        v=v*m
        print("v",v)

        #计算y轴
        maxindex=maxI( np.absolute(up(v)-down(v)) )
        if  maxindex==0:
            m=np.array([
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1]
            ])
        elif  maxindex==1:
            m=np.eye(3)
        else:
            m=np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]
            ])
        result=np.dot(result,m)
        v=v*m

        #计算z轴
        #x轴和y轴确定后z轴自然确定了，所以这里不用计算 实际上前后由于特征点的问题难以确定
        
        a = 1
        b = 1
        c = 1
        if (right(v)-left(v))[0]<0:a=-1 #计算x轴正方向
        if (up(v)-down(v))[1]<0:b=-1 #上下经常颠倒？
        if (forword(v)-back(v))[2]<0:c=-1
        result=np.dot(result,np.array([
            [a, 0, 0],
            [0, b, 0],
            [0, 0, c]
        ]))
        print(result)

        return result

    topNfeat=3# topNfeat 降维后的维度

    # 去均值，将样本数据的中心点移到坐标原点
    meanVals = np.mean(dataMat, axis=0)  # 按列求均值，即每一列求一个均值，不同的列代表不同的特征
    meanRemoved = dataMat - meanVals  

    # 计算协方差矩阵
    # print("meanRemoved",meanRemoved)
    covMat = np.cov(meanRemoved, rowvar=0)
    # print("covMat",covMat)

    # 计算协方差矩阵的特征值和特征向量 # 确保方差最大,构造新特征两两独立
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)             # 排序,并获取排序后的下标                      #将特征值按从小到大排列
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # cut off unwanted dimensions                #选择维度为topNfeat的特征值
    redEigVects = eigVects[:, eigValInd]        # reorganize eig vects largest to smallest   #选择与特征值对应的特征向量

    if drict_mode==1:
        direct = getDirect1(meanRemoved * redEigVects)
    else:
        direct = getDirect2(meanRemoved * redEigVects)

    redEigVects = redEigVects * direct

    normalization = meanRemoved * redEigVects
    return [redEigVects, meanVals, normalization.tolist()]


_,_,vertices=pca(vertices,2)
save_obj(vertices,triangles,outpath+"3.test_pca.obj")

print("4.将特征网格标准化")
max=np.max(np.array(vertices),0)
min=np.min(np.array(vertices),0)
mid=(max+min)/2
l00=(max-min)/2
for v in vertices:
    v_origin=[]
    for i in range(3):
        v[i]=(v[i]-mid[i])/l00[i]
        v_origin.append(v[i])

print("vertices.max",np.max(np.array(vertices),0))
print("vertices.min",np.min(np.array(vertices),0))

save_obj(vertices,triangles,outpath+"4.test_normal.obj")

print("5.获取模型面部")
import sys 
sys.path.append("objPreocess") 
from Mesh import Mesh
m0_origin = Mesh(inpath+'man2.obj')
m0 = Mesh(inpath+'man2.obj')
m0.download(outpath+"man_origin.obj")

print("m0.vertex.max",np.max(np.array(m0.vertex),0))
print("m0.vertex.min",np.min(np.array(m0.vertex),0))
print("face.min",np.min(np.array(m0.face)))
face_max=np.max(np.array(m0.face))
m0.vertexHeadFlag=np.zeros((1,face_max+1)).tolist()[0]
for i in range(len(m0.vertex)):
    step=len(m0.vertex[i])/3
    if (m0.vertex[i][1]>1.565  #上方
    and m0.vertex[i][1]<1.760  #去掉头发
    and m0.vertex[i][2]>0.035): #和前方
    # if (m0.vertex[i][2]>1.55 #上方
    #    and m0.vertex[i][2]<1.8
    #    and m0.vertex[i][1]<0.01):#和前方
        m0.vertexHeadFlag[i+1]=1
def getVertexHead():#需要输入对象m0
    vertexHead=[]
    for i in range(len(m0.vertex)):
        if m0.vertexHeadFlag[i+1]==1:
            vertexHead.append(m0.vertex[i])
    return vertexHead
m0.getVertexHead=getVertexHead

print("头部顶点个数：",np.sum(np.array(m0.vertexHeadFlag)))
face2=[]
for f in m0.face:
    step=int(len(f)/3)
    if m0.vertexHeadFlag[f[0]]==1 and m0.vertexHeadFlag[f[step]]==1 and m0.vertexHeadFlag[f[step*2]]==1:
        face2.append(f)
print("头部三角面个数：",len(face2))
m0.face=face2
m0.updateFace()
m0.download(outpath+"5.man_head.obj")


print("6.将模型面部标准化")
redEigVects,_,_=pca(m0.getVertexHead(),1)
# print("redEigVects",redEigVects)
m0.vertex=( np.array(m0.vertex)*redEigVects ).tolist() 
# print("head.max",np.max(np.array(m0.getVertexHead()),0))
# print("head.min",np.min(np.array(m0.getVertexHead()),0))
max=np.max(np.array(m0.getVertexHead()),0)
min=np.min(np.array(m0.getVertexHead()),0)
mid=(max+min)/2
l00=(max-min)/2
for v in m0.vertex:#getVertexHead():
    for i in range(3):
        v[i]=(v[i]-mid[i])/l00[i]
    #v[2]=0#不进行拍扁

#使用矩阵affine记录将OBJ拍扁的变换过程
affine=np.insert(redEigVects, 3, values=[0,0,0],   axis=0)#插入一行
affine=np.insert(affine,      3, values=[1,1,1,1], axis=1)#插入一列
affine=affine*np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [-mid[0],-mid[1],-mid[2],1]
])
affine=affine*np.array([
    [1/l00[0],0,0,0],
    [0,1/l00[1],0,0],
    [0,0,1/l00[2],0],
    [0,0,0,1]
])


print("head.max",np.max(np.array(m0.getVertexHead()),0))
print("head.min",np.min(np.array(m0.getVertexHead()),0))
#研究点：用深度学习自动标记三维模型的关键点
m0.updateVertex()
m0.download(outpath+"6.man_normal.obj")

print("7.每个OBJ顶点找出距离最近的4个特征点")
####开始输出特征点集和几何点集####
# m0.vertex
####完成输出特征点集和几何点集####
distance={}
for i in range(len(m0.vertex)):#遍历OBJ顶点
    if m0.vertexHeadFlag[i+1]==1:#顶点在面部
        v=m0.vertex[i]
        d0=[]
        for w in vertices:
            pos2=v
            d0.append( (v[0]-w[0])**2+(v[1]-w[1])**2+(v[2]-w[2])**2 )
        distance[""+str(i)]=d0
# print(np.array(distance))
nearest={}#特征点的索引
for i in distance:
    arg=np.argsort( distance[i] )[0:4]#获取最近的三个特征点的索引
    nearest[i]=arg.tolist()

print("8.计算权重")
# def computeWeight(p1,p2,p3,p4,y):
#     A=np.array([
#         [ p1[0],p2[0],p3[0],p4[0] ],
#         [ p1[1],p2[1],p3[1],p4[1] ],
#         [ p1[2],p2[2],p3[2],p4[2] ],
#         [ 1,    1,    1,    1     ]
#     ])
#     B=np.array([
#         [y[0]],
#         [y[1]],
#         [y[2]],
#         [1]])
#     # A=np.array([p1,p2,p3])
#     # B=np.array([y]).T
#     w0=np.linalg.solve(A, B)#ax=b #a'ax=a'b
#     return [ w0[0][0],w0[1][0],w0[2][0],w0[3][0] ]
def computeWeight(p1,p2,p3,p4,y):
    e=0.000001#极小值
    def distance(pa,pb):
        return e+(pa[0]-pb[0])**2+(pa[1]-pb[1])**2+(pa[2]-pb[2])**2
    a=1/distance(p1,y)
    b=1/distance(p2,y)
    c=1/distance(p3,y)
    d=1/distance(p4,y)
    s=a+b+c+d
    return [ a/s,b/s,c/s,d/s ]
def computeBias(p1,p2,p3,p4,y):
    def getBias0(pa,pb):
        return np.array([ pa[0]-pb[0],pa[1]-pb[1],pa[2]-pb[2] ])
    return [getBias0(y,p1),getBias0(y,p2),getBias0(y,p3),getBias0(y,p4)]


weight={}
bias={}
for i in nearest: #i是OBJ点的编号
    y=m0.vertex[int(i)]
    n=nearest[i]#n中是三个特征点的编号
    p1=vertices[ n[0] ]
    p2=vertices[ n[1] ]
    p3=vertices[ n[2] ]
    p4=vertices[ n[3] ]
    weight[i]=computeWeight(p1,p2,p3,p4,y)
    bias[i]=computeBias(p1,p2,p3,p4,y)
    
    if i=="705":
        print("i",i)
        print("nearest[i]",nearest[i])
        p1=np.array(vertices[ n[0] ])
        p2=np.array(vertices[ n[1] ])
        p3=np.array(vertices[ n[2] ])
        p4=np.array(vertices[ n[3] ])
        w1,w2,w3,w4=weight[i]
        b1,b2,b3,b4=bias[i]
        print("p1,p2,p3,p4",p1,p2,p3,p4)
        print((p1+b1),"\n",(p2+b2),"\n",(p3+b3),"\n",(p4+b4))

# print("9.将特征网格映射到OBJ网格上")
# i_new=np.array(vertices)
# i_new=np.insert(i_new, 3, values=np.ones(len(vertices)), axis=1)#插入一列
# i_new=i_new*np.linalg.inv(affine)
# i_new=np.delete(i_new, -1, axis=1)#删除一列
# vertices=i_new.tolist()
# save_obj(vertices,triangles,outpath+"9.test_affine.obj")

print("10.更新OBJ模型的顶点坐标")
for i in nearest: #i是OBJ点的编号
    n=nearest[i]#n中是三个特征点的编号

    p1=np.array(vertices[ n[0] ])
    p2=np.array(vertices[ n[1] ])
    p3=np.array(vertices[ n[2] ])
    p4=np.array(vertices[ n[3] ])

    # p1=np.array([ 1,2,3 ])
    # p2=np.array([ 4,5,6 ])
    # p3=np.array([ 7,8,9 ])
    w1,w2,w3,w4=weight[i]
    b1,b2,b3,b4=bias[i]
    # w1,w2,w3=np.array([0,1,2])
    p=w1*(p1+b1)+w2*(p2+b2)+w3*(p3+b3)+w4*(p4+b4)
    
    # print("w1,w2,w3,w4",w1,w2,w3,w4)
    # print("b1,b2,b3,b4",b1,b2,b3,b4)
    print("p1,p2,p3,p4",p1,p2,p3,p4)
    print("i",i)
    print("nearest[i]",nearest[i])
    print((p1+b1),"\n",(p2+b2),"\n",(p3+b3),"\n",(p4+b4))

    print("p",p)
    p2=np.array([[
        p[0],p[1],p[2],1
    ]])*np.linalg.inv(affine)
    print("p2",p2)
    print("m0_origin.vertex[int(i)]",m0_origin.vertex[int(i)])



    # print(p1,"\n",p2,"\n",p3)
    # print(w1,"\n",w2,"\n",w3)
    # print(p)
    for j in range(3):
        # m0_origin.vertex[int(i)][j]=p[j]
        # print(j)
        # print(p2.tolist()[0][j])
        m0_origin.vertex[int(i)][j]=p2.tolist()[0][j]
m0_origin.updateVertex()
m0_origin.download(outpath+"10.man_update.obj")
    

#################################数据的准备############################################
affine#仿射变换矩阵 #已获取
nearest#四个特征点编号 #已完成计算
weight#权重 #已完成计算
#################################建立服务器############################################
from flask import Flask
from flask import request
import numpy as np
app=Flask(__name__)

# 跨域支持
def after_request(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
app.after_request(after_request)


@app.route("/")
def process():
    #分四步
    #一.特征点的主方向
    #二.特征点的标准化
    #三.将特征网格映射到OBJ网格上
    #四.更新OBJ模型的顶点坐标
    text=request.args.get('text')#解析地址栏
    arr=text.split(",")
    vertices=[]
    for i in range(int(len(arr)/3)):
        vertices.append([
            float(arr[3*i]),
            float(arr[3*i+1]),
            float(arr[3*i+2])
            ])
    #一.特征点的主方向
    print("3.提取特征点的主方向")
    _,_,vertices=pca(vertices,2)

    #二.特征点的标准化
    print("4.将特征网格标准化")
    max=np.max(np.array(vertices),0)
    min=np.min(np.array(vertices),0)
    mid=(max+min)/2
    l00=(max-min)/2
    for v in vertices:
        v_origin=[]
        for i in range(3):
            v[i]=(v[i]-mid[i])/l00[i]
            v_origin.append(v[i])
    # print("vertices.max",np.max(np.array(vertices),0))
    # print("vertices.min",np.min(np.array(vertices),0))

    # #三.将特征网格的映射到OBJ网格上 #是不是先计算完再映射好一些？
    # print("9.将特征网格的映射到OBJ网格上")
    # i_new=np.array(vertices)
    # i_new=np.insert(i_new, 3, values=np.ones(len(vertices)), axis=1)#插入一列
    # i_new=i_new*np.linalg.inv(affine)
    # i_new=np.delete(i_new, -1, axis=1)#删除一列
    # vertices=i_new.tolist()

    #四.更新OBJ模型的顶点坐标
    print("10.更新OBJ模型的顶点坐标")
    face_vertices={}
    for i in nearest: #i是OBJ点的编号
        n=nearest[i]#n中是三个特征点的编号

        p1=np.array(vertices[ n[0] ])
        p2=np.array(vertices[ n[1] ])
        p3=np.array(vertices[ n[2] ])
        p4=np.array(vertices[ n[3] ])

        w1,w2,w3,w4=weight[i]
        b1,b2,b3,b4=bias[i]
        p=w1*(p1+b1)+w2*(p2+b2)+w3*(p3+b3)+w4*(p4+b4)
        p2=np.array([[
            p[0],p[1],p[2],1
        ]])*np.linalg.inv(affine)
        
        # for j in range(3):
        #     m0_origin.vertex[int(i)][j]=p[j]
        face_vertices[i]=p2.tolist()[0][0:3]#p.tolist()

        for j in range(3):
            m0_origin.vertex[int(i)][j]=p2.tolist()[0][j]#p[j]
    m0_origin.updateVertex()
    m0_origin.download(outpath+"11.man_update.obj")#用于测试

    str0 = json.dumps(face_vertices)
    
    #arr=text.
    return str0

if __name__=="__main__":
    print(app.url_map)
    app.run()
