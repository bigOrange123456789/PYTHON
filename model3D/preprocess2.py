import glob as glob
import numpy as np
import os
#import pymesh
import meshio

def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    #faces：三角面（面索引）
    #faces_contain_this_vertex：记录每一个点的相邻三角面
    #vf1：顶点1（点索引）
    #vf2：顶点2
    #except_face：三角面的索引（排除的面？）
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        #遍历与某个边相邻的三角面
        if i != except_face:#如果这个三角面是相邻面（不是这个三角面本身）
            face = faces[i].tolist()#获取这个三角形
            face.remove(vf1)
            face.remove(vf2)
            return i#返回某个边相邻三角形的顶点

    return except_face



def getN(vertices,faces):
    ns=[];
    for f in faces:
        [v1, v2, v3] = f
        x1, y1, z1 = vertices[v1]#获取3个点的坐标
        x2, y2, z2 = vertices[v2]
        x3, y3, z3 = vertices[v3]
        
        p1 = np.array([x1, y1, z1])
        p2 = np.array([x2, y2, z2])
        p3 = np.array([x3, y3, z3])
        n=np.cross(p1-p2, p2-p3)
        l=np.dot(n,n)**0.5
        ns.append((n/l).tolist())
    return ns;

if __name__ == '__main__':
                # load mesh
                #mesh = pymesh.load_mesh(file)#加载文件
                mesh = meshio.read("test.obj",file_format="obj")

                # clean up   #简化
                #mesh, _ = pymesh.remove_isolated_vertices(mesh)   #移除孤立点
                #mesh, _ = pymesh.remove_duplicated_vertices(mesh) #移除重复点

                # get elements 获取顶点信息和三角面信息
                #vertices = mesh.vertices.copy()
                #faces = mesh.faces.copy()
                vertices=mesh.points#顶点
                for i in mesh.cells:
                     faces=i.data #三角面
                
                # move to center
                center = (np.max(vertices, 0) + np.min(vertices, 0)) / 2
                vertices -= center#将顶点集合移动到中心对齐原点
                
                # normalize #将包围盒归一化
                max_len = np.max(vertices[:, 0]**2 + vertices[:, 1]**2 + vertices[:, 2]**2)
                vertices /= np.sqrt(max_len)

                # get normal vector #计算并获取平面法向量
                face_normal = getN(vertices,faces)
                #mesh = pymesh.form_mesh(vertices, faces)
                #mesh.add_attribute('face_normal')
                #face_normal = mesh.get_face_attribute('face_normal')
                
                
                # get neighbors   #获取邻居
                faces_contain_this_vertex = []#记录每一个点的相邻三角面
                for i in range(len(vertices)):
                    faces_contain_this_vertex.append(set([]))
                centers = []#中心点
                corners = []#角点
                
                for i in range(len(faces)):#遍历三角面
                    [v1, v2, v3] = faces[i]
                    x1, y1, z1 = vertices[v1]#获取3个点的坐标
                    x2, y2, z2 = vertices[v2]
                    x3, y3, z3 = vertices[v3]
                    #获取3个点的中心
                    centers.append([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3])
                    #三个点的坐标就是这三个点的角点
                    corners.append([x1, y1, z1, x2, y2, z2, x3, y3, z3])#三个点
                    faces_contain_this_vertex[v1].add(i)
                    faces_contain_this_vertex[v2].add(i)
                    faces_contain_this_vertex[v3].add(i)#v1,v2,v3这3个点都与这个面相邻
                
                neighbors = []#表示相邻三角形的情况
                for i in range(len(faces)):
                    [v1, v2, v3] = faces[i]
                    n1 = find_neighbor(faces, faces_contain_this_vertex, v1, v2, i)#发现邻居？
                    n2 = find_neighbor(faces, faces_contain_this_vertex, v2, v3, i)
                    n3 = find_neighbor(faces, faces_contain_this_vertex, v3, v1, i)
                    neighbors.append([n1, n2, n3])#每个点获取3个数字？
                    
                
                centers = np.array(centers)
                corners = np.array(corners)
                faces = np.concatenate([centers, corners, face_normal], axis=1)
                neighbors = np.array(neighbors)

                
                #_, filename = os.path.split(file)
                
                #使用np存储
                np.savez('test.npz',faces=faces, neighbors=neighbors)
