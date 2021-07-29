#obj ply off 
import meshio
print(dir(meshio))
mesh = meshio.read("test.obj",file_format="obj")
#顶点：mesh.points
print(mesh.points)
for i in mesh.cells:
    print(i.data)#三角面
    print(dir(i))
print(dir(meshio))