#obj ply off 
import meshio
mesh = meshio.read("test.obj",file_format="obj")
print(dir(mesh))
#顶点：mesh.points
print(mesh.points)
for i in mesh.cells:
    print("cells:")
    print(i.data)#三角面

