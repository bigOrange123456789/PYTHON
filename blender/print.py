import bpy

obj=bpy.data.objects[2]       
li=dir(obj)

str="["
for i in li:
    str=str+"\""+i+"\","
str=str+"]"

with  open('D:\\gitHubRepositorys\\PYTHON\\blender\\MESH.json','w') as f:
    f.write(str)
    f.close()

'''
import bpy
from random import randint
bpy.ops.object.delete(use_global=False)

print(bpy)
print(bpy.ops)

for name in vars(bpy.ops).items():
      print(name)   
      
'''
