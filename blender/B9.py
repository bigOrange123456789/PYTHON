import bpy

k0=0;
for i in bpy.context.selectable_objects:
    if i.type == 'MESH':
        i.name ="mesh"+str(k0);# "新改名网格物体"+i.name
        k0=k0+1;
        print("name:",i.name)
obj=bpy.data.objects[2]
print(obj.name)         
li=dir(obj)
print(li[0])

for i in li:
    print(i+": ")
print(len(li))

obj.scale.x=obj.scale.x-5;
obj.location.x=10;