import bpy

k0=0;
for i in bpy.context.selectable_objects:
    if i.type == 'MESH':
        i.name ="mesh"+str(k0);# "新改名网格物体"+i.name
        k0=k0+1;
        print("物体名称：",i.name)
        print(dir(i))  
        