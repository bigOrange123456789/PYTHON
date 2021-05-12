import bpy
import os
from bpy import ops

__ratio = 0.01
__fileType = "*.obj"
__dstPath = "D:\\test1"
__srcPath = "D:\\test2"

dstFile = os.path.join(__dstPath, "dst" + ".obj")

'''
#clear scene
for o in bpy.data.objects:		
    bpy.context.scene.objects.unlink(o) 
    o.user_clear() 
    bpy.data.objects.remove(o)	
'''

attrName = "dec"
attrType = "DECIMATE"
i = 0
paths = os.listdir(__srcPath)
for path in paths:
    srcFile = os.path.join(__srcPath, path, path + ".obj")
    #import	obj
    ops.import_scene.obj(filepath = srcFile, filter_glob = __fileType)
    obj = bpy.data.objects[i]
    #active obj
    bpy.context.scene.objects.active = obj
    #attribute
    obj.modifiers.new(attrName, type = attrType)
    obj.modifiers[attrName].ratio = __ratio
    #apply
    ops.object.modifier_apply(apply_as = "DATA", modifier = attrType)
    i += 1


#merge
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.join()

#split

__level=1

bpy.ops.object.mode_set(mode='EDIT')

l = 0
while l < __level:
  bpy.ops.mesh.subdivide(number_cuts=1)
  l=l+1

bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_mode(type ='FACE',action='ENABLE')

bpy.ops.object.mode_set(mode='OBJECT')

#select one by one 
__obj_polygons=bpy.data.objects[0].data.polygons
len=len(__obj_polygons)
t=0
while t < len:
  bpy.ops.object.mode_set(mode='OBJECT')
  __obj_polygons[t].select = True
  bpy.ops.object.mode_set(mode='EDIT')
  bpy.ops.mesh.separate(type="SELECTED")
  bpy.ops.object.mode_set(mode='OBJECT')
  t=t+1


#export one by one 

#file name
divider = 2**__level
factor = 32/divider
j=0
k=0

bpy.ops.object.select_all(action='DESELECT')
scene = bpy.context.scene
for ob in scene.objects:
  scene.objects.active = ob
  ob.select = True
  
  if ob.type == 'MESH':    
    tileName='tile_' + __level + '_' + j * factor + '_' + k * factor + '_tex.obj'
    bpy.ops.export_scene.obj(filepath=os.path.join(__dstPath, tileName), use_selection=True, filter_glob ="*.obj")    
    k=k+1
    if k>=divider:
      k=0
      j=j+1

  ob.select = False