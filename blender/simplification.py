#https://zhuanlan.zhihu.com/p/77157611
#对大的分块模型进行压缩并分层分块
import bpy
import os
from bpy import ops

__ratio = 0.01
__fileType = "*.obj"
__dstPath = "D:\\test1"
__srcPath = "D:\\test2"

dstFile = os.path.join(__dstPath, "dst" + ".obj")


#clear scene 清空场景
def clear():#clear scene
    for o in bpy.data.objects:	
        bpy.data.objects.remove(o)
clear()


#文件读取
attrName = "dec"
attrType = "DECIMATE"#decimate 毁灭#精简
i = 0
paths = os.listdir(__srcPath)#获取文件目录
for path in paths:
    srcFile = os.path.join(__srcPath, path, path + ".obj")
    #import	obj
    ops.import_scene.obj(filepath = srcFile, filter_glob = __fileType)
    obj = bpy.data.objects[i]
    #active obj
    bpy.context.scene.objects.active = obj
    #attribute
    ###bpy.ops.object.modifier_add(type='DECIMATE')
    obj.modifiers.new(attrName, type = attrType)
    obj.modifiers[attrName].ratio = __ratio
    #apply
    ###bpy.ops.object.modifier_apply()
    ops.object.modifier_apply(apply_as = "DATA", modifier = attrType)
    i += 1


#merge
def merge():
    bpy.ops.object.select_all(action='DESELECT')#取消选择
    bpy.ops.object.select_by_type(type='MESH')#选中所有mesh对象
    bpy.ops.object.join()#合并
merge()


#split
__level=1
bpy.ops.object.mode_set(mode='EDIT')#进入编辑模式

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
  
  
def simplification(obj,r):
    obj.modifiers.new("dec", type = "DECIMATE")#decimate 毁灭#精简
    obj.modifiers["dec"].ratio = r
    bpy.ops.object.mode_set(mode='OBJECT')