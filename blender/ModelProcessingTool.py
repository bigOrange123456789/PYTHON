import bpy

class ModelProcessingTool:
    def __init__(self):
        self.start()
        print("finished!")

    def __getObj():
        return bpy.data.objects;
        
    def start(self):
        bpy.ops.object.mode_set(mode='OBJECT')
        self.merge()
        for obj in bpy.data.objects:
            if obj.type=="MESH":
                self.simplification(0.1,obj)
        self.separate()
        self.reName()


    def merge(self):
        bpy.ops.object.select_all(action='DESELECT')#取消选择
        bpy.ops.object.select_by_type(type='MESH')#选中所有mesh对象
        bpy.ops.object.join()#合并

    def simplification(self,r,obj):
        obj.modifiers.new("dec", type = "DECIMATE")#decimate 毁灭#精简
        obj.modifiers["dec"].ratio = r
        bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.modifier_apply(modifier='dec')

    def separate(self):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.separate(type="LOOSE")#SELECTED
        bpy.ops.object.mode_set(mode='OBJECT')

    def reName(self):
        k0=0;
        for i in bpy.context.selectable_objects:
            if i.type == 'MESH':
                i.name ="mesh"+str(k0);# "改名网格物体"+i.name
                k0=k0+1;

ModelProcessingTool()