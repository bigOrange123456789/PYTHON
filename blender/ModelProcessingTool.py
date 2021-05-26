import bpy

class ModelProcessingTool:
    def __init__(self):
        self.start()
        print("finished!")

    def __getObj():
        return bpy.data.objects;
        
    def start(self):
        self.delete_all()#删除场景中的全部对象
        self.load_all("E:\\myModel3D\\2")
        self.merge()
        self.simplification_all(0.8)#输入网格的压缩比
        self.separate('MATERIAL')
        self.reName()
        self.separate('LOOSE')
        self.delete_noMesh()#删除非mesh对象
        #self.download("E:\\myFile\\test\\test")
        
    def delete_all(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
    def merge(self):
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')#取消选择
        bpy.ops.object.select_by_type(type='MESH')#选中所有mesh对象
        bpy.ops.object.parent_clear(type='CLEAR')#清空所有mesh的父级关系
        bpy.ops.object.join()#合并

    def simplification_all(self,r):
        for obj in bpy.data.objects:
            if obj.type=="MESH":
                self.simplification(r,obj)
    def simplification(self,r,obj):
        obj.modifiers.new("dec", type = "DECIMATE")#decimate 毁灭#精简
        obj.modifiers["dec"].ratio = r
        bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.modifier_apply(modifier='dec')

    def separate(self,type0):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.separate(type=type0)#SELECTED  "LOOSE"
        bpy.ops.object.mode_set(mode='OBJECT')

    def reName(self):
        k0=0;
        for i in bpy.context.selectable_objects:
            if i.type == 'MESH':
                i.name ="mesh"+str(k0);# "改名网格物体"+i.name
                k0=k0+1;

    def delete_noMesh(self):#只保留mesh类型的对象
        bpy.ops.object.select_all(action='DESELECT')#取消选择
        for i in bpy.context.selectable_objects:
            if not i.type == 'MESH':
                i.select_set(True)
        bpy.ops.object.delete()

    def download(self,url):
        bpy.ops.export_scene.gltf(filepath=url, export_format="GLB", export_tangents=False,
                             export_image_format="JPEG", export_cameras=False, export_lights=False)

    def load_all(self,path):
        self.load(path,"gltf")
        self.load(path,"obj")
    def load(self,path,type):
        filters = [] # 过滤的fbx文件
        need_file_items = []
        need_file_names = []

        filterDict = {}
        for item in filters:
            filterDict[item] = True;

        import os
        file_lst = os.listdir(path)#获取文件目录

        for item in file_lst:
            fileName, fileExtension = os.path.splitext(item)#将文件名和后缀名分离
            if fileExtension == ("."+type) and (not item in filterDict):
                need_file_items.append(item)
                need_file_names.append(fileName)

        n = len(need_file_items)
        for i in range(n):
            item = need_file_items[i]
            itemName = need_file_names[i]
            ufilename = path + "\\" + item
            
            if type=="fbx":
                bpy.ops.import_scene.fbx(filepath=ufilename, directory=path,filter_glob=("*."+type))
            if type=="gltf":
                bpy.ops.import_scene.gltf(filepath=ufilename, filter_glob='*.glb;*.gltf')
                
            
ModelProcessingTool()