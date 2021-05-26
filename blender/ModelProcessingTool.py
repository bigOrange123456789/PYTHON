import bpy

class ModelProcessingTool:
    def __init__(self):
        #self.start()
        self.load("E:\\myModel3D\\2","gltf")
        print("finished!")

    def __getObj():
        return bpy.data.objects;
        
    def start(self):
        bpy.ops.object.mode_set(mode='OBJECT')
        self.merge()
        self.simplification_all()
        self.separate('MATERIAL')
        self.reName()
        self.separate('LOOSE')
        #self.download("E:\\myFile\\test\\test")

    def merge(self):
        bpy.ops.object.select_all(action='DESELECT')#取消选择
        bpy.ops.object.select_by_type(type='MESH')#选中所有mesh对象
        bpy.ops.object.parent_clear(type='CLEAR')#清空所有mesh的父级关系
        bpy.ops.object.join()#合并

    def simplification_all(self):
        for obj in bpy.data.objects:
            if obj.type=="MESH":
                self.simplification(0.1,obj)

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
    def download(self,url):
        bpy.ops.export_scene.gltf(filepath=url, export_format="GLB", export_tangents=False,
                             export_image_format="JPEG", export_cameras=False, export_lights=False)

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
            
            

ModelProcessingTool()
