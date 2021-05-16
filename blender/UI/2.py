import bpy

test_num=10;
n=10;
def t(test_num):
    bpy.data.objects[1].name=str(test_num);
    test_num=test_num+1;
    return test_num
    


n=t(n);




class ObjectSelectPanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_select"
    bl_label = "Select003"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):#每次点击都被执行
        print("poll")
        return (context.object is not None)#如果当前有选中的物体

    def draw_header(self, context):#只是执行一次
        print("draw_header")
        layout = self.layout
        layout.label(text="My Select Panel")#设置布局的标签文本

    def draw(self, context):#只是执行一次
        print("draw")
        layout = self.layout

        box = layout.box()
        box.label(text="Selection Tools")
        box.operator("object.select_all").action = 'TOGGLE'
        row = box.row()
        row.operator("object.select_all").action = 'INVERT'
        row.operator("object.select_random")


bpy.utils.register_class(ObjectSelectPanel)