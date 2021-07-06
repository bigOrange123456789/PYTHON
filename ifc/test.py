#：https://blog.csdn.net/shanmama2434/article/details/103990322
import ifcopenshell
ifc_file = ifcopenshell.open(r'C:\Users\18640\Desktop\IFC+RVT\total model\小别墅.ifc') #自己的ifc路径自己找
products = ifc_file.by_type('IfcProduct')
beam = ifc_file.by_type('IfcBeam')[0]
print("第一根构件的ID是："+ beam.GlobalId)
print("第一根构件的名字是："+beam.Name)
products = ifc_file.by_type('IfcProduct')
for product in products:
    print("文件中含有以下几种产品："+product.is_a())
