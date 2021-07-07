#：https://blog.csdn.net/shanmama2434/article/details/103990322
import ifcopenshell
ifc_file= ifcopenshell.open("test.ifc")
products = ifc_file.by_type('IfcProduct')

beams=ifc_file.by_type('IfcBeam')
product = products[0]
print(ifc_file.by_type("ifcwall")[0])
print("第一根构件的ID是："+ product.GlobalId)
print("第一根构件的名字是："+product.Name)

print(dir(product))
for name,value in vars(product).items():
      print('%s=%s'%(name,value))
#print(vars(product).items())


def att(product,tag):
    for name,value in vars(product).items():
        if tag==name:
            return value;
    return None;
for i in products:
    print(att(i,"id"))

types=[]
print(len(products))
for i in range(0,len(products)-1):
    obj=products[i]
    if not(hasattr(obj,"type")):
        continue;
    for j in range(0,len(types)):
        if types[j]==obj.type:
            continue;
    #types[len(types)]=obj.type;
    #print(obj)
    #print(dir(obj))
    #print("type:"+obj.type)
    types.append(obj.type)
print(types)
#for product in products:
#    print("文件中含有以下几种产品："+product.is_a())


    
    
    