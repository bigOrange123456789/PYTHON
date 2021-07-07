import ifcopenshell
#https://blog.csdn.net/weixin_39904587/article/details/112573503

file = ifcopenshell.open("test.ifc")

def GetIfcClass(ifc_file):#获取文件中所有构件的种类信息
    products = ifc_file.by_type('IfcProduct')
    classList = []
    for product in products:
        type=product.is_a()
        flag=True;
        for t in classList:
            if type==t:
                flag=False;
        if flag:
            classList.append(product.is_a())
    return classList

print(GetIfcClass(file))
