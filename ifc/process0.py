#https://blog.csdn.net/weixin_39904587/article/details/112573503
import ifcopenshell
class IFC_File:
   #file ifc模型文件
   def __init__(self, url):
      self.url=url;
      self.file = ifcopenshell.open(url)  
   def _products(self):
       return self.file.by_type('IfcProduct')
   def getAllID(self):
      for i in self._products():
          print(self.att(i,"id"))
   def getClassName(self):#获取文件中所有构件的种类信息
      products = self._products()
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
   def remove(self,obj):
       self.file.remove(obj)
   def save(self):
       #self.file.write(self.url);
       self.file.write("output.ifc");
   @staticmethod
   def att(product,tag):#获取一个构件的某个属性值
      for name,value in vars(product).items():
            if tag==name:
                return value;
      return None;
ifc= IFC_File("test.ifc")
print(ifc.getClassName())

print(dir(ifc.file))
print(len(ifc.file.by_type('IfcProduct')))
while len(ifc.file.by_type('IfcProduct'))>200:
    ifc.remove(ifc.file.by_type('IfcProduct')[0])
ifc.save()

