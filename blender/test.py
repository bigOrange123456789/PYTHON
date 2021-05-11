class Site(object):
  def __init__(self):
    #self.a
    self.title = 'jb51 js code'
    self.url = 'https://www.jb51.net'
  def list_all_member(self):
    for name,value in vars(self).items():
      print('%s=%s'%(name,value))
  def list_all_member2(self):
    for name in vars(self).items():
      print(name)
      
def myPrint(obj):
    for name in vars(obj).items():
      print(name)      
if __name__== '__main__':
  site = Site()
  myPrint(site)