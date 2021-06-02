import json
'''
file_handle=open('1.txt',mode='w')
print(file_handle)
file_handle.close()
'''

with  open('1.txt') as f:
    value=f.readlines()
    print(value)
    #print(f.readline())
    f.close()

with  open('2.txt','w') as f:
    f.write("test")
    f.close()