import xlrd #xls文件的read
import numpy as np 
import math
bk = xlrd.open_workbook("sum.xlsx")
sh = bk.sheet_by_name("Sheet1")

nrows = sh.nrows#获取行数
ncols = sh.ncols#获取列数

import xlwt #xls文件的write
excelpath =('out.xlsx')  #新建excel文件
workbook = xlwt.Workbook(encoding='utf-8')  #写入excel文件
sheet = workbook.add_sheet('Sheet1',cell_overwrite_ok=True)  #新增一个sheet工作表 
#复制
for i in range(0,nrows):
    for j in range(0,ncols):
        cell_value = sh.cell_value(i,j)
        sheet.write(i,j,cell_value)

#计算个人得分        
for i in range(1,nrows):
    arr=[1,1,1,1]#成员贡献
    if sh.cell_value(i,8):#成员贡献
        str=sh.cell_value(i,8)
        list=str.split("、")
        arr=np.zeros([1,len(list)])
        for k in range(0,len(arr[0]),1):
            arr[0,k]=float(list[k])
            
        arr=arr/sum(sum(arr))
        arr=arr*len(arr[0])
        arr=arr[0]
        print(arr)
    for j in range(9,13):
        cell_value = sh.cell_value(i,j)
        if cell_value:
            cell_value0 = sh.cell_value(i,7)#小组总分
            score=cell_value0*arr[j-9]
            score=math.ceil(score);
            print(score)
            if score>50:
                score=50;
            
            sheet.write(i,j+4,score)
        

workbook.save(excelpath) #保存
