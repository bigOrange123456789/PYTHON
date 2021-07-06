import xlrd #xls文件的read
import numpy as np 
import math
bk = xlrd.open_workbook("list.xlsx")
sh = bk.sheet_by_name("Sheet1")

nrows = sh.nrows#获取行数
ncols = sh.ncols#获取列数

bk2 = xlrd.open_workbook("contribution.xlsx")
sh2 = bk2.sheet_by_name("Sheet1")

nrows2 = sh2.nrows#获取行数
ncols2 = sh2.ncols#获取列数

import xlwt #xls文件的write
excelpath =('out.xlsx')  #新建excel文件
workbook = xlwt.Workbook(encoding='utf-8')  #写入excel文件
sheet = workbook.add_sheet('Sheet1',cell_overwrite_ok=True)  #新增一个sheet工作表 
#复制list
for i in range(0,nrows):
    for j in range(0,ncols):
        cell_value = sh.cell_value(i,j)
        sheet.write(i,j,cell_value)

#计算个人得分        
for k in range(0,nrows):
    id=sh.cell_value(k,1)
    id=id.split("'")[0]
    id=int(id)
    for i in range(1,nrows2):
        for j in range(9,13):
            cell_value = sh2.cell_value(i,j)
            if cell_value:
                if cell_value==id:
                    sheet.write(k,5,sh2.cell_value(i,j+4))

workbook.save(excelpath) #保存
