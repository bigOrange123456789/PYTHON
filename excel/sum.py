import xlrd #xls文件的read
import numpy as np 
bk = xlrd.open_workbook("in.xlsx")
sh = bk.sheet_by_name("Sheet1")

nrows = sh.nrows#获取行数
ncols = sh.ncols#获取列数

sum=np.zeros([nrows,1])
for i in range(1,nrows):
    for j in range(2,7):
        cell_value = sh.cell_value(i,j)
        sum[i,0]=sum[i,0]+cell_value


import xlwt #xls文件的write
excelpath =('out.xlsx')  #新建excel文件
workbook = xlwt.Workbook(encoding='utf-8')  #写入excel文件
sheet = workbook.add_sheet('Sheet1',cell_overwrite_ok=True)  #新增一个sheet工作表 
for i in range(0,nrows):
    for j in range(0,ncols):
        cell_value = sh.cell_value(i,j)
        sheet.write(i,j,cell_value)
sheet.write(0,ncols,"总分(50分)")
for i in range(1,nrows):
    sheet.write(i,ncols,sum[i,0])
workbook.save(excelpath) #保存
