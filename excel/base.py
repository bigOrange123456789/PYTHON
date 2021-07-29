import xlrd #xls文件的read
bk = xlrd.open_workbook("out.xlsx")
sh = bk.sheet_by_name("Sheet1")

nrows = sh.nrows#获取行数
ncols = sh.ncols#获取列数

cell_value = sh.cell_value(1,1)#获取第一行第一列数据 
row_data = sh.row_values(1)#获取第一行数据

import xlwt #xls文件的write
excelpath =('out2.xlsx')  #新建excel文件
workbook = xlwt.Workbook(encoding='utf-8')  #写入excel文件
sheet = workbook.add_sheet('Sheet1',cell_overwrite_ok=True)  #新增一个sheet工作表 
for i in range(0,nrows):
    for j in range(0,ncols):
        cell_value = sh.cell_value(i,j)
        sheet.write(i,j,cell_value)

workbook.save(excelpath) #保存
