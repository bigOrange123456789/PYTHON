import xlrd #xls文件的read
book0 = xlrd.open_workbook("out.xlsx")
sheet0 = book0.sheet_by_name("Sheet1")

import xlwt #xls文件的write
book = xlwt.Workbook(encoding='utf-8')  #创建一个excel文件
sheet = book.add_sheet('Sheet1',cell_overwrite_ok=True) #新增一个sheet工作表 
#sh.row_values(1) 获取第一行数据
for i in range(0,sheet0.nrows):
    for j in range(0,sheet0.ncols):
        cell_value = sheet0.cell_value(i,j)
        sheet.write(i,j,float(cell_value))

book.save('out2.xlsx') #保存