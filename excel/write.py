import xlrd #xls文件的read
import xlwt #xls文件的write
excelpath =('out.xlsx')  #新建excel文件
workbook = xlwt.Workbook(encoding='utf-8')  #写入excel文件
sheet = workbook.add_sheet('Sheet1',cell_overwrite_ok=True)  #新增一个sheet工作表 
for i in range(0,5):
    for j in range(0,5):
        cell_value = i*10+j;
        sheet.write(i,j,cell_value)
workbook.save(excelpath) #保存