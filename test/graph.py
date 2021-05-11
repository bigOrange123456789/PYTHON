# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import math

def lineChart():
    t=[845.8,849.94,845.08,847.01,847.27,854.52,845.22,851.44,843.79,846.64,847.22,846.24,846.37,847.4,852.89,849.1,850.42,840.18,847.78,844.86,845.09,848.67,850.34,849.79,848.36,849.62,846.17,852.87,853.21,852.72,850.66,849.82,848.95,849.02,850.05,853.09,851.88,850.45,849.57,849.62,846.49,847.3,851.49,846.81,847.94,849.98,847.52,846.83,854.31,854.61,853.96,854.19,852.84,848.31,848.88,855.01,857.76,842.07,849.98,851,848.62,849.37,852.68,844.81,846,850.66,849.85,850.23,853.2,850.48,847.22,848.39,848.53,841.12,844.39,849.6,849.41,851.91,852.06,856.9,847.36,845.79,845.68,850.52,850.25,850.63,849.95,846.92,849.81,852.41,848.63,846.56,845.63,847.68,851.34,855.3,855.06,854.42,843.9,846.44,850.36,847.61,844.89,847.83,849.68,845.66,846.73,850.42,853.19,853.34,846.58,847.4,850.81,849.32,848.43,851.87,849.69,845.17,851.15,847.92,844.9,848.68,847.3,841.71,848.43,846.33,849.87,850.95,854.57,850.59,848.26,848.39,846.57,851.05,848.65,844.13,848.52,843.9,848.97,851.01,845.88,853.94,851.12,849.27,852.69,848.01,850.46,849.33,846.99,843.29,845.7,847.67,849.95,850.7,844.83,851.08,846.97,854.62,850.24,853.65,851.86,841.56,846.53,849.76,851.11,847.58,850.61,849.74,848.22,851.32,854.4,847.69,848.49,847.9,846.57,847.66,847.34,850.39,847.12,843.56,842.64,843.39,850,849.17,847.61,849.79,849.72,848.29,850.26,847.73,850.34,851.55,850.49,851.16,848.2,846.01,849.01,839.6,844.06,851.67,845.46,849.76,849.77,837.83,852.73,848.83,848.01,846.59,848.27,844.96,849.12,843.46,843.2,847.04,846.69,846.78,852.67,846.99,846.44,843.02,850.43,847.55,846.63,846.6,850.77,845.38,850.52,849.45,849.39,847.68,849.14,852.66,852.05,852.55,852.66,845.39,852.08,836.93,848.25,848.8,851.45,848.13,848.47,847.31,847.98,857.58,852.47,845.96,843.14,845.36,838.45,850.51,842.61,847.95,850.45,849.59,846.59,847.66,846.02,848.02,847.74,849.57,853.6,840.93,845.33,847.99,851.92,841.2,843.02,843.49,847.15,846.53,844.66,843.14,854.51,850.28,846.91,851.73,853.92,845.07,855.21,850.34,847.15,845.61,849.22,844.98,849.51,852.6,851.47,846.07,850.65,854.74,849.59,845.72,849.22,851.39,850.39,849.2,844.85,853.23,849.12,849.33,850.25,841.82,849.03,847.93,850.63,855.26,849.34,845.39,845.34,842.79,846.03,843.76,848,845.65,850.22,845.34,849.1,848.4,852.32,850.11,844.7,844.75,840.67,852.75,846.47,850.22,850.01,848.11,851.79,849.54,847.62,844.94,848.06,846.42,847.93,845.42,847.79,845.29,843.41,842.86,849.41,852.49,849.46,851.6,851.24,845.94,847.86,847.55,845.38,846.24,853.49,847.07,850.81,848.44,847.53,848.66,845.71,844.73,845.8]
    t=[441.8,442.7,443.37,445.64,443.88,443.46,443.97,442.65,441.09,441.75,442.85,442.49,442.38,442.96,444.97,445.53,449.03,452.97,457.75,460.11,463.34,467.33,471.58,473.76,476.55,480.68,484.35,488.66,494.59,498.96,503.84,507.72,512.07,516.41,520.22,525.22,528.77,532.04,534.19,538.63,541.01,544.12,546.59,548.61,550.13,550.46,550.63,551.83,553.02,553.37,554.28,554.94,555.23,555.37,555.69,556.2,556.74,556.86,557.1,557.42,557.67,557.94,558.18,558.57,558.67,558.68,558.6,558.86,558.87,557.04,547.04,538.48,530.45,525.96,523.44,524.3,525.14,527.13,528.04,528.69,525.62,525.8,522.5,520.51,518.76,514.85,512.95,511.5,509.22,506.23,506.32,503.94,501.52,499.46,495.91,492.77,489.28,484.49,480.48,477.17,472.75,468.61,465.57,461.68,457.56,454.12,450.21,446.02,440.46,436.59,432.22,426.64,422.37,417.88,414.95,409.44,404.51,400.18,393.87,389.31,384.79,380.07,375.52,370.11,366.79,362.3,360.6,355.15,351.16,345.83,340,335.94,330.65,327.54,322.09,319.01,314.96,311.5,305.29,298.99,295.76,292.07,288.19,283.64,278.74,273.62,267.97,261.66,259.75,255.32,249.21,245.97,240.69,238.81,234.16,228.75,225.24,220.7,218.08,210.34,205.06,203.62,202.11,197.4,194.88,192.18,194.26,195.36,191.78,188.98,186.83,191.7,191.65,190.47,188.41,189.3,190.78,190.25,191.34,189.81,189.41,192.26,195.02,193.81,193.84,192.04,196.19,201.23,202.43,198.93,198.51,198.61,196.5,195.16,192.18,193.71,194.67,194.85,191.37,186.18,182.86,179.08,181.31,179.31,177.86,176.47,170.59,168.22,165.03,159.68,154.14,149.65,142.81,138.34,136.66,134.8,131.89,125.99,122.83,115.49,111,106.92,104.74,101.77,98.48,93.83,87.69,85.43,81.46,76.92,71.89,67.98,64.7,61.8,59.05,57.02,55.26,53.2,52.49,52.67,52.03,52.27,52.11,52.27,52.2,50.54,48.07,44.5,38.64,37.22,38.13,38.25,38.44,39.09,39.64,39.54,40.89,41.07,41.89,42.39,43.44,43.02,43.72,44.83,45.6,46.91,47.81,49.33,51.73,53.32,55.37,57.85,59.94,63.78,66.29,67.83,71.6,74.84,77.41,79.56,82.16,85.06,87.83,91.68,93.88,96.91,101.41,103.32,108.15,111.4,115.04,117.98,117.11,118.37,120.06,121.24,122.27,121.74,124.33,125.62,126.56,129.03,134.02,139.61,145.19,152.57,161.41,166.73,173.5,178.09,182.48,188.79,195.71,199.19,203.54,209.56,215.08,226,233.5,241.42,248.19,256.31,267.66,275.83,281.59,289.68,298.28,306.69,317.05,326.03,332.29,339.19,344.18,352.62,359.96,366.25,372.08,376.03,380.84,387.97,394.11,399.1,404.31,408.64,413.04,416.48,419.53,423.51,425.97,427.13,427.43,428.85,428.94,429.17,432.19,433.86,435.75,436.61,438.02,439.33,441.8]
    
    t=[219.48,220.78,222.13,223.58,224.94,226.08,227.35,228.67,229.93,231.36,232.81,234.05,235.23,236.49,238.01,239.01,240.09,241.27,242.56,243.71,244.97,246.07,247.07,248.08,248.84,249.68,250.58,251.37,252.08,252.88,253.52,254.34,255.5,255.57,256.27,256.74,256.62,256.94,257.44,257.32,257.76,257.78,257.95,257.99,258.45,258.32,257.99,257.18,256.65,256.15,255.95,255.42,254.8,254,254.05,253.72,252.54,252.05,251.73,250.89,250.32,249.6,248.72,248.35,247.55,246.94,246.06,245.06,244.27,232.14,227.38,225.68,223.75,221.63,220.28,218.58,216.87,215.06,213.23,211.26,209.49,207.71,205.94,203.99,201.87,199.9,197.68,195.6,193.83,191.83,189.49,187.44,185.17,182.98,180.22,177.64,175.23,172.35,169.76,166.9,164.49,161.99,159.05,156.02,153.43,150.71,147.65,144.48,141.41,137.91,134.95,131.48,128.1,125.12,121.52,118.15,115.36,112.19,109.4,106.25,102.64,99.94,97.3,95.4,93.44,92.5,91.27,90.6,88.98,88.18,87.29,86.54,86.09,84.73,83.93,84.04,83.49,83.32,83.17,82.82,82.47,82.2,82.64,82.35,83.43,82.74,83.11,84.04,84.12,84.46,84.82,84.09,83.46,83.92,83.59,83.82,83.68,81.67,81.03,81.08,80.37,79.65,78.77,79.63,80.85,82.03,84.24,86.41,87.23,89.58,92.83,95.62,97.4,99.94,103.03,110.87,110.92,112.36,114.37,117.01,118.55,120.71,122.34,119.73,117.99,117.64,117.56,116.43,116.16,114.11,114.38,112.8,112.95,111.87,108.9,105.33,104.38,101.41,101.38,100.57,99.91,97.02,96.39,94.2,92.87,89.9,87.58,83.89,81.43,79.19,77.57,74.88,71.96,67.77,63.92,61.96,58.54,55.23,52.8,49.92,46.77,44.54,42.23,39.56,37.75,35.85,34.16,32.42,31.36,31.02,30.87,30.45,30.09,30.12,30.24,30.62,31.01,31.44,31.76,32.08,32.21,32.48,32.67,32.63,32.42,31.61,29.92,26.84,22.16,21.51,21.46,21.73,21.76,21.91,22.1,22.35,22.49,22.7,22.74,22.9,23.03,22.89,23.13,23.27,23.33,23.34,23.46,23.58,23.84,23.95,24.19,24.24,24.51,24.77,25.13,25.64,25.78,26.3,27.05,27.75,28.37,28.8,29.78,30.52,31.27,32.12,32.92,34.18,35.13,36.1,37.47,38.63,39.89,41.3,42.68,43.78,45.34,47.06,48.38,50.21,51.72,53.53,54.99,57.31,59.04,60.8,62.09,63.79,65.94,67.96,70.46,72.5,75.24,78.09,81.05,84.1,87.51,90.74,93.89,97.87,100.55,105.46,109.73,112.64,117.35,123.22,125.93,129.95,135.57,140.28,143.81,147.46,152.87,158.09,161.09,163.89,170.19,174.06,176.57,178.94,181.72,184.98,187.3,188.95,190.85,192.77,194.55,196.97,198.63,200.35,202.25,203.93,205.76,207.61,209.26,210.85,212.64,214.21,216.01,217.57,219.48]
    
    y=t
    x=range(0,len(y));
    #x = [a*1-100 for a in x]
    
    print(len(y))
    # 绘制图形
    plt.plot(x, y)
#lineChart()


def barChart():
    # 这两行代码解决 plt 中文显示的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    waters = ('碳酸饮料', '绿茶', '矿泉水', '果汁', '其他')
    buy_number = [6, 7, 6, 1, 2]

    plt.bar(waters, buy_number)
    plt.title('男性购买饮用水情况的调查结果')

    plt.show()
    
    
#barChart()



def histogram():
    matrix=[
        [0,5,7,7,5,6,7,8],
        [7,2,6,2,6,5,6,8],
        [6,9,7,7,0,7,2,7],
        [6,6,1,7,6,7,7,5],
        [9,6,0,7,8,2,6,7],
        [2,8,8,2,7,6,7,8],
        [7,3,2,6,1,7,5,8],
        [9,9,5,6,7,7,7,7]
        ]
    n=np.zeros((10))
    print(n)
    for i in range(len(matrix)):
       for  j in range(len(matrix[i])):
             n[matrix[i][j]]+=1;
             
    plt.bar(np.arange(10), n)


def cdf(x,arr):
    all=0
    for i in range(len(arr)):
        if i<=x:
            all+=arr[i];
    return all;
def cdfMin(arr):
    for i in range(len(arr)):
        if arr[i]>0:
            return i;
def t(x,arr):
    c=cdf(x,arr)
    c_min=cdfMin(arr)
    c_max=cdf(len(arr)-1,arr)
    
    t=round(len(arr)*(c-c_min)/(c_max-c_min))
    
    if(t<0):
        t=0;
    elif(t>len(arr)-1):
        t=len(arr)-1;
        
    return int(t);
def histogram2():
    matrix=[
        [0,5,7,7,5,6,7,8],
        [7,2,6,2,6,5,6,8],
        [6,9,7,7,0,7,2,7],
        [6,6,1,7,6,7,7,5],
        [9,6,0,7,8,2,6,7],
        [2,8,8,2,7,6,7,8],
        [7,3,2,6,1,7,5,8],
        [9,9,5,6,7,7,7,7]
        ]
    n=np.zeros((10))
    print(n)
    for i in range(len(matrix)):
       for  j in range(len(matrix[i])):
             n[matrix[i][j]]+=1;
    
    n2=np.zeros((10))
    for i in range(len(n2)):
        j=t(i,n)
        n2[j]+=n[i]
        print(i,j)
    plt.bar(np.arange(10), n2)


histogram()

