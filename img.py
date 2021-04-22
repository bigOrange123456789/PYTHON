import cv2   #导入数据包


if __name__ == '__main__': 
    image = cv2.imread("1.png")#读取数据
    cv2.imshow("1.jpg",image)#显示图片
    cv2.waitKey(0)
    cv2.destroyAllWindows()