import cv2  

image = cv2.imread('../1.png')
print(image[0,0,0])
'''
for a in image:
    for b in a:
        for c in b:
            print(c);
            
'''
print("test:",len([1,2,3]))
print("test:",len(image[0,0]))
for i in range(len(image)):
    a=image[i];
    for j in range(len(a)):
        b=a[j];
        for k in range(len(b)):
            c=b[k];
            if(image[i,j,k]<100):
                image[i,j,k]=image[i,j,k]*2;
            
cv2.imshow('cat',image)




from  matplotlib import pyplot as plt
plt.imshow(image,cmap='gray', vmin=0, vmax=255)

