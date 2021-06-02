#%读图----------------------------------------------
mul=imread('QB8011_MS_128_RGB_1.bmp');
mul=imresize(mul,[512,512]);
pan=imread('QB8011_PAN_512_8_1.bmp');
#%显示原多光谱图--------------------------------------
figure,imshow(mul);
#%预处理----------------------------
mul=double(mul)/255;
pan=double(pan)/255;
#%求相关矩阵--------------------------------
[r c bands]=size(mul);
pixels=r*c;
mul=reshape(mul, [pixels bands]);
correlation=(mul'*mul)/pixels;
#%求特征向量与特征值-------------------------------
[vector value]=eig(correlation);
#%求主分量------------------------------------------------
PC=mul*vector;
PC=reshape(PC,[r,c,bands]);
#%根据第一主分量直方图配准pan后代替第一主分量-------------------------------------
[counts,X]=imhist(PC(:,:,bands)/max(max(PC(:,:,bands))));
pan=histeq(pan,counts);
PC(:,:,bands)=double(pan*max(max(PC(:,:,bands))));
#%重构融合图象---------------------------
PC=reshape(PC,[pixels bands]);
fusion=PC*vector';
fusion=reshape(fusion,[r,c,bands]);
#%显示融合图象------------------------------------
figure,imshow(fusion(:,:,1:3));title('PCA\_RGB');
#http://blog.sina.com.cn/s/blog_d845c4360101oxx5.html