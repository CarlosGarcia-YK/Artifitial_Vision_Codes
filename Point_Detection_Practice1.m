%Read  the picture 

img = imread("./pictures/coin.jpeg");
gray_img = rgb2gray(img);

%Apply mask 
point_mask = [-1 -1 -1;
               -1 8 -1;
              -1 -1 -1];
point_detected = imfilter(gray_img,point_mask);

%Show the points 
figure;
imshow(point_detected,[]);
title('Point detection');