img = imread('pictures\puerta.jpeg');
gray_img = rgb2gray(img);

%Apply masks
horizontal_line = [-1 -1 -1;
               2 2 2;
              -1 -1 -1];
vertical_line = [-1 2 -1;
               -1 2 -1;
              -1 2 -1];

diagonal_line = [-1 -1 2;
               -1 2 -1;
              2 -1 -1];

mdiagonal_line = [2 -1 -1;
               -1 2 -1;
              -1 -1 2];

horizontal_detected = imfilter(gray_img,horizontal_line);
vertical_detected = imfilter(gray_img,vertical_line);
diagonal_detected = imfilter(gray_img,diagonal_line);
mdiagonal_detected = imfilter(gray_img,mdiagonal_line);

%Show the points 
figure;
subplot(2,2,1)
imshow(horizontal_detected,[]);
title('Horizontal Detection');
subplot(2,2,2)
imshow(vertical_detected,[]);
title('Vertical Detection');
subplot(2,2,3)
imshow(diagonal_detected,[]);
title('Diagonal Detection');
subplot(2,2,4)
imshow(mdiagonal_detected,[]);
title('mdiagonal Detection');