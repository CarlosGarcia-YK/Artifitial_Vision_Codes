img = imread('pictures\puerta.jpeg');
gray_img = rgb2gray(img);

edges_canny = edge(gray_img,"canny");
edges_sobel = edge(gray_img, "sobel");
edges_prewwitt = edge(gray_img, "prewitt");
edges_log = edge(gray_img,"log");

figure;
subplot(2,2,1);
imshow(edges_canny);
title('Canny Edge');

subplot(2,2,2);
imshow(edges_sobel);
title('Sobel Edge');

subplot(2,2,3);
imshow(edges_prewwitt);
title(' Prewwit Edge');

subplot(2,2,4);
imshow(edges_log);
title(' log Edge');


