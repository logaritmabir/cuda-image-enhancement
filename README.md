An implementation of parallel version of gamma correction, histogram equalization, gaussian filter and sobel filter algorithms written in CUDA C++.OpenCV is used for including image into main program and saving output image.

GPU Specs : GTX 1660 TI Mobile (TU116 Core), 1536 CUDA cores, 24 SM.  
CPU Specs : i7-9750H 2.4GHz base clock.

Speedups obtained by averaging 100 times execution on 1024x1024 image.  
gamma correction : 2.29579 times faster  

histogram equalization : 2.06096 times faster  
histogram equalization (shared memory) : 2.53584 times faster  

gaussian filter : 59.6874 times faster  
gaussian filter (shared memory): 49.5195 times faster (bank conflict,gonna fix later)  

sobel filter : 89.1415 times faster  
sobel filter (shared memory) : 76.5014 (bank conflict, gonna fix later)  




