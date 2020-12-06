# glcmwithcuda
In pattern recognition and image processing feature extraction is one of essential parts as it facilitates the subsequent
learning and generalization steps. Feature extraction aims to get the special characteristics of an object that can be applied
to produce the desired information. One of commonly used algorithm for feature extraction is Gray Level Co-occurrence
Matrix (GLCM). Though this algorithm produces a good result, the twelve Haralick texture features requires high
computation if CPUs are used. The emergence of the Graphics Processing Unit (GPU) helps cope with all that requires high
computation and running GLCM calculations in GPU is just a matter of seconds. To benefit greatly from thousands of CUDA
cores, one should be able to appropriately implement GLCM algorithm. GPU for GLCM had been done in several studies
and each offered various improvement in computation times. In this research, we propose another technique of
implementing GLCM algorithm and it shows that the computation time of the twelve Haralick texture features is 450 times
faster than using CPU.
