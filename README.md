# Regent-CUDA-Reproducer
Reproducer Tests for Regent-CUDA Kernel Generation

## Flags to run the Regent Reproducer
You need to pass the following flags to the Regent compiler to run `reproducer.rg`.
- ```-ll:gpu 1```
- ```-ll:fsize 2048```
- ```-ll:csize 2048```


To run `reproducer_q_derivatives.rg`, you need to pass the following flags:
- ```-ll:gpu 1```
- ```-ll:fsize 5120```
- ```-ll:csize 5120```

## Steps to run the CUDA C Reproducer
- Create a build directory to keep all your object files.
- `cd build`
- `cmake ..`
- `make`  


Pretty standard stuff. We tested and built our object files using NVHPC 22.5 and CUDA 11.7.64.

## Results
The average cudaTask runtimes are as follows:
- `C`: 1.248ms (obtained from NSight Systems stats view).
- `Regent`: 1.380ms (obtained from Legion_prof and a script to read tsv files).
The average qsim runtimes are as follows:
- `C` : 7.245ms
` `Regent` : 9.180ms (9.44ms with conn being `regentlib.array`)
