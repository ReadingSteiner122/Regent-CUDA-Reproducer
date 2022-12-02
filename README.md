# Regent-CUDA-Reproducer
Reproducer Tests for Regent-CUDA Kernel Generation

## Flags to run the Regent Reproducer
You need to pass the following flags to the Regent compiler to run `reproducer.rg`.
- ```-ll:gpu 1```
- ```-ll:fsize 2048```
- ```-ll:csize 2048```

## Steps to run the CUDA C Reproducer
- Create a build directory to keep all your object files.
- `cd build`
- `cmake ..`
- `make`
Pretty standard stuff. We tested and built our object files using NVHPC 22.5 and CUDA 11.7.64.

## Results
The average cudaTask runtimes are as follows:
- `C`: 1.248ms (obtained from NSight Systems stats view).
- `Regent`: 2.701ms (obtained from Legion_prof and a script to read tsv files).
