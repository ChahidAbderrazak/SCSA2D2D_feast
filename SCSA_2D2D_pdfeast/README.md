# SCSA2D2D:  Feast3.0 and MKL parallelism


The project is an optimization of the Semi-Classical Signal Analysis (SCSA) method for image denoinsing. The SCSA algorithm is used for [signal and image processing](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=Eigenfunctions%20of%20the%20Schr%C3%B6dinger%20Operator).  it maily  decomposes the input image into the squared eigenfunctions parametrized by `h`, `fe`, and `gm` using the SCSA operator 

This project accelerates the SCSA operator eigenfunctions decomposition using [Feast3.0 solver](https://www.feast-solver.org/) with enabled  MKL parallelism




# COMPILATION
  To build this program's executable file, please follow these steps:

  - **load intel compiler** : make sure you have intel compiler (icc) and intell MKL library installed on your environment, you can simply load these by typing:
  ```
  $  module load intel/15
  ```
  in case a different intel compiler is loaded, please edit the make.inc file to reflect the used compiler. 

  - **build/compile the project**:
  ```
  $ make all
  ```



# DEMO
The code needs to run on cluster. please find `sbatch` queries in the `exe_commands` folder.
Select the execution schemes in `prun_command_Cluster.sh` and run :

```
$ ./prun_command_Cluster.sh
```

    
# IMPORTANT:
Please make sure that :
  - the input image is saved in binary format `.dat` file 
  - the `h` parameter is a positive float value
  - the `gm` and `fe` parameters are integer value
  - The `d` parameter is an integer value divisible by 2

