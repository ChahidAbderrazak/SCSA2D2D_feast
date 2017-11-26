#!/bin/bash
#SBATCH --account=k1251
#SBATCH --job-name=pfeast_nc64
#SBATCH --output=./sensTV/64/pfeast_nc_64-%j.out
#SBATCH --error=./sensTV/64/errors/pfeast_nc_64-%j.err
#SBATCH --nodes=1
#SBATCH --time=23:59:00

export MKL_NUM_THREADS=32 OMP_NUM_THREADS=32
echo""
echo " MKL_NUM_THREADS=32 OMP_NUM_THREADS=32"
echo""

#srun --ntasks=1 --ntasks-per-node=1    ./SCSA_2D2D_dfeast_parallel   -nc 8 --data ../dat_data/img64_Lena64.dat ../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5
srun --ntasks=4 --ntasks-per-node=4  --ntasks-per-socket=2  ./SCSA_2D2D_dfeast_parallel   -nc 8 --data ../dat_data/img64_Lena64.dat ../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5
srun --ntasks=8 --ntasks-per-node=8  --ntasks-per-socket=4  ./SCSA_2D2D_dfeast_parallel   -nc 8 --data ../dat_data/img64_Lena64.dat ../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5
srun --ntasks=16 --ntasks-per-node=16  --ntasks-per-socket=8  ./SCSA_2D2D_dfeast_parallel   -nc 8 --data ../dat_data/img64_Lena64.dat ../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5

export MKL_NUM_THREADS=16 OMP_NUM_THREADS=16
echo""
echo " MKL_NUM_THREADS=16 OMP_NUM_THREADS=16"
echo""

#srun --ntasks=1 --ntasks-per-node=1    ./SCSA_2D2D_dfeast_parallel   -nc 8 --data ../dat_data/img64_Lena64.dat ../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5
srun --ntasks=4 --ntasks-per-node=4  --ntasks-per-socket=2  ./SCSA_2D2D_dfeast_parallel   -nc 8 --data ../dat_data/img64_Lena64.dat ../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5
srun --ntasks=8 --ntasks-per-node=8  --ntasks-per-socket=4  ./SCSA_2D2D_dfeast_parallel   -nc 8 --data ../dat_data/img64_Lena64.dat ../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5
srun --ntasks=16 --ntasks-per-node=16  --ntasks-per-socket=8  ./SCSA_2D2D_dfeast_parallel   -nc 8 --data ../dat_data/img64_Lena64.dat ../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5

echo""
echo "                            The run.sh command was:                                 "
echo "##########################################################################################"
cat run_64.sh
echo""