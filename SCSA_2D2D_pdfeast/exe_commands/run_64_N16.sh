#!/bin/bash
#SBATCH --account=k1251
#SBATCH --job-name=pfeast64_N16_Nc8
#SBATCH --output=./sensTV/64/pfeast64_N16_Nc8-%j.out
#SBATCH --error=./sensTV/64/errors/pfeast64_N16_Nc8-%j.err
#SBATCH --nodes=16
#SBATCH --time=23:59:00

for value in 64 32 16
do

export MKL_NUM_THREADS=$value OMP_NUM_THREADS=$value
srun --ntasks=64 --ntasks-per-node=4 --ntasks-per-socket=2 ../SCSA_2D2D_dfeast_parallel -MKL_TH $MKL_NUM_THREADS --data ../../dat_data/img64_Lena64.dat ../../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5
srun --ntasks=32 --ntasks-per-node=2 --ntasks-per-socket=1  ../SCSA_2D2D_dfeast_parallel -MKL_TH $MKL_NUM_THREADS --data ../../dat_data/img64_Lena64.dat ../../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5
srun --ntasks=16 --ntasks-per-node=1 ../SCSA_2D2D_dfeast_parallel -MKL_TH $MKL_NUM_THREADS --data ../../dat_data/img64_Lena64.dat ../../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5

done

echo""
echo "                            The run.sh command was:                                 "
echo "##########################################################################################"
cat run_64.sh
echo""