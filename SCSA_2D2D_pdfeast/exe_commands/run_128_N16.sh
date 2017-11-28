#!/bin/bash
#SBATCH --account=k1251
#SBATCH --job-name=pfeast_N16_nc_128
#SBATCH --output=./sensTV/128/pfeast_N16_nc_128-%j.out
#SBATCH --error=./sensTV/128/errors/pfeast_N16_nc_128-%j.err
#SBATCH --nodes=16
#SBATCH --time=23:59:00

for value in 64 32
do
export MKL_NUM_THREADS=$value OMP_NUM_THREADS=$value
srun --ntasks=160 --ntasks-per-node=10 --ntasks-per-socket=5 ../SCSA_2D2D_dfeast_parallel -MKL_TH $MKL_NUM_THREADS --data ../../dat_data/img64_Lena64.dat ../../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5
srun --ntasks=128 --ntasks-per-node=8 --ntasks-per-socket=4  ../SCSA_2D2D_dfeast_parallel -MKL_TH $MKL_NUM_THREADS --data ../../dat_data/img64_Lena64.dat ../../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5
srun --ntasks=64 --ntasks-per-node=4 --ntasks-per-socket=2 ../SCSA_2D2D_dfeast_parallel -MKL_TH $MKL_NUM_THREADS --data ../../dat_data/img64_Lena64.dat ../../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5
srun --ntasks=16 --ntasks-per-node=1   ../SCSA_2D2D_dfeast_parallel -MKL_TH $MKL_NUM_THREADS --data ../../dat_data/img64_Lena64.dat ../../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5

done

echo""
echo""
echo "                            The run.sh command was:                                 "
echo "##########################################################################################"
cat run_128.sh
echo""

