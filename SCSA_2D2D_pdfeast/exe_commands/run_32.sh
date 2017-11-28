#!/bin/bash
#SBATCH --account=k1251
#SBATCH --job-name=pfeast32_N1_Nc8
#SBATCH --output=./sensTV/32/pfeast32_N1_Nc8-%j.out
#SBATCH --error=./sensTV/32/errors/pfeast32_N1_Nc8-%j.err
#SBATCH --nodes=1
#SBATCH --time=23:59:00

for value in 64 32 16
do
export MKL_NUM_THREADS=$value OMP_NUM_THREADS=$value	
srun --ntasks=2 --ntasks-per-node=2 --ntasks-per-socket=1 ../SCSA_2D2D_dfeast_parallel -MKL_TH $MKL_NUM_THREADS --data ../../dat_data/img32_Lena32.dat ../../dat_data/img32_Lena32.dat  -N 32 -h 0.26 -gm 0.5
srun --ntasks=4 --ntasks-per-node=4 --ntasks-per-socket=2 ../SCSA_2D2D_dfeast_parallel -MKL_TH $MKL_NUM_THREADS --data ../../dat_data/img32_Lena32.dat ../../dat_data/img32_Lena32.dat  -N 32 -h 0.26 -gm 0.5
srun --ntasks=8 --ntasks-per-node=8 --ntasks-per-socket=4 ../SCSA_2D2D_dfeast_parallel -MKL_TH $MKL_NUM_THREADS --data ../../dat_data/img32_Lena32.dat ../../dat_data/img32_Lena32.dat  -N 32 -h 0.26 -gm 0.5
srun --ntasks=16 --ntasks-per-node=16 --ntasks-per-socket=8 ../SCSA_2D2D_dfeast_parallel -MKL_TH $MKL_NUM_THREADS --data ../../dat_data/img32_Lena32.dat ../../dat_data/img32_Lena32.dat  -N 32 -h 0.26 -gm 0.5

done

echo""
echo""
echo "                            The run.sh command was:                                 "
echo "##########################################################################################"
cat run_32.sh