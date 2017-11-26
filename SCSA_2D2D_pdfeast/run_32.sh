#!/bin/bash
#SBATCH --account=k1251
#SBATCH --job-name=pfeast_nc32
#SBATCH --output=./sensTV/32/pfeast_nc_32-%j.out
#SBATCH --error=./sensTV/32/errors/pfeast_nc_32-%j.err
#SBATCH --nodes=1
#SBATCH --time=23:59:00


export MKL_NUM_THREADS=32 OMP_NUM_THREADS=32
echo""
echo " MKL_NUM_THREADS=32 OMP_NUM_THREADS=32"
echo""

srun --ntasks=1 --ntasks-per-node=1   ./SCSA_2D2D_dfeast_parallel   -nc 8  --data ../dat_data/img32_Lena32.dat ../dat_data/img32_Lena32.dat  -N 32 -h 0.26 -gm 0.5
srun --ntasks=2 --ntasks-per-node=2 --ntasks-per-socket=1   ./SCSA_2D2D_dfeast_parallel   -nc 8  --data ../dat_data/img32_Lena32.dat ../dat_data/img32_Lena32.dat  -N 32 -h 0.26 -gm 0.5
srun --ntasks=4 --ntasks-per-node=4 --ntasks-per-socket=2   ./SCSA_2D2D_dfeast_parallel   -nc 8  --data ../dat_data/img32_Lena32.dat ../dat_data/img32_Lena32.dat  -N 32 -h 0.26 -gm 0.5

export MKL_NUM_THREADS=16 OMP_NUM_THREADS=16
echo""
echo " MKL_NUM_THREADS=16 OMP_NUM_THREADS=16"
echo""

srun --ntasks=1 --ntasks-per-node=1   ./SCSA_2D2D_dfeast_parallel   -nc 8  --data ../dat_data/img32_Lena32.dat ../dat_data/img32_Lena32.dat  -N 32 -h 0.26 -gm 0.5
srun --ntasks=2 --ntasks-per-node=2 --ntasks-per-socket=1   ./SCSA_2D2D_dfeast_parallel   -nc 8  --data ../dat_data/img32_Lena32.dat ../dat_data/img32_Lena32.dat  -N 32 -h 0.26 -gm 0.5
srun --ntasks=4 --ntasks-per-node=4 --ntasks-per-socket=2   ./SCSA_2D2D_dfeast_parallel   -nc 8  --data ../dat_data/img32_Lena32.dat ../dat_data/img32_Lena32.dat  -N 32 -h 0.26 -gm 0.5

echo""
echo""
echo "                            The run.sh command was:                                 "
echo "##########################################################################################"
cat run_32.sh
