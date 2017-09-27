#echo " Execution N=32, one interval !" && make clean && make all && NUM_MKL_THREADS=36 mpirun -ppn 1 -n 1 ./SCSA_2D2D_dfeast_parallel -nc 8 -m0 500 --data ../dat_data/img32_Lena32.dat ../dat_data/img32_Lena32.dat  -N 32 -h 0.26 -gm 0.5 >> ./sensTV/test32_Mintv.txt && echo  " Execution N=32 is Done with -h 0.26 -gm 0.5 -fe=1 !"
#echo " Execution N=32, two intervals!" && make clean && make all && NUM_MKL_THREADS=36 mpirun -ppn 1 -n 2 ./SCSA_2D2D_dfeast_parallel -nc 8 -m0 200 --data ../dat_data/img32_Lena32.dat ../dat_data/img32_Lena32.dat  -N 32 -h 0.26 -gm 0.5 >> ./sensTV/test32_Mintv.txt && echo  " Execution N=32 is Done with -h 0.26 -gm 0.5 -fe=1 !"
#echo " Execution N=32, two intervals!" && make clean && make all && NUM_MKL_THREADS=18 mpirun -ppn 1 -n 2 ./SCSA_2D2D_dfeast_parallel -nc 8 -m0 200 --data ../dat_data/img32_Lena32.dat ../dat_data/img32_Lena32.dat  -N 32 -h 0.26 -gm 0.5 >> ./sensTV/test32_Mintv.txt && echo  " Execution N=32 is Done with -h 0.26 -gm 0.5 -fe=1 !"
echo " Execution N=64 Two intervals!" && make clean && make all && NUM_MKL_THREADS=36 mpirun -ppn 1 -n 2 ./SCSA_2D2D_dfeast_parallel -nc 8 -m0 1000 --data ../dat_data/img64_Lena64.dat ../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5 >> ./sensTV/test64_2intv_2.txt && echo  " Execution N=64 is Done with -h 0.26 -gm 0.5 -fe=1 !"
echo " Execution N=64 Two intervals!" && make clean && make all && NUM_MKL_THREADS=18 mpirun -ppn 1 -n 2 ./SCSA_2D2D_dfeast_parallel -nc 8 -m0 1000 --data ../dat_data/img64_Lena64.dat ../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5 >> ./sensTV/test64_2intv_2.txt && echo  " Execution N=64 is Done with -h 0.26 -gm 0.5 -fe=1 !"
echo " Execution N=64 Four intervals!" && make clean && make all && NUM_MKL_THREADS=36 mpirun -ppn 1 -n 4 ./SCSA_2D2D_dfeast_parallel -nc 8 -m0 500 --data ../dat_data/img64_Lena64.dat ../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5 >> ./sensTV/test64_2intv_4.txt && echo  " Execution N=64 is Done with -h 0.26 -gm 0.5 -fe=1 !"
echo " Execution N=64 Four intervals!" && make clean && make all && NUM_MKL_THREADS=18 mpirun -ppn 1 -n 4 ./SCSA_2D2D_dfeast_parallel -nc 8 -m0 500 --data ../dat_data/img64_Lena64.dat ../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5 >> ./sensTV/test64_2intv_4.txt && echo  " Execution N=64 is Done with -h 0.26 -gm 0.5 -fe=1 !"
echo " Execution N=64 Four intervals!" && make clean && make all && NUM_MKL_THREADS=9 mpirun -ppn 1 -n 4 ./SCSA_2D2D_dfeast_parallel -nc 8 -m0 500 --data ../dat_data/img64_Lena64.dat ../dat_data/img64_Lena64.dat  -N 64 -h 0.26 -gm 0.5 >> ./sensTV/test64_2intv_4.txt && echo  " Execution N=64 is Done with -h 0.26 -gm 0.5 -fe=1 !"
