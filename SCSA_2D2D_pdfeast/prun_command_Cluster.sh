echo "   Running the SCSA_2D2D_dfeast_parallel on SHAHEEN "
echo " Please make sure the the script is compiled: make all"

cd exe_commands

## ---------------- LIST OF EXECUTIONS  -----------------------
#sbatch run_32.sh && echo " Run SCSA2D2D using an image of size 32X32 where Nh=571, One Node"
#sbatch run_32_N8.sh  && echo " Run SCSA2D2D using an image of size 32X32 where Nh=571 , 8 Nodes"
#sbatch run_64.sh && echo " Run SCSA2D2D using an image of size 64X64 where Nh=2560, one Node"
#sbatch run_64_N16.sh && echo " Run SCSA2D2D using an image of size 64X64 where Nh=2560, 16 Nodes"
# sbatch run_128_N8.sh && echo " Run SCSA2D2D using an image of size 128X128 where Nh=7462, 8 Nodes"
sbatch run_128_N16.sh && echo " Run SCSA2D2D using an image of size 128X128 where Nh=7462, 16 Nodes"
