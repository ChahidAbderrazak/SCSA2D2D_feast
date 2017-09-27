/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! FEAST Driver sparse example !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! solving Ax=eBx with A real and B spd --- A and B sparse matrix!!!
  !!!!!!! by Eric Polizzi- 2009-2012!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

#include <stdio.h> 
#include <stdlib.h> 
#include <sys/time.h>
#include <mpi.h>

#include "feast.h"
#include "feast_sparse.h"
int main(int argc, char **argv) { 
  /*!!!!!!!!!!!!!!!!! Feast declaration variable */
  int  feastparam[64]; 
  float epsout;
  int loop;
  char UPLO='F'; 

  /*!!!!!!!!!!!!!!!!! Matrix declaration variable */
  FILE *fp;
  int size_img=128, tolrnce=2;
  char name[]="../data_matrix/Lena128_A.mtx";

//  char name[]="./data_matrix/Lena128_A.mtx";  
  int  N,N0,nnz;
  float *sa;
  int *isa,*jsa;
  /*!!!!!!!!!!!!!!!!! Others */
  struct timeval t1, t2,tt1,tt2;
  int  i,k,err;
  int  M0,M,info;
  float Emin,Emax,trace;
  float *X; //! eigenvectors
  float *E,*res; //! eigenvalue+residual

/*********** MPI *****************************/
int lrank,lnumprocs,color,key;
int rank,numprocs;
MPI_Comm NEW_COMM_WORLD;
MPI_Init(&argc,&argv);
MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
MPI_Comm_rank(MPI_COMM_WORLD,&rank); 
/*********************************************/

  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!! read input file in csr format!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

  // !!!!!!!!!! form CSR arrays isa,jsa,sa 
  fp = fopen (name, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N0,&nnz);
  sa=calloc(nnz,sizeof(float));
  isa=calloc(N+1,sizeof(int));
  jsa=calloc(nnz,sizeof(int));

  for (i=0;i<=N;i++){
    *(isa+i)=0;
  };
  *(isa)=1;
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d%f\n",&i,jsa+k,sa+k);
    *(isa+i)=*(isa+i)+1;
  };
  fclose(fp);
  for (i=1;i<=N;i++){
    *(isa+i)=*(isa+i)+*(isa+i-1);
  };

  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!! INFORMATION ABOUT MATRIX !!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/*
if (rank==0) {
  printf("sparse matrix -system1- size %.d\n",N);
  printf("nnz %d \n",nnz);
}
*/

 gettimeofday(&tt1,NULL);
 
    /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! FEAST in sparse format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

 //printf("rank= %.d\n",rank);  ; // Added
//printf("numprocs/2-1 = %.d\n",numprocs/2-1); // Added


/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!! Definition of the two intervals 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/


if (rank<=numprocs/2-1) {
    color=1;} // first interval
  else {
    color=2; //! second interval
  }

  //!!!!!!!!!!!!!!!!! create new_mpi_comm_world
  key=0;
 MPI_Comm_split(MPI_COMM_WORLD,color,key,&NEW_COMM_WORLD);
 MPI_Comm_rank(NEW_COMM_WORLD,&lrank);
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  /*!!! search interval [Emin,Emax] including M eigenpairs*/

 

if (size_img==128){

 if (color==1) { // 1st interval
  Emin=(double) -1.7;
  Emax=(double) -0.3;
  M0=8000; // !! M0>=M
  }
 else if(color==2){ // 2nd interval

  Emin=(double) -0.3;
  Emax=(double) 0.0;
  M0=8000; // !! M0>=M
  }

}


if (size_img==64){

if (color==1) { // 1st interval
  Emin= -0.82;
  Emax=-0.3;
  M0=1500; // !! M0>=M
  }
 else if(color==2){ // 2nd interval

  Emin=(double) -0.3;
  Emax=(double) 0.0;
  M0=1850; // !! M0>=M
  }
}





//!!!!!!!!!!!!!!!!!! RUN INTERVALS in PARALLEL
 gettimeofday(&t1,NULL);

  //Emin=(float) -3.2; // -0.01; //-3.2; // -1.6;
  //Emax=(float) 0.0;
  //M0=12300;// 15300; //12300;// !! M0>=M

  /*!!!!!!!!!!!!! ALLOCATE VARIABLE */
  E=calloc(M0,sizeof(float));  // eigenvalues
  res=calloc(M0,sizeof(float));// eigenvectors 
  X=calloc(N*M0,sizeof(float));// residual


  /*!!!!!!!!!!!!  FEAST */
  feastinit(feastparam);
 // feastparam[0]=1;  /*change from default value */
 // feastparam[1]=14; 
 // feastparam[17]=10;
  feastparam[6]=tolrnce; //Stopping convergence criteria for single precision
  //feastparam[15]=2;
  feastparam[8]=NEW_COMM_WORLD;  /*change from default value */

// Estimate :x

//  sfeast_scsrgv(&UPLO,&N,sa,isa,jsa,sb,isa,jsa,feastparam,&epsout,&loop,&Emin,&Emax,&M0,E,X,&M,res,&info);
  sfeast_scsrev(&UPLO,&N,sa,isa,jsa,feastparam,&epsout,&loop,&Emin,&Emax,&M0,E,X,&M,res,&info);

  gettimeofday(&t2,NULL);


  /*!!!!!!!!!! REPORT !!!!!!!!!*/
if (lrank==0) {
//  printf("interval # %d\n",color);
//  printf("FEAST OUTPUT INFO %d\n",info);
//  if (info==0) {
//    printf("*************************************************\n");
//   printf("************** REPORT ***************************\n");
//    printf("*************************************************\n");
    MPI_Comm_size(NEW_COMM_WORLD,&lnumprocs);
//    printf("# of processors %d \n",lnumprocs);
//    printf("SIMULATION TIME %f\n",(t2.tv_sec-t1.tv_sec)*1.0+(t2.tv_usec-t1.tv_usec)*0.000001);
//    printf("# Search interval [Emin,Emax] %.15e %.15e\n",Emin,Emax);
//    printf("# mode found/subspace %d %d \n",M,M0);
//    printf("# iterations %d \n",loop);
//    trace=(double) 0.0;
//    for (i=0;i<=M-1;i=i+1){
//      trace=trace+*(E+i);
//    }
//    printf("TRACE %.15e\n", trace);
//    printf("Relative error on the Trace %.15e\n",epsout );
//    printf("Eigenvalues/Residuals\n");
//    for (i=0;i<=M-1;i=i+1){
//      for (i=0;i<2;i=i+1){
//      printf("   %d %.15e %.15e\n",i,*(E+i),*(res+i));
//    }
//      for (i=M-2;i<M-1;i=i+1){
//      printf("   %d %.15e %.15e\n",i,*(E+i),*(res+i));
//    }
//  }
 printf("Matrix size= %d with  Single precision = %d: Interval # %d = [%.15e %.15e] Error INFO= %d M/M0= %d / %d Relatice error on the Trace %.15e Iter= %d Time  %f ",size_img*size_img,tolrnce,color,Emin,Emax,info,M,M0,epsout,loop,(t2.tv_sec-t1.tv_sec)*1.0+(t2.tv_usec-t1.tv_usec)*0.000001);



}

MPI_Finalize(); /************ MPI ***************/
//gettimeofday(&tt2,NULL);
  //  printf("Sum SIMULATION TIME %f\n",(tt2.tv_sec-tt1.tv_sec)*1.0+(tt2.tv_usec-tt1.tv_usec)*0.000001);

  return 0;

}




