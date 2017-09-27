/**
 ==============================================================
|   SCSA_2D2D_pFeast Image Analysis [Feast3.0 and MKL solver]         |
 ==============================================================
*
- -* (C) Copyright 2016 King Abdullah University of Science and Technology
Authors:
Abderrazak Chahid (abderrazak.chahid@kaust.edu.sa)
Taous-Meriem Laleg (taousmeriem.laleg@kaust.edu.sa)

Under  assistance of :
 *
Hatem Ltaief (hatem.ltaief@kaust.edu.sa),
dalal.sukkari@kaust.edu.sa
 * 
* Partially Inspired by a program written on 2015 by :  

Ali Charara (ali.charara@kaust.edu.sa)
David Keyes (david.keyes@kaust.edu.sa)
Hatem Ltaief (hatem.ltaief@kaust.edu.sa)


Redistribution  and  use  in  source and binary forms, with or without
modification,  are  permitted  provided  that the following conditions
are met:

* Redistributions  of  source  code  must  retain  the above copyright
* notice,  this  list  of  conditions  and  the  following  disclaimer.
* Redistributions  in  binary  form must reproduce the above copyright
* notice,  this list of conditions and the following disclaimer in the
* documentation  and/or other materials provided with the distribution.
* Neither  the  name of the King Abdullah University of Science and
 Technology nor the names of its contributors may be used to endorse
* or promote products derived from this software without specific prior
* written permission.
*
THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**/

#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <mkl_lapack.h>
#include <mkl_blas.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include "feast.h"
#include "feast_sparse.h"
#include <string>
#include <sys/stat.h>
#include <ctime>
#include <iostream>
const time_t ctt = time(0);

using namespace std;
#define NUM_THREADS 40
#define sqr(a_) ((a_)*(a_))
#define PI 3.141592653589793

#define disp_msg_0 0           // Display The Fundamental Steps n the script
#define disp_msg_1 0	       // Display Subsection
#define disp_msg_2 0	       // Display Sub_Subsection
#define disp_msg_3 0	       // Display Sub_Sub_Subsection
#define disp_msg_4 0	       // Display Sub_Sub_Sub_Subsection
#define disp_matrices 0
#define msg_disp_performance 0
#define show_mkl_eigvlue 0
#define Nb_mkl_eigen 20
#define disp_msg_split 0
#define saveSC_hhD_matrix 0

#define USAGE printf("usage:\t./SCSA_2D2D_dfeast  \n \
\t --data filename_noisy filename_original \n \
\t -N image_dimension\n \
\t -d value_for_d (defaults to 2)\n \
\t -s value_for_s  [used solver: 0)MKL 1)feast](defaults to 0)\n \
\t -h value_for_h (defaults to 0.2)\n \
\t -gm value_for_gm (defaults to 4)\n \
\t -fe value_for_fe (defaults to 1)\n \
\t [-v] verbose_messages_ON\n \
\t --help generate this help message\n \
\n \
please consult the README file for detailed instructions.\n");

// *********************** Type DEFINITION  ***********************
struct Class_SCSA
{	// name of needed data in .dat format
	string  dataFileName;
	string  deltaFileName;
	string  originImage;
	int x_dim, y_dim;   			// matrix size
	int d;              			// SCSA Prameter
	int s;            			    // The used Solver: 0)MKL 1)feast
	float h;
	float gm;
	float fe;
	bool verbose;     				//paramters and flags
	unsigned long buffer_size;
	char jobvl;
	char jobvr;
	int NB_cntour_per_Interval;    // The number of  search sub-interval

	};

struct Feast_param
{
	int show__feast;   				//    	   feastparam[0]=show__feast;  //Print runtime comments on screen (0: No; 1: Yes)
	int Nb_cntour_points; 				//         feastparam[1]=10;  // # of contour points for Hermitian FEAST (half-contour) 8
									//                         		if fpm(15)=0,2, values permitted (1 to 20, 24, 32, 40, 48, 56)
									//                           	if fpm(15)=1, all values permitted
	int eps; 						//		   feastparam[2]=6 // Stopping convergence criteria for double precision 10^-epsilon
	int max_refin_loop; 			//         feastparam[3]=9; // Maximum number of FEAST refinement loop allowed
	int guess_M0; 					//         feastparam[4]=1; // Provide initial guess subspace (0: No; 1: Yes)
	int cnvgce_trace_Resdl; 		//         feastparam[5]=1; /*Convergence criteria (for the eigenpairs in the search interval) 1
									//                            0: Using relative error on the trace epsout i.e. epsout< epsilon
									//                            1: Using relative residual res i.e. maxi res(i) < epsilon
	int intgr_Gauss_Trapez_Zolot; 	//         feastparam[15]=1;  // Integration type for Hermitian (0: Gauss; 1: Trapezoidal; 2: Zolotarev)
	int cntour_ratio;					//         feastparam[17]=1;  // Ellipse contour ratio - fpm(17)/100 = ratio 'vertical axis'/'horizontal axis'
	int Run_Q_M0; 					//         feastparam[13]=1;  // 0: FEAST normal execution; 1: Return subspace Q after 1 contour; 0
									//                         	   2: Estimate #eigenvalues inside search interval
	int M0; 						//  M0 the expeced # of eigenvalues
	double Emin; 					//  the Lower bound of the search interval  [Emin,Emax]
	double Emax; 					//  the Upper bound of the search interval  [Emin,Emax]
	int Scaling_Emin;				// Emin= Scaling_Emin*EMin0
	int M_min;                      // Minimum number of eigenvalue in feast search interval
	int NB_Intervals;               // The number of  search sub-interval
};



/************************    Ali's FUNCTION ROUTINES ************************/
bool readInput(Class_SCSA* Objet_SCSA, float* &Input_Image, float* &original);
template<typename T> bool writeBuffer(Class_SCSA* Objet_SCSA, long int size, T* buffer, string fileName);
bool delta(int n, float* &deltaBuffer, float fex, float feh);
int parse_Objet_SCSA( int argc, char** argv, Class_SCSA *Objet_SCSA, Feast_param *Objet_feast );
double gettime(void);
inline float square(float v){return v*v;}

// *********************** CHAHID's FUNCTION ROUTINES   *********************
bool Display_Objet_SCSA( Class_SCSA *Objet_SCSA );
bool Objet_Feast_param(Feast_param *Objet_feast, int show__feast,int Nb_cntour_points,int eps,int max_refin_loop,int guess_M0,int cnvgce_trace_Resdl,int intgr_Gauss_Trapez_Zolot,int cntour_ratio,int Run_Q_M0, int M0, double Emin, double Emax, int Scaling_Emin, int M_min, int NB_Intervals );
bool Display_Objet_Feast_param( Feast_param *Objet_feast );
int Find_nmbr_Eigvalues(Feast_param *Objet_feast, int N, double* &sa,int* &isa,int* &jsa,double Emin,double Emax);
template<typename T> bool Build_SC_matrix( int MKL_Feast, float hp, int N,float *D, float *V,int *nnz,T * &SC_hhD,int * &iSC_hhD,int * &jSC_hhD,float *max_img, T *Emin,T * &SC_full);
template<typename T> void Disp_matrx(int n,int m,T *Mtx_A,char* name);
template<typename T> T Simp3(T* f, int n, float dx);
template<typename T> bool SCSA_2D2D_Reconstruction(float h, int N2, int M, float gm, float fe,float* Input_Image, float* Ref_Image, float max_img, T* &X,  T* &E ,T* &Output_Image , double* time_Psinnor, float* MSE0_p, float* PSNR0_p, float* MSE_p, float* PSNR_p);
template<typename T> bool MKL_solver( const char jobz, const char uplo, T* a, const int lda, T* &E, int* M, int* info_p);

string Name_SCSA(int N, float h, float gm, int fe, float Emin, float stop_eps, int MKL_Feast, float PSNR, int npoc);

template<typename T> bool SCSA_2D2D(int show_optimal, T Scan_Run, Class_SCSA *Objet_SCSA, Feast_param *Objet_feast, int MKL_Feast, float* PSNR_p, T* Emin0_p, int* M_p, int My_counter );

template<typename T> bool SCSA_2D2D_pFeast(int argc, char** argv, int show_optimal, T Scan_Run, Class_SCSA *Objet_SCSA, Feast_param *Objet_feast, int MKL_Feast, float* PSNR_p, T* Emin0_p, int* M_p, int My_counter);

template<typename T> bool feast_solver(Feast_param *Objet_feast, int N2, int nnz, T* &SC_hhd, int* &iSC_hhd, int* &jSC_hhd, T Emin, T Emax,  T* &X,  T* &E,T* &res, int* M, int* Iter_p,int* info_p);

template<typename T> bool Split_interval_4_pFeast(Feast_param *Objet_feast, int* indxes, int N, T* &sa,int* &isa,int* &jsa,double Emin,double Emax, int M_min, int* M0_list, T* Emin_list );

template<typename T> bool Get_the_Sub_Interval( Feast_param *Objet_feast,int N, T* &sa,int* &isa,int* &jsa,double Emin,double Emax, int M_min, int* &M0_list, T* &Emin_list, int* N_M0_p, int* M0_p );

template<typename T> bool pFeast_Solver(MPI_Comm NEW_COMM_WORLD, Feast_param *Objet_feast, int N2, int nnz, T* &SC_hhd, int* &iSC_hhd, int* &jSC_hhd,  T* &X,  T* &E,T* &res, int* M, int* Iter_p,int* info_p, T* Emin_p, T* Emax_p);

template<typename T> float evaluate_results( int N2, float* &ref,  T* &signal);


// *********************** ONLINE FUNCTION   ***********************
void ftoa(float n, char *res, int afterpoint);   // from: http://www.geeksforgeeks.org/convert-floating-point-number-string/
int intToStr(int x, char str[], int d);			 // from: http://www.geeksforgeeks.org/convert-floating-point-number-string/
bool display_time();                             // from: http://www.cplusplus.com/forum/beginner/73057/


// *********************** MAIN FUNCTION   ***********************

int main(int argc, char* argv[]){

	if(argc < 2){USAGE return 0;}

//####################    Preparing the images and Data   ############################

	Class_SCSA Objet_SCSA;
	Feast_param Objet_feast;
	if(!parse_Objet_SCSA(argc, argv, &Objet_SCSA, &Objet_feast)) return -1;
	int N2=Objet_SCSA.x_dim*Objet_SCSA.x_dim,  stop_eps, MKL_Feast = Objet_SCSA.s;
//	if(!Display_Objet_SCSA( &Objet_SCSA )) return -1;


	int show__feast=1, Nb_cntour_points=Objet_SCSA.NB_cntour_per_Interval , eps=6, max_refin_loop=20, guess_M0=0, cnvgce_trace_Resdl=1, intgr_Gauss_Trapez_Zolot=0, cntour_ratio=100, Run_Q_M0=0, M0=N2,Scaling_Emin=1, M=N2, M_min=Objet_feast.M_min, NB_Intervals=N2/M_min;
	double  Emin=3,Emax=0.0;
//	if(!Objet_Feast_param(&Objet_feast,  show__feast, Nb_cntour_points, eps, max_refin_loop, guess_M0, cnvgce_trace_Resdl, intgr_Gauss_Trapez_Zolot, cntour_ratio, Run_Q_M0, M0, Emin, Emax, Scaling_Emin, M_min, NB_Intervals)) return -1;

	double  Emin0, Scan_Run;              // Scan_Run: 0) Scan for an optimal h , gm, fe    1) Run the SCSA using h=Objet_SCSA.h   , gm =Objet_SCSA.gm, , fe =Objet_SCSA.fe
	int show_optimal=1, scaling_Emin=3,scaling_op=3;
	int scal_min, scal_max, scal_step, eps_min, eps_max, eps_step=1;
	float h_min, h_max, h_step, gm_min, gm_max, gm_step, PSNR_new;
	int fe_min, fe_max, fe_step;
	int scaling_Emin_new=scaling_Emin-2, step_M0_N2=N2/20, My_counter=0,step_M_min=1;

////#################### Scanning  for the optimal hop, gm_op   #########################
//	show_optimal=0;Scan_Run=0.0;M0=N2;
//	h_min=0.01; h_max=1; h_step=0.1; gm_min=0.5; gm_max=1; gm_step=0.5; fe_min=1; fe_max=2; fe_step=1;


//#################### Run with the found Optimal Parameters MKL parallel  #########################
	Scan_Run=1.0;show_optimal=1; MKL_Feast=0; 				// precises the typr of Sparse solver: 0) MKL   1) Feast
	My_counter=0;

//	if( ! SCSA_2D2D(show_optimal, Scan_Run,  &Objet_SCSA, &Objet_feast, MKL_Feast, &PSNR_new, &Emin0, &M,My_counter )) return -1;

	MKL_Feast=1;

////#################### Run with the found Optimal Parameters Fest parallel  #########################
//	Scan_Run=1.0;show_optimal=1; MKL_Feast=1; 				// precises the typr of Sparse solver: 0) MKL   1) Feast
//	My_counter=0;
//
////		for(Objet_feast.M_min=20; Objet_feast.M_min<=350 ; Objet_feast.M_min+=80){
////			for(Objet_feast.Scaling_Emin=1; Objet_feast.Scaling_Emin<=5 ; Objet_feast.Scaling_Emin++){
////				for(Objet_feast.M0=M-3;Objet_feast.M0<=2*M;Objet_feast.M0+=step_M0_N2){
////					for(Objet_feast.cntour_ratio=10;Objet_feast.cntour_ratio<=100;Objet_feast.cntour_ratio+=20){
////						for(Objet_feast.intgr_Gauss_Trapez_Zolot=0; Objet_feast.intgr_Gauss_Trapez_Zolot<=2 ; Objet_feast.intgr_Gauss_Trapez_Zolot++){
////							for(Objet_feast.cnvgce_trace_Resdl=0;Objet_feast.cnvgce_trace_Resdl<=1;Objet_feast.cnvgce_trace_Resdl++){
////								for(Objet_feast.eps=6; Objet_feast.eps<=12 ; Objet_feast.eps+=3){
////									for(Objet_feast.Nb_cntour_points=1;Objet_feast.Nb_cntour_points<=20;Objet_feast.Nb_cntour_points+=4){
////										for(Objet_feast.guess_M0=0; Objet_feast.guess_M0<=1 ; Objet_feast.guess_M0++){
////											for(Objet_feast.max_refin_loop=20;Objet_feast.max_refin_loop<=40;Objet_feast.max_refin_loop+=6){
////												for(Objet_feast.Run_Q_M0=0; Objet_feast.Run_Q_M0<=1 ; Objet_feast.Run_Q_M0++){
////
														if( ! SCSA_2D2D_pFeast(argc, argv,  show_optimal, Scan_Run,  &Objet_SCSA, &Objet_feast, MKL_Feast, &PSNR_new, &Emin0, &M,My_counter )) return -1;
//														My_counter++;
//
////													}
////												}
////											}
////										}
////									}
////								}
////							}
////						}
////					}
////				}
////		}
//
//
//	MPI_Finalize(); /************ MPI ***************/

	 if(disp_msg_0){cout<<endl<<endl<<"==> The End,  loops ="<< My_counter <<endl;}

return 0;
}
// *********************** FUNCTION ROUTINES   ***********************


/*============================= DELTA MATRIX   ==================================
____________________________________________________________________________
| This function returns D matrix, where deltaBuffer is a second order        |
| differentiation matrix obtained by using the Fourier pseudospectral mehtod |
| __________________________________________________________________________ |*/

bool delta(int n, float* &deltaBuffer, float fex, float feh){

	int ex, p;//float ex[n-1];
	float dx, test_bx[n-1], test_tx[n-1], factor = feh/fex;
	factor *= factor;

	if(n%2 == 0){

		p = 1;
		dx = -(PI*PI)/(3.0*feh*feh)-1.0/6.0;

		for(int i = 0; i < n-1; i+=2){
			ex = n-i-1;
			p *= -1;
			test_bx[i] = test_tx[i] = -0.5 * p / square(sin( ex * feh * 0.5));
			ex = n-i-2;
			p *= -1;
			test_bx[i+1] = test_tx[i+1] = -0.5 * p / square(sin( ex * feh * 0.5));
		}
	}
	else{
			dx = -(PI*PI) / (3.0*feh*feh) - 1.0/12.0;

			for(int i = 0; i < n-1; i++){
				ex = n-i-1;
				test_bx[i] = -0.5 * pow(-1,  ex) * cot( ex * feh * 0.5) / (sin( ex * feh * 0.5));
				test_tx[i] = -0.5 * pow(-1, -ex) * cot(-ex * feh * 0.5) / (sin(-ex * feh * 0.5));

				//$            test_bx[i] = -0.5 * pow(-1,  ex) * cos( ex * feh * 0.5) / (sin( ex * feh * 0.5));
				//$            test_tx[i] = -0.5 * pow(-1, -ex) * cos(-ex * feh * 0.5) / (sin(-ex * feh * 0.5));

				}
	}


	unsigned long buffer_size = n * n;
	deltaBuffer = new float[ buffer_size ];
	if(!deltaBuffer){
		cout << "out of memory, could not allocate "<< buffer_size <<" of memory!" << endl;
		return false;
	}

	int lda = n+1;

	for(int r = 0; r < n; r++){
		deltaBuffer[r*lda] = dx * factor;
	}

	float vL, vU;
	for(int r = 1; r < n; r++){

		vL = test_bx[n-r-1] * factor;
		vU = test_tx[n-r-1] * factor;

		for(int c = 0; c < n-r; c++){
			deltaBuffer[r   + c*lda] = vL;
			deltaBuffer[r*n + c*lda] = vU;
		}
	}


	if(disp_msg_3){cout<<endl<<endl<<"\t\t\t--> Delta matrix generated OK!"<<endl;}

	if(disp_matrices){
				cout<<" -->  The  Delta matrix has been read succesfully and stored in deltaBuffer"<<endl;
				Disp_matrx(n,n,deltaBuffer," Delta Matrix");
	}

	return true;
}

// ######################   functions for BLAS  ######################

/*================================ READ DATA    ===============================
____________________________________________________________________________
|         This function  reads the ".dat" file stored in Objet_SCSA          |
|                  to the buffer (table Pointer )  Input_Image                 |
| __________________________________________________________________________ |*/


bool readInput(Class_SCSA* Objet_SCSA, float* &Input_Image, float* &original){

	if(Objet_SCSA->verbose) cout << "reading data from file: " << Objet_SCSA->dataFileName << ", data of size: " << Objet_SCSA->x_dim << "x" << Objet_SCSA->y_dim << endl;


// ******************** load the noisy Image ********************
	FILE* infile;
	if(disp_msg_2){cout<<"\t\t--> Noisy Image : "<< Objet_SCSA->dataFileName.c_str();}
	infile = fopen(Objet_SCSA->dataFileName.c_str(), "rb");
	if(!infile){cout << "could not open input file!" << endl; return 0;}
	Objet_SCSA->buffer_size = Objet_SCSA->x_dim * Objet_SCSA->y_dim;
	if(Objet_SCSA->verbose) cout << "reading buffer of size: " << Objet_SCSA->buffer_size << endl;
	Input_Image = new float[ Objet_SCSA->buffer_size ];
	if(!Input_Image)	{
		cout << "out of memory, could not allocate "<< Objet_SCSA->buffer_size <<" of memory!" << endl;
		fclose(infile);
		return false;
	}

	unsigned long res = fread((void*)Input_Image, sizeof(float), Objet_SCSA->buffer_size, infile);
	if(ferror(infile)){	cout << "error in reading file!" << endl;	}
	if(Objet_SCSA->verbose) cout << "did read " << res << " entries, content not checked though!" << endl;
	fclose(infile);

// ******************** load the original Image ********************
	if(disp_msg_2){cout<<"\t\t--> Reference Image :  "<< Objet_SCSA->originImage.c_str()<<endl;}
	infile = fopen(Objet_SCSA->originImage.c_str(), "rb");
	if(!infile){
		cout << "could not open input file!" << endl;
		return 0;
	}

	Objet_SCSA->buffer_size = Objet_SCSA->x_dim * Objet_SCSA->y_dim;

	if(Objet_SCSA->verbose) cout << "reading buffer of size: " << Objet_SCSA->buffer_size << endl;
	original = new float[ Objet_SCSA->buffer_size ];
	if(!original){
		cout << "out of memory, could not allocate "<< Objet_SCSA->buffer_size <<" of memory!" << endl;
		fclose(infile);
		return false;
	}

	unsigned long res2 = fread((void*)original, sizeof(float), Objet_SCSA->buffer_size, infile);
	if(ferror(infile)){	cout << "error in reading file!" << endl;}
	if(Objet_SCSA->verbose) cout << "did read " << res2 << " entries, content not checked though!" << endl;
	fclose(infile);

// *************************     Displays    *************************

	if(disp_msg_3){cout<<endl<<endl<<"\t\t\t--> Images Loaded OK!"<<endl;}
	if(disp_matrices){
			cout<<" -->  The input images has been read succesfully and stored in Input_Image, Ref_Image.. OK! "<<endl;
			Disp_matrx(Objet_SCSA->x_dim,Objet_SCSA->y_dim,original,"Noisless Image");
			Disp_matrx(Objet_SCSA->x_dim,Objet_SCSA->y_dim,Input_Image,"Noisy Image");
	}

	return true;
}


/*========================= WRITE results  DATA    ===========================
____________________________________________________________________________
|       This function writes the the buffer (table Pointer )  buffer         |
|                   to  "fileName.dat" of Objet_SCSA                         |
| __________________________________________________________________________ |*/

template<typename T> bool writeBuffer(Class_SCSA* Objet_SCSA, long int size, T* buffer, string fileName){

	if(Objet_SCSA->verbose) printf("allocating memory...\n");
	if(Objet_SCSA->verbose) cout << "writeing data to file: " << fileName << ", data of size: " << size << endl;
	FILE* outfile;
	//load matrix data
// ******************** load the data to save ********************
	outfile = fopen(fileName.c_str(), "wb");
	if(!outfile){
		cout << "could not save the output file! please check the directoy:" << fileName<< endl;
		return 0;
	}

	unsigned long res = fwrite((void*)buffer, sizeof(T), size, outfile);

	if(ferror(outfile)){
		cout << "error writing to file!" << endl;
		return 0;
	}
	fclose(outfile);
// *************************     Displays    *************************

	if(disp_msg_2){
		cout<<"\t\t--> Saving data : " << endl;
		cout<<"\t\t--> "<<fileName<<" has been created succefully."<<endl;
	}

return true;
}

/*===================== SCSA Object reconstruction     =======================
____________________________________________________________________________
| This function contains the differents information of about the input data  |
| to process, Moreover it stores also SCSA parameters to use in the Process  |
|              All in type structure object called  Objet_SCSA               |
| __________________________________________________________________________ |*/


int parse_Objet_SCSA( int argc, char** argv, Class_SCSA *Objet_SCSA, Feast_param *Objet_feast )
{
// **************** fill in default values  ****************
	Objet_SCSA->x_dim = 64;
	Objet_SCSA->y_dim = 64;

	Objet_SCSA->d = 2;
	Objet_SCSA->s = 0;
	Objet_SCSA->h = 0.245;
	Objet_SCSA->gm = 0.5;
	Objet_SCSA->fe = 1.0;
	//Objet_SCSA->L = 0.015915494309190;
	Objet_SCSA->dataFileName = "scsa_input64.dat";
	Objet_SCSA->deltaFileName = "baboon_D.dat";
	Objet_SCSA->originImage = "scsa_original64.dat"; ;
	Objet_SCSA->jobvl = 'N';
	Objet_SCSA->jobvr = 'V';
	Objet_SCSA->verbose= false;
	Objet_SCSA->NB_cntour_per_Interval=8;               // The number of  search sub-interval



// **************** FEAST fill in default values  ****************


	Objet_feast->show__feast=0;   						//    	   feastparam[0]=show__feast;  //Print runtime comments on screen (0: No; 1: Yes)
	Objet_feast->Nb_cntour_points=8; 			    	//         feastparam[1]=8;  // # of contour points for Hermitian FEAST (half-contour) 8
																//                         		if fpm(15)=0,2, values permitted (1 to 20, 24, 32, 40, 48, 56)
																//                           	if fpm(15)=1, all values permitted
	Objet_feast->eps=12; 										//		   feastparam[2]=12 // Stopping convergence criteria for double precision 10^-epsilon
	Objet_feast->max_refin_loop=20; 					//         feastparam[3]=20; // Maximum number of FEAST refinement loop allowed
	Objet_feast->guess_M0=0; 							//         feastparam[4]=0; // Provide initial guess subspace (0: No; 1: Yes)
	Objet_feast->cnvgce_trace_Resdl=1; 					//         feastparam[5]=1; /*Convergence criteria (for the eigenpairs in the search interval) 1
																	//		0: Using relative error on the trace epsout i.e. epsout< epsilon
																	//		1: Using relative residual res i.e. maxi res(i) < epsilon
	Objet_feast->intgr_Gauss_Trapez_Zolot=0;			//         feastparam[15]=0;  // Integration type for Hermitian (0: Gauss; 1: Trapezoidal; 2: Zolotarev)
	Objet_feast->cntour_ratio=100;			  			//         feastparam[17]=100  // Ellipse contour ratio - fpm(17)/100 = ratio 'vertical axis'/'horizontal axis'
	Objet_feast->Run_Q_M0=0; 						    //         feastparam[13]=0;  // 0: FEAST normal execution; 1: Return subspace Q after 1 contour; 0
													   	//      	                   	   2: Estimate #eigenvalues inside search interval
	Objet_feast->Scaling_Emin=1;					   	// Emin= Scaling_Emin*EMin0
	Objet_feast->M_min =500;						   	// Minimum number of eigenvalue in feast search interval
	Objet_feast->NB_Intervals=1;                       	// The number of  search sub-interval




	int info;
	for( int i = 1; i < argc; ++i ) {

// **************** each -N fills in next entry of size  *****

	if ( strcmp("-N", argv[i]) == 0 && i+1 < argc ) {
		i++;
		int m, n;
		info = sscanf( argv[i], "%d:%d", &m, &n);

		if ( info == 2 && m > 0 && n > 0 ) {
			Objet_SCSA->x_dim = m;
			Objet_SCSA->y_dim = n;
		}else if ( info == 1 && m >= 0 ) {
				Objet_SCSA->x_dim = m;
				Objet_SCSA->y_dim = m;// implicitly
				}else {
					fprintf( stderr, "error: -N %s is invalid; ensure m >= 0, n >= 0, info=%d, m=%d, n=%d.\n",
					argv[i],info,m,n);
					exit(1);
				}
	}else if ( strcmp("--data", argv[i]) == 0 && i+1 < argc ) {
		Objet_SCSA->dataFileName = argv[++i];
		Objet_SCSA->originImage = argv[++i];

		}else if ( strcmp("--delta", argv[i]) == 0 && i+1 < argc ) {
			Objet_SCSA->deltaFileName = argv[++i];

			}else if ( strcmp("-d",    argv[i]) == 0 && i+1 < argc ) {
				Objet_SCSA->d = atoi( argv[++i] );

				}else if ( strcmp("-s",    argv[i]) == 0 && i+1 < argc ) {
					Objet_SCSA->s = atoi( argv[++i] );

					}else if ( strcmp("-h", argv[i]) == 0 && i+1 < argc ) {
							Objet_SCSA->h = atof( argv[++i] );

							}else if ( strcmp("-fe", argv[i]) == 0 && i+1 < argc ) {
								Objet_SCSA->fe = atof( argv[++i] );

								}else if ( strcmp("-gm", argv[i]) == 0 && i+1 < argc ) {
									Objet_SCSA->gm = atof( argv[++i] );

									}else if ( strcmp("-v",  argv[i]) == 0 ) {
										Objet_SCSA->verbose= true;

										}else if ( strcmp("-nc", argv[i]) == 0 && i+1 < argc ) {
											Objet_SCSA->NB_cntour_per_Interval = atof( argv[++i] );

											}else if ( strcmp("-m0", argv[i]) == 0 && i+1 < argc ) {
												Objet_feast->M_min = atof( argv[++i] );

													}else if ( strcmp("--help", argv[i]) == 0 ) {//----- usage
															USAGE
															exit(0);
													}else {
													fprintf( stderr, "error: unrecognized option %s\n", argv[i] );
													exit(1);

											}
	}

	return 1;
}


/*========================== Time measurement     ============================
____________________________________________________________________________
|       This function returns the actual time in secondes                   |
| __________________________________________________________________________ |*/

double gettime(void)
{
	struct timeval tp;
	gettimeofday( &tp, NULL );

	return tp.tv_sec + 1e-6 * tp.tv_usec;
}


// #########################  CHAHID FUNCTION ROUTINES   #######################


/*======================== Build SC_hhD  matrix   ===================
 __________________________________________________________________
|            This function returns the SC_hhD  Matrix              |
|                    in sparse and dense format                    |
|__________________________________________________________________|*/

template<typename T> bool Build_SC_matrix(int MKL_Feast, float hp, int N,float *D, float *V,int *nnz,T * &SC_hhD,int * &iSC_hhD,int * &jSC_hhD,float *max_img, T *Emin,T * &SC_full){

//M2
//M2 I = eye(n);
//M2 L = sparse(n*n,n*n);         % Reduce the memory starage
//M2 L = kron(I,D1) + kron(D1,I); % The 2D Laplacian operator
//M2 V = V(:) ;
//M2 SC = -h*h*L-diag(V); % The 2D Schr\"odinger operator
	if(disp_msg_2){  cout<<"\t\t--> The SCSA Data preparation has started. "<<endl;}

	long int k,i,j,I,J,bi,bj,indx,nnz0=2*N*N*N-(N*N),Np=N;
	int step=0,N2=N*N;
	float max_img0=0.0,hp2=hp*hp;
	T val;

	if (MKL_Feast==0){
	  SC_full = new T[ N2*N2 ];

	  }else{
		  SC_full = new T[ 1 ];
		  SC_full[1]= (T) 20.0;
	  }

    SC_hhD = new T[ nnz0 ];
    iSC_hhD = new int[ nnz0 ]; 
    jSC_hhD = new int[ nnz0 ];

    if(!iSC_hhD ||!jSC_hhD ||!SC_hhD || !SC_full  ){
		cout << "out of memory, could not allocate "<< (2+2+8)*nnz0 <<" of memory!" << endl;
		return false;

    }else{

    	if(disp_msg_3){cout <<endl<< "\t\t\t--> Memory allocation ~ "<<(2+2+8)*nnz0/1000 <<" Ko of memory OK!" << endl;

    	}
    }
    

	nnz0 = 0;

	for (i = 0; i < N2; i ++){
		for (j = 0; j < N2; j ++) {

			bi=i%Np ;
			bj=j%Np;
			I=i/Np ;
			J=j/Np ;
			val=0.0;

			//cout<<"(i,j)=("<<i<<","<<j<<")"<<"==>> (I,J)=("<<I<<","<<J<<")"<<"==>> (bi,bj)=("<<bi<<","<<bj<<")"<<endl;

			if (i==j ){
						val -= V[i];				// add the iamge value to the diagonal
						if (V[i]>max_img0){max_img0=V[i];}
			}

			if (I==J){val -=  hp2*D[bj*N+bi];}// Add Kron(I,D)
			if (bi==bj){val -=  hp2*D[J*N+I];}// Add Kron(D,I)

			if (val!=0.0){  //				  // Full version
				if (MKL_Feast==0){ SC_full[j*N2+i] = val;
				}

				SC_hhD[nnz0] = val; //				  // Sparse versio
				jSC_hhD[nnz0] = j;
				iSC_hhD[nnz0] = i;
				nnz0++;
			 }
		}
	}

	if(disp_msg_3){cout<<"\t\t\t-->  NON zeros element = "<<nnz0<< " out of "<< N2*N2<<"  => "<<100*(N*N-nnz0)/(N2*N2) <<" % of  Sparsity."<<endl<<endl;}
 
	FILE *fp;

//********************  Save file to SC_hhD3  *******************

//	if(saveSC_hhD_matrix){
//
//		int err;
//		char name[]="SC_hhD3_A.mtx";
//
//		fp = fopen (name, "w");
//		err=fprintf (fp, "%d %d %d\n",N2,N2,nnz0);
//
//		for (i=0; i<nnz0; i++){
//			err=fprintf(fp, "%d %d %f\n", *(iSC_hhD+i)+1,*(jSC_hhD+i)+1, *(SC_hhD+i));
//		}
//
//		fclose(fp);
//
//		if(disp_msg_2){ printf( "\t\t--> The   SChhD   matrix has been generated!\n" );}
//	}


// ********************Prepare the Matrix for FEAST decomposition ********************

  int *isa0,ith=0,ii;
  T h,r,min_lmba=0.0;
  N=N2;

  isa0=(int *)calloc(N2+1,sizeof(int));

  if(!isa0){printf( " --> isa: out of memory, could not allocate  of memory!\n" );return -1;}

  for (i=0;i<=N;i++){
    *(isa0+i)=0;
  };

  *(isa0)=1;
  
	for (k=0;k<=nnz0-1;k++){
//     err=fscanf(fp,"%d%d%lf\n",&i,jsa+k,sa+k);
		i=*(iSC_hhD+k)+1;
		*(jSC_hhD+k)=*(jSC_hhD+k)+1;
		*(iSC_hhD+k)=*(iSC_hhD+k)+1;
		*(isa0+i)=*(isa0+i)+1; // added to convert

// Fin d the estimated min eigenvalues of the sparse matrix

		if (ith==*(jSC_hhD+k)){h=*(SC_hhD+k);}

		if (ith!=i){
			r=*(SC_hhD+k);
			ith=i;
			if (min_lmba > (h-r)){min_lmba=(h-r) ;}
		}else{r=r+*(SC_hhD+k);}

	};

  if(disp_msg_2){printf("\t\t--> The min neg eigenvalue = %f\n",min_lmba);}

  
	for (i=1;i<=N;i++){
//     *(isa+i)=*(isa+i)+*(isa+i-1);
	*(isa0+i)=*(isa0+i)+*(isa0+i-1);// added to convert
	};

	memcpy(iSC_hhD, isa0, (N2+1) * sizeof(int));
	*nnz=nnz0;
	*max_img=(float)max_img0;
	*Emin=3.0*min_lmba;
	delete isa0;

// *************************     Displays    *************************

	if(disp_msg_0){cout<<endl<<endl<<"==> The SCSA data prepared OK!"<<endl;}
	if(disp_matrices){
		cout<<" -->  The SC_hhD  matrix has been read succesfully and stored in SC_hhD"<<endl;
		if (MKL_Feast==0){Disp_matrx( N2,N2, SC_full,"SC_hhD");}
	}

return true;

}

/*======================== Displays   ===================
__________________________________________________________________
| This function Shows the buffer in  Matrix representation         |
| ________________________________________________________________ |*/

template<typename T> void Disp_matrx(int n,int m,T *Mtx_A,char* name) {

	cout<<endl<<endl<<" -----------  Displaying the  "<<  name<<" ( "<< n<<"X"<<m<<") -----------"<<endl<<endl;

	for (int i=0; i<n; i++) {
		for (int j=0; j<m; j++) {

			cout<<" | "<< Mtx_A [i+j*n]  << " | ";
		}

		cout<<endl<<endl;
	}
}


/*=========================== SCSA_2D2D_Reconstruction   =============================
 ___________________________________________________________________________
| This function returns 2D SCSA PROCESS with paramters stored in  Objet_SCSA|
| __________________________________________________________________________|

THE ORIGINAL MATLAB CODE:
==========================================================================
     Recostruction of Lena's image using 2D SCSA formula without using
 separation of variables for $h = ...$ and $\gamma=1$
% http://sipi.usc.edu/database/database

   Author: Zineb Kaisserli and Meriem Laleg
    December 20th, 2014
%=========================================================================*/

template<typename T> bool SCSA_2D2D_Reconstruction(float h, int N2, int M, float gm, float fe,float* Input_Image, float* Ref_Image, float max_img, T* &X,  T* &E ,T* &Output_Image , double* time_Psinnor, float* MSE0_p, float* PSNR0_p, float* MSE_p, float* PSNR_p){

	if(disp_matrices){
		cout<<endl<<endl<<" ==>  Negative eigenspectrum of the SC_hhD."<<endl;
		Disp_matrx(M,1,E,"Absolute values of the Negative Eigenvalues");
		Disp_matrx(M,N2,X,"Negative Eigenfunctions");
	}

//M2     UC = (1/(4*pi))*gamma(gm+1)/gamma(gm+2);
//M2     h2L = h^2/UC;

	float UC = (1.0/(4.0*PI))*tgamma(gm+1.0)/tgamma(gm+2.0);
	float h2L = (h*h)/UC;
	int i, j,N=sqrt(N2), N4=N2*N2;
	T reconst_imge;
	float ERR[N2],reconst_img,MSE0=0.0,PSNR0=0.0,MSE=0.0,PSNR=0.0;

	Output_Image = new T[ N2 ];

//M2  kappa= diag((-temp(ind)).^(gm))
//M2  Nh = length(kappa);

	for(int i=0;i<=M-1;i=i+1){*(E+i)=powf(-*(E+i),gm); }//*(E+i)=ipow(-*(E+i), gm) ;}

//M2  for j = 1:Nh
//M2       I = sqrt((simp(psin(:,j).^2,fe))^(-1));
//M2       psinnor(:,j) = (psin(:,j).*I).^2;
//M2  end    
    
    
	double comp_time0 = gettime();

    for(int i = 0; i < M; i++){

        float I = 1.0/sqrt(Simp3((X+i*(N2)), N2, fe));
//         cout<< "I= "<<I<<endl;
            for(int j = 0; j < N2; j++){

                *(X+i*(N2)+j)=powf(((*(X+i*(N2)+j))*I),2);
            }
        }

	if(disp_matrices){
		Disp_matrx(M,1,E," Eigenvalues ^gm");
		Disp_matrx(M,N2,X,"Normalized Negative Eigenfunctions");
	}

	double time_Psinnor0 = gettime() - comp_time0;

	if(disp_msg_2){cout<<"\t\t--> EigenFunctions Normalization  OK!"<< endl; }

//M2 V1 = (h2L*sum(psinnor*kappa,2)).^(1/(1+gm));

	for(int i = 0; i < N2; i++){
	   reconst_img=0.0;

	   for(int j = 0; j < M; j++){

		   reconst_img+= (*(X+j*(N2)+i)) *(*(E+j));
//             cout<<"lamda("<<j <<")[ "<<*(E+j)<<"] * Eig_function ("<< i <<","<< j<< ") ["<<*(X+j*(N2)+i) <<"]="<< (*(X+j*(N2)+i)) *(*(E+j))<<endl;
            }

	   Output_Image[i]=reconst_img;

//M2  ERR =(abs(img-Output_Image))./max(max(img)).*100
//M2  MSE = mean2((img - Output_Image).^2);
//M2  PSNR = 10*log10(1/MSE);


//             cout<<"Output_Image("<<i <<")= "<<Output_Image[i]<<endl;
//             ERR[i]=((abs(Input_Image[i]-Output_Image[i]))/max_img)*100.0;
//		MSE0 += (Input_Image[i]-Ref_Image[i])*(Input_Image[i]-Ref_Image[i]);
//		MSE += (Ref_Image[i]-Output_Image[i])*(Ref_Image[i]-Output_Image[i]);
}

//// Evaluation PSNR comparing with the  Noisy image
//	MSE0=MSE0/N2;
//	PSNR0 = 10*log10(1.0/MSE0);
//// Evaliation PSNR comparing with the  Denoised image
//	MSE=MSE/N2;
//	PSNR = 10*log10(1.0/MSE);
	comp_time0 = gettime();

	if(disp_matrices){
		cout<<"-->  The  processed image  has been read succesfully computer"<<endl;
		Disp_matrx(N,N,Output_Image," Processed Image");
	}
	*time_Psinnor= time_Psinnor0;
	*MSE0_p=MSE0;
	*MSE_p=MSE;
	*PSNR0_p=PSNR0;
	*PSNR_p=PSNR;


// *************************     Displays    *************************

	if(disp_msg_0){cout<<endl<<endl<<"==>  Image Reconstructed OK!"<<endl;}

	if(msg_disp_performance){
		cout<<endl<<endl<<"************************   2D2D SCSA Diagnosis  ***************************";
		cout<<endl<<" The computations are done with the following results :  "<< endl;
		cout<< "==> Performances with:  h= "<<h<<" ,  gm= "<<gm<< " , fe= "<<fe<<endl;
		cout<<"Original:   MSE = "<<MSE0<<  "  PSNR = "<<PSNR0<<endl;
		cout<<"Denoised:   MSE = "<<MSE<<  "  PSNR = "<<PSNR<<endl<< endl;
	}

return true;
}


/*======================= Display  Image information    =======================

___________________________________________________________________________
|   This function displays the image information stored in  Objet_SCSA     |
| _________________________________________________________________________|*/

bool Display_Objet_SCSA( Class_SCSA *Objet_SCSA )
{

	if(disp_msg_0){
		cout<< " ============== Image Informations =============="<<endl;
		cout<< "|  Dimmension : " <<   Objet_SCSA->x_dim<<  " X "<<   Objet_SCSA->y_dim<< " Pixels."<<endl;
		cout<< "|  Reference image   : "<<    Objet_SCSA->originImage <<" ."<<endl;
		cout<< "|  Input Noisy image : "<<    Objet_SCSA->dataFileName <<" ."<<endl;
		cout<< "|*************** SCSA Parameters *****************"<<endl;
		cout<< "| h== "<<Objet_SCSA->h << "    d="<< Objet_SCSA->d<< "    gm="<<Objet_SCSA->gm << "    fe="<<Objet_SCSA->fe << "."<<endl;
		cout<< "| jobvl= "<< Objet_SCSA->jobvl << "    jobvr="<< Objet_SCSA->jobvr<< "    verbose="<< Objet_SCSA->verbose<< "."<<endl;
		cout<< " ================================================="<<endl;
	}

	return 1;
}



/*============================= SIMPEOMS'S RULE  ============================/*
___________________________________________________________________________
| This function returns the numerical integration of a function f^2 using   |
| Simpson method ot compute the  associated  L^2 -normalized eigenfunctions.|
| _________________________________________________________________________ |*/

template<typename T> T Simp3(T* f, int n, float dx){
//M2      %  This function returns the numerical integration of a function f
//M2      %  using the Simpson method
//M2
//M2      [n,~]=size(f);
//M2      I=1/3*(f(1,:)+f(2,:))*dx;
//M2
//M2      for i=3:n
//M2          if(mod(i,2)==0)
//M2              I=I+(1/3*f(i,:)+1/3*f(i-1,:))*dx;
//M2          else
//M2              I=I+(1/3*f(i,:)+f(i-1,:))*dx;
//M2          end
//M2      end
//M2      y=I;

	T I;
	I = (f[0]*f[0]+f[1]*f[1])*dx/3.0;

	for(int i = 2; i < n; i++){

		if (i % 2==0){
			I = I+(((1.0/3.0*f[i]*f[i])+f[i-1]*f[i-1])*dx);

		}else{

			I = I+(f[i]*f[i]+f[i-1]*f[i-1])*(dx/3.0);
		}
	}

	return I;
}


/*============================= Number of Eigenvalues estimation   ============================/*
______________________________________________________________________________________________
|  This function returns an estimation of the Number of Eigenvalues in interval [Emin, Emax]  |
|_____________________________________________________________________________________________|*/

int Find_nmbr_Eigvalues(Feast_param *Objet_feast, int N, double* &sa,int* &isa,int* &jsa,double Emin,double Emax){

	int  feastparam[64];
	double epsout;
	int loop;
	char UPLO='F';
	int  i,k,err;
	int  M0,M,info;
	double trace, Emin_i=Emin, Emax_i=Emax;
	double *X; //! eigenvectors
	double *E,*res; //! eigenvalue+residual
	// for(i=0;i<64511;i=i+1){ printf("%d   %d  %lf \n",*(isa+i),*(jsa+i),*(sa+i));}
	M0=10;

/*!!!!!!!!!!!!! ALLOCATE VARIABLE */
	E=(double *) calloc(M0,sizeof(double));  // eigenvalues
	if(!E){printf( " --> E out of memory, could not allocate  of memory!\n" );return -1;}

	res=(double *)calloc(M0,sizeof(double));// residual
	if(!res){printf( " --> res out of memory, could not allocate  of memory!\n" );return -1;}

	X=(double *)calloc(N*M0,sizeof(double));// eigenvectors  // factor 2 for complex
	if(!X){printf( " --> X out of memory, could not allocate  of memory!\n" );return -1;}



/*!!!!!!!!!!!!  FInd number of eigenvalues in the interval.  */

	feastinit(feastparam);
//	feastparam[0]=Objet_feast->show__feast;  		//Print runtime comments on screen (0: No; 1: Yes)
//	feastparam[1]=8;//Objet_feast->Nb_cntour_points;  	/* # of contour points for Hermitian FEAST (half-contour) 8
//													if fpm(15)=0,2, values permitted (1 to 20, 24, 32, 40, 48, 56)
//													if fpm(15)=1, all values permitted*/
//	feastparam[2]=Objet_feast->eps; 			// Stopping convergence criteria for double precision 10^-epsilon
	feastparam[13]=2;      			 			// 1: FEAST normal execution; 1: Return subspace Q after 1 contour; 0
												// 2: Estimate #eigenvalues inside search interval

	dfeast_scsrev(&UPLO,&N,sa,isa,jsa,feastparam,&epsout,&loop,&Emin_i,&Emax_i,&M0,E,X,&M,res,&info);

// *************************     Displays    *************************
	if(disp_msg_4){printf("\t\t--> M estimated is  = %d   . OK!.\n",M);}
//	if (1.25*M>N){ M0=N; } else{M0=1.25*M;}//+M/2;
	if (1.05*M>N){ M0=N; } else{M0=1.05*M;}//+M/2;

	if(disp_msg_3){cout<<endl<<"\t\t--> The Estimated  number of eigenvalues in   ["<<Emin <<","<<Emax << "] is M="<<M<<" . The aopted M0="<<M0<<endl<<endl;}

	return M0;
}

/*====================  EIGENVALUES DECOMPOSITION =============================
 __________________________________________________________________________________
 | This function computes all eigenvalues and, optionally, eigenvectors of :        |
 |  -> "a" real symmetric matrix of dmnsn "lda" with Lower triangle  is stored.     |
 |  -> If INFO = 0, "W" contains eigenvalues in ascending order.                    |
 |  -> If JOBZ = 'V', then if INFO = 0,A contains the orthonormal eigenvectors of A |
 |  -> if INFO = 0, WORK(1) returns the optimal LWORK                               |
 |  -> If JOBZ = 'V' and N > 1, LWORK must be at least:  1 + 6*N + 2*N**2.          |
 |  -> If JOBZ  = 'V' and N > 1, LIWORK must be at least 3 + 5*N.                   |
 |  -> INFO is INTEGER                                                              |
 |     = 0:  successful exit                                                        |
 |     < 0:  if INFO = -i, the i-th argument had an illegal value                   |
 |     > 0:  if INFO = i and JOBZ = 'N', then the algorithm failed                  |
 |              to converge; i off-diagonal elements of an intermediate tridiagonal |
 |              form did not converge to zero;                                      |
 |             if INFO = i and JOBZ = 'V', then the algorithm failed to compute an  |
 |               eigenvalue while working on the submatrix lying in rows and columns|
 |               INFO/(N+1) through  mod(INFO,N+1).                                 |
 | ________________________________________________________________________________ |*/

template<typename T> bool MKL_solver( const char jobz, const char uplo, T* a, const int lda, T* &E, int* M,  int* info_p ){

	if(disp_msg_0){cout<<endl<<"==> SCSA_2D2D Eigen Decomposition using  [MKL]"<<endl;}

	int N2=lda, N4=N2*N2, lwork = 1 + 6*N2 + 2*N4, liwork = 3 + 5*N2, iwork[liwork], Nh_size=0, info;
	T lamda[N2];
	T*  work=NULL;
	work=(T *) calloc(lwork,sizeof(T));  // eigenvalues

	if(!work){printf( " --> work out of memory, could not allocate  of memory!\n" );return -1;}

	dsyevd( &jobz, &uplo, &lda, a, &lda, lamda, work, &lwork, iwork, &liwork, &info );


	for(int j=0;j<N2;j++){

		if (lamda[j]<0){
			Nh_size++;
		}
	}

	*M=Nh_size;

	E=(T *) calloc(Nh_size,sizeof(T));  // eigenvalues
	if(!E){printf( " --> E out of memory, could not allocate  of memory!\n" );return -1;}

	for( int j=0;j<Nh_size;j++){*(E+j)=lamda[j]; }

	if(disp_msg_0){cout<<endl<<endl<<"==> Eigen Analysis of the Matrix SC_hhD [MKL dense] OK! Nh="<<Nh_size<<endl;}
	if(disp_msg_2){cout<<"\t\t-->  The  SC_hhD matrix has been decomposed succesfully with Nh="<<Nh_size<< endl;}

	*info_p=info;

	if(info==0){
		if(show_mkl_eigvlue){ Disp_matrx(1,Nb_mkl_eigen,E,"[MKL] Absolute values of the Negative Eigenvalues");}
		return true;
	}
	else{return false;}

}


/*==================  SCSA 2D2D Algorithm with double precision   =========================/*
_______________________________________________________________________________
|  This function runs the SCSA2D2D algorithm weither with MKL or feast solver  |
|                 and shows the obtained performane                            |
|______________________________________________________________________________|*/

template<typename T> bool SCSA_2D2D(int show_optimal, T Scan_Run, Class_SCSA *Objet_SCSA, Feast_param *Objet_feast, int MKL_Feast, float* PSNR_p, T* Emin0_p, int* M_p, int My_counter ){

	int   N = Objet_SCSA->x_dim,info, *iSC_hhd=NULL, *jSC_hhd=NULL,i, j, k, nnz, nb_iter=-2, M = N, N2 = N*M, lda = N;
	float h=Objet_SCSA->h,gm = Objet_SCSA->gm, fe = Objet_SCSA->fe, h2 = h*h;
	float *Input_Image = NULL, *deltaBuffer = NULL,feh = 2.0*PI / N, *Ref_Image = NULL,max_img,MSE0,PSNR0,MSE,PSNR=-100;
	double Emin,Emin0,Emax=0.0,comp_time0 = 0.0,data_prep=0.0, time_eig=0.0, time_reconst=0.0, comp_end=0.0, time_Psinnor; // X: eigenvectors,  E:eigenvalue, res:residual
	T *Output_Image=NULL, *SC_hhd=NULL,*SC_full=NULL,*X,*E,*res;

// ##############  Load the data and Prepare SC_hhD Matrix  ####################
	comp_time0 = gettime();
	if(!readInput(Objet_SCSA, Input_Image,Ref_Image)) return -1;//	        if(!readInput(&Objet_SCSA, Input_Image,Ref_Image)) return -1;
//        if(!writeBuffer(&Objet_SCSA,Objet_SCSA->x_dim*Objet_SCSA->x_dim,Input_Image,"scsa_in_not_processed.dat")) return -1;
	if(!delta(N, deltaBuffer, fe, feh)) return -1;
	if(!Build_SC_matrix(MKL_Feast, h, N, deltaBuffer, Input_Image, &nnz, SC_hhd, iSC_hhd, jSC_hhd, &max_img, &Emin0, SC_full)) return -1;
	data_prep= gettime() - comp_time0;

// ########### Prepare the negative eigenvalues intervalsfor fest solver  ########

	Emin=Emin0*Objet_feast->Scaling_Emin; 				// Using Scalling from estimimatede Emin
	Emax=0.0;

//	Emin=-0.9;
//	Emax=0.0;

// ############## Set the sub Interval  ####################
		Objet_feast->Emin=Emin;
		Objet_feast->Emax=Emax;

// ############## Run the eigen solver  ####################

		if (My_counter==0){

				if (MKL_Feast==0){
					 if (show_optimal==1){cout<< "MKL Double "<<endl<<"\t h \t gm \t fe \t Nb_cntour_points \t eps \t max_refin_loop \t guess_M0 \t cnvgce_trace_Resdl \t intgr_Gauss_Trapez_Zolot \t cntour_ratio \t Run_Q_M0 \t M0 \t Emin \t Emax \t Scaling_Emin \t Found Emin \t Found Emax \t PSNR0 \t PSNR \t MSE0 \t MSE\t Solver time  \t Totale time \t EigenAnalysis % \t #Iterations \t Nh \t  info "<<endl;}

				}else{
					if (show_optimal==1){cout<< "Feast Double"<<endl<<"\t h \t gm \t fe \t Nb_cntour_points \t eps \t max_refin_loop \t guess_M0 \t cnvgce_trace_Resdl \t intgr_Gauss_Trapez_Zolot \t cntour_ratio \t Run_Q_M0 \t M0 \t Emin \t Emax \t Scaling_Emin \t Found Emin \t Found Emax \t PSNR0 \t PSNR \t MSE0 \t MSE\t Solver time  \t Totale time \t EigenAnalysis % \t #Iterations \t Nh \t  info"<<endl;}
				}
		}

	comp_time0 = gettime();

    if (show_optimal==1){
		cout<<std::fixed<<"\t"<< h<< "\t"<< gm<< "\t"<< fe;

    	if(MKL_Feast==1){

    		cout<<std::fixed<< "\t"<< Objet_feast->Nb_cntour_points << "\t"<< Objet_feast->eps << "\t"<< Objet_feast->max_refin_loop << "\t";
    		cout<<std::fixed<< Objet_feast->guess_M0 << "\t"<< Objet_feast->cnvgce_trace_Resdl << "\t"<< Objet_feast->intgr_Gauss_Trapez_Zolot << "\t";
    		cout<<std::fixed<< Objet_feast->cntour_ratio << "\t"<< Objet_feast->Run_Q_M0 << "\t"<< Objet_feast->M0 << "\t"<< Objet_feast->Emin << "\t";
    		cout<<std::fixed<< Objet_feast->Emax << "\t"<< Objet_feast->Scaling_Emin << "\t";

    	}else{	cout<< "\t N/A \t N/A \t N/A \t N/A \t N/A \t N/A \t N/A \t N/A \t N/A \t N/A \t N/A \t N/A \t ";}

    }

	if (MKL_Feast==0){X= SC_full;
	if(!MKL_solver( 'V', 'U', SC_full, N2, E,  &M,&info)) return -1;}//  ### MKL Solver ###
	else{
// ############## Run the MKL solver on  the interval [Emin,Emax]  ####################

		if(!feast_solver( Objet_feast, N2, nnz, SC_hhd, iSC_hhd, jSC_hhd, Emin, Emax, X,  E, res, &M, &nb_iter,&info)) return false;}  //  ### MKL Solver ###

		time_eig= gettime() - comp_time0;

		if ( show_optimal==1)cout<<std::scientific<< *(E+M-1)<< "\t"<< *(E)<< "\t";

// ############## Reconstruct the image  ####################
		comp_time0 = gettime();
		if ( !SCSA_2D2D_Reconstruction( h, N2, M, gm, fe, Input_Image, Ref_Image, max_img, X, E, Output_Image , &time_Psinnor, &MSE0, &PSNR0, &MSE, &PSNR)) return -1;

		time_reconst=gettime() - comp_time0;
		comp_end=data_prep+time_eig+time_reconst;

// ############## Display the performance  ####################

		if(msg_disp_performance){
			cout<< "Number of Negative Eigenvalues = \t"<<M<<endl;
			cout<< "==> Execution Time:   Total Time  = "<<comp_end<<" sec"<< endl;
			cout<< "Data Preparation = \t"<<100.0*data_prep/comp_end<< "%  "<<endl;
			cout<< "EigenAnalysis    = \t"<<100.0*time_eig/comp_end<< "%  "<<endl;
			cout<<"EigenFunction normalization = "<<100.0*(time_Psinnor)/comp_end<< "%  "<<endl;
			cout<<"Image Reconstruction time   = "<<100.0*time_reconst/comp_end<< " % "<<endl<<endl;
			cout<<endl<<endl<<"************************   End of SCSA process   ************************"<<endl;
		}

// ############## Add row to the performance table display ####################

	if ( show_optimal==1){
		cout<< std::fixed<< PSNR0<< "\t"<< PSNR<< "\t"<<MSE0<< "\t"<< MSE<< "\t"<<time_eig<< "\t"<<comp_end<< "\t"<< 100.0*time_eig/comp_end<<"\t";

		if(MKL_Feast==1){ cout<<nb_iter+1<<"\t";}else{cout<<"N/A"<<"\t";}

		cout<<M<<"\t"<< info <<endl;
	}


// ############################ Save output image  ############################
	string name_output;
	name_output=Name_SCSA(N, h, gm, fe , Emin,Objet_feast->eps ,MKL_Feast,PSNR,1);

	if (Scan_Run==1.0){if(!writeBuffer(Objet_SCSA,N2,Output_Image, name_output)) return -1;}

	*M_p=M;
	*PSNR_p=PSNR;
	*Emin0_p=Emin0;

	if (info!=0) {
//		cout<<endl<< " Solver Error !! info code="<< info<<endl;
		return true;}

	else{return true;}

}





/*=====  Parallet  SCSA 2D2D Algorithm with double precision Feast  ============/*
_______________________________________________________________________________
|  This function runs the SCSA2D2D algorithm weither with parallel feast solver  |
|                 and shows the obtained performane                            |
|______________________________________________________________________________|*/

template<typename T> bool SCSA_2D2D_pFeast(int argc, char** argv, int show_optimal, T Scan_Run, Class_SCSA *Objet_SCSA, Feast_param *Objet_feast, int MKL_Feast, float* PSNR_p, T* Emin0_p, int* M_p, int My_counter ){

	int   N = Objet_SCSA->x_dim,info, *iSC_hhd=NULL, *jSC_hhd=NULL,i, j, k, nnz, nb_iter=-2, M = N, N2 = N*M, lda = N;
	float h=Objet_SCSA->h,gm = Objet_SCSA->gm, fe = Objet_SCSA->fe, h2 = h*h;
	float *Input_Image = NULL, *deltaBuffer = NULL,feh = 2.0*PI / N, *Ref_Image = NULL,max_img;
	double Emin,Emin0,Emax=0.0,comp_time0 = 0.0,data_prep=0.0, time_eig=0.0, time_reconst=0.0, comp_end=0.0, time_Psinnor, time_Split=0.0; // X: eigenvectors,  E:eigenvalue, res:residual
	T *Output_Image=NULL, *Vi=NULL, *SC_hhd=NULL,*SC_full=NULL,*X,*E,*res;
	float MSE0,MSE,MSE0_i,MSE_i,PSNR0,PSNR=-100;
	Output_Image=(T *) calloc(N2,sizeof(T));  // Elementary Images for each interval
	float UC = (1.0/(4.0*PI))*tgamma(gm+1.0)/tgamma(gm+2.0);
	float h2L = (h*h)/UC;
// ##############  Load the data and Prepare SC_hhD Matrix  ####################
	comp_time0 = gettime();
	if(!readInput(Objet_SCSA, Input_Image,Ref_Image)) return -1;//	        if(!readInput(&Objet_SCSA, Input_Image,Ref_Image)) return -1;
//        if(!writeBuffer(&Objet_SCSA,Objet_SCSA->x_dim*Objet_SCSA->x_dim,Input_Image,"scsa_in_not_processed.dat")) return -1;
	if(!delta(N, deltaBuffer, fe, feh)) return -1;
	if(!Build_SC_matrix(MKL_Feast, h, N, deltaBuffer, Input_Image, &nnz, SC_hhd, iSC_hhd, jSC_hhd, &max_img, &Emin0, SC_full)) return -1;
	data_prep= gettime() - comp_time0;

// ########### Prepare the negative eigenvalues intervalsfor fest solver  ########
	int Nb_M0=Objet_feast->NB_Intervals, M0, indxes=0,*M0_list = NULL, M_min=Objet_feast->M_min;
	T *Emin_list = NULL;
	Objet_feast->Emin=Emin0;
	Emin=Emin0*Objet_feast->Scaling_Emin;                           // Using Scalling from estimimatede Emin
	Emax=0.0;
	int sum_M=-1,sum_info=0;

//############## Initialize MPI  ####################
	int lrank,lnumprocs,key, color=-1;
	int rank,numprocs;
	int Nb_cntour_max, NB_MPI_interval;

	MPI_Init(&argc,&argv);

//############## Set the sub Interval  ####################
	if(!Get_the_Sub_Interval(Objet_feast, N2, SC_hhd, iSC_hhd, jSC_hhd, Emin, Emax, M_min,  M0_list, Emin_list, &Nb_M0, &M0)) return false;

	time_Split= gettime() - comp_time0;
	Objet_feast->NB_Intervals=Nb_M0;

	if(disp_msg_2){cout<<"\t\t--> Splitting interval is ["<<Emin<<","<<Emax << "] with M0=" << M0 << endl;}
	comp_time0 = gettime();

	int Mi =0;
// ####################### Run the Feast  solver  #############################

//******************* MPI *************************** /
	MPI_Comm NEW_COMM_WORLD;
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

//****************  Sub-Intervals********************* /

//		color=rank/(Objet_feast->Nb_cntour_points/ Objet_feast->NB_Intervals); // Sub-intervals Each group solves one intervals

	Nb_cntour_max=(int) numprocs/Objet_feast->NB_Intervals;

	if (Objet_feast->Nb_cntour_points>Nb_cntour_max){
//	Objet_feast->Nb_cntour_points=(int) numprocs/Objet_feast->NB_Intervals;

	if (Objet_feast->Nb_cntour_points==0){
		cout<<endl<<endl<<"Warnning(2) :The Number  of intervals= " << Objet_feast->NB_Intervals << "  is too much comparing to the number of MPI processes ="<<numprocs<<endl<< "Please, Increase the the MPI processes or Increase the Minimum Number of eigenvalues per interval  -m0 >"<< M0/numprocs<<endl;
		return -1;
	}
//		cout<<endl<<endl<<"Warnning(1) :The Number  of contours has been adjusted to "<< Objet_feast->Nb_cntour_points<<"of intervals of the number"<<endl<<endl;
}

//!!!!!!!!!!!!!!!!! Define the # of MPI processes per  Sub-Intervals  !!!!!!!!!!!!!!!!! /
	NB_MPI_interval=(int) numprocs/Objet_feast->NB_Intervals;
	color=rank/NB_MPI_interval; 									// Sub-intervals Each group solves one intervals

//!!!!!!!!!!!!!!!!! create new_mpi_comm_world for each sub-interval
	key=0;
	MPI_Comm_split(MPI_COMM_WORLD,color,key,&NEW_COMM_WORLD);		//control of subset assignment (nonnegative integer). Processes with the same color are in the same new communicator
	MPI_Comm_rank(NEW_COMM_WORLD,&lrank);
	MPI_Comm_size(NEW_COMM_WORLD,&lnumprocs);

// ******** Define the subintervals [ai, bi] with m0 expeced eigenvalues in it.

	T Emin_i=*(Emin_list + color); 					//define ai
	T Emax_i=*(Emin_list + color+1);				//define bi
	Objet_feast->M0=*(M0_list +color);				// !! M0>=M
	Objet_feast->Emax=Emax_i;						// !! M0>=M
	Objet_feast->Emin=Emin_i;						// !! M0>=M

// ########### Run the Parallel Feast  solver on  the interval [Emin_list,Emax_list]  ##########
//	if (rank==0){Objet_feast->show__feast=1;}else{Objet_feast->show__feast=0;}
//	if (rank==0){display_time() ;}    // Marking the start point



//					if(!pFeast_Solver(NEW_COMM_WORLD, Objet_feast, N2, nnz, SC_hhd, iSC_hhd, jSC_hhd, Emin_i, Emax_i, X,  E, res, &M, &nb_iter,&info)) return false;
	if(!pFeast_Solver( NEW_COMM_WORLD, Objet_feast,  N2,  nnz, SC_hhd, iSC_hhd, jSC_hhd,  X,  E, res, &M, &nb_iter,&info, &Emin_i, &Emax_i )) return false;

//					cout<<endl<<  " IN  [" << Objet_feast->Emin<<","<<Objet_feast->Emax <<"]  Feast has found Nh=" << M << "[" << Emin_i<<","<<Emax_i <<"]  With info code= "<< info<<endl;
	Mi = M;

///////########################################################################

	time_eig= gettime() - comp_time0;
//	cout<< endl<<"the Emin "<<rank << endl;;
//	Disp_matrx(1,10,E," ");


//				if ( show_optimal==1)cout<<std::scientific<< *(E+M-1)<< "\t"<< *(E)<< "\t";


// ############## Reconstruct the image  ####################
	comp_time0 = gettime();
	if ( !SCSA_2D2D_Reconstruction( h, N2, M, gm, fe, Input_Image, Ref_Image, max_img, X, E, Vi , &time_Psinnor, &MSE0_i, &PSNR0, &MSE_i, &PSNR)) return -1;
	time_reconst=gettime() - comp_time0;

	comp_end=data_prep+time_eig+time_reconst+time_Split;


// *********  Reduce all of the local sums into the global sum  *****
	MPI_Reduce(&Mi, &sum_M, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&MSE0_i, &MSE0, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&MSE_i, &MSE, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&info, &sum_info, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(Vi, Output_Image, N2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


// ############## Display the performance  ####################

//	cout<<"- ID:  numprocs= "<< numprocs<<" rank="<<rank<<" lnumprocs= "<< lnumprocs<< " lrank= "<< lrank<<" key= "<<key <<" color= "<< color << " info= "<<info<<endl;

		if (rank==0){

			for(int i = 0; i < N2; i++){
			   Output_Image[i]=powf(h2L*Output_Image[i],(1/(1+gm)));
			}

//			Disp_matrx(1,10,Output_Image,"Vout After Summing all Vi");
			// Compute the Performance using PSNR metric
			PSNR0=evaluate_results(  N2, Ref_Image, Input_Image);
			PSNR=evaluate_results(  N2, Ref_Image, Output_Image);

			if (show_optimal==1){cout<< "Parallel Feast Double"<<endl<<"h \t gm \t fe \t M0 \t Emin \t Emax \t Nb_cntour_points \t cntour_ratio \t eps \t intgr_Gauss_Trapez_Zolot \t PSNR0 \t PSNR \t time_eig \t time_Split \t Totale time \t nb_iter  \t Nh % \t info \t Scaling_Emin \t  max_refin_loop \t guess_M0 % \t cnvgce_trace_Resdl \t Run_Q_M0"<<endl;}

			comp_time0 = gettime();

	// ############## Add row to the performance table display ####################
			cout<<std::fixed<< h<< "\t"<< gm<< "\t"<< fe <<"\t"<< Objet_feast->M0 << "\t"<< Objet_feast->Emin << "\t"<< Objet_feast->Emax ;
			cout<<std::fixed<< "\t"<< Objet_feast->Nb_cntour_points <<  "\t"<< Objet_feast->cntour_ratio <<  "\t"<< Objet_feast->eps << "\t"<< Objet_feast->intgr_Gauss_Trapez_Zolot ;
			cout<< std::fixed<< "\t"<< PSNR0<< "\t"<< PSNR<<  "\t"<<time_eig<< "\t"<<time_Split << "\t"<< comp_end<<"\t"<<nb_iter+1<<"\t"<<sum_M<<"\t"<< sum_info;
			cout<< std::fixed << "\t" <<Objet_feast->Scaling_Emin <<  "\t"<< Objet_feast->max_refin_loop << "\t"<< Objet_feast->guess_M0 << "\t"<< Objet_feast->cnvgce_trace_Resdl << "\t"<< Objet_feast->Run_Q_M0 << endl;

	// ############################ Save output image  ############################
		string name_output;
		name_output=Name_SCSA(N, h, gm, fe , Emin,Objet_feast->eps ,MKL_Feast,PSNR,numprocs );

		if(!writeBuffer(Objet_SCSA,N2,Output_Image, name_output)) return -1;


	}


	*M_p=sum_M;
	*PSNR_p=PSNR;
	*Emin0_p=Emin0;

	MPI_Finalize(); /************ MPI ***************/

	return true;


}


/*==========================Build the name for the output fils and display   ============================/*
 __________________________________________________________________________
|  This function returns the name for the denoised image,                  |
|                 and the displays for performance                         |
|__________________________________________________________________________|*/

string Name_SCSA(int N, float h, float gm, int fe, float Emin, float stop_eps, int MKL_Feast, float PSNR, int npoc){
	char name_proc[3], name_N[3], name_h[5],name_gm[5],name_fs[5],name_Emin[10],name_stop_eps[5], name_psnr[10];
	ftoa(npoc, name_proc, 0);
	ftoa(N, name_N, 0);
	ftoa(h, name_h, 2);
	ftoa(gm, name_gm, 2);
	ftoa(fe, name_fs, 0);
	ftoa(abs(Emin), name_Emin, 2);
	ftoa(stop_eps, name_stop_eps, 0);
	ftoa(abs(PSNR), name_psnr, 2);

	string name_full ="./Denoised_Images";
	const int dir_err = mkdir("./Denoised_Images", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);// to save the denoised images with the optimal parameters for each image

	name_full +="/d_img_N_";
	name_full +=name_N;

	if (MKL_Feast==0){
		name_full+= "_mkl_PSNR_";
	}else{name_full+= "_feast_PSNR_";}

	if(PSNR<0)	name_full+= "neg";

	name_full +=name_psnr;
	name_full +="_h_";
	name_full +=name_h;
	name_full +="_gm_";
	name_full +=name_gm;
	name_full +="_fs_";
	name_full +=name_fs;
	name_full +="_Emin_";
	name_full +=name_Emin;
	name_full +="_eps_";
	name_full +=name_stop_eps;
	name_full +="_MPI_";
	name_full +=name_proc;
	name_full +=".dat";

    return name_full;
}





/*===================== feast Object reconstruction     =======================
____________________________________________________________________________
| This function contains the differents information of about the feast      |
| 			parametersparameters to use in the Process               		|
| _________________________________________________________________________ |*/


bool Objet_Feast_param(Feast_param *Objet_feast, int show__feast,int Nb_cntour_points,int eps,int max_refin_loop,int guess_M0,int cnvgce_trace_Resdl,int intgr_Gauss_Trapez_Zolot,int cntour_ratio,int Run_Q_M0, int M0, double Emin, double Emax, int Scaling_Emin, int M_min, int NB_Intervals ){


	Objet_feast->show__feast=show__feast;   			//    	   feastparam[0]=show__feast;  //Print runtime comments on screen (0: No; 1: Yes)
	Objet_feast->Nb_cntour_points=Nb_cntour_points; 	//         feastparam[1]=8;  // # of contour points for Hermitian FEAST (half-contour) 8
														//                         		if fpm(15)=0,2, values permitted (1 to 20, 24, 32, 40, 48, 56)
														//                           	if fpm(15)=1, all values permitted
	Objet_feast->eps=eps; 								//		   feastparam[2]=12 // Stopping convergence criteria for double precision 10^-epsilon
	Objet_feast->max_refin_loop=max_refin_loop; 		//         feastparam[3]=20; // Maximum number of FEAST refinement loop allowed
	Objet_feast->guess_M0=guess_M0; 					//         feastparam[4]=0; // Provide initial guess subspace (0: No; 1: Yes)
	Objet_feast->cnvgce_trace_Resdl=cnvgce_trace_Resdl; //         feastparam[5]=1; /*Convergence criteria (for the eigenpairs in the search interval) 1
																	//		0: Using relative error on the trace epsout i.e. epsout< epsilon
																	//		1: Using relative residual res i.e. maxi res(i) < epsilon
	Objet_feast->intgr_Gauss_Trapez_Zolot=intgr_Gauss_Trapez_Zolot;//         feastparam[15]=0;  // Integration type for Hermitian (0: Gauss; 1: Trapezoidal; 2: Zolotarev)
	Objet_feast->cntour_ratio=cntour_ratio;			   //         feastparam[17]=100  // Ellipse contour ratio - fpm(17)/100 = ratio 'vertical axis'/'horizontal axis'
	Objet_feast->Run_Q_M0=Run_Q_M0; 				   //         feastparam[13]=0;  // 0: FEAST normal execution; 1: Return subspace Q after 1 contour; 0
													   //      	                   	   2: Estimate #eigenvalues inside search interval
	Objet_feast->M0=M0;								   //  M0 the expeced # of eigenvalues
	Objet_feast->Emin=Emin;  						   //  the Lower bound of the search interval  [Emin,Emax]
	Objet_feast->Emax=Emax;  						   //  the Upper bound of the search interval  [Emin,Emax]
	Objet_feast->Scaling_Emin=1;					   // Emin= Scaling_Emin*EMin0
	Objet_feast->M_min =500;						   // Minimum number of eigenvalue in feast search interval
	Objet_feast->NB_Intervals=1;                       // The number of  search sub-interval

	return true;
}


/*======================= Display  Image information    =======================

___________________________________________________________________________
|   This function displays the feast paramters in  Objet_feast             |
| _________________________________________________________________________|*/

bool Display_Objet_Feast_param( Feast_param *Objet_feast )
{
    cout<< " ============== Feast parameters are =============="<<endl;
    cout<< "|  show__feast \t\t\t= " <<   Objet_feast->show__feast<<endl<<"|  Nb_cntour_points \t\t= "<<   Objet_feast->Nb_cntour_points <<endl<<"|  eps \t\t\t\t= "<<   Objet_feast->eps<<endl;
    cout<< "|  max_refin_loop \t\t= "<<   Objet_feast->max_refin_loop<<endl<<"|  guess_M0 \t\t\t= "<<   Objet_feast-> guess_M0<<endl<<"|  cnvgce_trace_Resdl \t\t= "<<Objet_feast-> cnvgce_trace_Resdl<< endl;
    cout<< "|  intgr_Gauss_Trapez_Zolot \t= "<<   Objet_feast-> intgr_Gauss_Trapez_Zolot<<endl<<"|  cntour_ratio \t\t= "<<   Objet_feast->cntour_ratio <<endl<<"|  Run_Q_M0 \t\t\t= "<<   Objet_feast->Run_Q_M0<<endl;
    cout<< "|  M0 \t\t\t\t= "<<   Objet_feast->M0 <<endl<<"|  Emin \t\t\t= "<<   Objet_feast->Emin <<endl<<"|  Emax \t\t\t= "<<   Objet_feast->Emax<<endl<<"|  Scaling_Emin \t\t= "<<   Objet_feast->Scaling_Emin<<endl;
    cout<< "|  NB_Intervals \t\t= "<<   Objet_feast->NB_Intervals <<endl;
    cout<< " ================================================="<<endl;

return true;
}




/*============================= Split feast    ============================/*
_____________________________________________________________________________
|  This elementry function Splits the [Emin,Emax] into M0_min  intervals     |
|  and returns a vector of sub-intervals  Emin_list and coreponding     	 |
|  		       number of eigenvalues M0_list in each interval		 	     |
|____________________________________________________________________________|*/


template<typename T> bool Split_interval_4_pFeast(Feast_param *Objet_feast, int* indxes, int N, T* &sa,int* &isa,int* &jsa,double Emin,double Emax, int M_min, int* M0_list, T* Emin_list ){
		if (Emin>Emax){
			T Ein=Emax;
			Emax=Emin;Emin=Ein;
		}

		T Ecenter, A, B;
		T Ecenter0, Emin0, Emax0 ;
		int  M, m0, m1=m0+10, m0_plus, m0_minus;
		int flag_border=0;                  // 0) no error    1) Minimum M0     2) many eigenvalues are concentrated arround the  the half borders
			A=Emin; B=Emax ;


			M= Find_nmbr_Eigvalues(Objet_feast, N,  sa, isa, jsa, A, B);
//			if(disp_msg_2){cout<<"\t\t--> Splitting interval is ["<<A <<","<<B << "] with M=" << M << endl;}

			if(M>1.7*M_min ){

				m0=M;m1=M;
				m0_plus=(M/2)+M%2 +1;
				m0_minus=(M/2);

				if(disp_msg_3){cout<<"\t\t\t--> Search for C such that:   "<<m0_minus <<"<  m0 < "<<m0_plus << endl;}

				while( flag_border==0  &&( m0_minus>m0 ||m0>m0_plus || m0_minus>m1 ||m1>m0_plus )) {

//					if(m0>m0_plus ||m1< m0_minus ){
					if(m0>m0_plus  ){
						B=Ecenter;
						Ecenter=(A+B)/2;
					}

//					if(m0< m0_minus || m1>m0_plus ){
					if(m0< m0_minus  ){

						A=Ecenter;
						Ecenter=(A+B)/2;
					}


//					if(disp_msg_2){cout<<".";}

					Ecenter0=Ecenter; Emin0=Emin; Emax0=Emax;
					m0= Find_nmbr_Eigvalues(Objet_feast, N,  sa, isa, jsa, Emin0, Ecenter0);
					m1= Find_nmbr_Eigvalues(Objet_feast, N,  sa, isa, jsa, Ecenter0, Emax0);
					if(disp_msg_4){cout<<endl<< "\t\t\t\t A= "<<A<< " , B=" << B<< " The found C=" << Ecenter<< " ==> m0 =" << m0 << " , m1 =" << m1<<endl;}


					if (abs(100*(A-B))<1){
						flag_border=2;
						if(disp_msg_4){cout<<"\t\t\t\t Error code (2) : In the intervals   [A,B] = ["<<Emin <<","<<Emax << "], The  " << abs(m0-m1)<<" eigenvalues are concentrated arround the  the half borders   ["<<A <<","<<B << "] " <<endl;}

					}


				}

			m0=Find_nmbr_Eigvalues(Objet_feast, N,  sa, isa, jsa, Emin, Ecenter) ;
			m1= Find_nmbr_Eigvalues(Objet_feast, N,  sa, isa, jsa, Ecenter,Emax );

			if(disp_msg_3){
				cout<< endl<<"\t\t\t For [A,B] ["<<Emin <<","<<Emax << "] ==> C"<<*indxes <<"= "<<  Ecenter<<endl;
				cout<< "\t\t\t For [A,C] m0 =" <<m0 <<endl;
				cout<< "\t\t\t For [C,B] m0 =" <<m1 <<endl;
			}

				if (m0<M && m1<M){


					Split_interval_4_pFeast( Objet_feast, indxes, N,  sa, isa, jsa, Emin, Ecenter,M_min, M0_list,  Emin_list);
					Split_interval_4_pFeast( Objet_feast, indxes, N,  sa, isa, jsa, Ecenter, Emax, M_min,M0_list,  Emin_list);
				}else if (m0>=M){

						*(Emin_list + *indxes)=Emin;
						*(M0_list + *indxes)=0;
						*indxes=*indxes+1;
						Emin=Ecenter;

						Split_interval_4_pFeast( Objet_feast, indxes, N,  sa, isa, jsa, Emin, Emax,M_min, M0_list,  Emin_list);

					}else{

					  	Emax=Ecenter;
						Split_interval_4_pFeast( Objet_feast, indxes, N,  sa, isa, jsa, Emin, Emax, M_min, M0_list,  Emin_list);

						*(Emin_list + *indxes)=Ecenter;
						*(M0_list + *indxes)=0;
						*indxes=*indxes+1;

					}


			}else{

				flag_border=1;
				Ecenter=Emax;
				if(disp_msg_4){cout<<"\t\t\t\t Error code (1) : The Number of eigenvalue in  [A,B] = ["<<Emin <<","<<Emax << "] is M0="<< M<<" which is  less then Mmin="<<M_min<< endl;}


				*(Emin_list + *indxes)=Emin;
				*(M0_list + *indxes)=M;

				if(disp_msg_2){cout<<  "\t\t\t--> The sub-interval C"<<*indxes <<" ==>   ["<<Emin <<","<<Emax << "] contains  M0="<< M << endl;}
				*indxes=*indxes+1;
				return true;

			}

return true;
}




/*============================= Split the [Emin,Emax]    ============================/*
_____________________________________________________________________________
|  This function Splits the [Emin,Emax] into M0_min  intervals and and       |
|  returns a vector of sub-intervals  Emin_list and coreponding number of    |
|  					eigenvalues M0_list in each interval		  		     |
|____________________________________________________________________________|*/


template<typename T> bool Get_the_Sub_Interval(Feast_param *Objet_feast, int N, T* &sa,int* &isa,int* &jsa,double Emin,double Emax, int M_min, int* &M0_list, T* &Emin_list, int* N_M0_p , int* M0_p){
		double comp_time00 = gettime();

		int M0, N_M0, indxes=0;

		M0=Find_nmbr_Eigvalues(Objet_feast, N,sa, isa, jsa, Emin, Emax);

		N_M0=M0/M_min ;
		Emin_list = new T[N_M0+3];
		M0_list= (int*) calloc(N_M0+3, sizeof(int));

		if(disp_msg_1){cout <<endl<< "\t--> Splitting the  interval ["<<Emin<<","<<Emax << "] that contains  "<< M0<< " Estimated eigenvalues of minimal size M0~"<< M_min << ".  Please wait ... "<<endl<< endl;}

		if(!Split_interval_4_pFeast(Objet_feast, &indxes, N, sa, isa, jsa, Emin, Emax, M_min,  M0_list, Emin_list)) return false;

//		if(disp_msg_1){cout << "\t\t-->The inteval has been  split into ~ "<< N_M0<< " Sub-intervals of minimal size M0~"<< M_min << endl;}

		*(Emin_list + indxes)=Emax;
		indxes++;

		double time_spliting= gettime() - comp_time00;

		if(disp_msg_split){

			cout<<endl<< "\t-->Itervale Splitting time = "<< time_spliting<< " sec"<<endl;
			cout<<endl<<endl<<"\t-->The Found Sub-intervals "<<endl<<"\t|";
			for(int k=0; k<indxes;k++){
			cout<< *(Emin_list + k);
			if (k<indxes-1)cout<<" -- ";
			}

			cout<<"|"<<endl<<endl<<"\t-->The number of Eigenvalues in each Sub-intervals "<<endl<<"\t  |";
			for(int k=0; k<indxes-1;k++){
			cout<< *(M0_list + k);
			if (k<indxes-2)cout<<" -- ";
			}
			cout<<"|"<<endl;
		}



		*N_M0_p=indxes-1;
		*M0_p= M0;

		if (N_M0==0){
					*(Emin_list)=Emin;
					*(Emin_list+1)=Emax;
					*N_M0_p=0;}

return true;
}


/*============================= double FEAST SOLVER   ============================/*
 __________________________________________________________________________
|  This function returns double sparse decompistion of the matrix SC_hhd   |
|__________________________________________________________________________|*/

template<typename T> bool feast_solver(Feast_param *Objet_feast, int N2, int nnz, T* &SC_hhd, int* &iSC_hhd, int* &jSC_hhd, T Emin, T Emax,  T* &X,  T* &E,T* &res, int* M, int* Iter_p,int* info_p){

		if(disp_msg_0){cout<<endl<<"==> SCSA_2D2D Eigen Decomposition using  [Feast 3.0]"<<endl;}

		if (Emin>Emax){
				T Ein=Emax;
				Emax=Emin;Emin=Ein;
		}
/* ############################ Feast declaration variable  ############################ */
		int  feastparam[64];
		T epsout;
		int loop;
		char UPLO='F';

/*!!!!!!!!!!!!!!!!! Others */
		struct timeval t1, t2;
		int  i, j, k, err;
		int  M0,NB_eig, info;
		double trace;

		//     for (int i=0; i<innz; i++){
		//         fprintf(stdout, "%d %d %f\n", *(iSC_hhd+i)+1,*(jSC_hhd+i)+1, *(SC_hhd+i));
		//       }


		if(disp_msg_2){
			cout<<endl<<endl<<" \t\t--> Feast Eigen Analysis of SC_hhD Matrix"<< endl;
			printf("\t\tsparse matrix size %.d\n",N2);
			printf("\t\tnnz %d \n",nnz);
		}

		gettimeofday(&t1,NULL);
		M0=Objet_feast->M0; // !! M0>=M
/*!!!!!!!!!!!!! ALLOCATE VARIABLE */
		E=(T *) calloc(M0,sizeof(T));  // eigenvalues
		if(!E){printf( " --> E out of memory, could not allocate  of memory!\n" );return -1;}

		res=(T *)calloc(M0,sizeof(T));// residual
		if(!res){printf( " --> res out of memory, could not allocate  of memory!\n" );return -1;}

		X=(T *)calloc(N2*M0,sizeof(T));// eigenvectors  // factor 2 for complex
		if(!X){printf( " --> X out of memory, could not allocate  of memory!\n" );return -1;}

		gettimeofday(&t1,NULL);
		feastinit(feastparam);
		feastparam[0]=Objet_feast->show__feast;  		//Print runtime comments on screen (0: No; 1: Yes)
		feastparam[1]=Objet_feast->Nb_cntour_points;  	/* # of contour points for Hermitian FEAST (half-contour) 8
														if fpm(15)=0,2, values permitted (1 to 20, 24, 32, 40, 48, 56)
														if fpm(15)=1, all values permitted*/
		feastparam[15]=Objet_feast->intgr_Gauss_Trapez_Zolot;  // Integration type for Hermitian (0: Gauss; 1: Trapezoidal; 2: Zolotarev)
		feastparam[17]=Objet_feast->cntour_ratio; 	 	// Ellipse contour ratio - fpm(17)/100 = ratio 'vertical axis'/'horizontal axis'
		feastparam[2]=Objet_feast->eps;	 		 	// Stopping convergence criteria for double precision 10^-epsilon
		feastparam[3]=Objet_feast->max_refin_loop; 	// Maximum number of FEAST refinement loop allowed
		feastparam[4]=Objet_feast->guess_M0; 			// Provide initial guess subspace (0: No; 1: Yes)
		feastparam[5]=Objet_feast->cnvgce_trace_Resdl; /*Convergence criteria (for the eigenpairs in the search interval) 1
															0: Using relative error on the trace epsout i.e. epsout< epsilon
															    1: Using relative residual res i.e. maxi res(i) < epsilon*/
		feastparam[13]=Objet_feast->Run_Q_M0;  		/*   1: FEAST normal execution; 1: Return subspace Q after 1 contour; 0
																	 2: Estimate #eigenvalues inside search interval*/
		  //######### NON HERMITIAM CASE [NOT Used] ##############
		//         feastparam[7]=1; // # of contour points for non-Hermitian FEAST (full-contour)
		//         feastparam[8]=1; //User dened MPI communicator for a given search interval:  MPI_COMM_WORLD/
		//         feastparam[9]=1; //Store factorizations with the predened interfaces (0: No; 1: Yes).
		//         feastparam[16]=1;  // Integration type for non-Hermitian (0: Gauss, 1: Trapezoidal)
		//         feastparam[18]=1;  //Rotation angle in degree [-180:180] for ellipse using non-Hermitian FEAST : 0
		//                             Origin of the rotation is the vertical axis.

		  /* ################  feast Solver  ####<< "="<<#####################*/
		dfeast_scsrev(&UPLO,&N2,SC_hhd,iSC_hhd,jSC_hhd,feastparam,&epsout,&loop,&Emin,&Emax,&M0,E,X,M,res,&info);

		if(disp_msg_2){

				gettimeofday(&t2,NULL);

				if (info==0) {
					printf("\t\t*********************************************************\n");
					printf("\t\t*      Feast Double solver has converged successfully!  \n");
					printf("\t\t*********************************************************\n");
					printf("\t\tSIMULATION TIME %f\n",(t2.tv_sec-t1.tv_sec)*1.0+(t2.tv_usec-t1.tv_usec)*0.000001);
					printf("\t\t# Search interval [Emin,Emax]=[ %.1f , %.1f ]\n",Emin,Emax);
					printf("\t\t# mode found/subspace %d %d \n",*M,M0);
					printf("\t\t# iterations %d \n",loop);

					trace=(double) 0.0;
					for (i=0;i<=*M-1;i=i+1){ trace=trace+*(E+i);}

					printf("\t\tTRACE %.15e\n", trace);
					printf("\t\tRelative error on the Trace %.15e\n",epsout );
					printf("\n\n\t\t| ith  | Eigenvalues |    Residuals  \n");
					printf("-------------------------------------\n");

					i=*M-2;
					printf("\t\t|  %d  |   %.1e       |   %.1e        \n",i,*(E+i),*(res+i));
					printf("-------------------------------------\n");

					i=*M-1;
					printf("\t\t|  %d  |   %.1e       |   %.1e        \n",i,*(E+i),*(res+i));
					printf("-------------------------------------\n");

				}else {
						printf("*********************************************************\n");
						printf("*  Feast solver hasn failed to converge. Error info= %d    \n",info);
						printf("*********************************************************\n");
				}
		}

	if(disp_msg_2){cout<<"\t\t-->  The  SC_hhD matrix has been decomposed succesfully with Nh="<<*M<< ", info="<<info<<  endl;}

	*Iter_p=loop+1;
	*info_p=info;
	return true;
}




/*============================= double Parallel FEAST SOLVER   ============================/*
 __________________________________________________________________________
|  This function returns double sparse decompostion of the matrix SC_hhd   |
|                       Using  feast 3.0  solver                           |
|__________________________________________________________________________|*/

template<typename T> bool pFeast_Solver(MPI_Comm NEW_COMM_WORLD, Feast_param *Objet_feast, int N2, int nnz, T* &SC_hhd, int* &iSC_hhd, int* &jSC_hhd,  T* &X,  T* &E,T* &res, int* M, int* Iter_p,int* info_p, T* Emin_p, T* Emax_p){


		if(disp_msg_0){cout<<endl<<"==> SCSA_2D2D Eigen Decomposition using  [MPI Feast 3.0]"<<endl;}

/* ############################ Feast declaration variable  ############################ */
		int  feastparam[64];
		T epsout;
		int loop;
		char UPLO='F';

/*!!!!!!!!!!!!!!!!! Others */
		struct timeval tt1, tt2, t1, t2;
		int  i, j, k, err;
		int  NB_eig, info;
		double trace;

///*********** MPI ************************************/
//		int lrank,lnumprocs,key;
//		int rank,numprocs;
//		MPI_Comm NEW_COMM_WORLD;
//		MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
//		MPI_Comm_rank(MPI_COMM_WORLD,&rank);

/****************  uUb-Intervals*****************************/
		 gettimeofday(&tt1,NULL);

		if(disp_msg_2){
			cout<<" \t\t--> Feast Eigen Analysis of SC_hhD Matrix"<< endl;
			printf("\t\tsparse matrix size %.d\n",N2);
			printf("\t\tnnz %d \n",nnz);
		}

// ********Define the subintervals [ai, bi] with m0 expeced eigenvalues in it.

		T Emin=Objet_feast->Emin; 					//define ai
		T Emax=Objet_feast->Emax;					//define bi
		int M0=Objet_feast->M0 ;

//		M0=Find_nmbr_Eigvalues(Objet_feast, N2,SC_hhd,iSC_hhd,jSC_hhd, Emin, Emax);
		M0=1.3*M0;
		cout<<" FEAST 3.0: Search Interval ["<<Emin << ","<< Emax<< "] where M0="<< M0<<endl;

//!!!!!!!!!!!!!!!!!! RUN INTERVALS in PARALLEL
		gettimeofday(&t1,NULL);

/*!!!!!!!!!!!!! ALLOCATE VARIABLE */
		E=(T *) calloc(M0,sizeof(T));  // eigenvalues
		if(!E){printf( " --> E out of memory, could not allocate  of memory!\n" );return -1;}

		res=(T *)calloc(M0,sizeof(T));// residual
		if(!res){printf( " --> res out of memory, could not allocate  of memory!\n" );return -1;}

		X=(T *)calloc(N2*M0,sizeof(T));// eigenvectors  // factor 2 for complex
		if(!X){printf( " --> X out of memory, could not allocate  of memory!\n" );return -1;}

		gettimeofday(&t1,NULL);
		feastinit(feastparam);
		feastparam[0]=Objet_feast->show__feast;  		//Print runtime comments on screen (0: No; 1: Yes)
//		feastparam[1]=Objet_feast->Nb_cntour_points;   	// # of contour points for Hermitian FEAST (half-contour) 8
//															  // if fpm(15)=0,2, values permitted (1 to 20, 24, 32, 40, 48, 56)
//															  // if fpm(15)=1, all values permitted*/
//		feastparam[15]=Objet_feast->intgr_Gauss_Trapez_Zolot;  // Integration type for Hermitian (0: Gauss; 1: Trapezoidal; 2: Zolotarev)
//		feastparam[17]=Objet_feast->cntour_ratio; 	 	// Ellipse contour ratio - fpm(17)/100 = ratio 'vertical axis'/'horizontal axis'
//		feastparam[2]=Objet_feast->eps;	 		 		// Stopping convergence criteria for double precision 10^-epsilon
//		feastparam[3]=Objet_feast->max_refin_loop; 		// Maximum number of FEAST refinement loop allowed
//		feastparam[4]=Objet_feast->guess_M0; 			// Provide initial guess subspace (0: No; 1: Yes)
//		feastparam[5]=Objet_feast->cnvgce_trace_Resdl;  /*Convergence criteria (for the eigenpairs in the search interval) 1
//															0: Using relative error on the trace epsout i.e. epsout< epsilon
//															    1: Using relative residual res i.e. maxi res(i) < epsilon*/
//		feastparam[13]=Objet_feast->Run_Q_M0;  			/*   1: FEAST normal execution; 1: Return subspace Q after 1 contour; 0
//																	 2: Estimate #eigenvalues inside search interval*/
		feastparam[8]=NEW_COMM_WORLD;  						/*change from default value */

//######### NON HERMITIAM CASE [NOT Used] ##############
		//         feastparam[7]=1; // # of contour points for non-Hermitian FEAST (full-contour)
		//         feastparam[8]=1; //User dened MPI communicator for a given search interval:  MPI_COMM_WORLD/
		//         feastparam[9]=1; //Store factorizations with the predened interfaces (0: No; 1: Yes).
		//         feastparam[16]=1;  // Integration type for non-Hermitian (0: Gauss, 1: Trapezoidal)
		//         feastparam[18]=1;  //Rotation angle in degree [-180:180] for ellipse using non-Hermitian FEAST : 0
		//                             Origin of the rotation is the vertical axis.


/* ################  feast Solver  ####<< "="<<#####################*/
		dfeast_scsrev(&UPLO,&N2,SC_hhd,iSC_hhd,jSC_hhd,feastparam,&epsout,&loop,&Emin,&Emax,&M0,E,X,M,res,&info);

		gettimeofday(&t2,NULL);

/*!!!!!!!!!! REPORT !!!!!!!!!*/

//		if(disp_msg_2){

//			if (lrank==0) {
//					printf("interval # %d\n",color);
//					printf("FEAST OUTPUT INFO %d\n",info);
//					if (info==0) {
//						printf("\t\t*********************************************************\n");
//						printf("\t\t*      Feast Double solver has converged successfully!  \n");
//						printf("\t\t*********************************************************\n");
//						printf("# of processors %d \n",lnumprocs);
//						printf("SIMULATION TIME %f\n",(t2.tv_sec-t1.tv_sec)*1.0+(t2.tv_usec-t1.tv_usec)*0.000001);
//						printf("\t\t# Search interval [Emin,Emax]=[ %.1f , %.1f ]\n",Emin,Emax);
//						printf("\t\t# mode found/subspace %d %d \n",*M,M0);
//						printf("\t\t# iterations %d \n",loop);
//
//						trace=(double) 0.0;
//						for (i=0;i<=*M-1;i=i+1){ trace=trace+*(E+i);}
//
//						printf("\t\tTRACE %.15e\n", trace);
//						printf("\t\tRelative error on the Trace %.15e\n",epsout );
//						printf("\n\n\t\t| ith  | Eigenvalues |    Residuals  \n");
//						printf("-------------------------------------\n");
//
//						i=*M-2;
//						printf("\t\t|  %d  |   %.1e       |   %.1e        \n",i,*(E+i),*(res+i));
//						printf("-------------------------------------\n");
//
//						i=*M-1;
//						printf("\t\t|  %d  |   %.1e       |   %.1e        \n",i,*(E+i),*(res+i));
//						printf("-------------------------------------\n");
//
//					}else {
//							printf("*********************************************************\n");
//							printf("*  Feast solver hasn failed to converge. Error info= %d    \n",info);
//							printf("*********************************************************\n");
//					}
//
//			}

//		}

	if(disp_msg_2){cout<<"\t\t-->  The  SC_hhD matrix has been decomposed succesfully with Nh="<<*M<< ", info="<<info<<  endl;}

	*Iter_p =loop;
	*info_p=info;
	*Emin_p= *E;
	*Emax_p=*(E+*M-1);

	gettimeofday(&tt2,NULL);
//	printf("Sum SIMULATION TIME %f\n",(tt2.tv_sec-tt1.tv_sec)*1.0+(tt2.tv_usec-tt1.tv_usec)*0.000001);

	if (info!=0) {
		cout<<endl<< " Solver Error !! info code="<< info<<endl;
		return true;}

	else{return true;}
}


/*============================= Evaluate the results    ============================/*
 __________________________________________________________________________
|                  This function returns PSNR
|__________________________________________________________________________|*/

template<typename T> float evaluate_results( int N2, float* &ref,  T* &signal){

	float MSE =0.0, PSNR;
	for(int i = 0; i < N2; i++){

		MSE += ((T) ref[i]-signal[i])*((T) ref[i]-signal[i]);
}

// Evaluation PSNR comparing with the  Signal with its reference

	MSE=MSE/N2;
	PSNR = 10*log10(1.0/MSE);
	return PSNR;
}


/*==========================  ONLINE SOURCES   ============================/*


/*****************************************************************************
 * from: http://www.cplusplus.com/forum/beginner/73057/
 *****************************************************************************/
/* This c program prints current system date. e   */

bool display_time(){
    cout<< endl<<endl<<" ============== Excution Marking ================="<<endl;
    cout <<" \t"<< asctime(localtime(&ctt)) ;
	cout<< " ================================================="<<endl<< endl<<endl;


return true;
}




/*****************************************************************************
 * from: http://www.geeksforgeeks.org/convert-floating-point-number-string/
 *****************************************************************************/

// Converts a floating point number to string.
void ftoa(float h, char *res, int afterpoint)
{
    // Extract integer part
    int ipart = (int)h;

    // Extract floating part
    float fpart = h - (float)ipart;

    // convert integer part to string
    int i = intToStr(ipart, res, 0);

    // check for display option after point
    if (afterpoint != 0)
    {
        res[i] = '.';  // add dot

        // Get the value of fraction part upto given no.
        // of points after dot. The third parameter is needed
        // to handle cases like 233.007
        fpart = fpart * pow(10, afterpoint);

        intToStr((int)fpart, res + i + 1, afterpoint);
    }
}


/*****************************************************************************
 * from: http://www.geeksforgeeks.org/convert-floating-point-number-string/
 *****************************************************************************/
// reverses a string 'str' of length 'len'
void reverse(char *str, int len)
{
    int i=0, j=len-1, temp;
    while (i<j)
    {
        temp = str[i];
        str[i] = str[j];
        str[j] = temp;
        i++; j--;
    }
}


/*****************************************************************************
 * from: http://www.geeksforgeeks.org/convert-floating-point-number-string/
 *****************************************************************************/

 // Converts a given integer x to string str[].  d is the number
 // of digits required in output. If d is more than the number
 // of digits in x, then 0s are added at the beginning.
int intToStr(int x, char str[], int d)
{
    int i = 0;
    while (x)
    {
        str[i++] = (x%10) + '0';
        x = x/10;
    }

    // If number of digits required is more, then
    // add 0s at the beginning
    while (i < d)
        str[i++] = '0';

    reverse(str, i);
    str[i] = '\0';
    return i;
}


