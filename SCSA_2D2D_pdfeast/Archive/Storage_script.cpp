//	// CASE 1 : Manual
//	Emin_list = new T[NB_M0+3];
//	M0_list =  new int[NB_M0+2];
//
//	*(Emin_list)=Emin;
//	for(int k=1;k<=Objet_feast->NB_Intervals;k++){
////		cout<<"--"<<*(Emin_list +k-1);
//		*(Emin_list +k)=Emin-k*(Emin/Objet_feast->NB_Intervals);
//		*(M0_list +k)= Find_nmbr_Eigvalues( Objet_feast,N2, SC_hhd, iSC_hhd, jSC_hhd, *(Emin_list +k-1), *(Emin_list +k));
//
//	}
//	for(int k=0;k<Objet_feast->NB_Intervals;k++){
//			cout<<" |--| "<<*(Emin_list +k);
//	}




cout<< "Error"<< endl;


if (MKL_Feast==0){
	 X= SC_full;
	 if(!MKL_solver( 'V', 'U', SC_full, N2, E,  &M,&info)) return -1;


		time_eig= gettime() - comp_time0;

		if ( show_optimal==1)cout<<std::scientific<< *(E+M-1)<< "\t"<< *(E)<< "\t";

// ############## Reconstruct the image  ####################
		comp_time0 = gettime();
		Emin_disp=*E;
		if ( !SCSA_2D2D_Reconstruction( h, N2, M, gm, fe, matBuffer, Image_ref, max_img, X, E, V2 , &time_Psinnor, &MSE0, &PSNR0, &MSE, &PSNR)) return -1;
		sum_M=M;
		time_reconst=gettime() - comp_time0;
		comp_end=data_prep+time_eig+time_reconst+time_Split;
}

else{







	//	if(rank==0){
	//
	////		cout<<endl<< endl<< "==>I AM   numprocs="<< numprocs<<"  # of intervals="<< Objet_feast->NB_Intervals<<"  #MPI per interval="<< NB_MPI_interval<<  "  rank="<< rank<<   "  lrank="<< lrank<<  "  color="<< color<<endl<< endl;
	////		cout<<endl<<"The choosen sub interval C"<< color<<"=["<< Objet_feast->Emin<<" ,"<<Objet_feast->Emax<<"] and the estimated M0="<< Objet_feast->M0<<endl;
	////		cout<<"Local output"<< color<<" ="<<Mi<<"   Global output"<< color<<" ="<<sum_M<<endl;
	//
	//		cout<<endl<<endl<<"************************   2D2D SCSA Diagnosis  ***************************";
	//		cout<<endl<<" The computations are done with the following results :  "<< endl;
	//		cout<<std::fixed<< "==> Performances with:  h= "<<h<<" ,  gm= "<<gm<< " , fe= "<<fe<<endl;
	//		cout<< "Number of Negative Eigenvalues = \t"<<sum_M<<"  with the Emin=" <<*(E)<< endl;
	//		cout<<std::scientific<<"Original:   MSE = "<<MSE0<<std::fixed<<  "  PSNR = "<<PSNR0<<endl;
	//		cout<<std::scientific<<"Denoised:   MSE = "<<MSE<< std::fixed<< "  PSNR = "<<PSNR<<endl<< endl;
	//
	//////		if(msg_disp_performance){
	//////			cout<< "Number of Negative Eigenvalues = \t"<<M<<endl;
	////			cout<< "Number of Negative Eigenvalues = \t"<<sum_M<<endl;
	////			cout<< "==> Execution Time:   Total Time  = "<<comp_end<<" sec"<< endl;
	////			cout<< "Data Preparation = \t"<<100.0*data_prep/comp_end<< "%  "<<endl;
	////			cout<< "EigenAnalysis    = \t"<<100.0*time_eig/comp_end<< "%  "<<endl;
	////			cout<< "Splitting    = \t"<<100.0*time_Split/comp_end<< "%  "<<endl;
	////			cout<<"EigenFunction normalization = "<<100.0*(time_Psinnor)/comp_end<< "%  "<<endl;
	////			cout<<"Image Reconstruction time   = "<<100.0*time_reconst/comp_end<< " % "<<endl<<endl;
	////			cout<<endl<<endl<<"************************   End of SCSA process   ************************"<<endl;
	//////		}
	//
	//
	//	}
