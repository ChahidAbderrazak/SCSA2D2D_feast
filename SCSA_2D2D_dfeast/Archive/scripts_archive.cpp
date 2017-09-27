// ############## Sensitivity analysis  ####################
	float scal_Emin, scaling_op, PSNR_op=-1000.0;
int  My_count=0
//	Emin0=Emin;//-9.8e-3;//  real Emin fro Lena64 -N 6

	for(j=scal_min;j<=scal_max;j+=scal_step){
		scal_Emin=(float) j;						// Search interval
		 Emin=scal_Emin*Emin0;
		 for(i=eps_min;i<eps_max;i+=eps_step){My_count++;
		 
		  // ############## Sensitivity Parameters  ####################
		 
		     Objet_feast->eps =i;                                 // Stopping convergence criteria : 10^-epsilon
		     
		     
	        if (show_optimal==1){if(MKL_Feast==1){ cout<<std::fixed<< i <<  "\t "<< h<< "\t"<< gm<< "\t"<< fe<< "\t"<< M0<<"\t"<< Emin<< "\t"<< scal_Emin<< "\t";}
	        	else{ cout<<std::fixed<<"N/A \t "<< h<< "\t"<< gm<< "\t"<< fe<< "\t"<< M0<<"\t"<< Emin<< "\t"<< scal_Emin<< "\t";}}

.	if(My_count==1){PSNR_op=PSNR;}

	// #####  Check if the inerval is the best R  ###########

	if(PSNR>PSNR_op){
		PSNR_op=PSNR;
		scaling_op=scal_Emin;
	}
	

*scal_Emin_p=scaling_op;

	if(Objet_SCSA->x_dim>128){MKL_Feast=1;
	      cout<< "Scanning for the optimal  parameters h and gm for SCSA2D2D using [Feast 3.0].  Plese wait :)"<<endl;
		  } else{ cout<< "Scanning for the optimal  parameters h and gm for SCSA2D2D using [MKL].  Plese wait :)"<<endl;}
//      cout<<"The used parameters ranges are: h_min= "<<h_min <<" h_max "<<h_max <<" h_step "<<  h_step<<  " gm_min "<<  gm_min << " gm_max "<< gm_max <<" gm_step "<< gm_step <<" fe_min "<< fe_min <<" fe_max "<<fe_max << " fe_step"<< fe_step<<endl;

	if (show_optimal==1){if (MKL_Feast==0){ cout<< "MKL Double\t h \t gm \t fe \t Emin \t scal_Emin \t Found Emin \t Found Emax \t PSNR0 \t PSNR \t MSE0 \t MSE\t Solver time  \t Totale time \t EigenAnalysis % \t #Iterations \t Nh \t  info "<<endl;}
	else{ cout<< "Feast Double\t h \t gm \t fe \t Emin \t scal_Emin \t Found Emin \t Found Emax \t PSNR0 \t PSNR \t MSE0 \t MSE\t Solver time  \t Totale time \t EigenAnalysis % \t #Iterations \t Nh \t  info "<<endl;}}



if(My_count==1){PSNR_op=PSNR_new;}


   for(int fe=fe_min;fe<fe_max;fe+=fe_step){
		for(float gm=gm_min;gm<gm_max;gm+=gm_step){
			 for(float h=h_min;h<h_max;h+=h_step){My_count++;




				if(PSNR_new>PSNR_op){
					h_op=Objet_SCSA->h;     //check if the values are better
					gm_op=Objet_SCSA->gm;
					fe_op=Objet_SCSA->fe;
					PSNR_op=PSNR_new;
					M=*M_p;
					Emin0_op=Emin0;
					scaling_op=scal_Emin_local;}
