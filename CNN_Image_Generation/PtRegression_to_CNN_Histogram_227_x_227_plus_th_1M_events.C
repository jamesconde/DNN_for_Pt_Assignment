#include <iostream>
#include <iomanip>
using namespace std;
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TAttFill.h"
#include "TCanvas.h"
#include <vector>
#include "stdio.h"
#include <stdlib.h>
#include "math.h"
#include "TMath.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TPaveStats.h"
#include "TStyle.h"
#include "THStack.h"
#include "TFitResultPtr.h"
#include <fstream>
#include <time.h>
void PtRegression_to_CNN_Histogram_227_x_227_plus_th_1M_events()
{
	//Source data
	TString fileName="/home/elb8/EMTF_pT_Assign/EMTFPtAssign2017/PtRegression_for_DNN_Vars_MODE_15_noBitCompr_RPC.root";
	TString directoryName="f_MODE_15_invPtTarg_invPtWgt_noBitCompr_RPC/TrainTree";

	cout << "Opening "<< fileName << endl;

	TFile* myFile = new TFile(fileName);//load data
	TTree* myTree = (TTree*) myFile->Get(directoryName);

	//Target Variable
	Float_t GEN_pt;
	//Input Variables
	Float_t theta;
	Float_t dPhi_12;
	Float_t dPhi_23;
	Float_t dPhi_34;
	Float_t dTh_12;
	Float_t dTh_23;
	Float_t dTh_34;

	myTree->SetBranchAddress("dPhi_12",&dPhi_12);
	myTree->SetBranchAddress("dPhi_23",&dPhi_23);
	myTree->SetBranchAddress("dPhi_34",&dPhi_34);
	myTree->SetBranchAddress("dTh_12",&dTh_12);
	myTree->SetBranchAddress("dTh_23",&dTh_23);
	myTree->SetBranchAddress("dTh_34",&dTh_34);
	myTree->SetBranchAddress("GEN_pt",&GEN_pt);
	myTree->SetBranchAddress("theta",&theta);

	//Long64_t numEvents = myTree->GetEntries();//read the number of entries in myTree
	int numEvents = 1000000;//User set number of events
	cout<<">>>>>>>>>>>>>>>>>>>>>"<<endl;
        cout<<numEvents<<" events to process..."<<endl;
	
	//pT values file
	//TString outFolder = "/storage1/users/eb8/time_test/";
	TString outFolder = "/storage1/users/eb8/1M_events_test/";
	TString outFile = outFolder + "pT_Values.csv";
	ofstream output(outFile);
	cout<< "Results: " << outFolder << endl;	
	
	//Histogram parameters
	const int NX_BINS = 227;
	const int NY_BINS = 227;
	const int XMIN = -113;
	const int XMAX = 114;
	const int YMIN = -127;
	const int YMAX = 100;
	cout << "Set hist params"<<endl;
	//Initialize histograms
        TH2I *hist[int(numEvents)];
	cout << "Initialized hists" <<endl;
	TImage *img[int(numEvents)];
        cout << "Initialized images" <<endl;
        TCanvas *c[int(numEvents)];
	cout << "Initialized canvases"<<endl;

	//Set up color palette
	Int_t palette[8];
	palette[0] = 1;
	palette[1] = 2;
	palette[2] = 3;
	palette[3] = 4;
	palette[4] = 5;
	palette[5] = 6;
	palette[6] = 7;
	palette[7] = 9;
	cout << "Color palette set" << endl;
	//Begin timer
	time_t start_time;
	double seconds;
	time_t timer;
	start_time = time(NULL);
	cout << "Timer started, beginning loop" << endl;
	//Begin loop over events
	for(Long64_t iEntry = 0; iEntry <numEvents; iEntry++){
		cout << "Loop step: "<< iEntry << endl;
        	if (iEntry % 5000 == 0){
			timer = time(NULL);
			seconds = difftime(timer,start_time);	
        		cout<< "Analyzing event " << iEntry << endl;
			cout<< "Time since start (s): " << seconds << endl;
        	}
		//load the i-th entry
        	myTree->GetEntry(iEntry);
		
		//Set up canvas
   		TString canv;
   		canv.Form("c%d",int(iEntry));
   		c[iEntry] = new TCanvas(canv,canv,227,227);
		c[iEntry]->cd();
 		c[iEntry]->SetBorderMode(0);//remove borders	
		c[iEntry]->SetFrameLineColor(0);
		c[iEntry]->SetLeftMargin(0);
		c[iEntry]->SetRightMargin(0);
		c[iEntry]->SetTopMargin(0);
		c[iEntry]->SetBottomMargin(0);
		c[iEntry]->SetCanvasSize(227,227);//correct canvas size
		
		//Set up histogram
		TString h_title;
		h_title.Form("hist%d",int(iEntry));
		hist[iEntry] = new TH2I(h_title, h_title, NX_BINS, XMIN, XMAX, NY_BINS, YMIN, YMAX);
		Float_t dPhi_12_fill = dPhi_12-113;//adjust dPhi_12 to center around 0
		Float_t dPhi_23_fill = dPhi_23;
		Float_t dPhi_34_fill = dPhi_34;
		//condense large dPhi values into last bin
		if (dPhi_12>226){
			dPhi_12_fill = 113;}
		if (dPhi_23>113){
			dPhi_23_fill = 113;}
		if (dPhi_23<-113){
			dPhi_23_fill = -113;}
		if (dPhi_34>113){
			dPhi_34_fill = 113;}
		if (dPhi_34<-113){
			dPhi_34_fill = -113;}
		//Fill histogram, add theta value and weight
		hist[iEntry]->Fill(dPhi_12_fill,dTh_12+theta,1);
		hist[iEntry]->Fill(dPhi_23_fill,dTh_23+theta,2);
		hist[iEntry]->Fill(dPhi_34_fill,dTh_34+theta,4);
		
		//Draw reference line
		for (int k = -113; k<115; k++){
			hist[iEntry]->Fill(k,-10,9);}
		
		hist[iEntry]->SetStats(0);//remove stat box and title
		hist[iEntry]->SetTitle("");
		gStyle->SetPalette(8,palette);
		hist[iEntry]->GetZaxis()->SetRangeUser(0.9,8.1);//fix color range
		hist[iEntry]->Draw("ABBCOL");
		
		//Save images
                img[iEntry] = TImage::Create();
               	img[iEntry]->FromPad(c[iEntry]);
                TString img_name;
                img_name.Form("event%06d.png",int(iEntry));
		TString img_out = outFolder + "Plots/" + img_name;
                img[iEntry]->WriteImage(img_out);
                
		//Record pT
		if(iEntry< numEvents-1){
			output << GEN_pt << ", ";}
		else{
			output << GEN_pt;}

	}
cout << "Done looping over events" << endl;
timer = time(NULL);
seconds = difftime(timer,start_time);
cout << "Total time (s): "<< seconds<< endl;

}//end image generation
