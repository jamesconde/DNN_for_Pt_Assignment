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

void root_to_csv()
{
     TString fileName="/home/elb8/EMTF_pT_Assign/EMTFPtAssign2017/PtRegression_for_DNN_Vars_MODE_15_noBitCompr_RPC.root";
     TString directoryName="f_MODE_15_invPtTarg_invPtWgt_noBitCompr_RPC/TrainTree";
     TFile* myFile = new TFile(fileName);//load data
     TTree* myTree = (TTree*) myFile->Get(directoryName);
     Float_t GEN_pt;
     Float_t dPhi_12;
     Float_t dPhi_23;
     Float_t dPhi_34;
     Float_t dTh_12;
     Float_t dTh_23;
     Float_t dTh_34;
     Float_t theta;
     
     myTree->SetBranchAddress("GEN_pt",&GEN_pt);
     myTree->SetBranchAddress("theta",&theta);
     myTree->SetBranchAddress("dPhi_12",&dPhi_12);
     myTree->SetBranchAddress("dPhi_23",&dPhi_23);
     myTree->SetBranchAddress("dPhi_34",&dPhi_34);
     myTree->SetBranchAddress("dTh_12",&dTh_12);
     myTree->SetBranchAddress("dTh_23",&dTh_23);
     myTree->SetBranchAddress("dTh_34",&dTh_34);
     
     ofstream output("PtRegression_for_DNN_Vars_MODE_15_noBitCompr_RPC.csv");
     output<< "GEN_pt" << "," << "theta" << "," << "dPhi_12" << "," << "dPhi_23" << "," << "dPhi_34" << "," << "dTh_12" << "," << "dTh_23" << "," << "dTh_34" << endl;
     Long64_t numEvents = 10;
     for (Long64_t iEntry = 0;iEntry<numEvents;iEntry++){
       myTree->GetEntry(iEntry*100);
       output<< GEN_pt << "," << theta << "," << dPhi_12 << "," << dPhi_23 << "," << dPhi_34 << "," << dTh_12 << "," << dTh_23 << "," << dTh_34 << endl;
     }
      
}
