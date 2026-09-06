 //------------------------------------------------------------------------------------------------
TVector3 PropagateToZplane(Float_t Z, TVector3 A, TVector3 D)
  {
    Float_t delta=(Z-A.Z())/D.Z();
    TVector3 result(A.X()+D.X()*delta, A.Y()+D.Y()*delta, A.Z()+D.Z()*delta);
    return result;
  }
//----------------------------------------------------------------------------------------------------
  TVector3 EntryCluster(BGeomTel *geom, TVector3 TrackPos, TVector3 TrackDir, Int_t iCluster) {
  TVector3 EntryPoint =TVector3(0,0,0);
  double RLim=70.;
  double sensRadius=60+RLim; //cluster sensitivity radius   

  Int_t   str7LastChan=288*(iCluster)-1;
  Int_t   str7FirstChan=288*(iCluster)-36;
  Int_t   str7MidChanBot=288*(iCluster)-19;
  Int_t   str7MidChanTop=288*(iCluster)-18;

  double zTopPlane=geom->At(str7LastChan)->GetZ() + RLim;
  double zBotPlane=geom->At(str7FirstChan)->GetZ() - RLim;
  
  TVector3 p_BotPlane=PropagateToZplane(zBotPlane,TrackPos,TrackDir);
  TVector3 p_TopPlane=PropagateToZplane(zTopPlane,TrackPos,TrackDir);

  TVector3 midStringPoint=TVector3(0.5*(geom->At(str7MidChanBot)->GetX()+geom->At(str7MidChanTop)->GetX()),
                                   0.5*(geom->At(str7MidChanBot)->GetY()+geom->At(str7MidChanTop)->GetY()),
                                   0.5*(geom->At(str7MidChanBot)->GetZ()+geom->At(str7MidChanTop)->GetZ()));

  //find line intersection with cylinder
  double x0=TrackPos.X()-midStringPoint.X();
  double y0=TrackPos.Y()-midStringPoint.Y();

  double xd=TrackDir.X();
  double yd=TrackDir.Y();
  double zd=TrackDir.Z();
  double t1=0;
  double t2=0;
  double D=pow(2*x0*xd+2*y0*yd,2)-4*(xd*xd+yd*yd)*(x0*x0+y0*y0-pow(sensRadius,2));
  
  if (D<0) t1=0;
  else {
    t1=(-(2*x0*xd+2*y0*yd)+sqrt(D))/(2*(xd*xd+yd*yd));
    t2=(-(2*x0*xd+2*y0*yd)-sqrt(D))/(2*(xd*xd+yd*yd));

    TVector3 p_t1(TrackPos.X()+t1*xd,TrackPos.Y()+t1*yd,TrackPos.Z()+t1*zd);
    TVector3 p_t2(TrackPos.X()+t2*xd,TrackPos.Y()+t2*yd,TrackPos.Z()+t2*zd);

    if (p_t2.Z()<zTopPlane&&p_t2.Z()>zBotPlane){
      EntryPoint = p_t2;
    }
    if (p_t2.Z()>=zTopPlane){
      EntryPoint = p_TopPlane;
    }
     if (p_t2.Z()<=zBotPlane){
      EntryPoint = p_BotPlane;
    }

  }

   return EntryPoint;
}
//----------------------------------------------------------------------------------------------------
void energy()
{
        gROOT->Reset();
    
  TFile *f = new TFile("/mnt/data/data/Baikal/nue2_vertex/reco/cluster1/2020_cl1_run1000_scl_MC.root","READ"); // cluster01            
        
       if(!f) {
                cout << "file is not found" << endl;
                exit(1);
        }
        TTree *tree = (TTree*)f->Get("Events"); // получаем доступ к дереву Events
        if(!tree) {
                cout << "tree is not found" << endl;
                exit(1);
        }
//        BEventMask* mask=0;        //
        BEvent*     ev=0;          //    
        BMCEvent*   event = 0;     // Подготавливаем доступ к веткам
        BGeomTel*   geom=0;        //
        BRecoMuon*  RecoMuon = 0;  //
        BEventMask* mcmask=0;  

        TBranch *branch0  = tree->GetBranch("BEvent."); // 
        branch0->SetAddress(&ev);
        TBranch *branchg  = tree->GetBranch("BGeomTel."); // 
        branchg->SetAddress(&geom);     
        TBranch *branch2  = tree->GetBranch("BRecoMuon."); // 
        branch2->SetAddress(&RecoMuon);
        TBranch *branch1  = tree->GetBranch("BMCEvent."); // подключаем ветки из файла к подготовленным указателям.
        branch1->SetAddress(&event);
        TBranch *branch3  = tree->GetBranch("MCEventMask.");       
        branch3->SetAddress(&mcmask);        

        Float_t ETrue;
        
//         for (Int_t i = 0; i < tree->GetEntries(); i++) {    // цикл по всем "строкам" таблицы       
         for (Int_t i = 0; i < 46; i++) {    // цикл по всем "строкам" таблицы       
        
           tree->GetEntry(i);
           if(event->GetResponseMuonsN() != 1) continue   ;                      
//             cout << i << " ---> Event "<< event->GetRunN() << " " << event->GetEventN() << endl;
           if(i!=1) continue; 
//           if(event->GetEventN() != 910332) continue;


//calculate number of clusters in the event
  Int_t nClusterImp[100]={0};             Int_t chanID=0;
             for (Int_t ihit=0; ihit < RecoMuon->GetNHits(); ihit++) {
              chanID=RecoMuon->GetChanID(ihit);
//              cout << ihit << " ChanID=" << chanID  << endl;   
              Int_t clusterID=floor(chanID/288);
              nClusterImp[clusterID]++;                                      
             }
             
  Int_t nClusters=0;
  Int_t iCluster=1;  
  for (int i=0; i<100; i++){
    if (nClusterImp[i]>0) { 
                            nClusters++;
                            iCluster=i+1;                                      
                          }
  }
  std::cout<<"nClusters: "<<nClusters << " iCluster=" << iCluster<<std::endl;
              
  Int_t str7LastChan;
  Int_t str7FirstChan;
  Int_t str7MidChanBot;
  Int_t str7MidChanTop;
  if (geom->GetNumOMs()>288) {
    str7LastChan=288*(iCluster)-1;
    str7FirstChan=288*(iCluster)-36;
    str7MidChanBot=288*(iCluster)-19;
    str7MidChanTop=288*(iCluster)-18;
  }
  else {
    str7LastChan=287;
    str7FirstChan=252;
    str7MidChanBot=269;
    str7MidChanTop=270;
  }

 TVector3 midStringPoint=TVector3(0.5*(geom->At(str7MidChanBot)->GetX()+geom->At(str7MidChanTop)->GetX()),
                                   0.5*(geom->At(str7MidChanBot)->GetY()+geom->At(str7MidChanTop)->GetY()),
                                   0.5*(geom->At(str7MidChanBot)->GetZ()+geom->At(str7MidChanTop)->GetZ()));   
                      
   // create and open a canvas
   TCanvas *sky = new TCanvas( "sky", "Single cluster", 300, 10, 1000, 1000 );
   sky->SetFillColor(0);
 
   // creating view
   TView *view = TView::CreateView(1,0,0);

   Double_t xmin=-300, xmax=200, ymin=0, ymax=400, zmin=-400,zmax=170;     
   
   xmin=midStringPoint[0]-300;
   xmax=midStringPoint[0]+300;   
   ymin=midStringPoint[1]-300;
   ymax=midStringPoint[1]+300;   
   zmin=midStringPoint[2]-300;
   zmax=midStringPoint[2]+300;   
  

   view->SetRange( xmin, ymin, zmin, xmax, ymax, zmax);
             
      TPolyMarker3D *clust = new TPolyMarker3D(288);             
Int_t fNumOMs=geom->GetNumOMs();
Int_t j=0;
for (Int_t ich=0; ich<fNumOMs; ich++){
    //skip channels from clusters which are not in the event
    Int_t clusterID=floor(ich/288);
    if (nClusterImp[clusterID]==0) continue;
    //
    TVector3 chanPos((geom->At(ich))->GetX(),
                     (geom->At(ich))->GetY(),
                     (geom->At(ich))->GetZ());
                     
    clust->SetPoint(j, (geom->At(ich))->GetX(),
                     (geom->At(ich))->GetY(),
                     (geom->At(ich))->GetZ()    );  
    j++;                        
}
      clust->SetMarkerSize( 1 );
      clust->SetMarkerColor(1);
      clust->SetMarkerStyle( 1 );
 
      //draw
      clust->Draw();
//  
//------------------------------------------------------------------------------------------  
//
//    MC reference point
//
//             for (Int_t j=0; j < event->GetResponseMuonsN(); j++) {           
            j=0;
              BMCTrack *track = event->GetTrack(j);
              ETrue=track->GetMuonEnergy();
              cout << "Track " << j << " Muon energy= " << ETrue << " GeV" << endl;
//           }
   double XMC=track->GetX();
   double YMC=track->GetY();
   double ZMC=track->GetZ();
   TVector3 XYZMC(XMC,YMC,ZMC);   
//
//  draw MC track reference point xv,yv,zv
// 
        TPolyMarker3D *xmc = new TPolyMarker3D(1);
        xmc->SetPoint(0,XMC,YMC,ZMC);
        xmc->SetMarkerColor(4);
        xmc->SetMarkerStyle( 30 );
        xmc->SetMarkerSize( 2 );        
        xmc->Draw();
   
   double ThetaMC=track->GetTheta();
   double PhiMC=track->GetPhi();   
   cout << j <<" MC track X,Y,Z= " << XMC << ' ' << YMC << ' ' << ZMC 
   << " Theta, Phi =" << ThetaMC << ' ' << PhiMC << endl;
   
   double    theta=ThetaMC*TMath::DegToRad();
   double    phi=PhiMC*TMath::DegToRad();   
   double    xd=sin(theta)*cos(phi);
   double    yd=sin(theta)*sin(phi);
   double    zd=cos(theta);
    TVector3 DirMC=TVector3(xd,yd,zd);  
//        
//----------------------------------------------------------------------------
//             EntryPoint
        
     Double_t PathL=0.;
     TVector3 MCEntry=EntryCluster(geom, XYZMC, DirMC, iCluster);      
     cout <<"Entry Point="<<MCEntry[0]<<" "<<MCEntry[1]<<" "<<MCEntry[2]<<endl;
    
     TPolyMarker3D *point = new TPolyMarker3D(1);  
     point->SetPoint(0,MCEntry[0],MCEntry[1],MCEntry[2]); 
     point->SetMarkerColor(4);
     point->SetMarkerStyle(20);
     point->Draw();
 
     TVector3 AM = XYZMC - MCEntry; 
     Double_t TDist=AM.Mag();
     Double_t CorAng=AM.Unit().Angle(DirMC);
     Double_t CorDist=TDist*cos(CorAng);   
     cout << "CorDist=" << CorDist << endl;    
    
//------------------------------------------------------------------------------     
     Double_t Ecor=ETrue;   
     cout << "Initialized Ecor = "<< Ecor << endl;
          
//        
// Interactions chain     
// 
   Int_t InteractionN=track->GetInteractionN();
   cout << "Number of interactions= " << InteractionN << endl;
  
        TPolyMarker3D *xint = new TPolyMarker3D(InteractionN);

   for (Int_t k=0; k < InteractionN; k++) {
    BMCInteraction* Inter=track->GetInteraction(k);
    double Xint=Inter->GetX();
    double Yint=Inter->GetY();
    double Zint=Inter->GetZ();
            xint->SetPoint(k,Xint,Yint,Zint);                
    double Energy=Inter->GetEnergy();
   cout << k <<" Interaction X,Y,Z= " << Xint << ' ' << Yint << ' ' << Zint 
   << " Energy =" << Energy*1000. << " GeV" << endl;  
    
// make energy correction
   
     TVector3 XYZint(Xint,Yint,Zint);
     TVector3 ToInt =  XYZint - MCEntry;  
     Double_t Dint=ToInt.Mag();     
     Double_t AngInt=ToInt.Unit().Angle(DirMC);     
     Double_t DintA= Dint * cos(AngInt);     
     cout<<"Distance to interaction="<<DintA <<endl;
//
// Apply the energy correction
//    
     if(CorDist>0.) {     // add the energy of interaction
       if(DintA>0.&&DintA<CorDist) {
         Ecor+=Energy*1000.;
         cout << "Ecor="<< Ecor << endl;      
       }
     }
     else {
       if(DintA<0.&&DintA>=CorDist) {
         Ecor-=Energy*1000.;
         cout << "Ecor="<< Ecor << endl;
       }

     }
     
//     cout << " Ecor = " << Ecor << endl;
//     
   }  // loop over interactions
   
// correction for the ionisation
         Ecor+=0.24*CorDist ;   
         cout << "Ecor with ionisation="<< Ecor << endl;    
                 
        xint->SetMarkerColor(6);
        xint->SetMarkerStyle( 29 );
        xint->Draw();
// getchar(); 
 
}   // loop over events  
}     
        
        
        
        
        
        
        
        
        
        
