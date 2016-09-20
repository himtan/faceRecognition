// facerecognition.cpp : main project file.

#include "stdafx.h"
#include<cv.h>
#include<highgui.h>
#include<cxcore.h>
#include<cvaux.h>
#include<string>
#include<iostream>
#include<fstream>
#include<conio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include<exception>
#include<random>
#include<stdio.h>
#include "nn.h"
#include "facerec.h"
#include "Form1.h"
using namespace facerecognition;
using namespace std;
ref class FormContainer
{
public:
	static Form1^ f;
};
int bpnsavecount=0;
int bpndatfilePosition=0;
int detectfacecounter=0;
class Gaussian
    { 
	public:
		static double getRandomGaussian()
        {
            return getRandomGaussian(0.0, 1.0);
        }
        static double getRandomGaussian(double mean, double stddev)
        {
            double rval1, rval2;
            getRandomGaussian(mean, stddev, & rval1, & rval2);
            return rval1;
        }
        static void getRandomGaussian(double mean, double stddev,double* val1, double* val2)
        {   
			static std::tr1::ranlux_base_01 real_engine(time(NULL));
			/*
			static std::tr1::default_random_engine random_engine(time(NULL));
			static std::tr1::ranlux_base_01 real_engine2(random_engine);

			static std::tr1::normal_distribution<double> nd(0.0,1.0);
			static std::tr1::exponential_distribution<double> ed;
			static std::tr1::gamma_distribution<double> gd;
			static std::tr1::poisson_distribution<double> pd;
			*/
			static std::tr1::uniform_real_distribution<double> ud(-1,1);
			
            double u, v, s, t;
            do
			{ 
			u=ud(real_engine);    
			v=ud(real_engine);
            } while ((u * u + v * v > 1) || (u == 0 && v == 0));
            s = u * u + v * v;
            t = sqrt((-2.0 * log(s)) / s);
            *val1 = stddev * u * t + mean;
            *val2 = stddev * v * t + mean;


        }

    };
    class TransferFunctions
    {
	public :
		static double evaluate(TransferFunction tfunc, double input)
        {
            switch (tfunc)
            {
			case TransferFunction::Sigmoid: return sigmoid(input);
			case TransferFunction::Linear: return linear(input);
			case TransferFunction::Gaussian: return gaussian(input);
			case TransferFunction::RationalSigmoid: return rationalsigmoid(input);
			case TransferFunction::None:
                default: return 0.0;

            }
        }
        static double evaluateDerivative(TransferFunction tfunc, double input)
        {
            switch (tfunc)
            {
			case TransferFunction::Sigmoid: return sigmoid_derivative(input);
			case TransferFunction::Linear: return linear_derivative(input);
			case TransferFunction::Gaussian: return gaussian(input);
			case TransferFunction::RationalSigmoid: return rationalsigmoid_derivative(input);
			case TransferFunction::None:
                default: return 0.0;
            }
        }
	private:
        /*transfer function definitions*/
        static double sigmoid(double x)
        {
            return 1.0 / (1.0 + exp(-x));
        }
        static double sigmoid_derivative(double x)
        {
            return (sigmoid(x) * (1 - sigmoid(x)));
        }
         static double linear(double x)
        { 
            return x; 
        }
        static double linear_derivative(double x)
        {
            return 1;
        }
        static double gaussian(double x)
        {
            return exp(-x * x);
        }
        static double gaussian_derivative(double x)
        {
            return -2.0 * exp(-x * x) * x;
        }
        static double rationalsigmoid(double x)
        {
            return x / (1 + sqrt(1 + x * x));
        }
        static double rationalsigmoid_derivative(double x)
        {double val =sqrt(1+x*x);
        return 1.0 / (val * (1 + val));
        }


    };	
    BackPropagationNetwork::BackPropagationNetwork(int layerSizes[], TransferFunction transferFunctions[],int numberOfLayers)    //CONSTRUCTOR
        {   //parameter validation
            if (transferFunctions[0] != TransferFunction::None)
                {
					FormContainer::f->consoleOutput("input layer cannot have any activation function (should be None)");
					system("pause");
					exit(-1);
				}
            //inintilize network layer
            layerCount =numberOfLayers - 1;  //hidden+output
            inputSize = layerSizes[0];
            layerSize = new int[layerCount];
            for (int i = 0; i < layerCount; i++)
                layerSize[i] = layerSizes[i + 1];

            transferFunction = new TransferFunction[layerCount];
            for (int i = 0; i < layerCount; i++)
                transferFunction[i] = transferFunctions[i + 1];
            bias = new double*[layerCount];
            previousBiasDelta = new double*[layerCount];
            delta = new double*[layerCount];
            layerInput = new double*[layerCount];
            layerOutput = new double*[layerCount];
            weight= new double**[layerCount];
            previousWeightDelta=new double** [layerCount];
            //fill one dimensional array
            for (int l = 0; l < layerCount; l++)
            {
                bias[l] = new double[layerSize[l]];
                previousBiasDelta[l] = new double[layerSize[l]];
                delta[l]=new double[layerSize[l]];
                layerInput[l] = new double[layerSize[l]];
                layerOutput[l]=new double[layerSize[l]];
                weight[l] = new double*[l==0?inputSize:layerSize[l-1]];
                previousWeightDelta[l] = new double*[l == 0 ? inputSize : layerSize[l - 1]];
                for (int n = 0; n < (l == 0 ? inputSize : layerSize[l - 1]); n++)
                {
                    weight[l][n] = new double[layerSize[l]];
                    previousWeightDelta[l][n] = new double[layerSize[l]];
                }
            }//end of for
            //initilize the weights and biases
            for (int l = 0; l < layerCount; l++)
            {
                for (int j = 0; j < layerSize[l]; j++)
                {
					bias[l][j] =Gaussian::getRandomGaussian();

                    previousBiasDelta[l][j] = 0.0;
                    layerInput[l][j] = 0.0;
                    layerOutput[l][j] = 0.0;
                    delta[l][j] = 0.0;

                }
                for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
                {
                    for (int j = 0; j < layerSize[l]; j++)
                    {
                        weight[l][i][j] = Gaussian::getRandomGaussian();
                        previousWeightDelta[l][i][j] = 0.0;
                    }
                }
            }
			
            
        }//END OF CONSTRUCTOR
    void BackPropagationNetwork::run(double input[], double** output,int inputLayerSize)        //RUN METHOD (TAKES INPUT, GENERATES OUTPUT)
		{   //PARAMETER VALIDATION
			
			if (inputLayerSize!= inputSize)
			{
			FormContainer::f->consoleOutput("input size invalid (run)");
			system("pause");
			exit(-1);

			}
           
			*output=new double[layerSize[layerCount-1]];
			
            for (int l = 0; l < layerCount; l++)                    //loop through entire network
            {
                for (int j = 0; j < layerSize[l]; j++)
                {
                    double sum = 0;
                    for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
                        sum += weight[l][i][j] * (l == 0 ? input[i] : layerOutput[l - 1][i]);
                sum+=bias[l][j];
                layerInput[l][j] = sum;
                layerOutput[l][j] = TransferFunctions::evaluate(transferFunction[l], sum);

                }
            }
			
			//copy the o/p to o/p array
            
			for (int i = 0; i < layerSize[layerCount - 1]; i++)
                *output[i] = layerOutput[layerCount - 1][i];

        }//END OF RUN     
	double BackPropagationNetwork::train(double input[], double desired[], double TrainingRate, double momentum,int inputLayerSize,int outputLayerSize)
        { 
        //parameter validation
            if (inputLayerSize!= inputSize)
			{
			FormContainer::f->consoleOutput("input size invalid (train)");
			system("pause");
			exit(-1);

			}
            if (outputLayerSize!= layerSize[layerCount - 1])
               {
			FormContainer::f->consoleOutput("output size invalid (train)");
			system("pause");
					exit(-1);

			}
            //local variables
            double error = 0.0, sum = 0.0, weightDelta = 0.0, biasdelta = 0.0;
            double *output=NULL;
			
            run(input, &output,inputLayerSize);
           
			//back-propagate the error

            for (int l = layerCount - 1; l >= 0; l--)
            {
                if (l == layerCount - 1) //output layer
                {
                    for (int k = 0; k < layerSize[l]; k++)
                    {
                        delta[l][k] = output[k] - desired[k];
                        error += delta[l][k]*delta[l][k];
						
                        delta[l][k] *= TransferFunctions::evaluateDerivative(transferFunction[l], layerInput[l][k]);
                            
                    }
					
                }
                else //hidden layer
                { 
                    for(int j=0;j<layerSize[l];j++)
                    {
                    for(int k=0;k<layerSize[l+1];k++)
                    delta[l][j]+=delta[l+1][k]*weight[l+1][j][k];
                    delta[l][j]*=TransferFunctions::evaluateDerivative(transferFunction[l],layerInput[l][j]);
                    }
                }
            }
			
			
            //update the weights and biases

            for(int l=0;l<layerCount;l++)

                for(int i=0;i<(l==0?inputSize:layerSize[l-1]);i++)
                    for (int j = 0; j < layerSize[l]; j++)
                    {
                        weightDelta = TrainingRate * delta[l][j] * (l == 0 ? input[i] : layerOutput[l - 1][i]) + momentum * previousWeightDelta[l][i][j];
                        weight[l][i][j] -= weightDelta ;
                        previousWeightDelta[l][i][j] = weightDelta;
                    }
            for(int l=0;l<layerCount;l++)
                for(int i=0;i<layerSize[l];i++)
            {
                biasdelta = TrainingRate * delta[l][i];
                bias[l][i] -= biasdelta + momentum * previousBiasDelta[l][i];
                previousBiasDelta[l][i] = biasdelta;
            }
				if(!output)
				delete [] output;
            return error;
        
		}//end train()
	void BackPropagationNetwork::weightRandomize()
		{
		 for (int l = 0; l < layerCount; l++)
            {
                for (int j = 0; j < layerSize[l]; j++)
                {
					bias[l][j] =Gaussian::getRandomGaussian();

                    previousBiasDelta[l][j] = 0.0;
                    layerInput[l][j] = 0.0;
                    layerOutput[l][j] = 0.0;
                    delta[l][j] = 0.0;

                }
                for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
                {
                    for (int j = 0; j < layerSize[l]; j++)
                    {
                        weight[l][i][j] = Gaussian::getRandomGaussian();
                        previousWeightDelta[l][i][j] = 0.0;
                    }
                }
            }
		}
	void BackPropagationNetwork::save(char* file)
		{
		static std::ofstream off;
		(bpnsavecount==0)?off.open(file,ios::binary):off.open(file,ios::binary|ios::app);
		if(!off)     
		{
		FormContainer::f->consoleOutput("bpndata.dat file error , now exiting");
		system("pause");
		exit(-1);
		}
		bpnsavecount++;
		off.write((char*)&layerCount,sizeof(int));     //layerCount
		off.write((char*)&inputSize,sizeof(int));      //inputSize
		for(int i=0;i<layerCount;i++)
		off.write((char *)&layerSize[i],sizeof(int));     //layerSize[]
		for(int i=0;i<layerCount;i++)
		off.write((char *)&transferFunction[i],sizeof(TransferFunction));   //transferFunction[]
		for (int l = 0; l < layerCount; l++)
            {    //saving 2D arrays
                for (int j = 0; j < layerSize[l]; j++)
                {
					 off.write((char *)&bias[l][j],sizeof(double));             //bias[l][j]
					 off.write((char *)&previousBiasDelta[l][j],sizeof(double));  //previousBiasDelta[l][j] 
                     off.write((char *)&layerInput[l][j],sizeof(double));          //layerInput[l][j]
					 off.write((char *)&layerOutput[l][j],sizeof(double));        //layerOutput[l][j] 
                     off.write((char *)&delta[l][j],sizeof(double));               //delta[l][j]
                }
				//saving 3D arrays
                for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
                {
                    for (int j = 0; j < layerSize[l]; j++)
                    {	
						off.write((char *)&weight[l][i][j],sizeof(double));      //weight[l][i][j]
                        off.write((char *)&previousWeightDelta[l][i][j],sizeof(double));         //previousWeightDelta[l][i][j]
                    }
                }
            }
		off.close();
		}//end save()
	void BackPropagationNetwork::load(char* file,BackPropagationNetwork** bp)
	   {
		int lcnt;
	    int ipsize;
	    static std::ifstream iff;
		iff.open(file,ios::binary);
		if(!iff)     
		{
		FormContainer::f->consoleOutput("bpndata.dat file error , now exiting");
		system("pause");
		exit(-1);
		}
		iff.seekg(bpndatfilePosition);
		iff.read((char*)&lcnt,sizeof(int));
		iff.read((char *)& ipsize,sizeof(int));
		int* lsizetemp=new int[lcnt];
		TransferFunction* tfuntemp=new TransferFunction[lcnt];
		for(int i=0;i<lcnt;i++)
		iff.read((char *)&lsizetemp[i],sizeof(int));
		for(int i=0;i<lcnt;i++)
		iff.read((char *)&tfuntemp[i],sizeof(TransferFunction));
		int * lsize=new int[lcnt+1];
		TransferFunction* tfun=new TransferFunction[lcnt+1];
		lsize[0]=ipsize;
		tfun[0]=TransferFunction::None;
		for(int i=0;i<lcnt;i++)
			{
				tfun[i+1]=tfuntemp[i];
				lsize[i+1]=lsizetemp[i];
			}
		*bp=new BackPropagationNetwork(lsize,tfun,lcnt+1);
		for (int l = 0; l < (*bp)->layerCount; l++)
            {    //saving 2D arrays
                for (int j = 0; j < (*bp)->layerSize[l]; j++)
                {
					iff.read((char *)&(*bp)->bias[l][j],sizeof(double)); 
					iff.read((char *)&(*bp)->previousBiasDelta[l][j],sizeof(double)); 
                    iff.read((char *)&(*bp)->layerInput[l][j],sizeof(double));
					iff.read((char *)&(*bp)->layerOutput[l][j],sizeof(double)); 
                    iff.read((char *)&(*bp)->delta[l][j],sizeof(double));
                    

                }
				//saving 3D arrays
                for (int i = 0; i < (l == 0 ? (*bp)->inputSize : (*bp)->layerSize[l - 1]); i++)
                {
                    for (int j = 0; j < (*bp)->layerSize[l]; j++)
                    {	iff.read((char *)&(*bp)->weight[l][i][j],sizeof(double)); 
                        iff.read((char *)&(*bp)->previousWeightDelta[l][i][j],sizeof(double)); 
                        
                    }
                }
            }
		
		delete []tfuntemp;
		delete []tfun;
		delete []lsizetemp;
		delete []lsize;
		bpndatfilePosition=iff.tellg();
		iff.close();
		
	   }//end of load

int noOfPerson=0;       //correct
int imagesPerPerson=0;           //correct
char extension[]=".pgm";       //correct

IplImage** pca::faceImgArr=0;       //correct
IplImage* pca::pAvgTrainImg=0;       //correct
IplImage** pca::eigenVectArr=0;       //correct
CvMat* pca::eigenValMat=0;       //correct
CvMat * pca::projectedTrainFaceMat=0;       //correct
int pca::nTrainFaces=0;       //correct
int pca::nEigens=0;       //correct
BackPropagationNetwork** pca::bpn=0;       //correct

void pca::learn()
{
	nTrainFaces=noOfPerson*imagesPerPerson;
	faceImgArr=(IplImage**)cvAlloc(sizeof(IplImage*) * nTrainFaces);
	for(int i=0;i<noOfPerson;i++)
	{   
		
	for(int j=0;j<imagesPerPerson;j++)
	{       
			FormContainer::f->consoleOutput("\nperson no ");
			FormContainer::f->consoleOutput(i);
			FormContainer::f->consoleOutput(" image no ");
			FormContainer::f->consoleOutput(j);
			faceImgArr[i*imagesPerPerson+j]=detectFace();
			if(!faceImgArr[i*imagesPerPerson+j])
			{
			FormContainer::f->consoleOutput("\nimage loading problem, now exiting");
			system("pause");
		    exit(-1);
			}

	}
	FormContainer::f->changeColour();
	}
doPCA();               //will load eigenVectArr,pAvgTrainImg,eigenValMat
projectedTrainFaceMat = cvCreateMat(nTrainFaces, nEigens, CV_32FC1);
 for(int i=0; i<nTrainFaces; i++)
 {
 cvEigenDecomposite(faceImgArr[i],nEigens,eigenVectArr,0, 0,pAvgTrainImg,projectedTrainFaceMat->data.fl + i*nEigens);
 }
 trainBpn();
 storeTrainingData();
 FormContainer::f->consoleOutput("\ntraining complete !");
}
void pca::recognize()
{   float * projectedTestFace=0 ;
    int personNumber=0;
	IplImage* testFace=0;
	std::ifstream ifName("names.dat");
	char name[50];
	testFace=detectFace();
	
	if(!testFace) 
	{
	FormContainer::f->consoleOutput("\ntest face loading error, now exiting");
	system("pause");
	exit(-1);
	}
	loadTrainingData();
	projectedTestFace= new float[nEigens];
	//project the test image onto the PCA subspace
	if((eigenVectArr[0]->width!=testFace->width) || (eigenVectArr[0]->height!=testFace->height))
	{
	FormContainer::f->consoleOutput("\nresolutions don't match, now exiting");
	system("pause");
	exit(-1);
	}

	cvEigenDecomposite(testFace,nEigens,(void*)eigenVectArr,0,0,pAvgTrainImg,projectedTestFace);
	personNumber=runBpn(projectedTestFace);
	
	for(int i=0;i<=personNumber;i++)
		ifName.getline(name,50,'#');
	ifName.close();
	cvReleaseImage(&testFace);
	delete[] projectedTestFace;
	FormContainer::f->consoleOutput("\nperson number  ");
	FormContainer::f->consoleOutput(personNumber);
	FormContainer::f->consoleOutput(" ");
	FormContainer::f->consoleOutput(gcnew String(name));

	
}
void pca::doPCA()
{
 CvTermCriteria calcLimit;
 CvSize faceImgSize=cvSize(faceImgArr[0]->width,faceImgArr[0]->height);

 // set the number of eigenvalues to use
 nEigens = nTrainFaces-1;

 // allocate the eigenvector images
 eigenVectArr =(IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);
 for(int i=0; i<nEigens; i++)
 eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

 // allocate the eigenvalue array
 eigenValMat = cvCreateMat( 1, nEigens, CV_32FC1 );

 // allocate the averaged image
 pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

 // set the PCA termination criterion
 calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, nEigens, 1);

 // compute average image, eigenvalues, and eigenvectors
 cvCalcEigenObjects(nTrainFaces, (void*)faceImgArr, (void*)eigenVectArr, 0, 0, 0, &calcLimit, pAvgTrainImg, eigenValMat->data.fl);

}
void pca::trainBpn()
{

	int layerSizes[]={nEigens,nEigens/2,1};
	TransferFunction tfunc[] = {TransferFunction::None,TransferFunction::Sigmoid,TransferFunction::Sigmoid};
	int numberOfLayers=sizeof(layerSizes)/sizeof(int);
	bpn=new BackPropagationNetwork*[noOfPerson];
	double error=0.0;
	long counter=0;
	double** faceProjection=new double*[imagesPerPerson];
	for(int i=0;i<imagesPerPerson;i++)
		faceProjection[i]=new double[nEigens];
	double desired=0.5;
	double notdesired=0;
	for(int i=0;i<noOfPerson;i++)
		{bpn[i]=new BackPropagationNetwork(layerSizes,tfunc,numberOfLayers);
	     error=0.0;
		 counter=0;
		 
		for(int j=0;j<imagesPerPerson;j++)
			for(int k=0;k<nEigens;k++)
			{
				faceProjection[j][k]=(double)*(projectedTrainFaceMat->data.fl+i*nEigens*imagesPerPerson+j*nEigens+k);
			}
		 do
		 {
			 error=0;
			 for(int j=0;j<imagesPerPerson;j++)
             error+=bpn[i]->train(faceProjection[j],&desired,0.2,0.0,nEigens,1);

		     counter++;
			 if(counter%1000==0)
				 {
				 FormContainer::f->consoleOutput("\ntraining neural network ");
				 FormContainer::f->consoleOutput(i);
				 FormContainer::f->consoleOutput(" epic = ");
				 FormContainer::f->consoleOutput(counter);
				 FormContainer::f->consoleOutput(" error = ");
				 FormContainer::f->consoleOutput(error);
				 if(counter%100000==0 && error>1e-6)
					 bpn[i]->weightRandomize();
			     }
		 }while(error>1e-8 && counter<1000000);
			 FormContainer::f->consoleOutput("\nXXXX---trained neural network----XXX ");
			 FormContainer::f->consoleOutput(i);
			 FormContainer::f->consoleOutput(" epic = ");
			 FormContainer::f->consoleOutput(counter);
			 FormContainer::f->consoleOutput(" error = ");
			 FormContainer::f->consoleOutput(error);
	    }
}
void pca::storeTrainingData()
{
 CvFileStorage * fileStorage;
 int i;

 // create a file-storage interface
 fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_WRITE );

 // store all the data
 cvWriteInt( fileStorage,"noOfPerson",noOfPerson);
 cvWriteInt( fileStorage, "nEigens", nEigens );
 cvWriteInt( fileStorage, "nTrainFaces", nTrainFaces );
 cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0,0));
 cvWrite(fileStorage, "projectedTrainFaceMat", projectedTrainFaceMat, cvAttrList(0,0));
 cvWrite(fileStorage, "avgTrainImg", pAvgTrainImg, cvAttrList(0,0));
 
 for(i=0; i<nEigens; i++)
 {
 char varname[200];
 sprintf( varname, "eigenVect_%d", i );
 cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0,0));
 }
  
 for(i=0; i<noOfPerson; i++)
 {
	 (**(bpn+i)).save("bpndata.dat");
 }
 // release the file-storage interface
 cvReleaseFileStorage( &fileStorage );
  
}
void pca::loadTrainingData()
{
	
 CvFileStorage * fileStorage;
 int i;

 // create a file-storage interface
 fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_READ );
 if( !fileStorage )
 {
 FormContainer::f->consoleOutput("\nCan't open facedata.xml");
 system("pause");
 exit(-1);
 }
 noOfPerson=cvReadIntByName(fileStorage, 0, "noOfPerson", 0);
 nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
 nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
 eigenValMat = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
 projectedTrainFaceMat =(CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
 pAvgTrainImg = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
 imagesPerPerson=nTrainFaces/noOfPerson;
 eigenVectArr =(IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);
 bpn=new BackPropagationNetwork*[nTrainFaces];
 for(i=0; i<nEigens; i++)
 {
 char varname[200];
 sprintf( varname, "eigenVect_%d", i );
 eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
 }
 bpn=new BackPropagationNetwork*[noOfPerson];
 for(i=0; i<noOfPerson; i++)
 {
	 BackPropagationNetwork::load("bpndata.dat",(bpn+i));
 }

 // release the file-storage interface
 cvReleaseFileStorage( &fileStorage );


}
int pca::runBpn(float * projectedTestFace)
{
double* input=new double[nEigens];
double** op=new double*[noOfPerson];
for(int i=0;i<nEigens;i++)
	input[i]=(double)projectedTestFace[i];
for(int i=0;i<noOfPerson;i++)
	bpn[i]->run(input,&op[i],nEigens);
double min=0;
int faceNumber;
double error=0.0;
for(int i=0;i<noOfPerson;i++)
{
	//*op[i] is the output value of NN i corresponding to face i
	error=(*op[i]-0.5)*(*op[i]-0.5);   //error in output NN i
	FormContainer::f->consoleOutput("\n error in ");
	FormContainer::f->consoleOutput(i);
	FormContainer::f->consoleOutput(" th NN output = ");
	FormContainer::f->consoleOutput(error);
	if(i==0)
	{
	min=error;
	faceNumber=0;
	}
	else if(error<min)
		{
		min=error;
	    faceNumber=i;
	    }
}
	FormContainer::f->consoleOutput("\n\nminimum recognition error = ");
	FormContainer::f->consoleOutput(min);
	FormContainer::f->consoleOutput("\n");
delete []input;
for(int i=0;i<noOfPerson;i++)
	delete op[i];
delete[] op;
return faceNumber;
}
IplImage* pca::detectFace()
{ const CvSize resolution=cvSize(92,112);
  const char cascade_name[]="haarcascade_frontalface_default.xml";
  static CvMemStorage* storage = 0;
  static CvHaarClassifierCascade* cascade =(CvHaarClassifierCascade*)cvLoad(cascade_name,0,0,0 );
  static CvCapture* capture = cvCaptureFromCAM(0);
  if(detectfacecounter==0)   //initialization
  {
  storage = 0;
  cascade =(CvHaarClassifierCascade*)cvLoad(cascade_name,0,0,0 );
  capture = cvCaptureFromCAM(0);
  }
  detectfacecounter++;
  IplImage* insideFace=0;
  IplImage* resizedFace=0;
  IplImage* resizedGrayFace=0;
  IplImage* frame=0;
  CvSeq* face=0;
  CvPoint pt1, pt2;
  CvRect* r=0;
  char key;

  if( !cascade || !capture )
  {
	 FormContainer::f->consoleOutput("file load error");
	 cvReleaseImage(&insideFace);
	 cvReleaseImage(&resizedFace);
	 cvReleaseCapture( &capture );
	 cvReleaseMemStorage(&storage);
	 system("pause");
	 exit(-1);
  }

  storage = cvCreateMemStorage(0);
  cvNamedWindow( "face recognition (ESC to exit) (enter to capture)", 1 );
  while( 1 )
  {     cvReleaseImage(&insideFace);
		cvClearMemStorage( storage );
		frame = cvQueryFrame( capture );
		if( !frame )
		{
			FormContainer::f->consoleOutput("\nERROR: frame is null...\n" );
			system("pause");
			exit(-1);
		}
		cvFlip(frame,frame,1);
		face = cvHaarDetectObjects( frame, cascade, storage,1.1, 5, CV_HAAR_DO_CANNY_PRUNING,cvSize(150, 150) );	
	
		for(int i = 0; i < (face ? face->total : 0); i++ )
		{   
			r = (CvRect*)cvGetSeqElem( face, i );
			pt1.x = r->x;
			pt2.x = (r->x+r->width/2);
			pt1.y = r->y;
			pt2.y = (r->y+r->height/2);
			cvSetImageROI(frame,*r);
			insideFace=cvCreateImage(cvSize(r->width,r->height),IPL_DEPTH_8U,3);
			cvCopy(frame,insideFace);
			cvResetImageROI(frame);
			cvRectangle( frame, pt1, cvPoint((r->x+r->width),(r->y+r->height)), CV_RGB(230,20,232), 3, 8, 0 );	
		}//outer face loop

		cvShowImage( "face recognition (ESC to exit) (enter to capture)", frame );		
		//If ESC key pressed exit
		key=cvWaitKey(10);
		if( key == 27 )
		{   cvReleaseImage(&insideFace);
		    cvReleaseImage(&resizedFace);
			cvReleaseMemStorage(&storage);
			cvReleaseHaarClassifierCascade(&cascade);
			cvReleaseCapture( &capture );
			cvDestroyWindow("face recognition (ESC to exit) (enter to capture)");
			exit(-1);
		}	
		else if(key==13)
		{
			if(face->total==1)
			{   resizedFace=cvCreateImage(resolution,IPL_DEPTH_8U,3); 
				resizedGrayFace=cvCreateImage(resolution,IPL_DEPTH_8U,1);
				cvResize(insideFace,resizedFace);
				cvCvtColor(resizedFace,resizedGrayFace,CV_RGB2GRAY);
				cvEqualizeHist(resizedGrayFace,resizedGrayFace);
				cvReleaseImage(&resizedFace);
 				cvReleaseImage(&insideFace);
				//cvReleaseMemStorage(&storage);
				//cvReleaseHaarClassifierCascade(&cascade);
				if(detectfacecounter==nTrainFaces || noOfPerson==0)
				cvReleaseCapture( &capture );
				cvDestroyWindow("face recognition (ESC to exit) (enter to capture)");
				
				return resizedGrayFace;
			}
			else if(face->total<1)
			{
				FormContainer::f->consoleOutput("\nno face detected\n");
				cvShowImage( "face recognition (ESC to exit) (enter to capture)", frame );
			}
			else
			{
			FormContainer::f->consoleOutput("multiple face detected\n");
			}

		}
  }//end while

  
}//end detectFace()
void safeEntry()
{
bpnsavecount=0;
bpndatfilePosition=0;
detectfacecounter=0;
}
int pca::pcamain(int nop, int ipp,int mode)
{
safeEntry();
 if(mode==0)
	{
		 noOfPerson=nop;
		 imagesPerPerson=ipp;
		 pca::learn();
		 return 0;
		
		
	}
  else if(mode==1)
	{
		 pca::recognize();
		 return 0;
	} 

 return 0;
}

[STAThreadAttribute]

int main(array<System::String ^> ^args)
{   
	

	// Enabling Windows XP visual effects before any controls are created
	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false); 
	// Create the main window and run it
	FormContainer::f=gcnew Form1();
	Application::Run(FormContainer::f);


	return 0;
}
