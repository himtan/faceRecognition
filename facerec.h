#ifndef pca_h
#define pca_h

class pca
{
private:
	
static IplImage** faceImgArr;
static IplImage* pAvgTrainImg;
static IplImage** eigenVectArr;
static CvMat* eigenValMat;
static CvMat * projectedTrainFaceMat;
static int nTrainFaces;
static int nEigens;
static BackPropagationNetwork** bpn;

static void doPCA();
static void storeTrainingData();
static void loadTrainingData();
static void trainBpn();
static int runBpn(float * projectedTestFace);
static IplImage* detectFace();
public:
static void learn();            // used:- bpn,nEigens,nTrainFaces,personNumTruthMat,projectedTrainFaceMat,eigenValMat,eigenVectArr,pAvgTrainImg
static void recognize();        //used:-  bpn,eigenVectArr,pAvgTrainImg,nEigens,personNumTruthMat,eigenValMat----computed---->projectedTrainFaceMat(applied to bpn)
static void printUsage();
static int pcamain(int,int,int);

};
#endif