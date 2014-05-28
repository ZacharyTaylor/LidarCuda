/* function for projecting lidar points
 *
 */

#include "../common.h"

__global__ void CameraTransformKernel(const float* const tform,
									  const float* const cam,
									  const size_t imWidth,
									  const size_t imHeight,
									  const float* const xIn,
									  const float* const yIn,
									  const float* const zIn,
									  const float* const xDIn,
									  const float* const yDIn,
									  const float* const zDIn,
									  const size_t numPoints,
									  float* const xOut,
									  float* const yOut,
									  float* const xDOut,
									  float* const yDOut,
									  bool* const valid){

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= numPoints){
		return;
	}

	//transform points
	float x = xIn[i]*tform[0] + yIn[i]*tform[4] + zIn[i]*tform[8] + tform[12];
	float y = xIn[i]*tform[1] + yIn[i]*tform[5] + zIn[i]*tform[9] + tform[13];
	float z = xIn[i]*tform[2] + yIn[i]*tform[6] + zIn[i]*tform[10] + tform[14];
	float xD = xDIn[i]*tform[0] + yDIn[i]*tform[4] + zDIn[i]*tform[8] + tform[12];
	float yD = xDIn[i]*tform[1] + yDIn[i]*tform[5] + zDIn[i]*tform[9] + tform[13];
	float zD = xDIn[i]*tform[2] + yDIn[i]*tform[6] + zDIn[i]*tform[10] + tform[14];

	bool v = true;
	if((z > 0)){
		//apply projective camera matrix
		x = cam[0]*x + cam[3]*y + cam[6]*z + cam[9];
		y = cam[1]*x + cam[4]*y + cam[7]*z + cam[10];
		z = cam[2]*x + cam[5]*y + cam[8]*z + cam[11];
		xD = cam[0]*xD + cam[3]*yD + cam[6]*zD + cam[9];
		yD = cam[1]*xD + cam[4]*yD + cam[7]*zD + cam[10];
		zD = cam[2]*xD + cam[5]*yD + cam[8]*zD + cam[11];

		//pin point camera model
		y = y/z;
		x = x/z;
		yD = yD/zD;
		xD = xD/zD;

		if((x < 0) || (y < 0) || (x >= imWidth) || (y >= imHeight)){
			v = false;
		}
	}
	else{
		v = false;
	}

	//output points
	xOut[i] = x;
	yOut[i] = y;
	xDOut[i] = xD-x;
	yDOut[i] = yD-y;
	valid[i] = v;
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
//initialize the MathWorks GPU API.
    mxInitGPU();

    //read data
    mxGPUArray const * tformMat = mxGPUCreateFromMxArray(prhs[0]);
    mxGPUArray const * camMat = mxGPUCreateFromMxArray(prhs[1]);
    mxGPUArray const * pointsMat = mxGPUCreateFromMxArray(prhs[2]);
    size_t imWidth = ((uint32_T *) mxGetData(prhs[3]))[1];
    size_t imHeight = ((uint32_T *) mxGetData(prhs[3]))[0];
    size_t numPoints = mxGPUGetDimensions(pointsMat)[0];

	
    //get input pointers
    float* tformPtr = (float*)(mxGPUGetDataReadOnly(tformMat));
    float* camPtr = (float*)(mxGPUGetDataReadOnly(camMat));

    float* xInPtr = (float*)(mxGPUGetDataReadOnly(pointsMat));
	float* yInPtr = &(xInPtr[numPoints]);
    float* zInPtr = &(yInPtr[numPoints]);
	float* xDInPtr = &(zInPtr[numPoints]);
	float* yDInPtr = &(xDInPtr[numPoints]);
	float* zDInPtr = &(yDInPtr[numPoints]);
	
    //create output
	mwSize outSize[] = {numPoints,4};
    mxGPUArray* outMat = mxGPUCreateGPUArray(2, outSize, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	plhs[1] = mxGPUCreateMxArrayOnGPU(outMat);
    outSize[1] = 1;
    mxGPUArray* validMat = mxGPUCreateGPUArray(2, outSize, mxLOGICAL_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	plhs[0] = mxGPUCreateMxArrayOnGPU(validMat);

    float* xOutPtr = (float*)(mxGPUGetData(outMat));
	float* yOutPtr = &(xOutPtr[numPoints]);
	float* xDOutPtr = &(yOutPtr[numPoints]);
	float* yDOutPtr = &(xDOutPtr[numPoints]);
	bool* validPtr = (bool*)(mxGPUGetData(validMat));

    //run and get ouputs
	CameraTransformKernel<<<gridSize(numPoints), BLOCK_SIZE>>>(tformPtr, camPtr, imWidth, imHeight, xInPtr, yInPtr, zInPtr, xDInPtr, yDInPtr, zDInPtr, numPoints, xOutPtr, yOutPtr, xDOutPtr, yDOutPtr,validPtr);
	CudaCheckError();
	
    //destroy reference structures
    mxGPUDestroyGPUArray(tformMat);
    mxGPUDestroyGPUArray(camMat);
    mxGPUDestroyGPUArray(pointsMat);
    mxGPUDestroyGPUArray(outMat);
	mxGPUDestroyGPUArray(validMat);
}
