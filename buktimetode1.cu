
   #include <cuda_runtime.h>
   #include <cuda.h>
   #include <stdio.h>
   #include <stdlib.h>
   #include <time.h>
   #include "kuda.h"
   #include "lodepng.h"
// Banyak nx * nx Matrix
// Banyak Max * Max Matrix
int Max;

void printMatrixGlcm(int *C, const int Max,int degree)
{
    int *ic = C;
    FILE * fp = NULL;
    if(degree==0){
        fp = fopen("matrix_glcm_0_method1_bukti.txt", "w+");
    }
    else if(degree==90){
        fp = fopen("matrix_glcm_90.txt", "w+");
    }
    else if(degree==180){
        fp = fopen("matrix_glcm_180.txt", "w+");
    }
    else if(degree==270){
        fp = fopen("matrix_glcm_270.txt", "w+");
    }
    else if(degree==45){
        fp = fopen("matrix_glcm_45.txt", "w+");
    }
    else if(degree==135){
        fp = fopen("matrix_glcm_135.txt", "w+");
    }
    else if(degree==225){
        fp = fopen("matrix_glcm_225.txt", "w+");
    }
    else if(degree==315){
        fp = fopen("matrix_glcm_315.txt", "w+");
    }

    if(fp == NULL){
        printf("Error creating results file\n");
        exit(1);
    }
    for (int iy = 0; iy <Max; iy++)
    {
        for (int ix = 0; ix <Max; ix++)
        {
            fprintf(fp, "%d  ", ic[ix]);

        }
        fprintf(fp, "\n\n");
        ic += (Max);

    }

    printf("\n");
    fclose(fp);
    return;
}

void printMatrixnxormalization(float *C, const int Max,int degree)
{
    float *ic = C;
    FILE * fp = NULL;
    if(degree==0){
        fp = fopen("matrix_normalisasi_0_metode1bukti.txt", "w+");
    }
    else if(degree==90){
        fp = fopen("matrix_normalisasi_90.txt", "w+");
    }
    else if(degree==180){   
        fp = fopen("matrix_normalisasi_180.txt", "w+");
    }
    else if(degree==270){
        fp = fopen("matrix_normalisasi_270.txt", "w+");
    }
    else if(degree==45){
        fp = fopen("matrix_normalisasi_45.txt", "w+");
    }
    else if(degree==135){
        fp = fopen("matrix_normalisasi_135.txt", "w+");
    }
    else if(degree==225){
        fp = fopen("matrix_normalisasi_225.txt", "w+");
    }
    else if(degree==315){
        fp = fopen("matrix_normalisasi_315.txt", "w+");
    }

    if(fp == NULL){
        printf("Error creating results file\n");
        exit(1);
    }
    for (int iy = 0; iy < Max; iy++)
    {
        for (int ix = 0; ix <Max; ix++)
        {

            fprintf(fp, "%.7f  ", ic[ix]);

        }
        fprintf(fp, "\n\n");
        ic +=Max;

    }

    printf("\n");
    fclose(fp);
    return;
}

// void calculate_glcm_host(int *matrix,int *glcm,int nx,int ny,int Max){
//     int i,j;
//     for(i=0;i<nx;i++){
//         for(j=0;j<ny;j++){
//             glcm[Max*matrix[i]+matrix[j]] +=1;
//         }
//     }
// }

//calculate glcm
__global__ void glcm_calculation_nol(int *A,int *glcm, const int nx, const int ny,int maxx)
{

    unsigned int idx =blockIdx.x*nx+threadIdx.x;
    int i;
    int k=0;
    for(i=0;i<nx;i++){
        if(idx>=i*nx && idx<((i+1) *nx)-1){
            k=maxx*A[idx]+A[idx+1];
            atomicAdd(&glcm[k],1);
        }
    }

}


__global__ void glcm_calculation_180(int *A,int *glcm, const int nx, const int ny,int max){
    //int iy = threadIdx.y + blockIdx.y* blockDim.y;
    unsigned int idx =blockIdx.x*nx+threadIdx.x;
    int i;
    int k=0;
    for(i=0;i<nx;i++){
        if(idx>=i*nx && idx<((i+1) *nx)-1){
            k=max*A[idx+1]+A[idx];
            atomicAdd(&glcm[k],1);
        }
    }
}

__global__ void glcm_calculation_270(int *A,int *glcm, const int nx, const int ny,int max){
    int ix = threadIdx.x + blockIdx.x* blockDim.x;
    int iy = threadIdx.y + blockIdx.y* blockDim.y;
    unsigned int idx =iy*nx+ix;
    int i;
    int k=0;
    for(i=0;i<nx-1;i++){
        if(idx>=i*nx && idx<((i+1) *nx)){
            k=max*A[idx]+A[idx+nx];
            atomicAdd(&glcm[k],1);           
        }
    }
    __syncthreads();
}

__global__ void glcm_calculation_90(int *A,int *glcm, const int nx, const int ny,int max){
    int ix = threadIdx.x + blockIdx.x* blockDim.x;
    int iy = threadIdx.y + blockIdx.y* blockDim.y;
    unsigned int idx =iy*nx+ix;
    int i;
    int k=0;
    for(i=0;i<nx-1;i++){
        if(idx>=i*nx && idx<((i+1) *nx)){
            k=max*A[idx+nx]+A[idx];
            atomicAdd(&glcm[k],1);          
        }
    }
    __syncthreads();
}

__global__ void glcm_calculation_45(int *A,int *glcm, const int nx, const int ny,int max){
    int ix = threadIdx.x + blockIdx.x* blockDim.x;
    int iy = threadIdx.y + blockIdx.y* blockDim.y;
    unsigned int idx =iy*nx+ix;
    int i;
    int k=0;
    for(i=1;i<nx;i++){
        if(blockIdx.x==i && idx <((i+1)*nx)-1){
            k=max*A[idx]+A[idx-(nx-1)];
            atomicAdd(&glcm[k],1);
        }
    }
    __syncthreads();
}

__global__ void glcm_calculation_135(int *A,int *glcm, const int nx, const int ny,int max){
    int ix = threadIdx.x + blockIdx.x* blockDim.x;
    int iy = threadIdx.y + blockIdx.y* blockDim.y;
    unsigned int idx =iy*nx+ix;
    int i;
    int k=0;
    for(i=1;i<nx;i++){
        if(blockIdx.x==i && idx >i*nx){
            k=max*A[idx]+A[idx-(nx+1)];
            atomicAdd(&glcm[k],1);
        }
    }
    __syncthreads();
}

__global__ void glcm_calculation_225(int *A,int *glcm, const int nx, const int ny,int max){
    int ix = threadIdx.x + blockIdx.x* blockDim.x;
    int iy = threadIdx.y + blockIdx.y* blockDim.y;
    unsigned int idx =iy*nx+ix;
    int i;
    int k=0;
    for(i=0;i<nx-1;i++){
        if(blockIdx.x==i && idx >i*nx){
            k=max*A[idx]+A[idx+(nx-1)];
            atomicAdd(&glcm[k],1);
        }
    }
    __syncthreads();
}

__global__ void glcm_calculation_315(int *A,int *glcm, const int nx, const int ny,int max){
    int ix = threadIdx.x + blockIdx.x* blockDim.x;
    int iy = threadIdx.y + blockIdx.y* blockDim.y;
    unsigned int idx =iy*nx+ix;
    int i;
    int k=0;
    for(i=0;i<nx-1;i++){
        if(blockIdx.x==i && idx <((i+1)*nx)-1){
            k=max*A[idx]+A[idx+(nx+1)];
            atomicAdd(&glcm[k],1);
        }
    }
    __syncthreads();
}

__global__ void Mul(float *newMatrix,float *mulMatrix,int Max,float *sumMatrix){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // int Index = iy * nx + ix;

    for (int k = 0; k < Max; k++) {
        // Accumulate results for a single element
        // c[row * nx + col] += a[row * nx + k] * b[k * nx + col];
        // printf("C[%d] = a[%d] * b[%d]\n",row * nx + col,row * nx + k, k * nx + col);
        atomicAdd(&mulMatrix[row * Max + col],newMatrix[row * Max + k] * newMatrix[k * Max + col]);
        // atomicAdd(&sumMatrix[0],mulMatrix[row * Max + col]);
    }
}


__global__ void Jumlah(float *sumMatrix,float *mulMatrix){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    // if(Index<1) printf("%f",mulMatrix[0]);
    atomicAdd(&sumMatrix[0],mulMatrix[Index]);

}

__global__ void AddToitTranspose(int *transposed,int *glcm,int Max){
    int col = blockIdx.x * blockDim.x + threadIdx.x;                
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    //printf("%d  %d\n",row*Max+col,col*Max+row);
    
    transposed[row*Max+col]=glcm[row*Max+col]+glcm[col*Max+row];
    
    
}

__global__ void normalization(int *glcm,float *norm,int Max,int sum){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * Max + ix;
    __syncthreads();
    if(idx<(Max+1)*(Max+1)&&glcm[idx]!=0){
        norm[idx]=float(glcm[idx])/float(sum);
    }
}


__global__ void calculate_contrast(float *norm,float *contrast,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // int Index = iy * N + ix;

    // for (int k = 0; k < Max; k++) {
        // // Accumulate results for a single element
        // // c[row * N + col] += a[row * N + k] * b[k * N + col];
        // // printf("C[%d] = a[%d] * b[%d]\n",row * N + col,row * N + k, k * N + col);
        // atomicAdd(&mulMatrix[row * Max + col],norm[row * Max + k] * norm[k * Max + col]);
    // }
    
    if(norm[row*Max+col]>0){
        //printf("%f\n",norm[row*Max+col]);
        atomicAdd(&contrast[0],((row-col)*(row-col))*norm[row*Max+col]);
        //printf("nilai contrast %d %d %d %f\n",((row-col)*(row-col)),row,col,norm[row*Max+col]);
        //atomicAdd(&ASM[0],norm[row*Max+col]*norm[row*Max+col]);
        //printf("%f\n",contrast[0]);
    }
    

    

    // if (Index == 0){

    //     printf("ASM %f\n",ASM[0]);
    // }
}

__global__ void calculate_IDM(float *norm,float *IDM,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(norm[row*Max+col]>0){
        atomicAdd(&IDM[0],norm[row*Max+col] / (1+((row-col)*(row-col))) );
        //printf("nilai IDM %d %d %d %f\n",((row-col)*(row-col)),row,col,norm[row*Max+col]);
        //atomicAdd(&ASM[0],norm[row*Max+col]*norm[row*Max+col]);
        //printf("%f\n",IDM[0]);
    }

}

__global__ void calculate_entropy(float *norm,float *entropy,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(norm[row*Max+col]>0){
        atomicAdd(&entropy[0],(norm[row*Max+col] * log10f(norm[row*Max+col])) );
        //printf("nilai entropy %d %d %d %f\n",((row-col)*(row-col)),row,col,norm[row*Max+col]);
        //atomicAdd(&ASM[0],norm[row*Max+col]*norm[row*Max+col]);
        //printf("%f\n",entropy[0]);
    }

}


__global__ void calculate_ASM(float *norm,float *ASM,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    //printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
    if(norm[row*Max+col]>0){
       // printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
        atomicAdd(&ASM[0],norm[row*Max+col]*norm[row*Max+col]);
        //printf("%f\n",ASM[0]);
    }
}

__global__ void calculate_miu_i(float *norm,float *miu_i,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(norm[row*Max+col]>0){
        //printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
        atomicAdd(&miu_i[0],row*norm[row*Max+col]);
        //printf("nilai miu_i %d %d %d %f\n",((row-col)*(row-col)),row,col,norm[row*Max+col]);
        //atomicAdd(&ASM[0],norm[row*Max+col]*norm[row*Max+col]);
        //printf("%f\n",miu_i[0]);
    }
}



__global__ void calculate_miu_j(float *norm,float *miu_j,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(norm[row*Max+col]>0){
        //printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
        atomicAdd(&miu_j[0],col*norm[row*Max+col]);
        //printf("nilai miu_i %d %d %d %f\n",((row-col)*(row-col)),row,col,norm[row*Max+col]);
        //atomicAdd(&ASM[0],norm[row*Max+col]*norm[row*Max+col]);
        //printf("%f\n",miu_i[0]);
    }
}

__global__ void calculate_std_i(float *norm,float *std_i,float*miu_i,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(norm[row*Max+col]>0){
        //printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
        atomicAdd(&std_i[0],norm[row*Max+col] * ((row-miu_i[0])*(row-miu_i[0])));
        //printf("nilai miu_i %d %d %d %f\n",((row-col)*(row-col)),row,col,norm[row*Max+col]);
        //atomicAdd(&ASM[0],norm[row*Max+col]*norm[row*Max+col]);
        //printf("%f\n",miu_i[0]);
    }
}


__global__ void calculate_std_j(float *norm,float *std_i,float *miu_j,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(norm[row*Max+col]>0){
        //printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
        atomicAdd(&std_i[0],norm[row*Max+col]*(((col-miu_j[0])*(col-miu_j[0]))));
        //printf("nilai miu_i %d %d %d %f\n",((row-col)*(row-col)),row,col,norm[row*Max+col]);
        //atomicAdd(&ASM[0],norm[row*Max+col]*norm[row*Max+col]);
        //printf("%f\n",miu_i[0]);
    }

}__global__ void calculate_korelasi(float *norm,float *korelasi,float *miu_i,float *std_i,float *miu_j,float *std_j,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(norm[row*Max+col]>0){
        //printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
        atomicAdd(&korelasi[0],(((row-miu_i[0])*(col-miu_j[0]))*norm[row*Max+col])/(std_i[0]*std_j[0]));
        //printf("nilai korelasi %f %f \n",(row-miu_i[0]),(col-miu_j[0]));
        //atomicAdd(&ASM[0],norm[row*Max+col]*norm[row*Max+col]);
        //printf("%f\n",miu_i[0]);
    }
}

__global__ void calculate_variance(float *norm,float *variance,float *miu_i,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(norm[row*Max+col]>0){
        //printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
        atomicAdd(&variance[0],((row-miu_i[0])*(row-miu_i[0]))*norm[row*Max+col]);
        //printf("nilai korelasi %f %f \n",(row-miu_i[0]),(col-miu_j[0]));
        //atomicAdd(&ASM[0],norm[row*Max+col]*norm[row*Max+col]);
        //printf("%f\n",miu_i[0]);
    }
}

__global__ void calculate_sumaverage(float *norm,float *sav,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int k;
    for(k=2;k<2*Max;k++){
        if((row+col)==k){
            atomicAdd(&sav[0],k*(1*norm[row*Max+col]));
        }
        else{
            atomicAdd(&sav[0],0);
        }
    }
}

__global__ void calculate_sumentropy(float *norm,float *sen,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int k;
    for(k=2;k<2*Max;k++){
        if((row+col)==k && norm[row*Max+col]>0){
            //printf("%f\n",norm[row*Max+col]);
            atomicAdd(&sen[0],(1*norm[row*Max+col])*(log10(1*norm[row*Max+col])));
        }
        else{
            atomicAdd(&sen[0],0);
        }
    }
}

__global__ void calculate_sumvariance(float *norm,float *sva,float *sen,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int k;
    for(k=2;k<2*Max;k++){
        if((row+col)==k && norm[row*Max+col]>0){
            //printf("sva%f\n",norm[row*Max+col]);
            atomicAdd(&sva[0],((k-sen[0])*(k-sen[0]))*(1*norm[row*Max+col]));
        }
        else{
            atomicAdd(&sva[0],0);
        }
    }
}

__global__ void calculate_differenceentropy(float *norm,float *den,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int k=0;
   
    
    for(k=0;k<Max-1;k++){
        if(abs(row-col)==k && norm[row*Max+col]>0){
           
            atomicAdd(&den[0],(1*norm[row*Max+col])*(log10(1*norm[row*Max+col])));
            //printf("apa %f\n",den[0]);
        }
        else{
            
            atomicAdd(&den[0],0);
        }
    }
}

__global__ void calculate_HX(float *norm,float *HX,int Max){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(norm[row*Max+col]>0){
        atomicAdd(&HX[0],norm[row*Max+col]*log10f(norm[row*Max+col]));
    }
}

__global__ void calculate_HY(float *norm,float *HY,int Max){
    //printf("%d\n",max);
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(norm[row*Max+col]>0){
        atomicAdd(&HY[0],norm[row*Max+col]*log10f(norm[row*Max+col]));
        
    }
}

__global__ void calculate_HXY1(float *norm,float *HXY1,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(norm[row*Max+col]>0){
        //printf("%.13f %.13f %f %f \n",norm[row],norm[col],norm[row*Max+col],log10f(norm[row]*norm[col]));
        atomicAdd(&HXY1[0],norm[row*Max+col]*log10f(norm[row*Max+col]));
    }
}

__global__ void calculate_dva(float *norm,float *dva,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int k;
    for(k=0;k<Max-1;k++){
        if(abs(row-col)==k && norm[row*Max+col]>0){
            //printf("%f\n",norm[row*Max+col]);
            atomicAdd(&dva[0],(k*k)*(1*norm[row*Max+col]));
        }
        else{
            atomicAdd(&dva[0],0);
        }
    }
}

// void takeimagevalue(const char* filename, rgb_image *img)
// {

//      unsigned error;
//      unsigned char* png;
//      size_t pngsize;;

//      lodepng_load_file(&png, &pngsize, filename);
//      error = lodepng_decode32(&img->image, &img->width, &img->height, png, pngsize);

//      if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

// }

// void transformToGrayCuda(rgb_image *img){
// 	unsigned char* image = img->image;
//     unsigned char* image_d;
//     unsigned int  width = img->width;
//     unsigned int height = img->height;
//     int n =width*height;
//     size_t size = n * 4 * sizeof(unsigned char);


// 	int device_count = 0;
// 	cudaError_t status = cudaGetDeviceCount(&device_count);

// 	status = cudaMalloc((void **) &image_d, size);


// 	cudaMemcpy(image_d, image,  size, cudaMemcpyHostToDevice);

// 	dim3 block_size(16, 16);
// 	dim3 num_blocks(img->width / block_size.x, img->height / block_size.y);
//     setPixelToGrayscale<<<num_blocks, block_size>>>(image_d, img->width, img->height);



// 	cudaMemcpy(image, image_d, size, cudaMemcpyDeviceToHost);

// 	cudaFree(image_d);
// }


// __global__
// void setPixelToGrayscale(unsigned char *image, unsigned width, unsigned height)
// {
//     float gray;
//     float r, g, b;

// 	int x = blockIdx.x * blockDim.x + threadIdx.x;
// 	int y = blockIdx.y * blockDim.y + threadIdx.y;

// 	if (x < width && y < height) {
// 		r = image[4 * width * y + 4 * x + 0];
// 		g = image[4 * width * y + 4 * x + 1];
// 		b = image[4 * width * y + 4 * x + 2];
// 		gray =.299f*r + .587f*g + .114f*b;
// 		image[4 * width * y + 4 * x + 0] = gray;
// 		image[4 * width * y + 4 * x + 1] = gray;
// 		image[4 * width * y + 4 * x + 2] = gray;
// 		image[4 * width * y + 4 * x + 3] = 255;
// 	}

// }

// void saveimagegray(const char* filename, rgb_image *img)
// {
//   /*Encode the image*/
//   unsigned error = lodepng_encode32_file(filename, img->image, img->width, img->height);

//   /*if there's an error, display it*/
//   if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
// }

int main(int argc, char *argv[]){


    char *d;
    long deg =strtol(argv[1],&d,10);
    int degree=deg;
    // char *data;
    // long datas =strtol(argv[2],&data,10);
    // int ukuran=datas;
    
    //const char* filename = argc > 1 ? argv[1] : "test.png";
    //rgb_image img;
   // takeimagevalue(filename, &img);
   // transformToGrayCuda(&img);
    //int nx =img.width;
    //int ny =img.height;
    int ukuran,graylevel;
    printf("ukuran gambar: ");
    scanf("%d",&ukuran);
    printf("max graylevel : ");
    scanf("%d",&graylevel);

    printf("calculate glcm for image size %dx%d %d degre Starting...\n", ukuran,ukuran,degree);
    int nx,ny;
    nx=ukuran;
    ny=ukuran;
    
    int *matrix,*glcm,*transposed;
    float *norm,*mulMatrix,*sumMatrix;

    cudaMallocManaged(&matrix, (nx * ny) * sizeof(int));

    for(int i = 0 ; i < (nx * nx) ; ++i){
        matrix[i] = rand() %graylevel;
        if(matrix[i] > Max){
            Max = matrix[i];
        }
    }

    for(int i = 0 ; i < nx ; ++i){
        for(int j = 0 ; j < nx ; ++j){
            printf("%4d",matrix[i * nx + j]);
        }
        printf("\n");
    }
    //printf("\n\n");
    Max = Max + 1; // karena index dimulai dari 0 dan Maximum 3 ( 0 - 3 = 4 ) jadi Max ditambah 1;
    int kBytes = Max * Max * sizeof(float);
    int nBytes =  nx * ny * sizeof(float);
    cudaMallocManaged(&glcm, (Max * Max) * sizeof(int));
    cudaMallocManaged(&transposed, (Max * Max) * sizeof(int));
    cudaMallocManaged(&mulMatrix, (Max * Max) * sizeof(float));
    cudaMallocManaged(&sumMatrix, (Max * Max) * sizeof(float));
    cudaMallocManaged(&norm, (Max * Max) * sizeof(float));
    for(int i = 0 ; i < (Max * Max) ; ++i){
        glcm[i] = 0;
        mulMatrix[i] = 0;
    }

    float*ASM,*contrast,*IDM,*entropy,*miu_i,*miu_j,*std_i,*std_j,*korelasi,*variance,*sav,*sen,*sva,*den,*HX,*HY,*HXY1,*dva;


     cudaMallocManaged(&ASM, (Max * Max) * sizeof(float));
     cudaMallocManaged(&contrast, (Max * Max) * sizeof(float));
     cudaMallocManaged(&IDM, (Max * Max) * sizeof(float));
     cudaMallocManaged(&entropy, (Max * Max) * sizeof(float));
     cudaMallocManaged(&miu_i, (Max * Max) * sizeof(float));
     cudaMallocManaged(&miu_j, (Max * Max) * sizeof(float));
     cudaMallocManaged(&std_i, (Max * Max) * sizeof(float));
     cudaMallocManaged(&std_j, (Max * Max) * sizeof(float));
     cudaMallocManaged(&korelasi, (Max * Max) * sizeof(float));
     cudaMallocManaged(&variance, (Max * Max) * sizeof(float));
     cudaMallocManaged(&sav, (Max * Max) * sizeof(float));
     cudaMallocManaged(&sen, (Max * Max) * sizeof(float));
     cudaMallocManaged(&sva, (Max * Max) * sizeof(float));
     cudaMallocManaged(&den, (Max * Max) * sizeof(float));
     cudaMallocManaged(&HX, (Max * Max) * sizeof(float));
     cudaMallocManaged(&HY, (Max * Max) * sizeof(float));
     cudaMallocManaged(&dva, (Max * Max) * sizeof(float));
     cudaMallocManaged(&HXY1, (Max * Max) * sizeof(float));


    dim3 block(ny);
    dim3 grid((nx + block.x - 1) / block.x, (nx + block.y - 1) / block.y);
    dim3 blocks(2,2);
    dim3 grids((Max + blocks.x - 1) / blocks.x, (Max + blocks.y - 1) / blocks.y);

    cudaGetLastError();
    clock_t start, end;
    double t = 0;
    start = clock();
    // invoke kernel for calculation
    // glcm_calculation_nol<<<ny,nx>>>(matrix,glcm, nx, ny,Max);
    // cudaDeviceSynchronize();
   

    // AddToitTranspose<<<grids,blocks>>>(transposed,glcm,Max);
    // cudaDeviceSynchronize();
    // printf("hasil transpose\n");
    // for(int i = 0 ; i < Max ; ++i){
	// 	for(int j = 0 ; j < Max ; ++j){
	// 		printf("%4d  ",transposed[i * Max + j]);
	// 	}
	// 	printf("\n");
	// }
     if(degree==0){
         glcm_calculation_nol<<<ny,nx>>>(matrix,glcm, nx, ny,Max);
         cudaDeviceSynchronize();
         end = clock();
         AddToitTranspose<<<grid,block>>>(transposed,glcm,Max);
         cudaDeviceSynchronize();
         printMatrixGlcm(transposed,Max,degree);
     }
     else if(degree ==180){
         glcm_calculation_180<<<ny,nx>>>(matrix,glcm, nx, ny,Max);
         cudaDeviceSynchronize();
         end = clock();
         AddToitTranspose<<<grid,block>>>(transposed,glcm,Max);
         cudaDeviceSynchronize();
         printMatrixGlcm(transposed,Max,degree);
     }
     else if(degree==270){
         dim3 block(1, nx);
         dim3 grid((ny + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

         glcm_calculation_270<<<grid,block>>>(matrix,glcm, nx, ny,Max);
         cudaDeviceSynchronize();
         end = clock();
         AddToitTranspose<<<grid,block>>>(transposed,glcm,Max);
         cudaDeviceSynchronize();
         printMatrixGlcm(transposed,Max,degree);
     }
     else if(degree==90){
         dim3 block(1, nx);
         dim3 grid((ny + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

         glcm_calculation_90<<<grid,block>>>(matrix,glcm, nx, ny,Max);
         cudaDeviceSynchronize();
         end = clock();
         AddToitTranspose<<<grid,block>>>(transposed,glcm,Max);
         cudaDeviceSynchronize();
         printMatrixGlcm(transposed,Max,degree);
     }
     else if(degree==45){
         glcm_calculation_45<<<ny,nx>>>(matrix,glcm, nx, ny,Max);
         cudaDeviceSynchronize();
         end = clock();
         AddToitTranspose<<<grid,block>>>(transposed,glcm,Max);
         cudaDeviceSynchronize();
         printMatrixGlcm(transposed,Max,degree);
     }
     else if(degree==135){
         glcm_calculation_135<<<ny,nx>>>(matrix,glcm, nx, ny,Max);
         cudaDeviceSynchronize();
         end = clock();
         AddToitTranspose<<<grid,block>>>(transposed,glcm,Max);
         cudaDeviceSynchronize();
         printMatrixGlcm(transposed,Max,degree);
     }
     else if(degree==225){
         glcm_calculation_225<<<ny,nx>>>(matrix,glcm, nx, ny,Max);
         cudaDeviceSynchronize();
         end = clock();
         AddToitTranspose<<<grid,block>>>(transposed,glcm,Max);
         cudaDeviceSynchronize();
         printMatrixGlcm(transposed,Max,degree);
     }
     else if(degree==315){
         glcm_calculation_315<<<ny,nx>>>(matrix,glcm, nx, ny,Max);
         cudaDeviceSynchronize();
         end = clock();
         AddToitTranspose<<<grid,block>>>(transposed,glcm,Max);
         cudaDeviceSynchronize();
         printMatrixGlcm(transposed,Max,degree);
     }
    // printMatrixGlcm(glcm,Max,0);
     printf("hasil glcm\n");
     for(int i = 0 ; i < Max ; ++i){
	 	for(int j = 0 ; j < Max ; ++j){
	 		printf("%4d  ",glcm[i * Max + j]);
	 	}
	 	printf("\n");
	 }
    
    t = ((double) (end - start))/CLOCKS_PER_SEC;
    
   

    
    int sum;
    sum=0;
    for(int i=0;i<Max*Max;i++){
        sum +=transposed[i];
        //if(transposed[i]>0){
        //    printf("%d\n",transposed[i]);
        //}
    }
    printf("sum %d",sum);
    normalization<<<Max,Max>>>(transposed,norm,Max,sum);


    cudaDeviceSynchronize();
    printf("Hasil normalisasi : \n");
    // for(int i = 0 ; i < Max ; ++i){
    //     for(int j = 0 ; j < Max ; ++j){
    //         //if(norm[i * Max + j]>0) 
    //        printf("%.7f ",norm[i * Max + j]);
    //     }
    //     printf("\n");
    // }
    printMatrixnxormalization(norm,Max,0);
    float sums;
    sums=0;
    for(int i=0;i<Max*Max;i++){
        sums  +=norm[i];

    }
    //Jumlah <<< Max,Max >>>(sumMatrix,norm);
    printf("jumlah %f\n",sums);
    int *dif;
    dif = (int *)malloc(kBytes);
    int *d_dif;
    (cudaMalloc((void **)&d_dif, nBytes));

    // transfer data from host to device
    (cudaMemcpy(d_dif, dif, kBytes, cudaMemcpyHostToDevice));


    dim3 b(32,32);
    dim3 g((Max + b.x - 1) / b.x, (Max + b.y - 1) / b.y);

    //Step1
    calculate_contrast<<<g,b>>>(norm,contrast,Max);
    calculate_entropy<<<g,b>>>(norm,entropy,Max);
    calculate_IDM<<<g,b>>>(norm,IDM,Max);
    calculate_ASM<<<g,b>>>(norm,ASM,Max);
    calculate_miu_i<<<g,b>>>(norm,miu_i,Max);
    calculate_miu_j<<<g,b>>>(norm,miu_j,Max); 
    cudaDeviceSynchronize();
    //Step2
    calculate_std_i<<<g,b>>>(norm,std_i,miu_i,Max);
    calculate_std_j<<<g,b>>>(norm,std_j,miu_j,Max);
    calculate_variance<<<g,b>>>(norm,variance,miu_i,Max);
    calculate_sumaverage<<<g,b>>>(norm,sav,Max);
    calculate_sumentropy<<<g,b>>>(norm,sen,Max);
    calculate_differenceentropy<<<g,b>>>(norm,den,Max);
    calculate_HX<<<g,b>>>(norm,HX,Max);
    calculate_HY<<<g,b>>>(norm,HY,Max);
    calculate_HXY1<<<g,b>>>(norm,HXY1,Max);
    cudaDeviceSynchronize();
    //Step3
    calculate_sumvariance<<<g,b>>>(norm,sva,sen,Max);
    calculate_korelasi<<<g,b>>>(norm,korelasi,miu_i,std_i,miu_j,std_j,Max);
    calculate_dva<<<g,b>>>(norm,dva,Max);
    cudaDeviceSynchronize();


    
    printf("ASM : %.13f\n",ASM[0]);
    printf("Contrast : %.13f\n",contrast[0]);
    printf("IDM : %.13f\n",IDM[0]);
    printf("entropy : %.13f\n",-(entropy[0]));
    printf("miu_i : %.13f\n",(miu_i[0]));
    printf("miu_j : %.13f\n",(miu_j[0]));
    printf("std_i : %.13f\n",(std_i[0]));
    printf("std_j : %.13f\n",(std_j[0]));
    printf("variance : %.13f\n",(variance[0]));
    printf("SAV : %.13f\n",(sav[0]));
    printf("SEN : %.13f\n",-(sen[0]));
    printf("SVA : %.13f\n",(sva[0]));
    printf("DEN : %.13f\n",-(den[0]));
    printf("HX : %.13f\n",-(HX[0]));
    printf("HY : %.13f\n",-(HY[0]));
    printf("HXY1 : %.13f\n",-(HXY1[0]));
    printf("IMC : %.13f\n",(entropy[0]-HXY1[0])/max(-(HX[0]),-(HY[0])));
    printf("korelasi : %.13f\n",(korelasi[0]));
    printf("Differnece Variance : %.13f\n",(dva[0]));



    printf("matrix gambar disimpan di matrixgambarmetode1 bukti.txt\n");
    printf("matrix glcm disimpan di matrix_glcm_bukti_%d.txt\n",0);
    printf("matrix glcm normalisasi disimpan di matrix_ormalisasi_%d_bukti.txt\n",0);


    printf("waktu eksekusi: %f\n",t);
    // free host and devide memory
    cudaFree(matrix);cudaFree(glcm);cudaFree(norm);
    cudaFree(mulMatrix);
}