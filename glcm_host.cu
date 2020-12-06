
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


void printMatrixGlcm(int *C, const int Max,int degree,int gambar)
{
    int *ic = C;
    FILE * fp = NULL;
    if(gambar==128){
        if(degree==0){
            fp = fopen("./data/GLCM/Sample128/Sudut_0/matrix_glcm_0_host.txt", "w+");
        }
        else if(degree==90){
            fp = fopen("./data/GLCM/Sample128/Sudut_90/matrix_glcm_90_host.txt", "w+");
        }
        else if(degree==180){
            fp = fopen("./data/GLCM/Sample128/Sudut_180/matrix_glcm_180_host.txt", "w+");
        }
        else if(degree==270){
            fp = fopen("./data/GLCM/Sample128/Sudut_270/matrix_glcm_270_host.txt", "w+");
        }
        else if(degree==45){
            fp = fopen("./data/GLCM/Sample128/Sudut_45/matrix_glcm_45_host.txt", "w+");
        }
        else if(degree==135){
            fp = fopen("./data/GLCM/Sample128/Sudut_135/matrix_glcm_135_host.txt", "w+");
        }
        else if(degree==225){
            fp = fopen("./data/GLCM/Sample128/Sudut_225/matrix_glcm_225_host.txt", "w+");
        }
        else if(degree==315){
            fp = fopen("./data/GLCM/Sample128/Sudut_315/matrix_glcm_315_host.txt", "w+");
        }
    }
    else if(gambar==256){
        if(degree==0){
            fp = fopen("./data/GLCM/Sample256/Sudut_0/matrix_glcm_0_host.txt", "w+");
        }
        else if(degree==90){
            fp = fopen("./data/GLCM/Sample256/Sudut_90/matrix_glcm_90_host.txt", "w+");
        }
        else if(degree==180){
            fp = fopen("./data/GLCM/Sample256/Sudut_180/matrix_glcm_180_host.txt", "w+");
        }
        else if(degree==270){
            fp = fopen("./data/GLCM/Sample256/Sudut_270/matrix_glcm_270_host.txt", "w+");
        }
        else if(degree==45){
            fp = fopen("./data/GLCM/Sample256/Sudut_45/matrix_glcm_45_host.txt", "w+");
        }
        else if(degree==135){
            fp = fopen("./data/GLCM/Sample256/Sudut_135/matrix_glcm_135_host.txt", "w+");
        }
        else if(degree==225){
            fp = fopen("./data/GLCM/Sample256/Sudut_225/matrix_glcm_225_host.txt", "w+");
        }
        else if(degree==315){
            fp = fopen("./data/GLCM/Sample256/Sudut_315/matrix_glcm_315_host.txt", "w+");
        }
    }
    else if(gambar==512){
        if(degree==0){
            fp = fopen("./data/GLCM/Sample512/Sudut_0/matrix_glcm_0_host.txt", "w+");
        }
        else if(degree==90){
            fp = fopen("./data/GLCM/Sample512/Sudut_90/matrix_glcm_90_host.txt", "w+");
        }
        else if(degree==180){
            fp = fopen("./data/GLCM/Sample512/Sudut_180/matrix_glcm_180_host.txt", "w+");
        }
        else if(degree==270){
            fp = fopen("./data/GLCM/Sample512/Sudut_270/matrix_glcm_270_host.txt", "w+");
        }
        else if(degree==45){
            fp = fopen("./data/GLCM/Sample512/Sudut_45/matrix_glcm_45_host.txt", "w+");
        }
        else if(degree==135){
            fp = fopen("./data/GLCM/Sample512/Sudut_135/matrix_glcm_135_host.txt", "w+");
        }
        else if(degree==225){
            fp = fopen("./data/GLCM/Sample512/Sudut_225/matrix_glcm_225_host.txt", "w+");
        }
        else if(degree==315){
            fp = fopen("./data/GLCM/Sample512/Sudut_315/matrix_glcm_315_host.txt", "w+");
        }
    }
    else if(gambar==1024){
        if(degree==0){
            fp = fopen("./data/GLCM/Sample1024/Sudut_0/matrix_glcm_0_host.txt", "w+");
        }
        else if(degree==90){
            fp = fopen("./data/GLCM/Sample1024/Sudut_90/matrix_glcm_90_host.txt", "w+");
        }
        else if(degree==180){
            fp = fopen("./data/GLCM/Sample1024/Sudut_180/matrix_glcm_180_host.txt", "w+");
        }
        else if(degree==270){
            fp = fopen("./data/GLCM/Sample1024/Sudut_270/matrix_glcm_270_host.txt", "w+");
        }
        else if(degree==45){
            fp = fopen("./data/GLCM/Sample1024/Sudut_45/matrix_glcm_45_host.txt", "w+");
        }
        else if(degree==135){
            fp = fopen("./data/GLCM/Sample1024/Sudut_135/matrix_glcm_135_host.txt", "w+");
        }
        else if(degree==225){
            fp = fopen("./data/GLCM/Sample1024/Sudut_225/matrix_glcm_225_host.txt", "w+");
        }
        else if(degree==315){
            fp = fopen("./data/GLCM/Sample1024/Sudut_315/matrix_glcm_315_host.txt", "w+");
        }
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

void printMatrixnxormalization(float *C, const int Max,int degree,int gambar)
{
    float *ic = C;
    FILE * fp = NULL;
    if(gambar==128){
        if(degree==0){
            fp = fopen("./data/Normalisasi/Sample128/Sudut_0/matrix_normalisasi_0_host.txt", "w+");
        }
        else if(degree==90){
            fp = fopen("./data/Normalisasi/Sample128/Sudut_90/matrix_normalisasi_90_host.txt", "w+");
        }
        else if(degree==180){
            fp = fopen("./data/Normalisasi/Sample128/Sudut_180/matrix_normalisasi_180_host.txt", "w+");
        }
        else if(degree==270){
            fp = fopen("./data/Normalisasi/Sample128/Sudut_270/matrix_normalisasi_270_host.txt", "w+");
        }
        else if(degree==45){
            fp = fopen("./data/Normalisasi/Sample128/Sudut_45/matrix_normalisasi_45_host.txt", "w+");
        }
        else if(degree==135){
            fp = fopen("./data/Normalisasi/Sample128/Sudut_135/matrix_normalisasi_135_host.txt", "w+");
        }
        else if(degree==225){
            fp = fopen("./data/Normalisasi/Sample128/Sudut_225/matrix_normalisasi_225_host.txt", "w+");
        }
        else if(degree==315){
            fp = fopen("./data/Normalisasi/Sample128/Sudut_315/matrix_normalisasi_315_host.txt", "w+");
        }
    }
    else if(gambar==256){
        if(degree==0){
            fp = fopen("./data/Normalisasi/Sample256/Sudut_0/matrix_normalisasi_0_host.txt", "w+");
        }
        else if(degree==90){
            fp = fopen("./data/Normalisasi/Sample256/Sudut_90/matrix_normalisasi_90_host.txt", "w+");
        }
        else if(degree==180){
            fp = fopen("./data/Normalisasi/Sample256/Sudut_180/matrix_normalisasi_180_host.txt", "w+");
        }
        else if(degree==270){
            fp = fopen("./data/Normalisasi/Sample256/Sudut_270/matrix_normalisasi_270_host.txt", "w+");
        }
        else if(degree==45){
            fp = fopen("./data/Normalisasi/Sample256/Sudut_45/matrix_normalisasi_45_host.txt", "w+");
        }
        else if(degree==135){
            fp = fopen("./data/Normalisasi/Sample256/Sudut_135/matrix_normalisasi_135_host.txt", "w+");
        }
        else if(degree==225){
            fp = fopen("./data/Normalisasi/Sample256/Sudut_225/matrix_normalisasi_225_host.txt", "w+");
        }
        else if(degree==315){
            fp = fopen("./data/Normalisasi/Sample256/Sudut_315/matrix_normalisasi_315_host.txt", "w+");
        }
    }
    else if(gambar==512){
        if(degree==0){
            fp = fopen("./data/Normalisasi/Sample512/Sudut_0/matrix_normalisasi_0_host.txt", "w+");
        }
        else if(degree==90){
            fp = fopen("./data/Normalisasi/Sample512/Sudut_90/matrix_normalisasi_90_host.txt", "w+");
        }
        else if(degree==180){
            fp = fopen("./data/Normalisasi/Sample512/Sudut_180/matrix_normalisasi_180_host.txt", "w+");
        }
        else if(degree==270){
            fp = fopen("./data/Normalisasi/Sample512/Sudut_270/matrix_normalisasi_270_host.txt", "w+");
        }
        else if(degree==45){
            fp = fopen("./data/Normalisasi/Sample512/Sudut_45/matrix_normalisasi_45_host.txt", "w+");
        }
        else if(degree==135){
            fp = fopen("./data/Normalisasi/Sample512/Sudut_135/matrix_normalisasi_135_host.txt", "w+");
        }
        else if(degree==225){
            fp = fopen("./data/Normalisasi/Sample512/Sudut_225/matrix_normalisasi_225_host.txt", "w+");
        }
        else if(degree==315){
            fp = fopen("./data/Normalisasi/Sample512/Sudut_315/matrix_normalisasi_315_host.txt", "w+");
        }
    }
    else if(gambar==1024){
        if(degree==0){
            fp = fopen("./data/Normalisasi/Sample1024/Sudut_0/matrix_normalisasi_0_host.txt", "w+");
        }
        else if(degree==90){
            fp = fopen("./data/Normalisasi/Sample1024/Sudut_90/matrix_normalisasi_90_host.txt", "w+");
        }
        else if(degree==180){
            fp = fopen("./data/Normalisasi/Sample1024/Sudut_180/matrix_normalisasi_180_host.txt", "w+");
        }
        else if(degree==270){
            fp = fopen("./data/Normalisasi/Sample1024/Sudut_270/matrix_normalisasi_270_host.txt", "w+");
        }
        else if(degree==45){
            fp = fopen("./data/Normalisasi/Sample1024/Sudut_45/matrix_normalisasi_45_host.txt", "w+");
        }
        else if(degree==135){
            fp = fopen("./data/Normalisasi/Sample1024/Sudut_135/matrix_normalisasi_135_host.txt", "w+");
        }
        else if(degree==225){
            fp = fopen("./data/Normalisasi/Sample1024/Sudut_225/matrix_normalisasi_225_host.txt", "w+");
        }
        else if(degree==315){
            fp = fopen("./data/Normalisasi/Sample1024/Sudut_315/matrix_normalisasi_315_host.txt", "w+");
        }
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
//             glcm[matrix[i]][matrix[j]] +=1;
//         }
//     }
// }

//calculate glcm



void calculate_glcm_host(int *matrix,int *glcm,int N,int Max){
    int i,j,k,l;

    for(i=0;i<N;i++){
        for(j=0;j<N-1;j++){
            for(k=0;k<Max;k++){
                for(l=0;l<Max;l++){
                    if(matrix[Max*i+j]==k&&matrix[Max*i+(j+1)]==l){
                        //printf("%d,%d,%d,%d\n",matrix[Max*i+j],k,matrix[Max*i+(j+1)],l);
                        glcm[Max*k+l] +=1;
                    }
                }
            }
        }
    }
}


__global__ void blok1_0(int *matrix,int *glcm,int N, int Max){
    int i,j,k,l;

    for(i=0;i<N;i++){
        for(j=0;j<N-1;j++){
            for(k=0;k<Max;k++){
                for(l=0;l<Max;l++){
                    if(matrix[N*i+j]==k && matrix[N*i+(j+1)]==l){
                        //printf("%d,%d,%d,%d\n",matrix[Max*i+j],k,matrix[Max*i+(j+1)],l);
                        atomicAdd(&glcm[Max*k+l],1);
                    }
                }
            }
        }
    }

}


  __global__ void blok1_45(int *matrix,int *glcm,int N, int Max){
      int i,j,k,l;

      for(i=1;i<N;i++){
          for(j=0;j<N-1;j++){
            for(k=0;k<Max;k++){
                for(l=0;l<Max;l++){
                    if(matrix[N*i+j]==k && matrix[N*(i-1)+(j+1)]==l){
                        //printf("%d,%d,%d,%d\n",matrix[Max*i+j],k,matrix[Max*i+(j+1)],l);
                        atomicAdd(&glcm[Max*matrix[N*i+j]+matrix[N*(i-1)+(j+1)]] ,1);
                    }
                }
            }
             

          }
      }
  }

  __global__ void blok1_90(int *matrix,int *glcm,int N, int Max){
    int i,j,k,l;

    for(i=1;i<N;i++){
        for(j=0;j<N;j++){
            for(k=0;k<Max;k++){
                for(l=0;l<Max;l++){
                    if(matrix[N*i+j]==k &&matrix[N*(i-1)+j]==l){
                        //printf("%d,%d,%d,%d\n",matrix[Max*i+j],k,matrix[Max*i+(j+1)],l);
                        atomicAdd(&glcm[Max*matrix[N*i+j]+matrix[N*(i-1)+j]] ,1);
                    }                    
                }
            }  
          

        }
    }
}


__global__ void blok1_135(int *matrix,int *glcm,int N, int Max){
    int i,j,k,l;

    for(i=1;i<N;i++){
        for(j=1;j<N;j++){
            for(k=0;k<Max;k++){
                for(l=0;l<Max;l++){
                    if(matrix[N*i+j]==k &&matrix[N*(i-1)+(j-1)]==l){
                        //printf("%d,%d,%d,%d\n",matrix[Max*i+j],k,matrix[Max*i+(j+1)],l);
                        atomicAdd(&glcm[Max*matrix[N*i+j]+matrix[N*(i-1)+(j-1)]] ,1);
                    }                    
                }
            }

        }
    }
}

__global__ void blok1_180(int *matrix,int *glcm,int N, int Max){
    int i,j,k,l;

    for(i=0;i<N;i++){
        for(j=1;j<N;j++){
            for(k=0;k<Max;k++){
                for(l=0;l<Max;l++){
                    if(matrix[N*i+j]==k &&matrix[N*i+(j-1)]==l){
                        //printf("%d,%d,%d,%d\n",matrix[Max*i+j],k,matrix[Max*i+(j+1)],l);
                        atomicAdd(&glcm[Max*matrix[N*i+j]+matrix[N*i+(j-1)]] ,1);
                    }                    
                }
            }
           

        }
    }
}

__global__ void blok1_225(int *matrix,int *glcm,int N, int Max){
    int i,j,k,l;

    for(i=0;i<N-1;i++){
        for(j=1;j<N;j++){
            for(k=0;k<Max;k++){
                for(l=0;l<Max;l++){
                    if(matrix[N*i+j]==k &&matrix[N*(i+1)+(j-1)]==l){
                        //printf("%d,%d,%d,%d\n",matrix[Max*i+j],k,matrix[Max*i+(j+1)],l);
                        atomicAdd(&glcm[Max*matrix[N*i+j]+matrix[N*(i+1)+(j-1)]] ,1);
                    }                    
                }
            }
           

        }
    }
}


__global__ void blok1_270(int *matrix,int *glcm,int N, int Max){
    int i,j,k,l;

    for(i=0;i<N-1;i++){
        for(j=0;j<N;j++){
            for(k=0;k<Max;k++){
                for(l=0;l<Max;l++){
                    if(matrix[N*i+j]==k &&matrix[N*(i+1)+j]==l){
                        //printf("%d,%d,%d,%d\n",matrix[Max*i+j],k,matrix[Max*i+(j+1)],l);
                        atomicAdd(&glcm[Max*matrix[N*i+j]+matrix[N*(i+1)+j]] ,1);
                    }                    
                }
            }
           

        }
    }
}

__global__ void blok1_315(int *matrix,int *glcm,int N, int Max){
    int i,j,k,l;

    for(i=0;i<N-1;i++){
        for(j=0;j<N-1;j++){
            for(k=0;k<Max;k++){
                for(l=0;l<Max;l++){
                    if(matrix[N*i+j]==k &&matrix[N*(i+1)+(j+1)]==l){
                        //printf("%d,%d,%d,%d\n",matrix[Max*i+j],k,matrix[Max*i+(j+1)],l);
                        atomicAdd(&glcm[Max*matrix[N*i+j]+matrix[N*(i+1)+(j+1)]] ,1);
                    }                    
                }
            }
           

        }
    }
}

// __global__ void blok1_45(int *matrix,int *glcm,int N, int Max){
//     int i,j,k,l;

//     for(i=1;i<N;i+2){
//         for(j=1;j<N;j++){
//             for(k=0;k<Max;k++){
//                 for(l=0;l<Max;l++){
//                     if(matrix[N*i+j]==k&&matrix[N*i+(j+1)]==l){
//                         //printf("%d,%d,%d,%d\n",matrix[Max*i+j],k,matrix[Max*i+(j+1)],l);
//                         atomicAdd(&glcm[Max*k+l] ,1);
//                     }
//                 }
//             }
//         }
//     }

// }

__global__ void Mul(float *glcm,float *mulMatrix,int Max,float *sumMatrix){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // int Index = iy * nx + ix;

    for (int k = 0; k < Max; k++) {
        // Accumulate results for a single element
        // c[row * nx + col] += a[row * nx + k] * b[k * nx + col];
        // printf("C[%d] = a[%d] * b[%d]\n",row * nx + col,row * nx + k, k * nx + col);
        atomicAdd(&mulMatrix[row * Max + col],glcm[row * Max + k] * glcm[k * Max + col]);
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


    transposed[row*Max+col]=glcm[row*Max+col]+glcm[col*Max+row];

}

__global__ void normalization(int *glcm,float *norm,int Max,int sum){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * Max + ix;
    __syncthreads();
    if(idx<(Max+1)*(Max+1)){
        norm[idx]=float(glcm[idx])/float(sum);
    }
}
__global__ void calculate_contrast(float *norm,float *contrast,int Max){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(norm[row*Max+col]>0){
        atomicAdd(&contrast[0],((row-col)*(row-col))*norm[row*Max+col]);
    }

}

__global__ void calculate_IDM(float *norm,float *IDM,int Max){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(norm[row*Max+col]>0){
        atomicAdd(&IDM[0],norm[row*Max+col] / (1+((row-col)*(row-col))) );
    }

}

__global__ void calculate_entropy(float *norm,float *entropy,int Max){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(norm[row*Max+col]>0){
        atomicAdd(&entropy[0],(norm[row*Max+col] * log10f(norm[row*Max+col])) );

    }

}


__global__ void calculate_ASM(float *norm,float *ASM,int Max){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

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
    }
}



__global__ void calculate_miu_j(float *norm,float *miu_j,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(norm[row*Max+col]>0){
        //printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
        atomicAdd(&miu_j[0],col*norm[row*Max+col]);
    }
}

__global__ void calculate_std_i(float *norm,float *std_i,float*miu_i,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(norm[row*Max+col]>0){
        //printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
        atomicAdd(&std_i[0],norm[row*Max+col] * ((row-miu_i[0])*(row-miu_i[0])));
    }
}


__global__ void calculate_std_j(float *norm,float *std_i,float *miu_j,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(norm[row*Max+col]>0){
        //printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
        atomicAdd(&std_i[0],norm[row*Max+col]*(((col-miu_j[0])*(col-miu_j[0]))));
    }

}__global__ void calculate_korelasi(float *norm,float *korelasi,float *miu_i,float *std_i,float *miu_j,float *std_j,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(norm[row*Max+col]>0){
        //printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
        atomicAdd(&korelasi[0],(((row-miu_i[0])*(col-miu_j[0]))*norm[row*Max+col])/(std_i[0]*std_j[0]));

    }
}

__global__ void calculate_variance(float *norm,float *variance,float *miu_i,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(norm[row*Max+col]>0){
        //printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
        atomicAdd(&variance[0],((row-miu_i[0])*(row-miu_i[0]))*norm[row*Max+col]);

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
            //printf("%f\n",norm[row*Max+col]);
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

    int k;
    for(k=0;k<Max-1;k++){
        if(abs(row-col)==k && norm[row*Max+col]>0){
            //printf("%f\n",norm[row*Max+col]);
            atomicAdd(&den[0],(1*norm[row*Max+col])*(log10(1*norm[row*Max+col])));
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

void takeimagevalue(const char* filename, rgb_image *img)
{

     unsigned error;
     unsigned char* png;
     size_t pngsize;;

     lodepng_load_file(&png, &pngsize, filename);
     error = lodepng_decode32(&img->image, &img->width, &img->height, png, pngsize);

     if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

}

void transformToGrayCuda(rgb_image *img){
	unsigned char* image = img->image;
    unsigned char* image_d;
    unsigned int  width = img->width;
    unsigned int height = img->height;
    int n =width*height;
    size_t size = n * 4 * sizeof(unsigned char);


	int device_count = 0;
	cudaError_t status = cudaGetDeviceCount(&device_count);

	status = cudaMalloc((void **) &image_d, size);


	cudaMemcpy(image_d, image,  size, cudaMemcpyHostToDevice);

	dim3 block_size(16, 16);
	dim3 num_blocks(img->width / block_size.x, img->height / block_size.y);
    setPixelToGrayscale<<<num_blocks, block_size>>>(image_d, img->width, img->height);



	cudaMemcpy(image, image_d, size, cudaMemcpyDeviceToHost);

	cudaFree(image_d);
}


__global__
void setPixelToGrayscale(unsigned char *image, unsigned width, unsigned height)
{
    float gray;
    float r, g, b;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		r = image[4 * width * y + 4 * x + 0];
		g = image[4 * width * y + 4 * x + 1];
		b = image[4 * width * y + 4 * x + 2];
		gray =.299f*r + .587f*g + .114f*b;
		image[4 * width * y + 4 * x + 0] = gray;
		image[4 * width * y + 4 * x + 1] = gray;
		image[4 * width * y + 4 * x + 2] = gray;
		image[4 * width * y + 4 * x + 3] = 255;
	}

}

void saveimagegray(const char* filename, rgb_image *img)
{
  /*Encode the image*/
  unsigned error = lodepng_encode32_file(filename, img->image, img->width, img->height);

  /*if there's an error, display it*/
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
}

int main(int argc, char *argv[]){


    char *d;
    long deg =strtol(argv[2],&d,10);
    int degree=deg;

    const char* filename = argc > 1 ? argv[1] : "test.png";
    rgb_image img;
    takeimagevalue(filename, &img);
    transformToGrayCuda(&img);
    int nx =img.width;
    int ny =img.height;
    printf("%s %d degre Starting...\n", argv[0],degree,nx);
    printf("%d %d\n",nx,ny);
    int *matrix,*glcm,*transposed;
    float *norm,*mulMatrix,*sumMatrix;
    float*ASM,*contrast,*IDM,*entropy,*miu_i,*miu_j,*std_i,*std_j,*korelasi,*variance,*sav,*sen,*sva,*den,*HX,*HY,*HXY1,*dva;
    cudaMallocManaged(&matrix, (nx * ny) * sizeof(int));

    for(int i = 0 ; i < (nx * nx) ; ++i){
        matrix[i] = img.image[i];
        if(matrix[i] > Max){
            Max = matrix[i];
        }
    }

    // for(int i = 0 ; i < nx ; ++i){
    //     for(int j = 0 ; j < nx ; ++j){
    //         printf("%4d",matrix[i * nx + j]);
    //     }
    //     printf("\n");
    // }
    //printf("\n\n");
    Max = Max + 1; // karena index dimulai dari 0 dan Maximum 3 ( 0 - 3 = 4 ) jadi Max ditambah 1;

    cudaMallocManaged(&glcm, (Max * Max) * sizeof(int));
    cudaMallocManaged(&transposed, (Max * Max) * sizeof(int));
    cudaMallocManaged(&mulMatrix, (Max * Max) * sizeof(float));
    cudaMallocManaged(&sumMatrix, (Max * Max) * sizeof(float));
    cudaMallocManaged(&norm, (Max * Max) * sizeof(float));
    for(int i = 0 ; i < (Max * Max) ; ++i){
        glcm[i] = 0;
        transposed[i] = 0;
        mulMatrix[i] = 0;
    }



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

    dim3 block(2 ,2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    dim3 grids((Max + block.x - 1) / block.x, (Max + block.y - 1) / block.y);
    clock_t start, end;
    double t = 0;
    start = clock();
    // invoke kernel for calculation
    if(degree==0){
        blok1_0<<<1,1>>>(matrix,glcm,nx,Max);
        cudaDeviceSynchronize();
        AddToitTranspose<<<grids,block>>>(transposed,glcm,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(glcm,Max,degree,nx);
    }
    else if(degree ==180){
        blok1_180<<<1,1>>>(matrix,glcm,nx,Max);;
        cudaDeviceSynchronize();
        AddToitTranspose<<<grids,block>>>(transposed,glcm,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(glcm,Max,degree,nx);
    }
    else if(degree==270){
        blok1_270<<<1,1>>>(matrix,glcm,nx,Max);;
        cudaDeviceSynchronize();
        AddToitTranspose<<<grids,block>>>(transposed,glcm,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(glcm,Max,degree,nx);
    }
    else if(degree==90){
        blok1_90<<<1,1>>>(matrix,glcm,nx,Max);;
        cudaDeviceSynchronize();
        AddToitTranspose<<<grids,block>>>(transposed,glcm,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(glcm,Max,degree,nx);
    }
    else if(degree==45){
        blok1_45<<<1,1>>>(matrix,glcm,nx,Max);;
        cudaDeviceSynchronize();
        AddToitTranspose<<<grids,block>>>(transposed,glcm,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(glcm,Max,degree,nx);
    }
    else if(degree==135){
        blok1_135<<<1,1>>>(matrix,glcm,nx,Max);;
        cudaDeviceSynchronize();
        AddToitTranspose<<<grids,block>>>(transposed,glcm,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(glcm,Max,degree,nx);
    }
    else if(degree==225){
        blok1_225<<<1,1>>>(matrix,glcm,nx,Max);;
        cudaDeviceSynchronize();
        AddToitTranspose<<<grids,block>>>(transposed,glcm,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(glcm,Max,degree,nx);
    }
    else if(degree==315){
        blok1_315<<<1,1>>>(matrix,glcm,nx,Max);;
        cudaDeviceSynchronize();
        AddToitTranspose<<<grids,block>>>(transposed,glcm,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(glcm,Max,degree,nx);
    }
    end = clock();
    t = ((double) (end - start))/CLOCKS_PER_SEC;


    int sum;
    sum=0;
    for(int i=0;i<Max*Max;i++){
        sum +=transposed[i];
    }
    printf("sum %d",sum);
    normalization<<<Max,Max>>>(transposed,norm,Max,sum);


    cudaDeviceSynchronize();
    printMatrixnxormalization(norm,Max,degree,nx);
    float sums;
    sums=0;
    for(int i=0;i<Max*Max;i++){
        sums  +=norm[i];
    }
    //Jumlah <<< Max,Max >>>(sumMatrix,norm);
    printf("jumlah %f\n",sums);
    dim3 b(32,32);
    dim3 g((Max + b.x - 1) / b.x, (Max + b.y - 1) / b.y);
    //Step1
    calculate_contrast<<<g,b>>>(norm,contrast,Max);
    cudaDeviceSynchronize();
    calculate_entropy<<<g,b>>>(norm,entropy,Max);
    calculate_IDM<<<g,b>>>(norm,IDM,Max);
    calculate_ASM<<<g,b>>>(norm,ASM,Max);
    calculate_miu_i<<<g,b>>>(norm,miu_i,Max);
    cudaDeviceSynchronize();
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
    std_j[0]=sqrt(std_j[0]);
    std_i[0]=sqrt(std_i[0]);
    calculate_sumvariance<<<g,b>>>(norm,sva,sen,Max);
    calculate_korelasi<<<g,b>>>(norm,korelasi,miu_i,std_i,miu_j,std_j,Max);
    calculate_dva<<<g,b>>>(norm,dva,Max);
    cudaDeviceSynchronize();


    printf("ASM : %.3f\n",ASM[0]);
    printf("Contrast : %.3f\n",contrast[0]);
    printf("IDM : %.3f\n",IDM[0]);
    printf("entropy : %.7f\n",-(entropy[0]));
    printf("miu_i : %.3f\n",(miu_i[0]));
    printf("miu_j : %.3f\n",(miu_j[0]));
    printf("std_i : %.3f\n",(std_i[0]));
    printf("std_j : %.3f\n",(std_j[0]));
    printf("variance : %.3f\n",(variance[0]));
    printf("SAV : %.3f\n",(sav[0]));
    printf("SEN : %.3f\n",-(sen[0]));
    printf("SVA : %.3f\n",(sva[0]));
    printf("DEN : %.3f\n",-(den[0]));
    printf("HX : %.3f\n",-(HX[0]));
    printf("HY : %.3f\n",-(HY[0]));
    printf("HXY1 : %.7f\n",-(HXY1[0]));
    printf("IMC : %.7f\n",(entropy[0]-HXY1[0])/max(-(HX[0]),-(HY[0])));
    printf("korelasi : %.3f\n",(korelasi[0]));
    printf("Differnece Variance : %.3f\n",(dva[0]));

    printf("matrix gambar disimpan di matrix_gambar.txt\n");
    printf("matrix glcm disimpan di matrix_glcm_host.txt\n");
    printf("matrix glcm normalisasi disimpan di matrix_normalisasi_host.txt\n");


    printf("waktu eksekusi: %f\n",t);
    // free host and devide memory
    cudaFree(matrix);cudaFree(glcm);cudaFree(norm);
    cudaFree(mulMatrix);
}