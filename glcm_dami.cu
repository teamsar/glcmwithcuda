#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// Banyak N * N Matrix
#define N 4


// Banyak Max * Max Matrix
int Max;


void printMatrixGlcm(int *C, const int Max,int degree)
{
    int *ic = C;
    FILE * fp = NULL;
    if(degree==0){
        fp = fopen("matrix_glcm_0_host.txt", "w+");
    }
    else if(degree==90){
        fp = fopen("matrix_glcm_90_host.txt", "w+");
    }
    else if(degree==180){
        fp = fopen("matrix_glcm_180_Host.txt", "w+");
    }
    else if(degree==270){
        fp = fopen("matrix_glcm_270_host.txt", "w+");
    }
    else if(degree==45){
        fp = fopen("matrix_glcm_45_host.txt", "w+");
    }
    else if(degree==135){
        fp = fopen("matrix_glcm_135_host.txt", "w+");
    }
    else if(degree==225){
        fp = fopen("matrix_glcm_225_host.txt", "w+");
    }
    else if(degree==315){
        fp = fopen("matrix_glcm_315_host.txt", "w+");
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

// Fungsi membagi matrix

// ada perbaikan di 0 derajat
__global__ void Div0(int *matrix , int *newMatrix,int Max){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int Index = iy * N + ix;
    int posisi = 0;

    for(int i = 0 ; i < N ; i += 2){
        if(Index >= i * N && Index < ((i + 1) * N) - 1){

            posisi = matrix[Index] * Max + matrix[Index + 1];
            atomicAdd(&newMatrix[posisi],1);

            posisi = matrix[Index + N] * Max + matrix[Index + (N + 1)];
            atomicAdd(&newMatrix[posisi],1);
        }
    }
}

__global__ void Div45(int *matrix,int *newMatrix,int Max){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int Index = iy * N + ix;
    int posisi = 0;

    for(int i = 0 ; i < N - 1 ; i++){
        if(Index >= i * N && Index < ((i + 1) * N) - 1){
        posisi = matrix[Index + N] * Max + matrix[Index + 1];
        atomicAdd(&newMatrix[posisi],1);
        //printf("Index : %d %d\n",Index + N , Index + 1);
        }
    }
}

__global__ void Div90(int *matrix,int *newMatrix,int Max){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int Index = iy * N + ix;
    int posisi = 0;

    for(int i = 0 ; i < N - 1 ; ++i){
        if(Index >= i * N && Index < ((i + 1) * N) - 1){
            if(Index == 0 || Index % 2 == 0){
                posisi = matrix[Index + N] * Max + matrix[Index];
                atomicAdd(&newMatrix[posisi],1);

                posisi = matrix[Index + (N + 1)] * Max + matrix[Index + 1];
                atomicAdd(&newMatrix[posisi],1);
                //printf("Index : %d %d dan %d %d\n",Index + N , Index, Index + (N + 1),Index + 1);
            }
        }
    }
}

__global__ void Div135(int *matrix,int *newMatrix,int Max){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int Index = iy * N + ix;
    int posisi = 0;

    for(int i = 0 ; i < N - 1 ; ++i){
        if(Index >= i * N && Index < ((i + 1) * N) - 1){

            posisi = matrix[Index + (N + 1)] * Max + matrix[Index];
            atomicAdd(&newMatrix[posisi],1);
            //printf("Index : %d %d\n",Index + (N + 1), Index);
        }
    }
}

__global__ void Div180(int *matrix,int *newMatrix,int Max){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int Index = iy * N + ix;
    int posisi = 0;

    for(int i = 0 ; i < N ; i += 2){
        if(Index >= i * N && Index < ((i + 1) * N) - 1){
                
                posisi = matrix[Index + 1] * Max + matrix[Index];
                atomicAdd(&newMatrix[posisi],1);

                posisi = matrix[Index + (N + 1)] * Max + matrix[Index + N];
                atomicAdd(&newMatrix[posisi],1);
        }
    }
}

__global__ void Div225(int *matrix,int *newMatrix,int Max){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int Index = iy * N + ix;
    int posisi = 0;

    for(int i = 0 ; i < N - 1 ; ++i){
        if(Index >= i * N && Index < ((i + 1) * N) - 1){
            posisi = matrix[Index + 1] * Max + matrix[Index + N];
            atomicAdd(&newMatrix[posisi],1);
            //printf("Index : %d %d\n",Index + 1, Index + N);
        }
    }
}

__global__ void Div270(int *matrix,int *newMatrix,int Max){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int Index = iy * N + ix;
    int posisi = 0;

    for(int i = 0 ; i < N - 1 ; ++i){
        if(Index >= i * N && Index < ((i + 1) * N) - 1){
            if(Index == 0 || Index % 2 == 0){
                posisi = matrix[Index] * Max + matrix[Index + N];
                atomicAdd(&newMatrix[posisi],1);

                posisi = matrix[Index + 1] * Max + matrix[Index + (N + 1)];
                atomicAdd(&newMatrix[posisi],1);
                printf("Index : %d %d dan %d %d\n",Index,Index + N , Index + 1, Index + (N + 1));
            }
        }
    }
}

__global__ void Div315(int *matrix,int *newMatrix,int Max){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int Index = iy * N + ix;
    int posisi = 0;

    for(int i = 0 ; i < N - 1 ;  ++i ){
        if(Index >= i * N && Index < ((i + 1) * N) - 1){
            posisi = matrix[Index] * Max + matrix[Index + (N + 1)];
            atomicAdd(&newMatrix[posisi],1);
            printf("Index : %d %d\n",Index,Index + (N + 1));
        }
    }
}


__global__ void blok1_0(int *matrix,int *glcm, int Max){
    int i,j,k,l;

    for(i=0;i<N;i++){
        for(j=0;j<N-1;j++){
            for(k=0;k<Max;k++){
                for(l=0;l<Max;l++){
                    if(matrix[Max*i+j]==k&&matrix[Max*i+(j+1)]==l){
                        //printf("%d,%d,%d,%d\n",matrix[Max*i+j],k,matrix[Max*i+(j+1)],l);
                        atomicAdd(&glcm[Max*k+l],1);
                    }
                }
            }
        }
    }

}


__global__ void blok1_45(int *matrix,int *newMatrix, int Max){
    int i,j;

    for(i=1;i<N;i++){
        for(j=0;j<N-1;j++){
            atomicAdd(&newMatrix[Max*matrix[N*i+j]+matrix[N*(i-1)+(j+1)]] ,1);
    
        }
    }
}

__global__ void blok1_90(int *matrix,int *newMatrix, int Max){
    int i,j;

    for(i=1;i<N;i++){
        for(j=0;j<N;j++){
        atomicAdd(&newMatrix[Max*matrix[N*i+j]+matrix[N*(i-1)+j]] ,1);
    
        }
    }
}


__global__ void blok1_135(int *matrix,int *newMatrix, int Max){
    int i,j;

    for(i=1;i<N;i++){
        for(j=1;j<N;j++){
        atomicAdd(&newMatrix[Max*matrix[N*i+j]+matrix[N*(i-1)+(j-1)]] ,1);
    
        }
    }
}

__global__ void blok1_180(int *matrix,int *newMatrix, int Max){
    int i,j;

    for(i=0;i<N;i++){
        for(j=1;j<N;j++){
        atomicAdd(&newMatrix[Max*matrix[N*i+j]+matrix[N*i+(j-1)]] ,1);
    
        }
    }
}

__global__ void blok1_225(int *matrix,int *newMatrix, int Max){
    int i,j;

    for(i=0;i<N-1;i++){
        for(j=1;j<N;j++){
        atomicAdd(&newMatrix[Max*matrix[N*i+j]+matrix[N*(i+1)+(j-1)]] ,1);
    
        }
    }
}


__global__ void blok1_270(int *matrix,int *newMatrix, int Max){
    int i,j;

    for(i=0;i<N-1;i++){
        for(j=0;j<N;j++){
        atomicAdd(&newMatrix[Max*matrix[N*i+j]+matrix[N*(i+1)+j]] ,1);
    
        }
    }
}

__global__ void blok1_315(int *matrix,int *newMatrix, int Max){
    int i,j;

    for(i=0;i<N-1;i++){
        for(j=0;j<N-1;j++){
        atomicAdd(&newMatrix[Max*matrix[N*i+j]+matrix[N*(i+1)+(j+1)]] ,1);
    
        }
    }
}

__global__ void Mul(float *newMatrix,float *mulMatrix,int Max){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // int Index = iy * N + ix;

    for (int k = 0; k < Max; k++) {
        // Accumulate results for a single element
        // c[row * N + col] += a[row * N + k] * b[k * N + col];
        // printf("C[%d] = a[%d] * b[%d]\n",row * N + col,row * N + k, k * N + col);
        atomicAdd(&mulMatrix[row * Max + col],newMatrix[row * Max + col] * newMatrix[row * Max + col]);
    
        
    }
}

// void calculate_glcm_host(int *matrix,int *glcm,int Max){
//     int i,j,k,l;
//     printf("oksas");
//     int p=0;
//     for(i=0;i<N*N;i++){
//         for(k=0;k<=Max;k++){
//             //printf("oksas");
//             for(l=0;l<=Max;l++){
                
//                if((matrix[i]==k) && (matrix[i+1]==l)){
//                 //printf("oksas");
//                    p=((Max)*k) +l;
//                    glcm[p] +=1;
//                 }
//             }
//         }
//     }
//     printf("oksasoo");
// }


void calculate_glcm_host(int *matrix,int *glcm,int Max){
    int i,j;
    
    for(i=0;i<N;i++){
        for(j=0;j<N-1;j++){
            glcm[Max*matrix[Max*i+j]+matrix[Max*i+(j+1)]] +=1;
        }
    }
}

__global__ void AddToitTranspose(int *transposed,int *glcm,int Max){
    //if(Index<1) printf("%f",newMatrix[0]);
    int col = blockIdx.x * blockDim.x + threadIdx.x;                
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    //printf("%d  %d\n",row*Max+col,col*Max+row);
    
        transposed[row*Max+col]=glcm[row*Max+col]+glcm[col*Max+row];
        //printf("%d %d  %d\n",transposed[row*Max+col],glcm[row*Max+col],glcm[col*Max+row]);
    
    
}

__global__ void normalization(int *glcm,float *norm,int Max,int sum){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * Max + ix;
    __syncthreads();
    
    if(glcm[idx]>0){
        //printf("%d\n",glcm[idx]);
        norm[idx]=float(glcm[idx])/float(sum);
        //printf("%f\n",norm[idx]);
    }
}

__global__ void Jumlah(int *sumMatrix,int *newMatrix,int Max){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int Index =row*Max+col;
    //if(Index<1) printf("%f",newMatrix[0]);
    if(newMatrix[row*Max+col]>0)
    atomicAdd(&sumMatrix[0],newMatrix[Index]);
    
}

__global__ void Jumlah_norm(float *sumMatrix,float *newMatrix){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    
    //if(Index<1) printf("%f",newMatrix[0]);
    atomicAdd(&sumMatrix[0],newMatrix[Index]);
    
}

// __global__ void calculate_ASM(float *norm,float *ASM,float *mulMatrix,int Max){
//     //printf("%d\n",max);
//     int col = blockIdx.x * blockDim.x + threadIdx.x;                
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     //mulMatrix[row * Max + col]=0;
//     // int Index = iy * N + ix;
//     //printf("%f %d\n",norm[row*Max+col],row*Max+col);
//     for (int k = 0; k < Max ; k++) {
//         // Accumulate results for a single element
//         // c[row * N + col] += a[row * N + k] * b[k * N + col];
//         // printf("C[%d] = a[%d] * b[%d]\n",row * N + col,row * N + k, k * N + col);
//         printf("%f %d %f %d\n",norm[row*Max+k],row*Max+k,norm[col*Max+k],col*Max+k);
//         mulMatrix[row * Max + col]+=norm[row * Max + k] * norm[k * Max + col];
//         //printf("%f %d\n",mulMatrix[row*Max+col],row*Max+col);
//     }
//     //printf("%f index%d\n",mulMatrix[row*Max+col],row*Max+col);
//     int Index = blockIdx.x * blockDim.x + threadIdx.x;

//     for (int stride = 1; stride < Max*Max; stride *= 2)
//     {
//         if ((Index % (2 * stride)) == 0)
//         {
//             mulMatrix[Index] += mulMatrix[Index+ stride];
//             //printf("%d %f\n",Index,mulMatrix[Index]);
//         }
//         // synchronize within threadblock
//         __syncthreads();
//     }
//     __syncthreads();
//     if (Index == 0){

//         printf("ASM %f\n",mulMatrix[0]);
//     }
// }

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
        atomicAdd(&contrast[0],norm[row*Max+col]);
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
        //printf("nilai entropy %f %d %d %f\n",log10f(norm[row*Max+col]),row,col,norm[row*Max+col]);
        //atomicAdd(&ASM[0],norm[row*Max+col]*norm[row*Max+col]);
        //printf("%f\n",entropy[0]);
    }

}


__global__ void calculate_ASM(float *norm,float *ASM,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    //printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
    if(norm[row*Max+col]>0 ){
    // printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
        atomicAdd(&ASM[0],norm[row*Max+col]*norm[row*Max+col]);
        //printf("%f %f %d %d\n",norm[row*Max+col],norm[row*Max+col]*norm[row*Max+col],count,Index);
    }
}

__global__ void calculate_miu_i(float *norm,float *miu_i,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(norm[row*Max+col]>0){
        //printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
        atomicAdd(&miu_i[0],row*norm[row*Max+col]);
        //printf("nilai miu_i  %d %d %f\n",row,col,norm[row*Max+col]);
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


__global__ void calculate_std_j(float *norm,float *std_j,float *miu_j,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(norm[row*Max+col]>0){
        //printf("%d  %d  %d\n",row,col,Index);
        //printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
        atomicAdd(&std_j[0],norm[row*Max+col] * ((col-miu_j[0])*(col-miu_j[0])));
        //printf("nilai miu_i %d %d %d %f\n",((row-col)*(row-col)),row,col,norm[row*Max+col]);
        //atomicAdd(&ASM[0],norm[row*Max+col]*norm[row*Max+col]);
        //printf("%f\n",miu_i[0]);
    }

}
__global__ void calculate_korelasi(float *norm,float *korelasi,float *miu_i,float *std_i,float *miu_j,float *std_j,int Max){
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

__global__ void calculate_variance(float *norm,float *variance,float *miu_i,float *miu_j,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(norm[row*Max+col]>0){
        //printf("nilai %d %d %d %f\n",row*Max+col,row,col,norm[row*Max+col]);
        atomicAdd(&variance[0],((row-((miu_i[0]+miu_j[0])/2))*(row-((miu_i[0]+miu_j[0])/2)))*norm[row*Max+col]);
        //printf("nilai variance  %d %f %f \n",row,((row-((miu_i[0]+miu_j[0])/2))*(row-((miu_i[0]+miu_j[0])/2)))   ,(row-((miu_i[0]+miu_j[0])/2)));
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
    Max=Max-1;
    int k;
    for(k=2;k<2*Max;k++){
        if((row+col)==k && norm[row*Max+col]>0){
            //printf("sen %d %d %f\n",row+col,k,norm[row*Max+col]);
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
            //printf("sen %d %d  %f %f %f\n",row+col,k,(k-sen[0]),((k-sen[0])*(k-sen[0])),norm[row*Max+col]);
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

__global__ void calculate_HX(float *norm,float *HX,int Max){
    //printf("%d\n",max);

    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(norm[row]>0){
        atomicAdd(&HX[0],norm[row]*log10f(norm[row]));
    }
}

__global__ void calculate_HY(float *norm,float *HY,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(norm[col]>0){
        atomicAdd(&HY[0],norm[col]*log10f(norm[col]));
    }
}

__global__ void calculate_HXY1(float *norm,float *HXY1,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;


    if(norm[row]>0 && norm[col]>0){
        //printf("%f %f\n",norm[row*Max+col],log10f(norm[row]*norm[col]));
        atomicAdd(&HXY1[0],norm[row*Max+col]*log10f(norm[row]*norm[col]));
    }
}
__global__ void calculate_n_kuadrat(float *norm,float *n_kuadrat,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(norm[row*Max+col]>0 ){
        //printf("%f %f\n",norm[row*Max+col],log10f(norm[row]*norm[col]));
        atomicAdd(&n_kuadrat[0],(row-col)*(row-col));
        // /printf("%f  %d %d\n",n_kuadrat[0],row,col);

    }
}


float max(float *x,float *y){
    if(x[0]>y[0]){
        return x[0];
    }else{
        return y[0];
    }
    
}

int main(int argc, char *argv[]){
    char *d;
    long deg =strtol(argv[1],&d,10);
    int degree=deg;
    printf("%s %d degre Starting...\n", argv[0],degree);

    int *matrix,*newMatrix,*mulMatrix,*sumMatrix,*transposed;
    float *norm,*sum_norm,*ASM,*kali,*contrast,*IDM,*entropy,*miu_i,*miu_j,*std_i,*std_j,*korelasi,*variance,*sav,*sen,*sva,*den,*HX,*HY,*HXY1,*dva,*n_kuadrat;

    cudaMallocManaged(&matrix, (N * N) * sizeof(int));
    int data[16]={1,3,2,0,1,0,2,2,2,0,1,1,1,3,1,3};
    for(int i = 0 ; i < (N * N) ; ++i){
        matrix[i] =data[i];
        if(matrix[i] > Max){
            Max = matrix[i];
        }
    }
    
for(int i = 0 ; i < N ; ++i){
    for(int j = 0 ; j < N ; ++j){
        //if(matrix[i * N + j]>0)
        printf("%4d",matrix[i * N + j]);
    }
    printf("\n");
}
    //printf("\n\n");
    Max = Max + 1; // karena index dimulai dari 0 dan maximum 3 ( 0 - 3 = 4 ) jadi Max ditambah 1;

    cudaMallocManaged(&newMatrix, (Max * Max) * sizeof(int));
    cudaMallocManaged(&transposed, (Max * Max) * sizeof(int));
    cudaMallocManaged(&mulMatrix, (Max * Max) * sizeof(int));
    cudaMallocManaged(&sumMatrix, (Max * Max) * sizeof(int));
    cudaMallocManaged(&norm, (Max * Max) * sizeof(float));
    cudaMallocManaged(&sum_norm, (Max * Max) * sizeof(float));
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
    cudaMallocManaged(&HXY1, (Max * Max) * sizeof(float));
    cudaMallocManaged(&dva, (Max * Max) * sizeof(float));
    cudaMallocManaged(&n_kuadrat, (Max * Max) * sizeof(float));
    cudaMallocManaged(&kali, (Max * Max) * sizeof(float));

    for(int i = 0 ; i < (Max * Max) ; ++i){
        newMatrix[i] = 0;
        mulMatrix[i] = 0;
        transposed[i]=0;
    }

    dim3 block(2 ,2);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    dim3 grids((Max + block.x - 1) / block.x, (Max + block.y - 1) / block.y);
    // Mencari Pembagian dan kuadrat matrix
// printf("ok7\n");
    //Div0 <<< grid,block >>>(matrix,newMatrix);
    //printf("ok6\n");
    clock_t start, end;
    double t = 0;
    start = clock();
    //Div315<<< grid,block >>>(matrix,newMatrix,Max);
    if(degree==0){
        Div0 <<< grid,block >>>(matrix,newMatrix,Max);
        //blok1_0<<<1,1>>>(matrix,newMatrix,Max);;
        cudaDeviceSynchronize();
        AddToitTranspose<<<grids,block>>>(transposed,newMatrix,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(newMatrix,Max,degree);
    }
    else if(degree ==180){
        blok1_180<<<1,1>>>(matrix,newMatrix,Max);;
        cudaDeviceSynchronize();
        AddToitTranspose<<<grids,block>>>(transposed,newMatrix,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(newMatrix,Max,degree);
    }
    else if(degree==270){
        blok1_270<<<1,1>>>(matrix,newMatrix,Max);;
        cudaDeviceSynchronize();
        AddToitTranspose<<<grids,block>>>(transposed,newMatrix,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(newMatrix,Max,degree);
    }
    else if(degree==90){
        blok1_90<<<1,1>>>(matrix,newMatrix,Max);;
        cudaDeviceSynchronize();
        AddToitTranspose<<<grids,block>>>(transposed,newMatrix,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(newMatrix,Max,degree);
    }
    else if(degree==45){
        blok1_45<<<1,1>>>(matrix,newMatrix,Max);;
        cudaDeviceSynchronize();
        AddToitTranspose<<<grids,block>>>(transposed,newMatrix,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(newMatrix,Max,degree);
    }
    else if(degree==135){
        blok1_135<<<1,1>>>(matrix,newMatrix,Max);;
        cudaDeviceSynchronize();
        AddToitTranspose<<<grids,block>>>(transposed,newMatrix,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(newMatrix,Max,degree);
    }
    else if(degree==225){
        blok1_225<<<1,1>>>(matrix,newMatrix,Max);;
        cudaDeviceSynchronize();
        AddToitTranspose<<<grids,block>>>(transposed,newMatrix,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(newMatrix,Max,degree);
    }
    else if(degree==315){
        blok1_315<<<1,1>>>(matrix,newMatrix,Max);;
        cudaDeviceSynchronize();
        AddToitTranspose<<<grids,block>>>(transposed,newMatrix,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(newMatrix,Max,degree);
    }
    //calculate_glcm_host(matrix,newMatrix,Max);
    printf("okelals");
    end = clock();
    t = ((double) (end - start))/CLOCKS_PER_SEC;
    //Mul <<< grid,block >>>(newMatrix,mulMatrix,Max);
    printf("waktu eksekusi: %f\n",t);
    //cudaDeviceSynchronize();

printf("Hasil glcm : \n");
for(int i = 0 ; i < Max ; ++i){
    for(int j = 0 ; j < Max ; ++j){

        //transposed[i * Max + j]=newMatrix[i * Max + j];
        printf("%4d",newMatrix[i * Max + j]);
        
    }
    printf("\n");
}
    AddToitTranspose<<<grid,block>>>(transposed,newMatrix,Max);
    printf("ok8\n");
    cudaDeviceSynchronize();
    printf("Hasil penjumlahan transpose : \n");
    for(int i = 0 ; i < Max ; ++i){
        for(int j = 0 ; j < Max ; ++j){
    
            //transposed[i * Max + j]=newMatrix[i * Max + j];
            //if(transposed[i * Max + j]>0)
            //printf("%4d",transposed[i * Max + j]);
        }
        //printf("\n");
    }
    Jumlah <<< grids,block >>>(sumMatrix,transposed,Max);
    cudaDeviceSynchronize();

    printf("sum %d",sumMatrix[0]);
    normalization<<<grids,block>>>(transposed,norm,Max,sumMatrix[0]);
    printf("ok9\n");
    cudaDeviceSynchronize();
    int count=0;
    printf("Hasil normalisasi : \n");
    for(int i = 0 ; i < Max ; ++i){
        for(int j = 0 ; j < Max ; ++j){
            if(norm[i * Max + j]>0){
                count++; 
                norm[i*Max+j]=norm[i*Max+j];
            }
        
            // printf("%.7f ",norm[i * Max + j]);
        }
        //printf("\n");
    }
    Mul<<< grids,block >>>(norm,kali,Max);
    cudaDeviceSynchronize();
    printf("count %d\n",count);
    int THREADS = 32;
// Blocks per grid dimension (assumes THREADS divides N evenly)
int BLOCKS = Max / THREADS;
// Use dim3 structs for block  and grid dimensions
dim3 threads(THREADS, THREADS);
dim3 blocks(BLOCKS, BLOCKS);
dim3 b(32,32);
dim3 c(Max,Max);
dim3 g((Max + b.x - 1) / b.x, (Max +b.y - 1) / b.y);
printf("ok0\n");
Jumlah_norm<<< Max,Max >>>(sum_norm,norm);
printf("ok1\n");
cudaDeviceSynchronize();
calculate_n_kuadrat<<<g,c>>>(norm,n_kuadrat,Max);
calculate_ASM<<<Max,Max>>>(norm,ASM,Max);
calculate_contrast<<<Max,Max>>>(norm,contrast,Max);
calculate_IDM<<<g,b>>>(norm,IDM,Max);
calculate_entropy<<<g,c>>>(norm,entropy,Max);
calculate_miu_i<<<g,c>>>(norm,miu_i,Max);
printf("ok2\n");
cudaDeviceSynchronize();
calculate_miu_j<<<g,c>>>(norm,miu_j,Max);
printf("ok3\n");
cudaDeviceSynchronize();
calculate_std_i<<<g,b>>>(norm,std_i,miu_i,Max);
cudaDeviceSynchronize(); 
calculate_std_j<<<g,b>>>(norm,std_j,miu_j,Max);
printf("ok3\n");
calculate_variance<<<g,c>>>(norm,variance,miu_i,miu_j,Max);
cudaDeviceSynchronize();
std_i[0]=sqrt(std_i[0]);
std_j[0]=sqrt(std_j[0]);
calculate_korelasi<<<g,b>>>(norm,korelasi,miu_i,std_i,miu_j,std_j,Max);
calculate_sumaverage<<<g,c>>>(norm,sav,Max);
calculate_sumentropy<<<g,c>>>(norm,sen,Max);
calculate_differenceentropy<<<g,c>>>(norm,den,Max);
cudaDeviceSynchronize();
calculate_sumvariance<<<g,c>>>(norm,sva,sen,Max);
calculate_HX<<<g,b>>>(norm,HX,Max);
calculate_HY<<<g,b>>>(norm,HY,Max);
calculate_dva<<<g,b>>>(norm,dva,Max);
calculate_HXY1<<<g,b>>>(norm,HXY1,Max);
cudaDeviceSynchronize();
printf("contrast %f %f\n",contrast[0],n_kuadrat[0]);
//printf("\n");

    // printf("\n\nHasil Kuadrat : \n");
    // for(int i = 0 ; i < Max ; ++i){
    //     for(int j = 0 ; j < Max ; ++j){
    //        // printf("%4d",mulMatrix[i * N + j]);
    //         jlh += mulMatrix[i * N + j];
    //     }
    //    // printf("\n");
    // }
    //max(a,b);
    //printf("\n");
    printf("ASM : %.7f\n",ASM[0]);
    printf("Contrast : %.7f\n",contrast[0]*n_kuadrat[0]);
    printf("IDM : %.7f\n",IDM[0]);
    printf("entropy : %.7f\n",-(entropy[0]));
    printf("miu_i : %.7f\n",(miu_i[0]));
    printf("miu_j : %.7f\n",(miu_j[0]));
    printf("std_i : %.7f\n",(std_i[0]));
    printf("std_j : %.7f\n",(std_j[0]));
    printf("variance : %.7f\n",(variance[0]));
    printf("SAV : %.7f\n",(sav[0]));
    printf("SEN : %.7f\n",-(sen[0]));
    printf("SVA : %.7f\n",(sva[0]));
    printf("DEN : %.7f\n",-(den[0]));
    printf("HX : %.7f\n",-(HX[0]));
    printf("HY : %.7f\n",-(HY[0]));
    printf("HXY1 : %.7f\n",-(HXY1[0]));
    printf("IMC : %.7f\n",(entropy[0]-HXY1[0])/max(-(HX[0]),-(HY[0])));
    printf("korelasi : %.7f\n",(korelasi[0]));
    printf("Differnece Variance : %.7f\n",(dva[0]));
    cudaFree(matrix);
    cudaFree(newMatrix);
    cudaFree(mulMatrix);
}