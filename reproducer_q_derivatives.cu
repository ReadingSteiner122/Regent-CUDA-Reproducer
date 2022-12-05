#include "iostream"
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
#define max_points 16000204

using namespace std;
using namespace std::chrono;

struct point{
    double x[max_points];
    double y[max_points];
    int nbhs[max_points];
    int conn[20][max_points];
    double q[4][max_points];
    double dq[2][4][max_points];
    double qm[2][4][max_points];
};
point points;

void init_vals(point &points){
    for(int i=0;i<max_points;i++){
        points.x[i] = i*0.0001;
        points.y[i] = i*0.0001;
        points.nbhs[i] = 11;
        for(int j=0;j<11;j++){
            points.conn[j][i] = (i+j+1)%max_points;
        }
        points.q[0][i] = 1002.0;
        points.q[1][i] = 120.0;
        points.q[2][i] = 1242.0;
        points.q[3][i] = 0.0;

        for(int j=0;j<4;j++){
            // points.q[j][i] = 1.0;
            for(int k=0;k<2;k++){
                points.dq[k][j][i] = 0.0;
                points.qm[k][j][i] = 0.0;
            }
        }
    }
}
    

__global__ void qsim(point &points){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid >= max_points || tid < 0)
        return ;

    double x_i = points.x[tid];
    double y_i = points.y[tid];

    double sum_delx_sqr = 0.0;
    double sum_dely_sqr = 0.0;
    double sum_delx_dely = 0.0;

    double sum_delx_delq[4] = {0.0, 0.0, 0.0, 0.0};
    double sum_dely_delq[4] = {0.0, 0.0, 0.0, 0.0};

    for(int i = 0; i < 4; ++i){
        points.qm[0][i][tid] = points.q[i][tid];
        points.qm[1][i][tid] = points.q[i][tid];
    }

    for(int k=0; k<points.nbhs[tid]; ++k){
        int nbh = points.conn[k][tid];

        for(int r=0; r<4; r++){
            if(points.q[r][nbh] > points.qm[0][r][tid]){
                points.qm[0][r][tid] = points.q[r][nbh];
            }
            if(points.q[r][nbh] < points.qm[1][r][tid]){
                points.qm[1][r][tid] = points.q[r][nbh];
            }
        }

        double x_k = points.x[nbh];
        double y_k = points.y[nbh];

        double delx = x_k - x_i;
        double dely = y_k - y_i;

        double dist = sqrt(delx*delx + dely*dely);
        double weights = pow(dist, 2.0);

        double delx_weights = delx * weights;
        double dely_weights = dely * weights;

        sum_delx_sqr = sum_delx_sqr + delx_weights * delx;
        sum_dely_sqr = sum_dely_sqr + dely_weights * dely;
        sum_delx_dely = sum_delx_dely + delx_weights * dely;

        double delq[4];
        for(int r=0; r<4; r++){
            delq[r] = points.q[r][nbh] - points.q[r][tid];
            sum_delx_delq[r] = sum_delx_delq[r] + delx_weights * delq[r];
            sum_dely_delq[r] = sum_dely_delq[r] + dely_weights * delq[r];
        }
    }
    double det = sum_delx_sqr * sum_dely_sqr - sum_delx_dely * sum_delx_dely;
    double inv_det = 1.0 / det;

    for (int l=0; l<4; l++){
        points.dq[0][l][tid] = (sum_dely_sqr * sum_delx_delq[l] - sum_delx_dely * sum_dely_delq[l]) * inv_det;
        points.dq[1][l][tid] = (sum_delx_sqr * sum_dely_delq[l] - sum_delx_dely * sum_delx_delq[l]) * inv_det;
    }
}

void printPoint(point &points){
    for(int i=0;i<max_points;i++){
        cout << points.x[i] << " " << points.y[i] << " " << points.nbhs[i] << " ";
        cout << endl;
        for(int j=0;j<11;j++){
            cout << points.conn[j][i] << " ";
        }
        cout << endl;
        // for(int j=0;j<4;j++){
        //     cout << points.q[j][i] << " ";
        // }
        // cout << endl;
        // for(int j=0;j<4;j++){
        //     for(int k=0;k<2;k++){
        //         cout << points.dq[k][j][i] << " ";
        //     }
        //     cout << endl;
        // }
        // for(int j=0;j<4;j++){
        //     for(int k=0;k<2;k++){
        //         cout << points.qm[k][j][i] << " ";
        //     }
        //     cout << endl;
        // }
        // cout << endl;
    }
}

__global__ void oldCheck(point &points){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Check if thread is out of bounds
    if(tid >= max_points || tid < 0) return;

    // Perform x = y calculations
    points.qm[0][0][tid] = points.dq[0][0][tid];
    points.qm[0][1][tid] = points.dq[0][1][tid];
    points.qm[0][2][tid] = points.dq[0][2][tid];
    points.qm[0][3][tid] = points.dq[0][3][tid];
    points.qm[1][0][tid] = points.dq[1][0][tid];
    points.qm[1][1][tid] = points.dq[1][1][tid];
    points.qm[1][2][tid] = points.dq[1][2][tid];
    points.qm[1][3][tid] = points.dq[1][3][tid];
}

int main(){
    init_vals(points);

    dim3 threadsPerBlock(128, 1, 1);
    dim3 blocksPerGrid((max_points/threadsPerBlock.x)+1, 1, 1);

    point *point_d;
    unsigned long long point_size = sizeof(point);

    cudaMalloc(&point_d, point_size);
    cudaMemcpy(point_d, &points, point_size, cudaMemcpyHostToDevice);

    auto start = high_resolution_clock::now();
    for(int i=0; i<1000; i++){
        qsim<<<blocksPerGrid, threadsPerBlock>>>(*point_d);
        // oldCheck<<<blocksPerGrid, threadsPerBlock>>>(*point_d);
    }
    cudaDeviceSynchronize();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "Time taken by function: " << duration.count()/1e6 << " seconds" << endl;

    cudaMemcpy(&points, point_d, point_size, cudaMemcpyDeviceToHost);
    cudaFree(point_d);

    // printPoint(points);
}