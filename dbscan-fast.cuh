/*
 *@Author: Krutarth Patel
 *@Date: 16th May 2025
 *@Description : GDBSCAN header only library
 * using CUDA for speeeeeeeeeeeeeeed.
 */

#include <cwchar>
#include <algorithm>
#include <iostream>
#include <vector>
#include <exception>
#include <cuda_runtime.h>
#include <queue>
#include <fstream>
#include <unistd.h>
#define SIZE 64
// #define DEBUG

__device__ __managed__ int true_count = 1;
#define __START_TIMER__                                                                            \
    float gpu_elapsed_time_ms;                                                                     \
    cudaEvent_t start, stop;                                                                       \
    cudaEventCreate(&start);                                                                       \
    cudaEventCreate(&stop);                                                                        \
    cudaEventRecord(start, 0);

#define __END_TIMER__                                                                              \
    cudaEventRecord(stop, 0);                                                                      \
    cudaEventSynchronize(stop);                                                                    \
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);                                       \
    std::cout << gpu_elapsed_time_ms << std::endl;
    
template<typename T, int N>
struct Point
{
    std::vector<T> data;
    size_t dims;
    Point(): dims(N){}
    Point(std::vector<T> &input) 
    {
        if(N!=input.size()){
            std::cerr << "Dimensions do not match" << std::endl;
        }
        dims = N;
        data = std::vector<T>(N);
        std::copy(input.begin(), input.end(), data.begin());
    }
};

template<int N>
__global__ void num_neighbors(int *count_list, float *points, 
    int *adj_list, int offset, int no_of_nodes, float eps){
    int index = blockIdx.x*SIZE + threadIdx.x;
    int blk_ind = threadIdx.x;
    int glob_stride = offset*index;
    float p[N];
    int temp=0;
    float dist;
    if(index < no_of_nodes){
        // 
        // store this Point
        // in thread's register
        // 
        for(int i=0;i < N; ++i){
            p[i] = points[index * N + i];
        }
    }
    __shared__ float l1pts[SIZE][N];
    for(int blk_off=0; blk_off < no_of_nodes;blk_off+=SIZE){
        int tmp = blk_ind + blk_off;
        __syncthreads();
        if(tmp < no_of_nodes){
            // 
            // copy point to shared memory
            // Done by each block
            // 
            for(int i=0;i < N; ++i){
                l1pts[blk_ind][i] = points[tmp * N + i];
            }
        }
        __syncthreads();
        int size = min((no_of_nodes - blk_off), SIZE);
        if(index < no_of_nodes){
            for(int i=0;i<size;++i){
                dist = 0;
                for(int j=0;j < N; ++j){
                    dist += fabs(l1pts[i][j] - p[j]);
                }
                if(dist <= eps){
                    // this should be shared memory
                    adj_list[glob_stride + temp] = i + blk_off;
                    temp++;
                }
            }
        }
    }
    if(index < no_of_nodes){
        count_list[index]=temp;
    }
}

class DBSCAN{
private:
	float eps;
	int min_pts;
	long long no_nodes;
	int MAX_NEIGHBORS; // thereotical limit on maximum total neighbor
	int OFFSET;        // max neighbor for one node
	double dmin;       // min distance between two nodes
	int *labels;
	float *dev_nodes;
	int *adj_list; 
	int *dev_adj_list;
	int *dev_neighbor_count;
	std::vector<int> neighbor_count;
public:
    DBSCAN(float eps, int min_pts) :  eps(eps), min_pts(min_pts){ 
        labels = nullptr;
        adj_list = nullptr;
    }
    ~DBSCAN(){
        cudaFree(dev_adj_list);
        cudaFree(dev_nodes);
        cudaFree(dev_neighbor_count);
        delete []labels;
        delete []adj_list;
    }
    template<typename T, int N>
    int identify_cluster(std::vector<Point<T,N>> & nodes, float (*f)(T))
    {
        no_nodes = nodes.size();
        neighbor_count = std::vector<int>(no_nodes, 0);
        // flat array of points
        cudaMalloc(&dev_nodes, sizeof(float) * no_nodes * N);
        cudaMalloc(&dev_neighbor_count, sizeof(int)*no_nodes);
        // 
        // NOTE: this is not the best way to do this. But any 
        // other method will not improve the worst case.
        // 
        std::vector<float> transformed(N);
        // 
        // TODO: this should be done on the GPU
        // 
        for(int i=0;i < nodes.size(); ++i){
            for(int j=0;j < N; ++j)transformed[j] = f(nodes[i].data[j]);
            cudaMemcpy(dev_nodes + N*i, transformed.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
        }
        OFFSET = no_nodes;
        if(labels != nullptr)delete[] labels;
        if(adj_list != nullptr)delete[] adj_list;
        labels = new int[no_nodes];
        adj_list = new int[no_nodes * no_nodes];
        cudaMalloc(&dev_adj_list, sizeof(int)*(no_nodes * no_nodes));
        
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess){
            std::cerr << cudaGetErrorString(err) << std::endl;
        }
        
        //
        //find neighbors
        //
        dim3 dim_block(SIZE, 1, 1);
        dim3 dim_grid((no_nodes + SIZE-1)/SIZE, 1, 1);
        num_neighbors<N><<<dim_grid, dim_block>>>(dev_neighbor_count, dev_nodes, 
            dev_adj_list, this->OFFSET, no_nodes, eps);
        cudaDeviceSynchronize();
        cudaMemcpy(neighbor_count.data(), dev_neighbor_count, 
            sizeof(int)*no_nodes, cudaMemcpyDeviceToHost);
        cudaMemcpy(adj_list, dev_adj_list, sizeof(int) * no_nodes * no_nodes, cudaMemcpyDeviceToHost);
#if defined(DEBUG)
        err = cudaGetLastError();
        if(err != cudaSuccess){
            std::cerr << "ERROR:" << cudaGetErrorString(err) << std::endl;
        }
        std::cout << "no of nodes: " << no_nodes << std::endl; 
        for(int i=0;i < no_nodes; ++i){
             std::cout << neighbor_count[i] << ' ';
        }
        std::cout << std::endl;
        std::cout << "adj list :\n";
        for(int i=0;i < no_nodes; ++i){
        for(int j=i * OFFSET; j < i * OFFSET + neighbor_count[i]; ++j){
            std::cout << adj_list[j] << ' ';
        }
        std::cout << std::endl;
        }
#endif
        int cluster_id = 1;
        for(int i=0;i < no_nodes; ++i){
            labels[i]=0;
        }
        //
        // bfs to find clusters
        // 
        std::queue<int> q;
        for(int node=0;node < no_nodes; ++node){
            if(!labels[node] && neighbor_count[node] >= min_pts){
                q.push(node);
                labels[node] = cluster_id;
                if(node == 0){std::cout << labels[node] << std::endl;}
                while(!q.empty()){
                    int curr_node = q.front();q.pop();
                    for(int i=curr_node*this->OFFSET;i<curr_node*this->OFFSET+neighbor_count[curr_node];++i){
                        int neighbor = adj_list[i];
                        if(!labels[neighbor]){ 
                            labels[neighbor]=cluster_id;
                            if(neighbor_count[neighbor] >= min_pts)q.push(neighbor); 
                        }
                    }
                }
                cluster_id++;
            }
        }
        for(int node=0;node<no_nodes;++node){
            if(labels[node] == 0){labels[node]=cluster_id++;}
        }
        cudaFree(dev_adj_list);
        cudaFree(dev_nodes);
        cudaFree(dev_neighbor_count);
        // 
        // we return the number of nodes
        // considered in the algorithm
        // 
        return no_nodes;
    }

    void show_labels(){
        std::cout << "labels :\n";
        for(int i=0;i < no_nodes; ++i){
            std::cout << labels[i] << ' ';
        }
        std::cout << '\n';
    }

    int label(int index){
        if(index >= no_nodes){return -1;}
        return labels[index];
    }
    
};
