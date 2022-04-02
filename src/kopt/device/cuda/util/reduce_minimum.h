#pragma once
#include <float.h>
#include <stdio.h>

	template<class T>
	__device__
	T limit();
	
	template<>
	__device__
	float limit(){
		return FLT_MAX;
	}
	
	template<>
	__device__
	double limit(){
		return DBL_MAX;
	}
	

	template<unsigned int THREADS_X_PER_BLOCK, class F>
		__device__
		void k_warp_reduce_group_minimum(volatile F* sdata, int tx){

			if(THREADS_X_PER_BLOCK>=64){		
				if (sdata[tx].x>sdata[tx+32].x){
					sdata[tx].x=sdata[tx+32].x;
					sdata[tx].y=sdata[tx+32].y;
				}
			}
			if(THREADS_X_PER_BLOCK>=32){
				if (sdata[tx].x>sdata[tx+16].x){
					sdata[tx].x=sdata[tx+16].x;
					sdata[tx].y=sdata[tx+16].y;
				}
			}
			if(THREADS_X_PER_BLOCK>=16){
				if (sdata[tx].x>sdata[tx+8].x){
					sdata[tx].x=sdata[tx+8].x;
					sdata[tx].y=sdata[tx+8].y;
				}
			}
			if(THREADS_X_PER_BLOCK>=8){
				if (sdata[tx].x>sdata[tx+4].x){
					sdata[tx].x=sdata[tx+4].x;
					sdata[tx].y=sdata[tx+4].y;
				}
			}
			if(THREADS_X_PER_BLOCK>=4){
				if (sdata[tx].x>sdata[tx+2].x){
					sdata[tx].x=sdata[tx+2].x;
					sdata[tx].y=sdata[tx+2].y;
				}
			}
			if(THREADS_X_PER_BLOCK>=2){
				if (sdata[tx].x>sdata[tx+1].x){
					sdata[tx].x=sdata[tx+1].x;
					sdata[tx].y=sdata[tx+1].y;
				}
			}
		}

/*Reduces the particle minima of a PSO group to its group minima. Caution: One PSO group is mapped to one block*/
template<unsigned int THREADS_X_PER_BLOCK, class F, class F2>
__global__
void k_reduce_to_group_minimum(int n_particles,int n_groups, F* r, F* gmin_dest, int* gmin_id, int* pso_rep_counter){
	F n_particles_inv=1.0/n_particles;
		
	if (blockIdx.x>=n_groups){
		return;
	}
	/*The function k_warp_reduce_sum expects shared memory size to be minimum two times the size
	of a warp*/
	constexpr int memsize=(THREADS_X_PER_BLOCK<=64)?64:THREADS_X_PER_BLOCK;
	static __shared__ F2 sdata[memsize];
	int tx=threadIdx.x;
	sdata[tx].x=limit<F>(); // TODO: Max float
	 // TODO: Max float
	int rem=n_groups%gridDim.x;
	int groups_for_this_block=(blockIdx.x<rem)?1+(n_groups/static_cast<F>(gridDim.x)):(n_groups/static_cast<F>(gridDim.x));
	int iter=0;
	
	F* r_ptr=r+n_particles*blockIdx.x;
	for (int group=0;group<groups_for_this_block;group++){
		int index=tx;
		F min_val=limit<F>();
		F min_id=index;
		//printf("index: %d\n",groups_for_this_block);

		while (index<n_particles){
			sdata[tx].y=group*n_particles*gridDim.x+blockIdx.x*n_particles+index;
			if ((index+blockDim.x)<n_particles){
				F v1=r_ptr[index];
				F v2=r_ptr[index+blockDim.x];
				if (v1<v2){
					sdata[tx].x=v1;
					//sdata[index+blockDim.x]=index;
					
				}
				else{
					sdata[tx].x=v2;
					sdata[tx].y+=blockDim.x;
				}
			
			}else{
				sdata[tx].x=r_ptr[index];
			}
				
			__syncthreads();
			
			/*if (blockIdx.x==0 && tx==0){
				for (int i=0;i<8;i++){
					printf("%f\t",sdata[i].x);
				}
				printf("\n");
			}*/
		
			if (THREADS_X_PER_BLOCK>=512){
				if (tx<256){
					if (sdata[tx].x>sdata[tx+256].x){
						sdata[tx].x=sdata[tx+256].x;
						sdata[tx].y=sdata[tx+256].y;
					}									
				}
				__syncthreads();
			}
			if (THREADS_X_PER_BLOCK>=256){
				if (tx<128){
					if (sdata[tx].x>sdata[tx+128].x){
						sdata[tx].x=sdata[tx+128].x;
						sdata[tx].y=sdata[tx+128].y;
					}
				}
				__syncthreads();
			}	
			if (THREADS_X_PER_BLOCK>=128){
				if (sdata[tx].x>sdata[tx+64].x){
					sdata[tx].x=sdata[tx+64].x;
					sdata[tx].y=sdata[tx+64].y;
				}
				__syncthreads();
			}
			
			if (tx<32){
				
				k_warp_reduce_group_minimum<THREADS_X_PER_BLOCK,F2>(sdata,tx);
				if (sdata[0].x<min_val){
					
					min_val=sdata[0].x;
					min_id=sdata[0].y;
				}
				sdata[tx].x=limit<F>();
			}
			index+=2*blockDim.x;
			__syncthreads();	
		}
		//Because all threads write the same value, an if statement is not required
		if (tx==0){
			int rpos=group*gridDim.x+blockIdx.x;
			if (min_val<gmin_dest[rpos]){
				//printf("id: %d\n",min_id);
				gmin_dest[rpos]=min_val;
				gmin_id[rpos]=min_id;
				pso_rep_counter[rpos]=0;
			}
			else{
				pso_rep_counter[rpos]+=1;
			}
			//r[rpos]=min_val;
			//r[rpos+n_groups]=min_id;
		}	
		r_ptr+=n_particles*gridDim.x;
		iter+=n_particles*gridDim.x;
		__syncthreads();	
	}
}

template<unsigned int THREADS_X_PER_BLOCK, class F, class F2>
__global__
void k_reduce_to_global_minimum(int n_particles, int n_groups, int n_params, F* r, int* r_id, F* min_params, F* min_prev, F* dest_params){
	int index=2*blockIdx.x*blockDim.x+threadIdx.x;
		
	if (index>=n_groups){
		return;
	}

	/*The function k_warp_reduce_sum expects shared memory size to be minimum two times the size
	of a warp*/
	constexpr int memsize=(THREADS_X_PER_BLOCK<=64)?64:THREADS_X_PER_BLOCK;
	static __shared__ F2 sdata[memsize];
	int tx=threadIdx.x;
	sdata[tx].x=limit<F>(); // TODO: Max float
	
	F* r_ptr=r+n_groups*blockIdx.x;
	int* r_id_ptr=r_id+n_groups*blockIdx.x;
	F min_val=limit<F>();
	//F min_id;
	while (index<n_groups){
		sdata[tx].y=r_id_ptr[index];
		if ((index+blockDim.x)<n_groups){
			F v1=r_ptr[index];
			F v2=r_ptr[index+blockDim.x];
			if (v1<v2){
				sdata[tx].x=v1;
				
			}
			else{
				sdata[tx].x=v2;
				sdata[tx].y=r_id_ptr[index+blockDim.x];
			}
		
		}else{
			sdata[tx].x=r_ptr[index];
						
		}
		__syncthreads();
	
		if (THREADS_X_PER_BLOCK>=512){
			if (tx<256){
				if (sdata[tx].x>sdata[tx+256].x){
					sdata[tx].x=sdata[tx+256].x;
					sdata[tx].y=sdata[tx+256].y;
				}									
			}
			__syncthreads();
		}
		if (THREADS_X_PER_BLOCK>=256){
			if (tx<128){
				if (sdata[tx].x>sdata[tx+128].x){
					sdata[tx].x=sdata[tx+128].x;
					sdata[tx].y=sdata[tx+128].y;
				}
			}
			__syncthreads();
		}	
		if (THREADS_X_PER_BLOCK>=128){
			if (sdata[tx].x>sdata[tx+64].x){
				sdata[tx].x=sdata[tx+64].x;
				sdata[tx].y=sdata[tx+64].y;
			}
			__syncthreads();
		}
		
		if (tx<32){
			//printf("v:%f and %d\n",sdata[index].x, index);
			k_warp_reduce_group_minimum<THREADS_X_PER_BLOCK,F2>(sdata,tx);
			//printf("after:%f and %d\n",sdata[index].x, index);
			if (sdata[0].x<min_val){
				min_val=sdata[0].x;
				//min_id=sdata[0].y;
			}
			//sdata[tx].x=limit<F>();
		}
		index+=2*blockDim.x*gridDim.x;
		__syncthreads();	
	}
	
	//printf("v:%f and %f and %d\n",min_val, min_prev[0],blockIdx.x);
	if (tx<32 && min_val<min_prev[0]){	
		min_prev[0]=min_val;
		//printf("New min: %f with id: %d\n",min_val, sdata[0].y);
		int stride_dest=(n_groups+(2*blockDim.x)-1)/(2*blockDim.x);
		//printf("Look: %d and %d\n",n_groups, ThreaDim.x);
		if(stride_dest>gridDim.x){
			stride_dest=gridDim.x;
		}
		for (int idx=tx;idx<n_params;idx+=blockDim.x){
			dest_params[blockIdx.x+idx*stride_dest]=min_params[idx*n_groups*n_particles+int(sdata[0].y)];
		}
	}
}