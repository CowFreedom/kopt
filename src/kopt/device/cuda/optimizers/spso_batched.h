#pragma once
#include <device/cuda/pcg.h>
#include <device/cuda/util/reduce_minimum.h>
#include <device/cuda/optimizers/spso_helpers.h>

namespace kf::opt{
	using namespace kf::rand;
	using namespace kf::opt::helpers::pso;		


	template<int BX, int TX, int NSTREAMS, class T, class T2, class F, class ...Args>
	KPSOError spso_kbatch(const int* n_particles, const int* n_groups, const int* n_vars, int batch_size, int iterations, int* pso_rep_counter_d, ulonglong2* pcg_data_d,  T* pmin_d, T* pmin_prev_d, T* pmin_vars_d, T* gmin_prev_d, int* gid_d, T* fmin_d,  T* velocity_d, T* vars_d,  T* vars_buf_d, T2* bounds_d, F* f, Args&& ... args){
		if (TX <=0 || BX <= 0 || NSTREAMS <= 0){
			return KPSOError::Error;
		}
		for (int i=0;i<batch_size;i++){
			if (n_vars[i]<=0 || ((n_groups[i]&((~n_groups[i])+1))!= n_groups[i]) || n_groups[i]<=0|| ((n_particles[i]&((~n_particles[i])+1))!= n_particles[i]) ||n_particles[i]<=0){
				return KPSOError::Error;
			}
		}

		cudaStream_t streams[NSTREAMS];
		T iterations_inv=1.0/iterations;
		T w;
		T c1;
		T c2;

		int* pso_rep_counter_ptr[NSTREAMS];
		T* pmin_vars_ptr[NSTREAMS];
		T* min_buf_ptr[NSTREAMS];
		T* fmin_ptr[NSTREAMS];
		T2* bounds_ptr[NSTREAMS];
		T* gmin_prev_ptr[NSTREAMS];
		int* gid_ptr[NSTREAMS];
		T* pmin_prev_ptr[NSTREAMS];
		ulonglong2* pcg_ptr[NSTREAMS];
		T* vars_ptr[NSTREAMS];
		T* vars_buf_ptr[NSTREAMS];
		T* velocity_ptr[NSTREAMS];
		int TXMAX=((TX*BX)<1024)?TX*BX:1024;
		int v=0;
		for (int i=0;i<batch_size;i++){
			v+=n_particles[i]*n_groups[i]*n_vars[i];
		}
	
		pso_rep_counter_ptr[0]=pso_rep_counter_d;
		pcg_ptr[0]=pcg_data_d;	
		min_buf_ptr[0]=pmin_d;
		pmin_prev_ptr[0]=pmin_prev_d;
		pmin_vars_ptr[0]=pmin_vars_d;
		gmin_prev_ptr[0]= gmin_prev_d;
		gid_ptr[0]=gid_d;
		fmin_ptr[0]=fmin_d;
		vars_ptr[0]=vars_d;
		vars_buf_ptr[0]=vars_buf_d;
		bounds_ptr[0]=bounds_d;
		velocity_ptr[0]=velocity_d;
		cudaStreamCreate(&streams[0]);

		int chunks=batch_size/NSTREAMS;
		int rem=batch_size%NSTREAMS;
		int used_streams=(chunks==0)?rem:NSTREAMS;
		
		for (int stream=1;stream<used_streams;stream++){
			cudaStreamCreate(&streams[stream]);
			pso_rep_counter_ptr[stream]=pso_rep_counter_ptr[stream-1]+n_groups[stream-1];
			pcg_ptr[stream]=pcg_ptr[stream-1]+TX*BX;
			min_buf_ptr[stream]=min_buf_ptr[stream-1]+n_groups[stream-1]*n_particles[stream-1];
			pmin_prev_ptr[stream]=pmin_prev_ptr[stream-1]+n_groups[stream-1]*n_particles[stream-1];
			pmin_vars_ptr[stream]=pmin_vars_ptr[stream-1]+n_groups[stream-1]*n_particles[stream-1]*n_vars[stream-1];
			gmin_prev_ptr[stream]= gmin_prev_ptr[stream-1]+n_groups[stream-1];
			gid_ptr[stream]=gid_ptr[stream-1]+n_groups[stream-1];
			fmin_ptr[stream]=fmin_ptr[stream-1]+1;
			velocity_ptr[stream]=velocity_ptr[stream-1]+n_groups[stream-1]*n_particles[stream-1]*n_vars[stream-1];
			vars_ptr[stream]=vars_ptr[stream-1]+n_vars[stream-1];	
			vars_buf_ptr[stream]=vars_buf_ptr[stream-1]+n_groups[stream-1]*n_particles[stream-1]*n_vars[stream-1];	
			bounds_ptr[stream]=bounds_ptr[stream-1]+n_vars[stream-1];
		}		
		
		for (int i=0;i<chunks;i++){			
			size_t pmin_buf_size=0;
			size_t gmin_buf_size=0;

			for (size_t stream=0;stream<NSTREAMS;stream++){
				pmin_buf_size+=n_groups[stream]*n_particles[stream];
				gmin_buf_size+=n_groups[stream];
			}
		
			cudaMemset(pmin_prev_d,127,sizeof(T)*pmin_buf_size); //TODO: This is not the maximum floating point value. set as constant somewhere
			cudaMemset(gmin_prev_d,127,sizeof(T)*gmin_buf_size); //TODO: This is not the maximum floating point value. set as constant somewhere
			cudaMemset(fmin_ptr[0],127,sizeof(T)*NSTREAMS);

			for (int stream=0;stream<NSTREAMS;stream++){
				rand_init<T,T2><<<dim3(BX,1,1),dim3(TX,1,1),0,streams[stream]>>>(n_particles[stream],n_groups[stream],n_vars[stream],pcg_ptr[stream],vars_buf_ptr[stream],bounds_ptr[stream]); //#1
			}
			for (int iteration=0;iteration<iterations;iteration++){
				
				for (int stream=0;stream<NSTREAMS;stream++){
					f[stream]<<<dim3(BX,1,1),dim3(TX,1,1),0,streams[stream]>>>(vars_buf_ptr[stream],min_buf_ptr[stream],args...); //#2
				}
				//
				for (int stream=0;stream<NSTREAMS;stream++){
					k_update_pmin_vars<<<dim3(BX,1,1),dim3(TX,1,1),0,streams[stream]>>>(n_particles[stream],n_groups[stream],n_vars[stream], min_buf_ptr[stream], pmin_prev_ptr[stream], pmin_vars_ptr[stream],vars_buf_ptr[stream]); //#3
				}
				for (int stream=0;stream<NSTREAMS;stream++){
					k_update_pmin<<<dim3(BX,1,1),dim3(TX,1,1),0,streams[stream]>>>(n_particles[stream],n_groups[stream],n_vars[stream], min_buf_ptr[stream], pmin_prev_ptr[stream]); //#3
				}
			
				for (int stream=0;stream<NSTREAMS;stream++){
					k_reduce_to_group_minimum<TX,T,T2><<<dim3(BX,1,1),dim3(TX,1,1),0,streams[stream]>>>(n_particles[stream],n_groups[stream],min_buf_ptr[stream],gmin_prev_ptr[stream],gid_ptr[stream],pso_rep_counter_ptr[stream]);//#4
				}
				
				for (int stream=0;stream<NSTREAMS;stream++){
					k_reduce_to_global_minimum<TX,T,T2><<<dim3(1,1,1),dim3(TXMAX,1,1),0,streams[stream]>>>(n_particles[stream],n_groups[stream], n_vars[stream], gmin_prev_ptr[stream], gid_ptr[stream], pmin_vars_ptr[stream], fmin_ptr[stream], vars_ptr[stream]);//#4
				}
		
				for (int stream=0;stream<((iteration<(iterations-1))?NSTREAMS:0);stream++){
					w=0.4*((iteration-iterations)*iterations_inv*iterations_inv)+0.4;
					c1=-1.0*iteration*iterations_inv+1.5;
					c2=0.5*iteration*iterations_inv+0.5;
					k_advance_particles<T><<<dim3(BX,1,1),dim3(TX,1,1),0,streams[stream]>>>(n_particles[stream],n_groups[stream],n_vars[stream],w,c1,c2,pso_rep_counter_ptr[stream],pcg_ptr[stream],pmin_prev_ptr[stream],pmin_vars_ptr[stream],gmin_prev_ptr[stream],gid_ptr[stream],velocity_ptr[stream],vars_buf_ptr[stream],bounds_ptr[stream]); //#5
				}
				for (int stream=0;stream<NSTREAMS;stream++){
					cudaStreamSynchronize(streams[stream]);
				}
			}
			f+=NSTREAMS;
			vars_ptr[0]=vars_ptr[NSTREAMS-1]+n_vars[NSTREAMS-1];	
			fmin_ptr[0]+=NSTREAMS;
			bounds_ptr[0]=bounds_ptr[NSTREAMS-1]+n_vars[NSTREAMS-1];	
			n_particles+=NSTREAMS;
			n_groups+=NSTREAMS;
			n_vars+=NSTREAMS;
		
			for (int stream=1;stream<((i<(chunks-1))?NSTREAMS:rem);stream++){
				pso_rep_counter_ptr[stream]=pso_rep_counter_ptr[stream-1]+n_groups[stream-1];
				min_buf_ptr[stream]=min_buf_ptr[stream-1]+n_groups[stream-1]*n_particles[stream-1];
				pmin_prev_ptr[stream]=pmin_prev_ptr[stream-1]+n_groups[stream-1]*n_particles[stream-1];
				pmin_vars_ptr[stream]=pmin_vars_ptr[stream-1]+n_groups[stream-1]*n_particles[stream-1]*n_vars[stream-1];
				gmin_prev_ptr[stream]= gmin_prev_ptr[stream-1]+n_groups[stream-1];
				//printf("off %d, %d, %d\n",n_groups[stream-1],n_vars[stream-1],n_particles[stream-1]);
				gid_ptr[stream]=gid_ptr[stream-1]+n_groups[stream-1];
				fmin_ptr[stream]=fmin_ptr[stream-1]+1;
				velocity_ptr[stream]=velocity_ptr[stream-1]+n_groups[stream-1]*n_particles[stream-1]*n_vars[stream-1];
				vars_ptr[stream]=vars_ptr[stream-1]+n_vars[stream-1];	
				vars_buf_ptr[stream]=vars_buf_ptr[stream-1]+n_groups[stream-1]*n_particles[stream-1]*n_vars[stream-1];	
				bounds_ptr[stream]=bounds_ptr[stream-1]+n_vars[stream-1];
			}
	
		}
		if (rem!=0){
			size_t pmin_buf_size=0;
			size_t gmin_buf_size=0;

			for (size_t stream=0;stream<rem;stream++){
				pmin_buf_size+=n_groups[stream]*n_particles[stream];
				gmin_buf_size+=n_groups[stream];
			}
			cudaMemset(pmin_prev_d,127,sizeof(T)*pmin_buf_size); //TODO: This is not the maximum floating point value. set as constant somewhere
			cudaMemset(gmin_prev_d,127,sizeof(T)*gmin_buf_size); //TODO: This is not the maximum floating point value. set as constant somewhere
			cudaMemset(fmin_ptr[0],127,sizeof(T)*rem);
			for (int stream=0;stream<rem;stream++){
				rand_init<T,T2><<<dim3(BX,1,1),dim3(TX,1,1),0,streams[stream]>>>(n_particles[stream],n_groups[stream],n_vars[stream],pcg_ptr[stream],vars_buf_ptr[stream],bounds_ptr[stream]); //#1
			}
			for (int iteration=0;iteration<iterations;iteration++){
			//	printf("\n\nIter %d\n",iteration);					
				for (int stream=0;stream<rem;stream++){
					f[stream]<<<dim3(BX,1,1),dim3(TX,1,1),0,streams[stream]>>>(vars_buf_ptr[stream],min_buf_ptr[stream],args...); //#2
				}
			
				for (int stream=0;stream<rem;stream++){
					k_update_pmin_vars<<<dim3(BX,1,1),dim3(TX,1,1),0,streams[stream]>>>(n_particles[stream],n_groups[stream],n_vars[stream], min_buf_ptr[stream], pmin_prev_ptr[stream], pmin_vars_ptr[stream],vars_buf_ptr[stream]); //#3
				}
				
				for (int stream=0;stream<rem;stream++){
					k_update_pmin<<<dim3(BX,1,1),dim3(TX,1,1),0,streams[stream]>>>(n_particles[stream],n_groups[stream],n_vars[stream], min_buf_ptr[stream], pmin_prev_ptr[stream]); //#3
				}
	//
				for (int stream=0;stream<rem;stream++){
					k_reduce_to_group_minimum<TX,T,T2><<<dim3(BX,1,1),dim3(TX,1,1),0,streams[stream]>>>(n_particles[stream],n_groups[stream],min_buf_ptr[stream],gmin_prev_ptr[stream],gid_ptr[stream],pso_rep_counter_ptr[stream]);//#4
				}
//print_data<<<1,1>>>(n_groups[0],min_buf_ptr[0]);				
			//print_data<<<1,1>>>(n_groups[0],gmin_prev_ptr[0]);
		//
				for (int stream=0;stream<rem;stream++){
					k_reduce_to_global_minimum<TX,T,T2><<<dim3(1,1,1),dim3(TXMAX,1,1),0,streams[stream]>>>(n_particles[stream],n_groups[stream], n_vars[stream], gmin_prev_ptr[stream], gid_ptr[stream], pmin_vars_ptr[stream], fmin_ptr[stream], vars_ptr[stream]);//#4
				}
				
		//		print_data<<<1,1>>>(1,fmin_ptr[0]);
				for (int stream=0;stream<((iteration<(iterations-1))?rem:0);stream++){
					w=0.4*((iteration-iterations)*iterations_inv*iterations_inv)+0.4;
					c1=-1.0*iteration*iterations_inv+1.5;
					c2=0.5*iteration*iterations_inv+0.5;
					//printf("w: %f, c1: %f, c2: %f\n",w,c1,c2);
					k_advance_particles<T><<<dim3(BX,1,1),dim3(TX,1,1),0,streams[stream]>>>(n_particles[stream],n_groups[stream],n_vars[stream],w,c1,c2,pso_rep_counter_ptr[stream],pcg_ptr[stream],pmin_prev_ptr[stream],pmin_vars_ptr[stream],gmin_prev_ptr[stream],gid_ptr[stream],velocity_ptr[stream],vars_buf_ptr[stream],bounds_ptr[stream]); //#5
				}
					
				for (int stream=0;stream<rem;stream++){
					cudaStreamSynchronize(streams[stream]);
				}
			}			
		}
		
		
		//	print_data<<<1,1>>>(8,pmin_d);

			//print_data<<<1,1>>>(4,min_buf);
			//Reduziere zu globalen minimum
			
		//print_data<<<1,1>>>(v,vars_d);
		//print_data<<<1,1>>>(batch_size,vars_d);
		
				
		for (int stream=0;stream<used_streams;stream++){
			cudaStreamDestroy(streams[stream]);
		}
		
		return KPSOError::OK;
		
	}
	/*
	template<int BX, int TX, int NSTREAMS, class T, class T2, class F, class ...Args>
	KPSOError spso_kbatch(const int* n_particles, const int* n_groups, const int* n_vars, int batch_size, int pso_iterations, T* fmin_d, T* vars_result_d, T2* bounds_d, F* f, Args&& ... args){
		T* vars_buf; //buffer that can hold variables per for each particle
		T* pmin; //buffer that holds minimum value (in terms of loss) of a particle
		T* pmin_prev; //buffer that holds previous minimum value (in terms of loss) of a particle
		T* pmin_vars; //buffer that holds the variable combinations responsible for the particle minimum values
		T* gmin_prev; //buffer that holds group minimum value (in terms of loss) of a group
		int* gid; //buffer that holds particle id of group's minimum value (in terms of loss) of a group
		T* velocity; //buffer that stores particle velocity values
		int* pso_rep_counter; //buffer that holds the number of subsequent iterations per group that did not change the group minimum
		//determines how many subsequent iterations with non changing group minimum are allowed, before the values of all group members are sampled anew
		ulonglong2* pcg_data; //initialisation data for random number engine
 
		int n_all_particles=0; //
		int n_all_groups=0; //
		int n_all_vars=0;
		for (int i=0;i<batch_size;i++){
			n_all_particles=n_particles[i];
			n_all_groups+=n_groups[i];
			n_all_vars+=n_vars[i];		
		}
		//TODO: Many of the routines allocate way too much memory. Need a proper subset sum implementation
		cudaError_t err;
		err=cudaMalloc((void**)&pcg_data,sizeof(ulonglong2)*TX*BX*NSTREAMS);
		if (err!=cudaSuccess){
			return KPSOError::CUDAMallocError;
		}
		cudaMalloc((void**)&pso_rep_counter,sizeof(int)*batch_size*n_all_groups);
		if (err!=cudaSuccess){
			cudaFree(pcg_data);
			return KPSOError::CUDAMallocError;
		}
		err=cudaMalloc((void**)&velocity,sizeof(T)*n_all_vars*n_all_particles*n_all_groups);
		if (err!=cudaSuccess){
			cudaFree(pso_rep_counter);
			cudaFree(pcg_data);
			return KPSOError::CUDAMallocError;
		}
		err=cudaMalloc((void**)&pmin,sizeof(T)*n_all_particles*n_all_groups*NSTREAMS);
		if (err!=cudaSuccess){
			cudaFree(pso_rep_counter);
			cudaFree(pcg_data);
			cudaFree(velocity);
			return KPSOError::CUDAMallocError;
		}
		err=cudaMalloc((void**)&pmin_prev,sizeof(T)*n_all_particles*n_all_groups*NSTREAMS);
		if (err!=cudaSuccess){
			cudaFree(pso_rep_counter);
			cudaFree(pcg_data);

			cudaFree(pmin);
			cudaFree(velocity);
			return KPSOError::CUDAMallocError;
		}
		err=cudaMalloc((void**)&gmin_prev,sizeof(T)*n_all_groups*NSTREAMS);
		if (err!=cudaSuccess){
			cudaFree(pso_rep_counter);
			cudaFree(pcg_data);
			cudaFree(pmin_prev);
			cudaFree(pmin);
			cudaFree(velocity);
			return KPSOError::CUDAMallocError;
		}
		err=cudaMalloc((void**)&gid,sizeof(int)*n_groups_per_problem*NSTREAMS);
		if (err!=cudaSuccess){
			cudaFree(pso_rep_counter);
			cudaFree(pcg_data);
			cudaFree(pmin_prev);
			cudaFree(gmin_prev);
			cudaFree(gid);
			cudaFree(pmin);
			cudaFree(velocity);
			return KPSOError::CUDAMallocError;
		}
		err=cudaMalloc((void**)&bounds,sizeof(T2)*n_all_vars);
		if (err!=cudaSuccess){
			cudaFree(pso_rep_counter);
			cudaFree(pcg_data);
			cudaFree(pmin_prev);
			cudaFree(gmin_prev);
			cudaFree(gid);
			cudaFree(pmin);
			cudaFree(velocity);
			return KPSOError::CUDAMallocError;
		}
		err=cudaMalloc((void**)&pmin_vars, sizeof(T)*NSTREAMS*n_all_vars*n_all_particles_per_group*n_all_groups);
		if (err!=cudaSuccess){
			cudaFree(pso_rep_counter);
			cudaFree(pcg_data);
			cudaFree(pmin_prev);
			cudaFree(gmin_prev);
			cudaFree(gid);
			cudaFree(pmin);
			cudaFree(bounds);
			cudaFree(velocity);
			return KPSOError::CUDAMallocError;
		}
		err=cudaMalloc((void**)&vars_buf, sizeof(T)*n_all_vars*n_all_particles_per_group*n_all_groups*NSTREAMS);
		if (err!=cudaSuccess){
			cudaFree(pso_rep_counter);
			cudaFree(pcg_data);
			cudaFree(pmin_vars);
			cudaFree(pmin_prev);
			cudaFree(gmin_prev);
			cudaFree(gid);
			cudaFree(pmin);
			cudaFree(bounds);
			cudaFree(velocity);
			return KPSOError::CUDAMallocError;
		}
	
		auto res= spso_kbatch<BX,TX,NSTREAMS>(n_particles_per_group, n_groups,n_vars,batch_size,pso_iterations,pso_rep_counter, pcg_data, pmin,pmin_prev,pmin_vars,gmin_prev,gid,fmin_d, velocity,vars_result_d, vars_buf, bounds, functions,tend, args...);
		cudaFree(pso_rep_counter);
		cudaFree(pcg_data);
		cudaFree(pmin_vars);
		cudaFree(pmin_prev);
		cudaFree(gmin_prev);
		cudaFree(gid);
		cudaFree(pmin);
		cudaFree(bounds);
		cudaFree(vars_buf);
		cudaFree(velocity);
		return res;
	}	
	*/
}