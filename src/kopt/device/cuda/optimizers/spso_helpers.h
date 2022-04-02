#pragma once
#include <device/cuda/pcg.h>
	#include <stdio.h>
namespace kf::opt{
	using namespace kf::rand;
	
	enum class KPSOError{
		OK,
		Error,
		CUDAMallocError
	};
	
	namespace helpers::pso{
		
		template<class T>
		__global__
		void k_update_le(int n, T* dest, int stride_dest, T* comp, int stride_comp){
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
			
			for (int i=idx;i<n;i+=gridDim.x*blockDim.x){
				T v=comp[i*stride_comp];
				if(dest[i*stride_dest]<v){
					dest[i*stride_dest]=v;
				}
			}
		}
		
		template<class T>
		__global__
		void k_update_pmin_vars(int n_particles,int n_groups,int n_vars, T* pmin, T* pmin_prev, T* pmin_vars, T* curr_vars){	
			int n_all=n_vars*n_particles*n_groups;
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
			int ppos;

			while (idx < n_all){
				ppos=idx%(n_particles*n_groups);
				T v2=pmin_prev[ppos];
				T v1=pmin[ppos];
				if(v1<v2){ 
					pmin_vars[idx]=curr_vars[idx];
				}
				idx+=gridDim.x*blockDim.x;
			}
		}
		
			template<class T>
		__global__
		void k_update_pmin(int n_particles,int n_groups,int n_vars, T* pmin, T* pmin_prev){	
			int n_all=n_particles*n_groups;
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
		
			while (idx < n_all){
				T v2=pmin_prev[idx];
				T v1=pmin[idx];
				if(v1<v2){ 
				
					pmin_prev[idx]=v1;		
				}
				else{
					pmin[idx]=v2; 			
				}
				idx+=gridDim.x*blockDim.x;
			}
		}	
		
		template<class T, class T2>
		__global__
		void rand_init(int n_particles, int n_groups, int n_vars, ulonglong2* pcg_data, T* vars, T2* bounds){
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
			int n_all=n_vars*n_particles*n_groups;

			if (idx<n_all){
				ulonglong2& pcg_thread_data=(pcg_data[idx]);
				int stride=n_groups*n_particles;
				T one_over_stride=1.0/stride;
				T bl;
				T bu;
				int bpos;
				do{
					bpos=(idx*one_over_stride);	
					bl=bounds[bpos].x;
					bu=bounds[bpos].y;		
					vars[idx]=(bu-bl)*pcg_xsh_rr_real(pcg_thread_data)+bl;		
					
					idx+=gridDim.x*blockDim.x;
				}
				while(idx<n_all);				
			}
		}
		
	//See https://towardsdatascience.com/particle-swarm-optimization-visually-explained-46289eeb2e14

		template<class T, class T2>
		__global__
		void k_advance_particles(int n_particles,int n_groups,int n_vars, T w, T c1, T c2, int* pso_rep_counter, ulonglong2* pcg_data, T* pmin_prev, T* pmin_vars, T* gmin, int* gmin_id, T* velocity, T* current_vars, T2* bounds){	
			int n=n_particles*n_groups*n_vars;
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
			if (idx<n){
						
				T n_particles_inv=1.0/n_particles;
				T n_particles_groups_inv=1.0/(n_particles*n_groups);
				//F w=0.4*(current_iteration-max_iterations)*(max_iterations_inv*max_iterations_inv)+0.4
				//F 
				//int rem=n_groups%blockDim.x;
				//int groups_for_this_block=(blockDim.x<rem)?rem+(n_groups/blockDim.x):(n_groups/blockDim.x);
				int group_id;
				int particle_id;

				ulonglong2& pcg_thread_data=(pcg_data[idx]);
				int max_tries=3; //maximum amount of local exploration is fitness does not decrease
				do{
					group_id=int(idx*n_particles_inv)%n_groups;
					particle_id=idx%(n_particles*n_groups);
					//Rethrow particles
					int bpos=idx*n_particles_groups_inv;
					//pcg_real<T>(pcg_state,pcg_inc);
					T bl=bounds[bpos].x;
					T bu=bounds[bpos].y;
					//printf("%f vs %d\n",gmin[group_id],pso_rep_counter[group_id]);
					if (pso_rep_counter[group_id]<=max_tries){
						T current_var=current_vars[idx];
						T r1=2.0*pcg_xsh_rr_real(pcg_thread_data);
						T r2=2.0*pcg_xsh_rr_real(pcg_thread_data);
						int gmin_vars_id=gmin_id[group_id];
						//printf("gminid: %d\n",gmin_vars_id);
					//	printf("cp: %f, pmin: %f\n",current_var,pmin_vars[gmin_vars_id+bpos*n_particles*n_groups]);
						T v=w*velocity[idx]+c1*r1*(pmin_vars[idx]-current_var)+c2*r2*(pmin_vars[gmin_vars_id+bpos*n_particles*n_groups]-current_var);
						//printf("v:%f\n",v);
						current_var+=v;
						velocity[idx]=v;
						if (current_var<bl){
							current_var=bl+0.1*(bu-bl)*pcg_xsh_rr_real(pcg_thread_data);
						//	printf("bl!\n");
						}
						else if (current_var>bu){
							current_var=bu-0.1*(bu-bl)*pcg_xsh_rr_real(pcg_thread_data);
							//		printf("bu!\n");
						}
						else{
						//	printf("Nobound\n");
						}
						current_vars[idx]=current_var;
					}
					
					else{
						pso_rep_counter[group_id]=0;
						//printf("bl: %f, bu: %f\n",bl,bu);
						
						current_vars[idx]=(bu-bl)*pcg_xsh_rr_real(pcg_thread_data)+bl;
						//printf("idx:%d\n",idx);
						pmin_prev[particle_id]=limit<T>();
						gmin[group_id]=limit<T>(); //multiple writes from threads at same location is OK	
					}					
					/*
					group_id=int(idx*n_particles_inv)%n_groups;
					particle_id=idx%(n_particles*n_groups);
					//Rethrow particles
					int bpos=idx*n_particles_groups_inv;
					//pcg_real<T>(pcg_state,pcg_inc);
					T bl=bounds[bpos].x;
					T bu=bounds[bpos].y;
					if (gmin[group_id]<gmin_prev[group_id]){
						pso_rep_counter[group_id]=0; //simultaneous write should be safe
						gmin_prev[group_id]=gmin[group_id];
						T current_var=current_vars[idx];
						T r1=2.0*pcg_xsh_rr_real(pcg_thread_data);
						T r2=2.0*pcg_xsh_rr_real(pcg_thread_data);
						int gmin_vars_id=gmin[group_id+n_groups];
					//	printf("cp: %f, pmin: %f\n",current_var,pmin_vars[gmin_vars_id+bpos*n_particles*n_groups]);
						T v=w*velocity[idx]+c1*r1*(pmin_vars[idx]-current_var)+c2*r2*(pmin_vars[gmin_vars_id+bpos*n_particles*n_groups]-current_var);
						//printf("v:%f\n",v);
						current_var+=v;
						velocity[idx]=v;
						if (current_var<bl){
							current_var=bl+0.1*(bu-bl)*pcg_xsh_rr_real(pcg_thread_data);
						//	printf("bl!\n");
						}
						else if (current_var>bu){
							current_var=bu-0.1*(bu-bl)*pcg_xsh_rr_real(pcg_thread_data);
							//		printf("bu!\n");
						}
						else{
						//	printf("Nobound\n");
						}
						current_vars[idx]=current_var;
					}
					else if (pso_rep_counter[group_id]<max_tries){
						pso_rep_counter[group_id]+=1; //simultaneous write should be safe
						gmin_prev[group_id]=gmin[group_id];
						T current_var=current_vars[idx];
						T r1=2.0*pcg_xsh_rr_real(pcg_thread_data);
						T r2=2.0*pcg_xsh_rr_real(pcg_thread_data);
						int gmin_vars_id=gmin[group_id+n_groups];
					//	printf("cp: %f, pmin: %f\n",current_var,pmin_vars[gmin_vars_id+bpos*n_particles*n_groups]);
						T v=w*velocity[idx]+c1*r1*(pmin_vars[idx]-current_var)+c2*r2*(pmin_vars[gmin_vars_id+bpos*n_particles*n_groups]-current_var);
						//printf("v:%f\n",v);
						current_var+=v;
						velocity[idx]=v;
						if (current_var<bl){
							current_var=bl+0.1*(bu-bl)*pcg_xsh_rr_real(pcg_thread_data);
						//	printf("bl!\n");
						}
						else if (current_var>bu){
							current_var=bu-0.1*(bu-bl)*pcg_xsh_rr_real(pcg_thread_data);
							//		printf("bu!\n");
						}
						else{
						//	printf("Nobound\n");
						}
						current_vars[idx]=current_var;				
					}
					else{
						pso_rep_counter[group_id]=0;
						//printf("bl: %f, bu: %f\n",bl,bu);
						
						current_vars[idx]=(bu-bl)*pcg_xsh_rr_real(pcg_thread_data)+bl;
						//printf("idx:%d\n",idx);
						pmin_prev[particle_id]=limit<T>();
						gmin[group_id]=limit<T>(); //multiple writes from threads at same location is OK	
			
					}
					*/
					idx+=gridDim.x*blockDim.x;			
				}
				while(idx<n);
				
			}
		}
	}

		__global__
	void print_data(int n, float* data){

		for (int i=0;i<n;i++){
			printf("%f\t",data[i]);
		}
			printf("\n\n");
	}

	template<class T>
		__global__
	void print_data_vec(int n, T* data);


	template<>
		__global__
	void print_data_vec(int n, float2* data){

		for (int i=0;i<n;i++){
			printf("%f\t",data[i].x);
			printf("%f\t",data[i].y);
		}
			printf("\n\n");
	}

	template<>
		__global__
	void print_data_vec(int n, ulonglong2* data){

		for (int i=0;i<n;i++){
			printf("%ull\t",data[i].x);
			printf("%ull\t",data[i].y);
		}
		printf("\n\n");
	}
	
}