#include <core/kopt.h>
#include <core/krand.h>
#include <cmath>
#include <chrono>
#include <stdio.h>
#include <utility>

//Helpful error checking macro
//See https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<class T>
__global__
void set_bounds(int batch_size,T* bounds){
	int n_vars[]={1};
	for (int j=0;j<batch_size;j++){
		for (int i=0;i<n_vars[j];i++){
			if (i==0){
				bounds[i].x=0;
				bounds[i].y=1.57;
			}
		}
		bounds+=n_vars[j];
	}
}

template<class T>
__global__
void set_pcg_data(int nthreads, T* pcg_data){
	for (int i=0;i<nthreads;i++){
		uint64_t state;
		uint64_t inc;
		kf::rand::set_sequence(0,i,state,inc);
		pcg_data[i].x=state;
		pcg_data[i].y=inc;
	}
}


__global__
void randsequences(){
	uint64_t pcg_state; //TODO Check if this is safe
	uint64_t pcg_inc;
	kf::rand::set_sequence(0,0,pcg_state,pcg_inc);
	for (int i=0;i<2;i++){
		printf("%f\t",kf::rand::pcg_xsh_rr_real(pcg_state,pcg_inc));
	
	}
	printf("\n");
}


template<class T, int STRIDE, int DRONE_START_X, int DRONE_START_Y, int DRONE_SPEED_X, int DRONE_SPEED_Y>
__global__
void ballistic_trajectory(const T* vars, T* fmin, T tend, T v0, T mass, T mu, T delta){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	T u1[4];
	T u_copy[4];
	T k1[4];
	T fu[4];
	T grav=9.8;
	T mu_over_m=mu/mass;
	T ah=1.0/(2.0+1.4142);
	while (idx<STRIDE){
		T curr_fmin=9999999999999;
		T alpha=vars[idx];
		u1[0]=0.0;
		u1[1]=0.0;
		u1[2]=v0*cos(alpha);
		u1[3]=v0*sin(alpha);
		T t=0.0;
		
		while (t<=tend){
			u_copy[0]=u1[0];
			u_copy[1]=u1[1];
			u_copy[2]=u1[2];
			u_copy[3]=u1[3];
			T vnorm=sqrt(u1[2]*u1[2]+u1[3]*u1[3]);
			T one_over_vnorm=1.0/vnorm;
			fu[0]=u1[2];
			fu[1]=u1[3];
			fu[2]=-mu_over_m*u1[2]*vnorm;
			fu[3]=-grav-mu_over_m*u1[3]*vnorm;
			
			T a=1.0;
			T b=-ah;
			T c=1.0;
			T d=-ah;
			T e=1.0+ah*mu_over_m*(vnorm+u1[2]*u1[2]*one_over_vnorm);
			T f=ah*mu_over_m*u1[2]*one_over_vnorm*u1[3];
			T g=ah*mu_over_m*u1[3]*one_over_vnorm*u1[2];
			T h=1.0+ah*mu_over_m*(vnorm+u1[3]*one_over_vnorm*u1[3]);
		
			k1[3]=(1.0/(h-(g*f/e)))*(fu[3]-(g/e)*fu[2]);
			k1[2]=(1.0/e)*(fu[2]-f*k1[3]);
			k1[1]=(1.0/c)*(fu[1]-d*k1[3]);
			k1[0]=(1.0/a)*(fu[0]-b*k1[2]);
			
			u1[0]=u_copy[0]+delta*k1[0];
			u1[1]=u_copy[1]+delta*k1[1];		
			u1[2]=u_copy[2]+delta*k1[2];
			u1[3]=u_copy[3]+delta*k1[3];
			T Dxi=DRONE_SPEED_X*(t)+DRONE_START_X;
			T Dyi=DRONE_SPEED_Y*(t)+DRONE_START_Y;
			t+=delta;
			T temp=(u1[0]-Dxi)*(u1[0]-Dxi)+(u1[1]-Dyi)*(u1[1]-Dyi);
		
			if(temp<curr_fmin){
				curr_fmin=temp;
			}
		}
//		printf("fmin f:%f\n",curr_fmin);
		fmin[idx]=curr_fmin;	
		idx+=gridDim.x*blockDim.x;			
	}
}


template<class T, class T2, int BATCH_SIZE>
void find_angles(){
	int batch_size=BATCH_SIZE;
	T* vars; //holds result variables
	T* vars_buf; //buffer that can hold variables per for each particle
	T* pmin; //buffer that holds minimum value (in terms of loss) of a particle
	T* pmin_prev; //buffer that holds previous minimum value (in terms of loss) of a particle
	T* pmin_vars; //buffer that holds the variable combinations responsible for the particle minimum values
	T* gmin_prev; //buffer that holds group minimum value (in terms of loss) of a group
	int* gid; //buffer that holds particle id of group's minimum value (in terms of loss) of a group
	T* fmin; //hols minimum value (in terms of loss) for each function in the batch
	T* velocity; //buffer that stores particle velocity values
	int* pso_rep_counter; //buffer that holds the number of subsequent iterations per group that did not change the group minimum
	//determines how many subsequent iterations with non changing group minimum are allowed, before the values of all group members are sampled anew
	int pso_iterations=7;//PSO iterations for each problem in the batch
	T2* bounds; //variable bounds for each particle
	ulonglong2* pcg_data; //initialisation data for random number engine
	constexpr int particles_in_each_group=16;
	int n_vars[BATCH_SIZE]; //number of function variables per problem in the batch
	int n_particles_per_group[BATCH_SIZE]; //Number of PSO particles per problem in the batch. Caution: Number of particles must be a power of two!
	int n_groups[BATCH_SIZE]; //Number of PSO groups per problem in the batch. 
	constexpr int n_groups_per_problem=4; //number of PSO groups per problem. Caution: Must be a power of two!
	int n_vars_per_problem=1; //number of variables per problem
	for (int i=0;i<batch_size;i++){
		n_vars[i]=n_vars_per_problem;
		n_particles_per_group[i]=particles_in_each_group;
		n_groups[i]=n_groups_per_problem;
	}
	int n_all_particles_per_group=0; //
	int n_all_groups=0; //
	int n_all_vars=0;
	for (int i=0;i<batch_size;i++){
		n_all_particles_per_group+=n_particles_per_group[i];
		n_all_groups+=n_groups[i];
		n_all_vars+=n_vars[i];		
	}
	
	constexpr int NSTREAMS=2; //number of concurrent CUDA streams used
	constexpr int TX=32; //number of threads per block
	constexpr int BX=2; //number of blocks per batch problem
	
	gpuErrchk(cudaMalloc((void**)&pcg_data,sizeof(ulonglong2)*TX*BX*NSTREAMS));
	gpuErrchk(cudaMalloc((void**)&pso_rep_counter,sizeof(int)*batch_size*n_groups_per_problem)); //For simplicity this allocates too much memory. Instead of using "batch_size*n_all_groups" we would have to multiply by the largest subset sum of size NSTREAMS in n_groups
	gpuErrchk(cudaMalloc((void**)&velocity,sizeof(T)*NSTREAMS*n_vars_per_problem*particles_in_each_group*n_groups_per_problem)); //For simplicity this allocates too much memory. Instead of using "*n_all_vars*n_all_particles_per_group*n_all_groups" we would have to multiply by the largest subset sum of size NSTREAMS in n_particles, n_groups
	gpuErrchk(cudaMalloc((void**)&pmin,sizeof(T)*particles_in_each_group*n_groups_per_problem*NSTREAMS)); //For simplicity this allocates too much memory. See above
	gpuErrchk(cudaMalloc((void**)&pmin_prev,sizeof(T)*particles_in_each_group*n_groups_per_problem*NSTREAMS));  //For simplicity this allocates too much memory. See above
	gpuErrchk(cudaMalloc((void**)&gmin_prev,sizeof(T)*n_groups_per_problem*NSTREAMS)); //For simplicity this allocates too much memory. See above
	gpuErrchk(cudaMalloc((void**)&gid,sizeof(int)*n_groups_per_problem*NSTREAMS));
	gpuErrchk(cudaMalloc((void**)&bounds,sizeof(T2)*n_vars_per_problem*batch_size));
	gpuErrchk(cudaMalloc((void**)&vars, sizeof(T)*batch_size*n_vars_per_problem));	
	gpuErrchk(cudaMalloc((void**)&pmin_vars, sizeof(T)*NSTREAMS*n_vars_per_problem*particles_in_each_group*n_groups_per_problem)); 
	gpuErrchk(cudaMalloc((void**)&vars_buf, sizeof(T)*n_vars_per_problem*particles_in_each_group*n_groups_per_problem*NSTREAMS));
	gpuErrchk(cudaMalloc((void**)&fmin, sizeof(T)*batch_size));

	//Filling the functions array with many test problems
	void (*functions[BATCH_SIZE])(const T*, T*, T, T, T, T, T);
	
	//Some drone trajectories. Currently we just duplicate the same drone trajectories
	for (int i=0;i<BATCH_SIZE;i++){
	functions[i]={ballistic_trajectory<T,particles_in_each_group*n_groups_per_problem,1000,100,-70,1>};
	}
	
	set_bounds<T2><<<1,1>>>(batch_size,bounds);
	set_pcg_data<<<1,1>>>(TX*BX*NSTREAMS,pcg_data);
	
	T v0=700; //Initial speed of the projectiles. Caution: The higher the speed, the smaller the solver step size delta variable has to be in order to get accurate results
	T mass=0.5;//mass of a shell
	T mu=0.001;//air friction coefficient
	T delta=0.0125; //step size of ODE solver
	T tend=5.0; //duration of ballistic simulation (tstart=0). If set too low, shells will never come near drone trajectories
	
	auto start = std::chrono::high_resolution_clock::now(); 
	auto res=kf::opt::spso_kbatch<BX,TX,NSTREAMS>(n_particles_per_group, n_groups,n_vars,batch_size,pso_iterations,pso_rep_counter, pcg_data, pmin,pmin_prev,pmin_vars,gmin_prev,gid,fmin, velocity,vars, vars_buf, bounds, functions,tend, v0,mass,mu,delta);
	auto stop = std::chrono::high_resolution_clock::now(); 
	
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
	printf("Time taken for optimization (in miliseconds):%llu\n",duration.count());
	if (res!=kf::opt::KPSOError::OK){
		printf("The optimization returned an error\n");	
	}
	else{
		T vars_h[BATCH_SIZE];
		T residuals_h[BATCH_SIZE];
		gpuErrchk(cudaMemcpy(vars_h,vars,sizeof(T)*batch_size*n_vars_per_problem,cudaMemcpyDeviceToHost));	
		gpuErrchk(cudaMemcpy(residuals_h,fmin,sizeof(T)*batch_size,cudaMemcpyDeviceToHost));
		
		for (int i=0;i<n_vars_per_problem*batch_size;i++){
			printf("angle %d (in radians):%f\tResidual: %f\n",i,vars_h[i],residuals_h[i]);
		}	
	}

	cudaFree(pso_rep_counter);
	cudaFree(pcg_data);
	cudaFree(pmin_vars);
	cudaFree(pmin_prev);
	cudaFree(gmin_prev);
	cudaFree(gid);
	cudaFree(pmin);
	cudaFree(bounds);
	cudaFree(vars);
	cudaFree(vars_buf);
	cudaFree(fmin);
	cudaFree(velocity);
}

int main(){
	constexpr int batch_size=25; //number of simultaneous problems
	find_angles<float,float2,batch_size>();
	printf("Program finished\n");
}