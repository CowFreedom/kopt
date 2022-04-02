#include <core/kopt.h>
#include <core/krand.h>
#include <chrono>

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
void set_bounds(T* bounds){
	bounds[0].x=0;
	bounds[0].y=2;
	bounds[1].x=-20;
	bounds[1].y=-15;
	bounds[2].x=-1;
	bounds[2].y=5;
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


template<class T,int STRIDE>
__global__
void f1(const T* vars, T* fmin){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	while (idx<STRIDE){
		T x=vars[idx];
		T y=vars[STRIDE+idx];
		fmin[idx]=(x-1.3)*(x-1.3)+(y+16.2)*(y+16.2);
		idx+=gridDim.x*blockDim.x;
	}
}

template<class T,int STRIDE>
__global__
void f2(const T* vars, T* fmin){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	while (idx<STRIDE){
		T x=vars[idx];
		T val=(x-4);
		fmin[idx]=val*val;
		idx+=gridDim.x*blockDim.x;
	}
}

template<class T, class T2>
void find_zeros(){
	const int batch_size=2;
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
	int pso_iterations=700;//PSO iterations for each problem in the batch
	T2* bounds; //variable bounds for each particle
	ulonglong2* pcg_data; //initialisation data for random number engine


	constexpr int n_vars[batch_size]={2,1}; //number of function variables per problem in the batch
	constexpr int n_particles[batch_size]={32,8}; //Number of PSO particles per problem in the batch. Caution: Number of particles must be a power of two!
	constexpr int n_groups[batch_size]={2,4}; //Number of PSO groups per problem in the batch. 
	
	constexpr int NSTREAMS=2; //number of concurrent CUDA streams used
	constexpr int TX=32; //number of threads per block
	constexpr int BX=1; //number of blocks per batch problem
	
	gpuErrchk(cudaMalloc((void**)&pcg_data,sizeof(ulonglong2)*TX*BX*NSTREAMS));
	gpuErrchk(cudaMalloc((void**)&pso_rep_counter,sizeof(int)*(n_groups[0]+n_groups[1]))); 
	gpuErrchk(cudaMalloc((void**)&velocity,sizeof(T)*(n_vars[0]*n_particles[0]*n_groups[0]+n_vars[1]*n_particles[1]*n_groups[1])));
	gpuErrchk(cudaMalloc((void**)&pmin,sizeof(T)*(n_particles[0]*n_groups[0]+n_particles[1]*n_groups[1]))); 
	gpuErrchk(cudaMalloc((void**)&pmin_prev,sizeof(T)*(n_particles[0]*n_groups[0]+n_particles[1]*n_groups[1]))); 
	gpuErrchk(cudaMalloc((void**)&gmin_prev,sizeof(T)*(n_groups[0]+n_groups[1]))); 
	gpuErrchk(cudaMalloc((void**)&gid,sizeof(int)*(n_groups[0]+n_groups[1])));
	gpuErrchk(cudaMalloc((void**)&bounds,sizeof(T2)*(n_vars[0]+n_vars[1])));
	gpuErrchk(cudaMalloc((void**)&vars, sizeof(T)*(n_vars[0]+n_vars[1])));	
	gpuErrchk(cudaMalloc((void**)&pmin_vars, sizeof(T)*(n_vars[0]*n_particles[0]*n_groups[0]+n_vars[1]*n_particles[1]*n_groups[1]))); 
	gpuErrchk(cudaMalloc((void**)&vars_buf, sizeof(T)*(n_vars[0]*n_groups[0]*n_particles[0]+n_vars[1]*n_groups[1]*n_particles[1])));
	gpuErrchk(cudaMalloc((void**)&fmin, sizeof(T)*batch_size));

	//Filling the functions array with many test problems
	void (*functions[batch_size])(const T*, T*);
	
	//Some drone trajectories. Currently we just duplicate the same drone trajectories
	functions[0]={f1<T,n_particles[0]*n_groups[0]>};
	functions[1]={f2<T,n_particles[1]*n_groups[1]>};	
	
	set_bounds<T2><<<1,1>>>(bounds);
	set_pcg_data<<<1,1>>>(TX*BX*NSTREAMS,pcg_data);
	
	auto start = std::chrono::high_resolution_clock::now(); 
	auto res=kf::opt::spso_kbatch<BX,TX,NSTREAMS>(n_particles, n_groups,n_vars,batch_size,pso_iterations,pso_rep_counter, pcg_data, pmin,pmin_prev,pmin_vars,gmin_prev,gid,fmin, velocity,vars, vars_buf, bounds, functions);
	auto stop = std::chrono::high_resolution_clock::now(); 
	
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
	printf("Time taken for optimization (in miliseconds):%llu\n",duration.count());
	if (res!=kf::opt::KPSOError::OK){
		printf("The optimization returned an error\n");	
	}
	else{
		T vars_h[batch_size];
		T residuals_h[batch_size];
		gpuErrchk(cudaMemcpy(vars_h,vars,sizeof(T)*(n_vars[0]+n_vars[1]),cudaMemcpyDeviceToHost));	
		gpuErrchk(cudaMemcpy(residuals_h,fmin,sizeof(T)*batch_size,cudaMemcpyDeviceToHost));
		
		printf("f1 zeros:x=%f, y=%f \tResidual: %f\n",vars_h[0],vars_h[1],residuals_h[0]);
		printf("f2 zeros:x=%f\tResidual: %f\n",vars_h[2],residuals_h[1]);
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
	find_zeros<float,float2>();
}