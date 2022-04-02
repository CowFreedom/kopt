#include <kf_random.h>
#include <stdio.h>

__global__
void print_ten_random_numbers_from_device(){
	uint64_t state;
	uint64_t inc;
	kf::rand::set_sequence(0,1,state,inc);
	printf("Numbers from device:\n");
	for (int i=0;i<10;i++){
		printf("%u\t",kf::rand::pcg_xsh_rr(state,inc));	
	}
	printf("\n\n");
}

void print_ten_random_numbers_from_host(){
	uint64_t state;
	uint64_t inc;
	kf::rand::set_sequence(0,1,state,inc);
	printf("Numbers from host:\n");
	for (int i=0;i<10;i++){
		printf("%u\t",kf::rand::pcg_xsh_rr(state,inc));	
	}
	printf("\n\n");
}

int main(){
	print_ten_random_numbers_from_device<<<1,1>>>();
	cudaDeviceSynchronize();
	print_ten_random_numbers_from_host();
	return 0;
}