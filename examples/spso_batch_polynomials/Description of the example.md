# spso_batch_drone_interception #
This example calculates the roots for the functions

$$\begin{align}f_1(x,y)&=\left(x-1.3\right)^2\left(y-16.2\right)^2 \\
f_2(x)&= \left(x-4\right)^2 \\
\end{align}$$


This example uses a GPU only particle swarm optimization algorithm to solve the roots problem. The loss function is evaluated in the kernel only.

## Why do I want to see this

You are solving $2$ independent optimization problems. Play around with the batch sizes, CUDA block grid size and independent CUDA streams, to see how this affects performance and memory utilization. Because all the problems in the batch are instances of each other, we could also use another version of kopt::spso that uses less memory.

## Notes about the code
The following lines

	constexpr int NSTREAMS=2; //number of concurrent CUDA streams used
	constexpr int TX=32; //number of threads per block
	constexpr int BX=2; //number of blocks per batch problem
do affect the performance at appropriate batch sizes and swarm configurations.
The problems from the batch are mapped to CUDA streams. Two independent streams for a batch size of two is a reasonable setting under many circumstances.

The grid layout parameters are mapped to the number of particles and groups for each problem in the batch. Setting them much higher than the numbers of particles per batch problem might not result in better performance.


## Building
You can manually build the files or use CMake to generate build files using the CMakeLists.txt.