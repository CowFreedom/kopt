# spso_batch_drone_interception #
This example calculates the angles $\alpha_i$ needed for  
each projectile $i$ in order to hit flying target $i$.

This entails numerically solves the ballistic equation
$$\begin{align}\frac{mdx}{dt}&=-\mu \frac{dx}{dt}\cdot \sqrt{ \left(\frac{dx}{dt}\right)^2 +\left(\frac{dy}{dt}\right)^2 } \\
\frac{mdy}{dt}&=-\mu \frac{dy}{dt}\cdot \sqrt{ \left(\frac{dx}{dt}\right)^2 +\left(\frac{dy}{dt}\right)^2 } -gm \\
\end{align}$$

. We assume that the projectiles are shot at $t_0=0$ from the origin $x_0=y_0=0$. Given a launch speed of $\|v_0 \| \in \mathbb{R}^+$, we retrieve the initial launch velocity vector by $v_0=\|v_0 \|\begin{pmatrix}\text{cos}\left(\alpha\right)\\\text{sin} \left(\alpha\right)\end{pmatrix}$.

Each drone has a user customizable linear trajectory $D_i$. In order to find the necessary $\alpha_i$, we solve for the minimal distance between projectile $i$ trajectory and the drone $i$ trajectory, subject to the launch angle. If we have $n$ drones, we have therefore $n$ independent optimization problems.
This example uses a GPU only particle swarm optimization algorithm to solve the optimization problem. The loss function is evaluated in the kernel only.

## Why do I want to see this

You are solving $n$ independent optimization problems. Play around with the batch sizes, CUDA block grid size and independent CUDA streams, to see how this affects performance and memory utilization. Because all the problems in the batch are instances of each other, we could also use another version of kopt::spso that uses less memory.

## Notes about the code
Currently, you'll find the following line in the code:

	for (int i=0;i<BATCH_SIZE;i++){
    	functions[i]={ballistic_trajectory<T,particles_in_each_group*n_groups_per_problem,1000,100,-70,1>};
	}
This means that all drones start at the same position $\begin{pmatrix}1000\\ 100\end{pmatrix}$ and move with the same velocity $\begin{pmatrix}-70\\ 1\end{pmatrix}$. Ergo, each resulting angle $\alpha_i$ should be the same. You can try to change these velocities.

Furthermore, the following lines

	constexpr int NSTREAMS=2; //number of concurrent CUDA streams used
	constexpr int TX=32; //number of threads per block
	constexpr int BX=2; //number of blocks per batch problem
do affect the performance at appropriate batch sizes and swarm configurations.
The problems from the batch are mapped to CUDA streams. Setting the NSTREAMS value higher might result in better performance.

The grid layout parameters are mapped to the number of particles and groups for each problem in the batch. Setting them much higher than the numbers of particles per batch problem might not result in better performance.

## What could be interesting

You may note, that if your launch speed is too low, your projectiles might never hit their given target. Instead of angle $\alpha$ only, in addition you could also try to find an optimal $\|v_0 \|$ to ensure that there will always be a way to hit the target.

## Building
You can manually build the files or use CMake to generate build files using the CMakeLists.txt.