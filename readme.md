# kOpt #
kOpt is a library that solves your GPU based optimization problems. Currently, the library has been implemented in CUDA. 

# Who needs this?
We believe the requirement to solve optimization problems in real time will grow. kOpt has been designed to offer enough performance and flexibility to be usable in a real time setting such as video games. For example, no method will ever allocate memory through malloc by itself. All necessary allocations such as memory buffers are handled by the programmer, which leads to a very transparent allocation scheme. In addition, data is packaged to (hopefully) avoid inefficient access or shared memory bank conflicts. 

 
## Overview of implemented algorithms
Currently, one algorithm is implemented. There is more to come!
|Name|Description|
|---|---|
|spso_batch|Batch implementation of the [Standard Particle Swarm](https://towardsdatascience.com/particle-swarm-optimization-visually-explained-46289eeb2e14) algorithm|


## What is the use for this?
We have found that in many situations the loss function of an optimization problem is simple enough to be efficiently evaluated on a single CUDA thread. On todays GPU's, hundreds or even thousands of such threads can efficiently run in parallel. Many of kOpt's algorithms are therefore batch based, where you can optimize multiple functions at once with negligible performance impact.

## Can I use this code in my commercial project?
You can use this code in your hobbyist or commercial endeavors without any restrictions. The development of this project costs money however, so if you are using kopt in your commercial project please consider reaching out at general@keinefirma.xyz.

## Building
The examples contain cmake build files to help you build the code for your system.

## Get in touch!
...via [https://keinefirma.xyz](https://keinefirma.xyz) or general@keinefirma.xyz!