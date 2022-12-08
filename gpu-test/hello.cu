#include <stdio.h>
// A simple 'hello-world' style CUDA program.

// The __global__ indicates that this is a GPU function (aka 'kernel') and can be called from either CPU or GPU.
// (You'd have the __device__ keyword for kernels that only other kernels can call.)
// This one doesnt take any params, but you could have it take arguments like other CPU functions. Do note however that those
// need to be references to objects residing in GPU memory, not CPU memory. Put them there before you call the kernel.
// Also, the __global__ kernels dont return anything. (How would you decide one return value to the CPU code
// among 1000s of GPU threads?) You can write back the outputs to global memory, thats usually how they output/return data.
__global__ void hello()
{
// Yes, CUDA kernels can execute printfs like CPU code.
// ThreadIdx.x is the x dimension in the thread array, similarly block in the grid ary
    printf("Hello from Thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

int main()
{
    hello<<<1,1>>>();
    // So, the above kernel call will return after the kernel is launched, but before it might be complete.
    cudaDeviceSynchronize(); // The above call is asynchronous, wait until it
                             // finishes before exiting the program!

    return 0;
}