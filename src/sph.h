#ifndef __SPH__
#define __SPH__

#define PARTICLE_COUNT ( 1024*16/*16*/ )
#define NEIGHBOR_COUNT 32


#define PCISPH 1 // change to 0 for ordinary SPH
//int PCISPH = 1;

#ifndef M_PI
#define M_PI 3.1415927f
#endif

#define XMIN 0
#define XMAX 80.16//120.24//120//200//50//100
#define YMIN 0
#define YMAX 80.16//80//330//40//80
#define ZMIN 0
#define ZMAX 45.09//180//10//30

//================================== for internal use ============

#define NO_PARTICLE_ID -1
#define NO_CELL_ID -1
#define NO_DISTANCE -1.0f


#endif

/*
OpenCL Hello World 

#define __CL_ENABLE_EXCEPTIONS
#define __NO_STD_VECTOR
#define __NO_STD_STRING

#include <CL/cl.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>

const char * helloStr = "__kernel void hello(void) { }\n";

int main(void) {
   try {
      cl::Context context(CL_DEVICE_TYPE_GPU, 0, NULL, NULL, &err);
      cl::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
      cl::CommandQueue queue(context, devices[0], 0, &err);
      cl::Program::Sources source(1, std::make_pair(helloStr,strlen(helloStr)));
      cl::Program program_ = cl::Program(context, source);
      program_.build(devices);
      cl::Kernel kernel(program_, "hello", &err);
      cl::KernelFunctor func = kernel.bind(queue, cl::NDRange(4, 4), cl::NDRange(2, 2));
      func().wait();
   } catch (cl::Error err) {
      std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
   }
   return EXIT_SUCCESS;
}

*/