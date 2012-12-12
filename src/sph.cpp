
#pragma comment( lib, "opencl.lib" )							// opencl.lib

//#define QUEUE_EACH_KERNEL//debugging feature

#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <math.h>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32) || defined (_WIN64)
	#include <windows.h>
#elif defined(__linux__)
	#include <time.h>
#endif
#include "sph.h"

using std::max;
using std::min;
//AP2012//#include "dx10_render.h"
/*
#ifdef NDEBUG
#define ENABLE_OPENCL_RADIXSORT
#include "radixsort.hpp"
#endif
*/

//see opencl.hpp to find this:
/*! \class BufferGL
 * \brief Memory buffer interface for GL interop.
 */

//AP2012//#define USE_DX_INTEROP
#if defined(__APPLE__) || defined(__MACOSX)
	#include <OpenCL/cl.hpp>
	#include <OpenCL/cl_d3d10.h>
#else
	#include <CL/cl.hpp>
//AP2012//#include <CL/cl_d3d10.h>
#endif

int ELASTIC_CONNECTIONS_COUNT;// calculated during initialization; depends on particle content and configuration of the scene

const float xmin = XMIN;
const float xmax = XMAX;
const float ymin = YMIN;
const float ymax = YMAX;
const float zmin = ZMIN;
const float zmax = ZMAX;

const float rho0 = 1000.0f;
const float stiffness = 0.75f;
const float h = 3.34f;
const float hashGridCellSize = 2.0f * h;
const float hashGridCellSizeInv = 1.0f / hashGridCellSize;
const float mass = 0.0003f;//0.0003
const float simulationScale = 0.004f;
const float simulationScaleInv = 1.0f / simulationScale;
const float mu = 10.0f;//why this value? Dynamic viscosity of water at 25 C = 0.89e-3 Pa*s
const float timeStep = 0.002f;//0.0005f;//0.0042f;
const float CFLLimit = 100.0f;
const int NK = NEIGHBOR_COUNT * PARTICLE_COUNT;
const float damping = 0.75f;

const float r0 = 0.5f * h; // distance between two boundary particle

const float beta = timeStep*timeStep*mass*mass*2/(rho0*rho0);// B. Solenthaler's dissertation, formula 3.6 (end of page 30)
const float betaInv = 1.f/beta;

const float Wpoly6Coefficient = 315.0f / ( 64.0f * M_PI * pow( h * simulationScale, 9.0f ) );
const float gradWspikyCoefficient= -45.0f / ( M_PI * pow( h * simulationScale, 6.0f ) );
const float del2WviscosityCoefficient = -gradWspikyCoefficient;

const float gravity_x = 0.0f;
const float gravity_y = -9.8f;
const float gravity_z = 0.0f;

float calcDelta()
{
    float x[] = { 1, 1, 0,-1,-1,-1, 0, 1, 1, 1, 0,-1,-1,-1, 0, 1, 1, 1, 0,-1,-1,-1, 0, 1, 2,-2, 0, 0, 0, 0, 0, 0 };
    float y[] = { 0, 1, 1, 1, 0,-1,-1,-1, 0, 1, 1, 1, 0,-1,-1,-1, 0, 1, 1, 1, 0,-1,-1,-1, 0, 0, 2,-2, 0, 0, 0, 0 };
    float z[] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 2,-2, 1,-1 };
    //float beta = dt * dt * m_particlemass * m_particlemass * 2 / (m_restDensity * m_restDensity);
    //Vector3 sum1 = Vector3.Zero;
	float sum1_x = 0.f;
	float sum1_y = 0.f;
	float sum1_z = 0.f;
    float sum1 = 0.f, sum2 = 0.f;
	float v_x = 0.f;
	float v_y = 0.f;
	float v_z = 0.f;
	float dist;
	float particleRadius = pow(mass/rho0,1.f/3.f);  // the value is about 0.01 instead of 
	float h_r_2;									// my previous estimate = simulationScale*h/2 = 0.0066

    for (int i = 0; i < 32; i++)
    {
		v_x = x[i] * 0.8f * particleRadius;
		v_y = y[i] * 0.8f * particleRadius;
		v_z = z[i] * 0.8f * particleRadius;

        dist = sqrt(v_x*v_x+v_y*v_y+v_z*v_z);//scaled, right?

        if (dist <= h)
        {
			h_r_2 = pow((h*simulationScale - dist),2);//scaled

            sum1_x += h_r_2 * v_x / dist;
			sum1_y += h_r_2 * v_y / dist;
			sum1_z += h_r_2 * v_z / dist;

            sum2 += h_r_2 * h_r_2;
        }
    }

	sum1 = sum1_x*sum1_x + sum1_y*sum1_y + sum1_z*sum1_z;

	//float result = 1.0f / (beta * (sum1_x*sum1_x + sum1_y*sum1_y + sum1_z*sum1_z + sum2));
    //float result = 1.0f / (beta * gradWspikyCoefficient * gradWspikyCoefficient * (sum1 + sum2));
	
	return  1.0f / (beta * gradWspikyCoefficient * gradWspikyCoefficient * (sum1 + sum2));
	 //delta = 0.82
}

const float delta = calcDelta();


//int particleCount = PARTICLE_COUNT;//AP2012

int gridCellsX;
int gridCellsY;
int gridCellsZ;
int gridCellCount;


float * positionBuffer;
float * velocityBuffer;
float * densityBuffer;
float * elasticConnectionsDataBuffer;

//Delete This After Fixing
float * neighborMapBuffer;
unsigned int * particleIndexBuffer;
unsigned int * gridNextNonEmptyCellBuffer;

cl::Context context;
std::vector< cl::Device > devices;
cl::CommandQueue queue;
cl::Program program;

// Buffers
cl::Buffer acceleration;// size * 2 // forceAcceleration and pressureForceAcceleration
cl::Buffer gridCellIndex;
cl::Buffer gridCellIndexFixedUp;
cl::Buffer neighborMap;
cl::Buffer particleIndex;// list of pairs [CellIndex, particleIndex]
cl::Buffer particleIndexBack;// list of indexes of particles before sort 
cl::Buffer position;
cl::Buffer pressure;
cl::Buffer rho;// size * 2
cl::Buffer rhoInv;// for basic SPH only
cl::Buffer sortedPosition;// size * 2
cl::Buffer sortedVelocity;
cl::Buffer velocity;
cl::Buffer elasticConnectionsData;//list of particle pairs connected with springs and rest distance between them

// Kernels
cl::Kernel clearBuffers;
cl::Kernel computeAcceleration;
cl::Kernel computeDensityPressure;
cl::Kernel findNeighbors;
cl::Kernel hashParticles;
cl::Kernel indexx;
cl::Kernel integrate;
cl::Kernel sortPostPass;
// additional kernels for PCISPH
cl::Kernel preElasticMatterPass;
cl::Kernel pcisph_computeDensity;
cl::Kernel pcisph_computeForcesAndInitPressure;
cl::Kernel pcisph_integrate;
cl::Kernel pcisph_predictPositions;
cl::Kernel pcisph_predictDensity;
cl::Kernel pcisph_correctPressure;
cl::Kernel pcisph_computePressureForceAcceleration;
cl::Kernel pcisph_computeElasticForces;

unsigned int
_runClearBuffers( cl::CommandQueue queue ){
	// Stage ClearBuffers
	clearBuffers.setArg( 0, neighborMap );
	queue.enqueueNDRangeKernel(
		clearBuffers, cl::NullRange, cl::NDRange( (int) ( PARTICLE_COUNT ) ),
#if defined( __APPLE__ )
		cl::NullRange, NULL, NULL );
#else
		cl::NDRange( (int)( 256 ) ), NULL, NULL );
#endif
#ifdef QUEUE_EACH_KERNEL
	queue.finish();
#endif
	return 0;
}



unsigned int
_runComputeAcceleration( cl::CommandQueue queue ){
	// Stage ComputeAcceleration
	computeAcceleration.setArg( 0, neighborMap );
	computeAcceleration.setArg( 1, pressure );
	computeAcceleration.setArg( 2, rho );
	computeAcceleration.setArg( 3, rhoInv );
	computeAcceleration.setArg( 4, sortedPosition );
	computeAcceleration.setArg( 5, sortedVelocity );
	computeAcceleration.setArg( 6, particleIndexBack );
	computeAcceleration.setArg( 7, CFLLimit );
	computeAcceleration.setArg( 8, del2WviscosityCoefficient );
	computeAcceleration.setArg( 9, gradWspikyCoefficient );
	computeAcceleration.setArg( 10, h );
	computeAcceleration.setArg( 11, mass );
	computeAcceleration.setArg( 12, mu );
	computeAcceleration.setArg( 13, simulationScale );
	computeAcceleration.setArg( 14, acceleration );
	computeAcceleration.setArg( 15, rho0 );
	queue.enqueueNDRangeKernel(
		computeAcceleration, cl::NullRange, cl::NDRange( (int) ( PARTICLE_COUNT ) ),
#if defined( __APPLE__ )
		cl::NullRange, NULL, NULL );
#else
		cl::NDRange( (int)( 256 ) ), NULL, NULL );
#endif
#ifdef QUEUE_EACH_KERNEL
	queue.finish();
#endif
	return 0;
}


unsigned int
_run_pcisph_computePressureForceAcceleration( cl::CommandQueue queue ){
	// Stage ComputeAcceleration
	pcisph_computePressureForceAcceleration.setArg( 0, neighborMap );
	pcisph_computePressureForceAcceleration.setArg( 1, pressure );
	pcisph_computePressureForceAcceleration.setArg( 2, rho );
	pcisph_computePressureForceAcceleration.setArg( 3, rhoInv );
	pcisph_computePressureForceAcceleration.setArg( 4, sortedPosition );
	pcisph_computePressureForceAcceleration.setArg( 5, sortedVelocity );
	pcisph_computePressureForceAcceleration.setArg( 6, particleIndexBack );
	pcisph_computePressureForceAcceleration.setArg( 7, CFLLimit );
	pcisph_computePressureForceAcceleration.setArg( 8, del2WviscosityCoefficient );
	pcisph_computePressureForceAcceleration.setArg( 9, gradWspikyCoefficient );
	pcisph_computePressureForceAcceleration.setArg( 10, h );
	pcisph_computePressureForceAcceleration.setArg( 11, mass );
	pcisph_computePressureForceAcceleration.setArg( 12, mu );
	pcisph_computePressureForceAcceleration.setArg( 13, simulationScale );
	pcisph_computePressureForceAcceleration.setArg( 14, acceleration );
	pcisph_computePressureForceAcceleration.setArg( 15, rho0 );
	pcisph_computePressureForceAcceleration.setArg( 16, position );
	pcisph_computePressureForceAcceleration.setArg( 17, particleIndex );
	queue.enqueueNDRangeKernel(
		pcisph_computePressureForceAcceleration, cl::NullRange, cl::NDRange( (int) ( PARTICLE_COUNT ) ),
#if defined( __APPLE__ )
		cl::NullRange, NULL, NULL );
#else
		cl::NDRange( (int)( 256 ) ), NULL, NULL );
#endif
#ifdef QUEUE_EACH_KERNEL
	queue.finish();
#endif
	return 0;
}

unsigned int
_runComputeDensityPressure( cl::CommandQueue queue){
	// Stage ComputeDensityPressure
	computeDensityPressure.setArg( 0, neighborMap );
	computeDensityPressure.setArg( 1, Wpoly6Coefficient );
	computeDensityPressure.setArg( 2, gradWspikyCoefficient );
	computeDensityPressure.setArg( 3, h );
	computeDensityPressure.setArg( 4, mass );
	computeDensityPressure.setArg( 5, rho0 );
	computeDensityPressure.setArg( 6, simulationScale );
	computeDensityPressure.setArg( 7, stiffness );
	computeDensityPressure.setArg( 8, sortedPosition );
	computeDensityPressure.setArg( 9, pressure );
	computeDensityPressure.setArg(10, rho );
	computeDensityPressure.setArg(11, rhoInv );
	computeDensityPressure.setArg(12, particleIndexBack );
	computeDensityPressure.setArg(13, delta );
	queue.enqueueNDRangeKernel(
		computeDensityPressure, cl::NullRange, cl::NDRange( (int) ( PARTICLE_COUNT ) ),
#if defined( __APPLE__ )
		cl::NullRange, NULL, NULL );
#else
		cl::NDRange( (int)( 256 ) ), NULL, NULL );
#endif
#ifdef QUEUE_EACH_KERNEL
	queue.finish();
#endif
	return 0;
}


unsigned int
_run_pcisph_computeDensity( cl::CommandQueue queue){
	// Stage ComputeDensityPressure
	pcisph_computeDensity.setArg( 0, neighborMap );
	pcisph_computeDensity.setArg( 1, Wpoly6Coefficient );
	pcisph_computeDensity.setArg( 2, gradWspikyCoefficient );
	pcisph_computeDensity.setArg( 3, h );
	pcisph_computeDensity.setArg( 4, mass );
	pcisph_computeDensity.setArg( 5, rho0 );
	pcisph_computeDensity.setArg( 6, simulationScale );
	pcisph_computeDensity.setArg( 7, stiffness );
	pcisph_computeDensity.setArg( 8, sortedPosition );
	pcisph_computeDensity.setArg( 9, pressure );
	pcisph_computeDensity.setArg(10, rho );
	pcisph_computeDensity.setArg(11, rhoInv );
	pcisph_computeDensity.setArg(12, particleIndexBack );
	pcisph_computeDensity.setArg(13, delta );
	queue.enqueueNDRangeKernel(
		pcisph_computeDensity, cl::NullRange, cl::NDRange( (int) ( PARTICLE_COUNT ) ),
#if defined( __APPLE__ )
		cl::NullRange, NULL, NULL );
#else
		cl::NDRange( (int)( 256 ) ), NULL, NULL );
#endif
#ifdef QUEUE_EACH_KERNEL
	queue.finish();
#endif
	return 0;
}

unsigned int
_runFindNeighbors( cl::CommandQueue queue/*, int nIter*/){
	// Stage FindNeighbors
	findNeighbors.setArg( 0, gridCellIndexFixedUp );
	findNeighbors.setArg( 1, sortedPosition );
	gridCellCount = ((gridCellsX) * (gridCellsY)) * (gridCellsZ);
	findNeighbors.setArg( 2, gridCellCount );
	findNeighbors.setArg( 3, gridCellsX );
	findNeighbors.setArg( 4, gridCellsY );
	findNeighbors.setArg( 5, gridCellsZ );
	findNeighbors.setArg( 6, h );
	findNeighbors.setArg( 7, hashGridCellSize );
	findNeighbors.setArg( 8, hashGridCellSizeInv );
	findNeighbors.setArg( 9, simulationScale );
	findNeighbors.setArg( 10, xmin );
	findNeighbors.setArg( 11, ymin );
	findNeighbors.setArg( 12, zmin );
	findNeighbors.setArg( 13, neighborMap );
	//findNeighbors.setArg( 14, nIter );
	queue.enqueueNDRangeKernel(
		findNeighbors, cl::NullRange, cl::NDRange( (int) ( PARTICLE_COUNT ) ),
#if defined( __APPLE__ )
		cl::NullRange, NULL, NULL );
#else
		cl::NDRange( (int)( 256 ) ), NULL, NULL );
#endif
#ifdef QUEUE_EACH_KERNEL
	queue.finish();
#endif
	return 0;
}
unsigned int
_runHashParticles( cl::CommandQueue queue ){
	// Stage HashParticles
	hashParticles.setArg( 0, position );
	hashParticles.setArg( 1, gridCellsX );
	hashParticles.setArg( 2, gridCellsY );
	hashParticles.setArg( 3, gridCellsZ );
	hashParticles.setArg( 4, hashGridCellSizeInv );
	hashParticles.setArg( 5, xmin );
	hashParticles.setArg( 6, ymin );
	hashParticles.setArg( 7, zmin );
	hashParticles.setArg( 8, particleIndex );
	queue.enqueueNDRangeKernel(
		hashParticles, cl::NullRange, cl::NDRange( (int) ( PARTICLE_COUNT ) ),
#if defined( __APPLE__ )
		cl::NullRange, NULL, NULL );
#else
		cl::NDRange( (int)( 256 ) ), NULL, NULL );
#endif
#ifdef QUEUE_EACH_KERNEL
	queue.finish();
#endif
	return 0;
}

unsigned int
_runIndexPostPass( ){
	// Stage IndexPostPass

	//28aug_Palyanov_start_block
	int err;
	err = queue.enqueueReadBuffer( gridCellIndex, CL_TRUE, 0, (gridCellCount+1) * sizeof( unsigned int ) * 1, gridNextNonEmptyCellBuffer );
	if( err != CL_SUCCESS ){ throw std::runtime_error( "could not enqueue gridCellIndex read" ); } 
	queue.finish();

	int recentNonEmptyCell = gridCellCount;
	for(int i=gridCellCount;i>=0;i--)
	{
		if(gridNextNonEmptyCellBuffer[i]==NO_CELL_ID)
			gridNextNonEmptyCellBuffer[i] = recentNonEmptyCell; 
		else recentNonEmptyCell = gridNextNonEmptyCellBuffer[i];
	}

	err = queue.enqueueWriteBuffer( gridCellIndexFixedUp, CL_TRUE, 0, (gridCellCount+1) * sizeof( unsigned int ) * 1, gridNextNonEmptyCellBuffer );
	//err = queue.enqueueWriteBuffer( gridNextNonEmptyCell, CL_TRUE, 0, (gridCellCount+1) * sizeof( unsigned int ) * 1, gridNextNonEmptyCellBuffer );
	if( err != CL_SUCCESS ){ throw std::runtime_error( "could not enqueue ??? write" ); }
	queue.finish();

	//_runIndexPostPass( queue ); queue.finish();// no need for this slow shit anymoar =)
	// for details look at what happens with gridNextNonEmptyCellBuffer
	//28aug_Palyanov_end_block

	return 0;
}


unsigned int
_runIndexx( cl::CommandQueue queue ){
	// Stage Indexx
	indexx.setArg( 0, particleIndex );
	gridCellCount = ((gridCellsX) * (gridCellsY)) * (gridCellsZ);
	indexx.setArg( 1, gridCellCount );
	indexx.setArg( 2, gridCellIndex );
	int gridCellCountRoundedUp = ((( gridCellCount - 1 ) / 256 ) + 1 ) * 256;
	queue.enqueueNDRangeKernel(
		indexx, cl::NullRange, cl::NDRange( (int) ( gridCellCountRoundedUp ) ),
#if defined( __APPLE__ )
		cl::NullRange, NULL, NULL );
#else
		cl::NDRange( (int)( 256 ) ), NULL, NULL );
#endif
#ifdef QUEUE_EACH_KERNEL
	queue.finish();
#endif
	return 0;
}

unsigned int
_runIntegrate( cl::CommandQueue queue){
	// Stage Integrate
	integrate.setArg( 0, acceleration );
	integrate.setArg( 1, sortedPosition );
	integrate.setArg( 2, sortedVelocity );
	integrate.setArg( 3, particleIndex );
	integrate.setArg( 4, particleIndexBack );
	integrate.setArg( 5, gravity_x );
	integrate.setArg( 6, gravity_y );
	integrate.setArg( 7, gravity_z );
	integrate.setArg( 8, simulationScaleInv );
	integrate.setArg( 9, timeStep );
	integrate.setArg( 10, xmin );
	integrate.setArg( 11, xmax );
	integrate.setArg( 12, ymin );
	integrate.setArg( 13, ymax );
	integrate.setArg( 14, zmin );
	integrate.setArg( 15, zmax );
	integrate.setArg( 16, damping );
	integrate.setArg( 17, position );
	integrate.setArg( 18, velocity );
	queue.enqueueNDRangeKernel(
		integrate, cl::NullRange, cl::NDRange( (int) ( PARTICLE_COUNT ) ),
#if defined( __APPLE__ )
		cl::NullRange, NULL, NULL );
#else
		cl::NDRange( (int)( 256 ) ), NULL, NULL );
#endif
#ifdef QUEUE_EACH_KERNEL
	queue.finish();
#endif
	return 0;
}

unsigned int
_run_pcisph_integrate( cl::CommandQueue queue){
	// Stage Integrate
	pcisph_integrate.setArg( 0, acceleration );
	pcisph_integrate.setArg( 1, sortedPosition );
	pcisph_integrate.setArg( 2, sortedVelocity );
	pcisph_integrate.setArg( 3, particleIndex );
	pcisph_integrate.setArg( 4, particleIndexBack );
	pcisph_integrate.setArg( 5, gravity_x );
	pcisph_integrate.setArg( 6, gravity_y );
	pcisph_integrate.setArg( 7, gravity_z );
	pcisph_integrate.setArg( 8, simulationScaleInv );
	pcisph_integrate.setArg( 9, timeStep );
	pcisph_integrate.setArg( 10, xmin );
	pcisph_integrate.setArg( 11, xmax );
	pcisph_integrate.setArg( 12, ymin );
	pcisph_integrate.setArg( 13, ymax );
	pcisph_integrate.setArg( 14, zmin );
	pcisph_integrate.setArg( 15, zmax );
	pcisph_integrate.setArg( 16, damping );
	pcisph_integrate.setArg( 17, position );
	pcisph_integrate.setArg( 18, velocity );
	pcisph_integrate.setArg( 19, rho );
	pcisph_integrate.setArg( 20, r0 );
	pcisph_integrate.setArg( 21, neighborMap );
	queue.enqueueNDRangeKernel(
		pcisph_integrate, cl::NullRange, cl::NDRange( (int) ( PARTICLE_COUNT ) ),
#if defined( __APPLE__ )
		cl::NullRange, NULL, NULL );
#else
		cl::NDRange( (int)( 256 ) ), NULL, NULL );
#endif
#ifdef QUEUE_EACH_KERNEL
	queue.finish();
#endif
	return 0;
}

unsigned int
_run_pcisph_predictPositions( cl::CommandQueue queue){
	//
	pcisph_predictPositions.setArg( 0, acceleration );
	pcisph_predictPositions.setArg( 1, sortedPosition );
	pcisph_predictPositions.setArg( 2, sortedVelocity );
	pcisph_predictPositions.setArg( 3, particleIndex );
	pcisph_predictPositions.setArg( 4, particleIndexBack );
	pcisph_predictPositions.setArg( 5, gravity_x );
	pcisph_predictPositions.setArg( 6, gravity_y );
	pcisph_predictPositions.setArg( 7, gravity_z );
	pcisph_predictPositions.setArg( 8, simulationScaleInv );
	pcisph_predictPositions.setArg( 9, timeStep );
	pcisph_predictPositions.setArg( 10, xmin );
	pcisph_predictPositions.setArg( 11, xmax );
	pcisph_predictPositions.setArg( 12, ymin );
	pcisph_predictPositions.setArg( 13, ymax );
	pcisph_predictPositions.setArg( 14, zmin );
	pcisph_predictPositions.setArg( 15, zmax );
	pcisph_predictPositions.setArg( 16, damping );
	pcisph_predictPositions.setArg( 17, position );
	pcisph_predictPositions.setArg( 18, velocity );
	pcisph_predictPositions.setArg( 19, r0 );
	pcisph_predictPositions.setArg( 20, neighborMap );
	queue.enqueueNDRangeKernel(
		pcisph_predictPositions, cl::NullRange, cl::NDRange( (int) ( PARTICLE_COUNT ) ),
#if defined( __APPLE__ )
		cl::NullRange, NULL, NULL );
#else
		cl::NDRange( (int)( 256 ) ), NULL, NULL );
#endif
#ifdef QUEUE_EACH_KERNEL
	queue.finish();
#endif
	return 0;
}


//#ifndef NDEBUG
int myCompare( const void * v1, const void * v2 ){
	int * f1 = (int *)v1;
	int * f2 = (int *)v2;
	if( f1[ 0 ] < f2[ 0 ] ) return -1;
	if( f1[ 0 ] > f2[ 0 ] ) return +1;
	return 0;
}
//#endif

int * _particleIndex = new int[ PARTICLE_COUNT * 2 ];

unsigned int
_runSort( cl::CommandQueue queue ){
/*#ifdef NDEBUG
	radixSort.sort( particleIndex );
	radixSort.wait();
#else*/
	
	queue.enqueueReadBuffer( particleIndex, CL_TRUE, 0, PARTICLE_COUNT * 2 * sizeof( int ), _particleIndex );
	queue.finish();
	qsort( _particleIndex, PARTICLE_COUNT, 2 * sizeof( int ), myCompare );
	queue.enqueueWriteBuffer( particleIndex, CL_TRUE, 0, PARTICLE_COUNT * 2 * sizeof( int ), _particleIndex );
	queue.finish();	
	//delete [] _particleIndex;
//#endif
	return 0;
}
unsigned int
_runSortPostPass( cl::CommandQueue queue ){
	// Stage SortPostPass
	sortPostPass.setArg( 0, particleIndex );
	sortPostPass.setArg( 1, particleIndexBack );
	sortPostPass.setArg( 2, position );
	sortPostPass.setArg( 3, velocity );
	sortPostPass.setArg( 4, sortedPosition );
	sortPostPass.setArg( 5, sortedVelocity );
	queue.enqueueNDRangeKernel(
		sortPostPass, cl::NullRange, cl::NDRange( (int) ( PARTICLE_COUNT ) ),
#if defined( __APPLE__ )
		cl::NullRange, NULL, NULL );
#else
		cl::NDRange( (int)( 256 ) ), NULL, NULL );
#endif
#ifdef QUEUE_EACH_KERNEL
	queue.finish();
#endif
	return 0;
}


unsigned int
_runPreElasticMatterPass( cl::CommandQueue queue ){
	// Stage SortPostPass
	preElasticMatterPass.setArg( 0, particleIndex );
	preElasticMatterPass.setArg( 1, particleIndexBack );
	preElasticMatterPass.setArg( 2, position );
	preElasticMatterPass.setArg( 3, velocity );
	preElasticMatterPass.setArg( 4, sortedPosition );
	preElasticMatterPass.setArg( 5, sortedVelocity );
	queue.enqueueNDRangeKernel(
		preElasticMatterPass, cl::NullRange, cl::NDRange( (int) ( PARTICLE_COUNT ) ),
#if defined( __APPLE__ )
		cl::NullRange, NULL, NULL );
#else
		cl::NDRange( (int)( 256 ) ), NULL, NULL );
#endif
#ifdef QUEUE_EACH_KERNEL
	queue.finish();
#endif
	return 0;
}


// additional functions used for PCISPH
unsigned int
_run_pcisph_computeForcesAndInitPressure( cl::CommandQueue queue ){

	pcisph_computeForcesAndInitPressure.setArg( 0, neighborMap );
	pcisph_computeForcesAndInitPressure.setArg( 1, rho );
	pcisph_computeForcesAndInitPressure.setArg( 2, pressure );
	pcisph_computeForcesAndInitPressure.setArg( 3, sortedPosition );
	pcisph_computeForcesAndInitPressure.setArg( 4, sortedVelocity );
	pcisph_computeForcesAndInitPressure.setArg( 5, acceleration );
	pcisph_computeForcesAndInitPressure.setArg( 6, particleIndexBack );
	pcisph_computeForcesAndInitPressure.setArg( 7, Wpoly6Coefficient );
	pcisph_computeForcesAndInitPressure.setArg( 8, del2WviscosityCoefficient );
	pcisph_computeForcesAndInitPressure.setArg( 9, h );
	pcisph_computeForcesAndInitPressure.setArg(10, mass );
	pcisph_computeForcesAndInitPressure.setArg(11, mu );
	pcisph_computeForcesAndInitPressure.setArg(12, simulationScale );
	pcisph_computeForcesAndInitPressure.setArg(13, gravity_x );
	pcisph_computeForcesAndInitPressure.setArg(14, gravity_y );
	pcisph_computeForcesAndInitPressure.setArg(15, gravity_z );
	//pcisph_computeForcesAndInitPressure.setArg(16, ELASTIC_CONNECTIONS_COUNT);
	//pcisph_computeForcesAndInitPressure.setArg(17, elasticConnectionsData );

	queue.enqueueNDRangeKernel(
		pcisph_computeForcesAndInitPressure, cl::NullRange, cl::NDRange( (int) ( PARTICLE_COUNT ) ),
#if defined( __APPLE__ )
		cl::NullRange, NULL, NULL );
#else
		cl::NDRange( (int)( 256 ) ), NULL, NULL );
#endif
#ifdef QUEUE_EACH_KERNEL
	queue.finish();
#endif
	return 0;
}

unsigned int
_run_pcisph_computeElasticForces( cl::CommandQueue queue ){

	pcisph_computeElasticForces.setArg( 0, neighborMap );
	pcisph_computeElasticForces.setArg( 1, sortedPosition );
	pcisph_computeElasticForces.setArg( 2, sortedVelocity );
	pcisph_computeElasticForces.setArg( 3, acceleration );
	pcisph_computeElasticForces.setArg( 4, particleIndexBack );
	pcisph_computeElasticForces.setArg( 5, h );
	pcisph_computeElasticForces.setArg( 6, mass );
	pcisph_computeElasticForces.setArg( 7, simulationScale );
	pcisph_computeElasticForces.setArg( 8, ELASTIC_CONNECTIONS_COUNT);
	pcisph_computeElasticForces.setArg( 9, elasticConnectionsData );

	int elasticConnectionsCountRoundedUp = ((( ELASTIC_CONNECTIONS_COUNT - 1 ) / 256 ) + 1 ) * 256;

	queue.enqueueNDRangeKernel(
		pcisph_computeElasticForces, cl::NullRange, cl::NDRange( (int) ( elasticConnectionsCountRoundedUp ) ),
#if defined( __APPLE__ )
		cl::NullRange, NULL, NULL );
#else
		cl::NDRange( (int)( 256 ) ), NULL, NULL );
#endif
#ifdef QUEUE_EACH_KERNEL
	queue.finish();
#endif
	return 0;
}


unsigned int
_run_pcisph_predictDensity( cl::CommandQueue queue ){
	// Stage ComputeDensityPressure
	pcisph_predictDensity.setArg( 0, neighborMap );
	pcisph_predictDensity.setArg( 1, particleIndexBack );
	pcisph_predictDensity.setArg( 2, Wpoly6Coefficient );
	pcisph_predictDensity.setArg( 3, gradWspikyCoefficient );
	pcisph_predictDensity.setArg( 4, h );
	pcisph_predictDensity.setArg( 5, mass );
	pcisph_predictDensity.setArg( 6, rho0 );
	pcisph_predictDensity.setArg( 7, simulationScale );
	pcisph_predictDensity.setArg( 8, stiffness );
	pcisph_predictDensity.setArg( 9, sortedPosition );
	pcisph_predictDensity.setArg(10, pressure );
	pcisph_predictDensity.setArg(11, rho );
	pcisph_predictDensity.setArg(12, rhoInv );
	pcisph_predictDensity.setArg(13, delta );
	queue.enqueueNDRangeKernel(
		pcisph_predictDensity, cl::NullRange, cl::NDRange( (int) ( PARTICLE_COUNT ) ),
#if defined( __APPLE__ )
		cl::NullRange, NULL, NULL );
#else
		cl::NDRange( (int)( 256 ) ), NULL, NULL );
#endif
#ifdef QUEUE_EACH_KERNEL
	queue.finish();
#endif
	return 0;
}

unsigned int
_run_pcisph_correctPressure( cl::CommandQueue queue ){
	// Stage ComputeDensityPressure
	pcisph_correctPressure.setArg( 0, neighborMap );
	pcisph_correctPressure.setArg( 1, particleIndexBack );
	pcisph_correctPressure.setArg( 2, Wpoly6Coefficient );
	pcisph_correctPressure.setArg( 3, gradWspikyCoefficient );
	pcisph_correctPressure.setArg( 4, h );
	pcisph_correctPressure.setArg( 5, mass );
	pcisph_correctPressure.setArg( 6, rho0 );
	pcisph_correctPressure.setArg( 7, simulationScale );
	pcisph_correctPressure.setArg( 8, stiffness );
	pcisph_correctPressure.setArg( 9, sortedPosition );
	pcisph_correctPressure.setArg(10, pressure );
	pcisph_correctPressure.setArg(11, rho );
	pcisph_correctPressure.setArg(12, rhoInv );
	pcisph_correctPressure.setArg(13, delta );
	pcisph_correctPressure.setArg(14, position );
	pcisph_correctPressure.setArg(15, particleIndex );
	queue.enqueueNDRangeKernel(
		pcisph_correctPressure, cl::NullRange, cl::NDRange( (int) ( PARTICLE_COUNT ) ),
#if defined( __APPLE__ )
		cl::NullRange, NULL, NULL );
#else
		cl::NDRange( (int)( 256 ) ), NULL, NULL );
#endif
#ifdef QUEUE_EACH_KERNEL
	queue.finish();
#endif
	return 0;
}


//======================

void writeLog_density()
{
	int err;

	err = queue.enqueueReadBuffer( rho, CL_TRUE, 0, PARTICLE_COUNT * sizeof( float ) * 1, positionBuffer );
	if( err != CL_SUCCESS ){
		throw std::runtime_error( "could not enqueue read" );
	}
	queue.finish();

	FILE* flog = fopen("density.txt","wt");
	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		fprintf(flog,"%e\t",(float)positionBuffer[i]);
	}
	fprintf(flog,"\n");
	fclose(flog);

}

void writeLog_neighbor_count()
{
	int err;

	err = queue.enqueueReadBuffer( rhoInv, CL_TRUE, 0, PARTICLE_COUNT * sizeof( float ) * 1, positionBuffer );
	if( err != CL_SUCCESS ){
		throw std::runtime_error( "could not enqueue read" );
	}
	queue.finish();

	float value = 0.f;

	FILE* flog = fopen("neighbor_count.txt","a+");
	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		value += (float)positionBuffer[i];
	}

	value /= (float)PARTICLE_COUNT;

	fprintf(flog,"%e\n",value);
	fclose(flog);

}

void writeLog_density_pred(int n_iter)
{
	int err;
	float aver_density = 0;
	float max_density = 0;
	float min_density = rho0;

	err = queue.enqueueReadBuffer( rho, CL_TRUE, 0, PARTICLE_COUNT * sizeof( float ) * 2, positionBuffer );
	if( err != CL_SUCCESS ){
		throw std::runtime_error( "could not enqueue read" );
	}
	queue.finish();
	

	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		aver_density += ((float)positionBuffer[i+PARTICLE_COUNT])/((float)PARTICLE_COUNT);
		max_density = max(max_density,(float)positionBuffer[i+PARTICLE_COUNT]);
		min_density = min(min_density,(float)positionBuffer[i+PARTICLE_COUNT]);
	}



	FILE* flog = fopen("density.txt","a+");
	/*for(int i=0;i<PARTICLE_COUNT;i++)
	{
		fprintf(flog,"%e\t",(float)positionBuffer[i+PARTICLE_COUNT]);
	}
	fprintf(flog,"\n");*/

	fprintf(flog,"%d\t%e\t%e\t%e\n",n_iter,max_density,aver_density,min_density);

	fclose(flog);

}

void writeLog_positions()
{
	int err;

	err = queue.enqueueReadBuffer( position, CL_TRUE, 0, PARTICLE_COUNT * sizeof( float ) * 4, positionBuffer );
	if( err != CL_SUCCESS ){
		throw std::runtime_error( "could not enqueue read" );
	}
	queue.finish();

	FILE* flog = fopen("positions.txt","wt");
	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		fprintf(flog,"%e\t",(float)positionBuffer[PARTICLE_COUNT*4*0+i*4]);
	}
	fprintf(flog,"\n");
	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		fprintf(flog,"%e\t",(float)positionBuffer[PARTICLE_COUNT*4*0+i*4+1]);
	}
	fprintf(flog,"\n");
	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		fprintf(flog,"%e\t",(float)positionBuffer[PARTICLE_COUNT*4*0+i*4+2]);
	}
	fprintf(flog,"\n");

	fclose(flog);

}

void writeLog_positions_pred()
{
	int err;

	err = queue.enqueueReadBuffer( sortedPosition, CL_TRUE, 0, PARTICLE_COUNT * sizeof( float ) * 8, positionBuffer );
	if( err != CL_SUCCESS ){
		throw std::runtime_error( "could not enqueue read" );
	}
	queue.finish();

	FILE* flog = fopen("positions.txt","wt");
	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		fprintf(flog,"%e\t",(float)positionBuffer[PARTICLE_COUNT*4+i*4]);
	}
	fprintf(flog,"\n");
	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		fprintf(flog,"%e\t",(float)positionBuffer[PARTICLE_COUNT*4+i*4+1]);
	}
	fprintf(flog,"\n");
	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		fprintf(flog,"%e\t",(float)positionBuffer[PARTICLE_COUNT*4+i*4+2]);
	}
	fprintf(flog,"\n");

	fclose(flog);

}

void writeLog_accelerations()
{
	int err;

	err = queue.enqueueReadBuffer( acceleration, CL_TRUE, 0, PARTICLE_COUNT * sizeof( float ) * 8, positionBuffer );
	if( err != CL_SUCCESS ){
		throw std::runtime_error( "could not enqueue read" );
	}
	queue.finish();

	FILE* flog = fopen("accelerations.txt","wt");
	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		fprintf(flog,"%e\t",(float)positionBuffer[PARTICLE_COUNT*0+i*4]);
	}
	fprintf(flog,"\n");
	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		fprintf(flog,"%e\t",(float)positionBuffer[PARTICLE_COUNT*0+i*4+1]);
	}
	fprintf(flog,"\n");
	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		fprintf(flog,"%e\t",(float)positionBuffer[PARTICLE_COUNT*0+i*4+2]);
	}
	fprintf(flog,"\n");

	fclose(flog);

}

void writeLog_accelerations_p()
{
	int err;

	err = queue.enqueueReadBuffer( acceleration, CL_TRUE, 0, PARTICLE_COUNT * sizeof( float ) * 8, positionBuffer );
	if( err != CL_SUCCESS ){
		throw std::runtime_error( "could not enqueue read" );
	}
	queue.finish();

	FILE* flog = fopen("accelerations_p.txt","wt");
	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		fprintf(flog,"%e\t",(float)positionBuffer[PARTICLE_COUNT*4+i*4]);
	}
	fprintf(flog,"\n");
	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		fprintf(flog,"%e\t",(float)positionBuffer[PARTICLE_COUNT*4+i*4+1]);
	}
	fprintf(flog,"\n");
	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		fprintf(flog,"%e\t",(float)positionBuffer[PARTICLE_COUNT*4+i*4+2]);
	}
	fprintf(flog,"\n");

	fclose(flog);

}


void writeLog_pressure()
{
	int err;

	err = queue.enqueueReadBuffer( pressure, CL_TRUE, 0, PARTICLE_COUNT * sizeof( float ) * 1, positionBuffer );
	if( err != CL_SUCCESS ){
		throw std::runtime_error( "could not enqueue read" );
	}
	queue.finish();

	FILE* flog = fopen("pressure.txt","wt");
	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		fprintf(flog,"%e\t",(float)positionBuffer[i]);
	}
	fprintf(flog,"\n");
	fclose(flog);

}

float getDensityError(int n_iter)
{
	int err;
	float aver_density = 0;
	float max_density = 0;
	float min_density = rho0;
//	float density;

	err = queue.enqueueReadBuffer( rho, CL_TRUE, 0, PARTICLE_COUNT * sizeof( float ) * 2, positionBuffer );
	if( err != CL_SUCCESS ){
		throw std::runtime_error( "could not enqueue read" );
	}
	queue.finish();

	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		max_density = max(max_density,(float)positionBuffer[i+PARTICLE_COUNT]);
		//min_density = min(min_density,(float)positionBuffer[i+PARTICLE_COUNT]);
	}

	return (max_density-rho0) / rho0;

}

void prepareElasticMatterData()
{
	int i,j;
	int id,jd;// id - ID for i-th particle, jd - ID for j-th particle
	float rij0;
	int particle_i_type;
	int particle_j_type;

	const int border = 0;
	const int liquid = 1;
	const int elastic_matter = 2;

	float xi,yi,zi;
	float xj,yj,zj;

	float rij_check;
	int id_back,jd_back;
	int elasticParticlesCounter = 0;

	int elasticConnectionsCounter = 0;

	// when running this function I expect that positionBuffer contains sortedPosition data 

	for(int mode=0;mode<=1;mode++)// run twice: first to determine size of array
	{// then allocate necessary amount of memory, then, at second run, fill the arraty with data

		for(i=0;i<PARTICLE_COUNT;i++)
		{
			//id = particleIndexBuffer[i*2+0];
			particle_i_type = (int)positionBuffer[i*4+3];
			xi = positionBuffer[i*4+0];
			yi = positionBuffer[i*4+1];
			zi = positionBuffer[i*4+2];

			id = i;
			id_back = particleIndexBuffer[id*2+1];
			//id_back = particleIndexBuffer[id];
			//id = particleIndexBack[id];
			if(id_back==0)
			{
				id_back = 0;
			}

			if(particle_i_type==elastic_matter)
			{
				elasticParticlesCounter++;

				for(j=0;j<NEIGHBOR_COUNT;j++)
				{
					jd = (int)neighborMapBuffer[id * NEIGHBOR_COUNT * 2 + j*2 + 0];
					rij0 =    neighborMapBuffer[id * NEIGHBOR_COUNT * 2 + j*2 + 1];

					jd_back = particleIndexBuffer[jd*2+1];
					//jd_back = particleIndexBuffer[jd];
					//jd_back = jd;

					particle_j_type = (int)positionBuffer[jd*4+3];
					xj = positionBuffer[jd*4+0];
					yj = positionBuffer[jd*4+1];
					zj = positionBuffer[jd*4+2];

					rij_check = simulationScale * sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));

					if(particle_j_type==elastic_matter)
					{
						//rij0 = rij0;
						//if(rij0<h*simulationScale/2)
						{
							if(mode==1)// here I expect that memory is already allocated
							{
								// elementary data cell: [i|j|r0_ij]
								elasticConnectionsDataBuffer[elasticConnectionsCounter*4+0] = (float)id_back + 0.1f;
								elasticConnectionsDataBuffer[elasticConnectionsCounter*4+1] = (float)jd_back + 0.1f;
								elasticConnectionsDataBuffer[elasticConnectionsCounter*4+2] = rij0;
								elasticConnectionsDataBuffer[elasticConnectionsCounter*4+3] = 0.f;
							}

							elasticConnectionsCounter++;
						}
					}
				}

				//j = j;
			}
		}

		if(mode==0)
		{
			// elementary data cell: [i|j|r0_ij]
			ELASTIC_CONNECTIONS_COUNT = elasticConnectionsCounter;
			elasticConnectionsDataBuffer = new float [ELASTIC_CONNECTIONS_COUNT*4];
			elasticConnectionsCounter = 0;
			elasticParticlesCounter = 0;
		}
	}

	int err;
	if(ELASTIC_CONNECTIONS_COUNT != 0){
	
		elasticConnectionsData = cl::Buffer( context, CL_MEM_READ_WRITE, ( ELASTIC_CONNECTIONS_COUNT * sizeof( float ) * 4 ), NULL, &err );
		if( err != CL_SUCCESS ){ throw std::runtime_error( "buffer elasticConnectionsData creation failed" ); }
	

		err = queue.enqueueWriteBuffer( elasticConnectionsData, CL_TRUE, 0, ELASTIC_CONNECTIONS_COUNT * sizeof( float ) * 4, elasticConnectionsDataBuffer );
		if( err != CL_SUCCESS ){ throw std::runtime_error( "could not enqueue elasticConnectionsData write" ); }
	}
	

	return;
}
double elapsedTime;
#if defined(_WIN32) || defined (_WIN64)
	LARGE_INTEGER frequency;				// ticks per second
	LARGE_INTEGER t0, t1, t2;				// ticks
	LARGE_INTEGER t3,t4;
#elif defined(__linux__)
	timespec t0, t1, t2;
	timespec t3,t4;
	double us;
#endif
double stopwatch_report(const char *str){
#if defined(_WIN32) || defined(_WIN64)
	QueryPerformanceCounter(&t2);
	printf(str,(t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart);
	t1 = t2;
	return (t2.QuadPart - t0.QuadPart) * 1000.0 / frequency.QuadPart;
#elif defined(__linux__)
	clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
	double us;
	time_t sec = t2.tv_sec - t1.tv_sec;
	long nsec;
	if (t2.tv_nsec >= t1.tv_nsec) {
			nsec = t2.tv_nsec - t1.tv_nsec;
	} else {
			nsec = 1000000000 - (t1.tv_nsec - t2.tv_nsec);
			sec -= 1;
	}
	printf(str,(float)sec * 1000.f + (float)nsec/1000000.f);
	t1 = t2;
	return (float)(t2.tv_sec - t0.tv_sec) * 1000.f + (float)(t2.tv_nsec - t0.tv_nsec)/1000000.f;
#endif
}
void step(int nIter)
{
	int err;
	float rho_err;
#if defined(_WIN32) || defined (_WIN64)
    QueryPerformanceFrequency(&frequency);	// get ticks per second
    QueryPerformanceCounter(&t1);			// start timer
	t0 = t1;
#elif defined(__linux__)
	clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
	t0 = t1;
#endif
	printf("\n");

	/* THIS IS COMMON PART FOR BOTH SPH AND PCISPH*/

	_runClearBuffers( queue ); queue.finish();			 stopwatch_report("_runClearBuffers: \t%9.3f ms\n");

	_runHashParticles( queue ); queue.finish();			 stopwatch_report("_runHashParticles: \t%9.3f ms\n");

	_runSort( queue ); queue.finish(); 					 stopwatch_report("_runSort: \t\t%9.3f ms\n");

	_runSortPostPass( queue ); queue.finish();			 stopwatch_report("_runSortPostPass: \t%9.3f ms\n");

	_runIndexx( queue ); queue.finish();				 stopwatch_report("_runIndexx: \t\t%9.3f ms\n");

	// this is completely new _runIndexPostPass, very fast, which replaced the previous one (slow, non-optimal) | single CPU thread
	_runIndexPostPass( ); /*queue.finish();*/	 stopwatch_report("_runIndexPostPass: \t%9.3f ms\n");

	_runFindNeighbors( queue); queue.finish();	 stopwatch_report("_runFindNeighbors: \t%9.3f ms\n");

	if(nIter==1)//run once only at first (initial) pass
	{
		_runPreElasticMatterPass( queue ); queue.finish();			// QueryPerformanceCounter(&t2); printf("_runPreElasticMatterPass: \t%9.3f ms\n",			(t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart); t1 = t2;

		err = queue.enqueueReadBuffer( neighborMap, CL_TRUE, 0, ( NK * sizeof( float ) * 2 ), neighborMapBuffer );
		if( err != CL_SUCCESS ){ throw std::runtime_error( "could not enqueue neighborMap read" ); } queue.finish();

		err = queue.enqueueReadBuffer( particleIndex, CL_TRUE, 0, ( PARTICLE_COUNT * sizeof( unsigned int ) * 2 ),  particleIndexBuffer);
		if( err != CL_SUCCESS ){ throw std::runtime_error( "could not enqueue particleIndexBuffer read" ); } queue.finish();

		err = queue.enqueueReadBuffer( sortedPosition, CL_TRUE, 0, PARTICLE_COUNT * sizeof( float ) * 4, positionBuffer );
		if( err != CL_SUCCESS ){ throw std::runtime_error( "could not enqueue position read" ); } queue.finish();

		prepareElasticMatterData(); 
	}

	// basic SPH
	if(PCISPH==0)
	{
		_runComputeDensityPressure( queue ); queue.finish(); stopwatch_report("_runComputeDensityPressure: \t%9.3f ms\n");

		_runComputeAcceleration( queue ); queue.finish();	 stopwatch_report("_runComputeAcceleration: \t%9.3f ms\n");

		_runIntegrate( queue ); queue.finish();	 stopwatch_report("_runIntegrate: \t%9.3f ms\n");
	}

	// PCISPH
	if(PCISPH==1)
	{
		// according to B. Solenthaler's dissertation, page 28
		// we need to compute forces (all except pressure forces), initialize pressure and pressure forces with 0.0
		// viscosity force needs density to be calculated preliminarily:

		_run_pcisph_computeDensity( queue ); queue.finish(); /*writeLog_neighbor_count();*/  //writeLog_density();
		//printf("_run_pcisph_computeDensity\n");
		 
		// compute forces F_viscous,gravity,external; initialize p(t)=0.0 and Fp(t)=0.0:
		_run_pcisph_computeForcesAndInitPressure( queue ); queue.finish();// writeLog_accelerations();
		//printf("_run_pcisph_computeForcesAndInitPressure\n");

		//QueryPerformanceCounter(&t3);
		// this function MUST follow _run_pcisph_computeForcesAndInitPressure, otherwise it will work incorrectly
		_run_pcisph_computeElasticForces( queue ); queue.finish();
		//QueryPerformanceCounter(&t4);
		//printf("_run_pcisph_computeElasticForces: \t%9.3f ms\n",(t4.QuadPart - t3.QuadPart) * 1000.0 / frequency.QuadPart);
		 
		// this is the main prediction-correction loop which works until rho_err_*_(t+1) > d_rho (~1%)
		int iter = 0;
		int maxIterations = 3;//3;
		do
		{
			_run_pcisph_predictPositions(queue); queue.finish();	//writeLog_positions_pred();
			//printf("_run_pcisph_predictPositions\n");
			_run_pcisph_predictDensity(queue); queue.finish();		//writeLog_density_pred(iter);
			//printf("_run_pcisph_predictDensity\n");
			_run_pcisph_correctPressure(queue); queue.finish();		//writeLog_pressure();
			//printf("_run_pcisph_correctPressure\n");
			_run_pcisph_computePressureForceAcceleration( queue ); queue.finish(); //writeLog_accelerations_p();
			//printf("_run_pcisph_computePressureForceAcceleration\n");

			iter++;
			/*
			rho_err=getDensityError(iter);
			printf("iter: %d \t\t\trho_err: %f\n",iter,rho_err);
			/**/
			/*if(iter>5)
			{
				iter = iter;
			}*/
		}
		while( (iter<maxIterations) /* && (rho_err >= 0.02f)*/); //(until error becomes less than 2%)

		// for all particles: compute new velocity v(t+1)
		// for all particles: compute new position x(t+1)
		_run_pcisph_integrate( queue ); queue.finish();	//writeLog_positions();

		stopwatch_report("_runPCISPH: \t\t%9.3f ms\t3 iteration(s)\n");
	}

	

	//...

	// END OF PCISPH


	//printf("_[0]\n");
	//printf("enter <queue.enqueueReadBuffer>, line 700 at main.cpp\n");
	err = queue.enqueueReadBuffer( position, CL_TRUE, 0, PARTICLE_COUNT * sizeof( float ) * 4, positionBuffer );
	if( err != CL_SUCCESS ){ throw std::runtime_error( "could not enqueue position read" ); } queue.finish();
	//printf("_[1]\n");
	if(nIter == 100){
		writeLog_positions();
	}
	err = queue.enqueueReadBuffer( rho, CL_TRUE, 0, PARTICLE_COUNT * sizeof( float ) * 1, densityBuffer );
	if( err != CL_SUCCESS ){ throw std::runtime_error( "could not enqueue read" ); } queue.finish();

	stopwatch_report("_readBuffer: \t\t%9.3f ms\n");

	//It should be removed after fixing
	/**/err = queue.enqueueReadBuffer( neighborMap, CL_TRUE, 0, ( NK * sizeof( float ) * 2 ), neighborMapBuffer );
	if( err != CL_SUCCESS ){ throw std::runtime_error( "could not enqueue neighborMap read" ); } queue.finish();
	//printf("_[2]\n");

	stopwatch_report("_readBuffer: \t\t%9.3f ms\n");

	/**/err = queue.enqueueReadBuffer( particleIndex, CL_TRUE, 0, ( PARTICLE_COUNT * sizeof( unsigned int ) * 2 ),  particleIndexBuffer);
	if( err != CL_SUCCESS ){ throw std::runtime_error( "could not enqueue particleIndexBuffer read" ); } queue.finish();
	//printf("_[3]\n");

	elapsedTime = stopwatch_report("_readBuffer: \t\t%9.3f ms\n");


	printf("------------------------------------\n");
	printf("_Total_step_time:\t%9.3f ms\n",elapsedTime);
	printf("------------------------------------\n");

}


void initializeOpenCL(
					  cl::Context & context,
					  std::vector< cl::Device > & devices,
					  cl::CommandQueue & queue,
					  cl::Program & program
					  )
{
	cl_int err;
	std::vector< cl::Platform > platformList;
	err = cl::Platform::get( &platformList );
	if( platformList.size() < 1 ){
		throw std::runtime_error( "no OpenCL platforms found" );
	}

	
///////////////////AP2012///////////////

	char cBuffer[1024];
	cl_platform_id clSelectedPlatformID = NULL;
	cl_platform_id cl_pl_id[10];
	cl_uint n_pl;
	clGetPlatformIDs(10,cl_pl_id,&n_pl);
		
	cl_int ciErrNum;// = oclGetPlatformID (&clSelectedPlatformID);
	//oclCheckError(ciErrNum, CL_SUCCESS);
	int sz;

	for(int i=0;i<(int)n_pl;i++)
	{
		// Get OpenCL platform name and version
		ciErrNum = clGetPlatformInfo (cl_pl_id[i], CL_PLATFORM_VERSION, sz = sizeof(cBuffer), cBuffer, NULL);

		if (ciErrNum == CL_SUCCESS)
		{
			printf(" CL_PLATFORM_VERSION [%d]: \t%s\n", i, cBuffer);
		} 
		else
		{
			printf(" Error %i in clGetPlatformInfo Call !!!\n\n", ciErrNum);
		}
	}

///////////////////AP2012////////////////
/*
	cl_context_properties *cprops;
	cprops = new cl_context_properties[ 6 ];
	cprops[ 0 ] = CL_CONTEXT_D3D10_DEVICE_KHR; 
	cprops[ 1 ] = (intptr_t) DXUTGetD3D10Device();
	cprops[ 2 ] = CL_CONTEXT_PLATFORM;
	cprops[ 3 ] = (cl_context_properties)(platformList[0])();
	cprops[ 4 ] = cprops[ 5 ] = 0;

#ifdef NDEBUG
	context = cl::Context( CL_DEVICE_TYPE_ALL, cprops, NULL, NULL, &err );
#else
	context = cl::Context( CL_DEVICE_TYPE_ALL, cprops, NULL, NULL, &err );
#endif
*/

	//0-CPU, 1-GPU// depends on order appropriet drivers was instaled
	int plList=1;//selected platform index in platformList array

	cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties) (platformList[plList])(), 0 };

	//cl_context context1 = clCreateContext(cprops, CL_DEVICE_TYPE_CPU, NULL, NULL, NULL,&err);//( cl::Context( CL_DEVICE_TYPE_ALL, cprops, NULL, NULL, &err );
	context = cl::Context( CL_DEVICE_TYPE_ALL, cprops, NULL, NULL, &err );
	
	devices = context.getInfo< CL_CONTEXT_DEVICES >();
	if( devices.size() < 1 ){
		throw std::runtime_error( "no OpenCL devices found" );
	}

	///////////////////AP2012////////////////
	int value;
	cl_int result = devices[0].getInfo(CL_DEVICE_NAME,&cBuffer);// CL_INVALID_VALUE = -30;		
	if(result == CL_SUCCESS) printf("CL_PLATFORM_VERSION [%d]: CL_DEVICE_NAME [%d]: \t%s\n",plList, 0, cBuffer);
	result = devices[0].getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE,&value);
	if(result == CL_SUCCESS) printf("CL_PLATFORM_VERSION [%d]: CL_DEVICE_MAX_WORK_GROUP_SIZE [%d]: \t%d\n",plList, 0, value);
	result = devices[0].getInfo(CL_DEVICE_MAX_COMPUTE_UNITS,&value);
	if(result == CL_SUCCESS) printf("CL_PLATFORM_VERSION [%d]: CL_DEVICE_MAX_COMPUTE_UNITS [%d]: \t%d\n",plList, 0, value);
	result = devices[0].getInfo(CL_DEVICE_GLOBAL_MEM_SIZE,&value);
	if(result == CL_SUCCESS) printf("CL_PLATFORM_VERSION [%d]: CL_DEVICE_GLOBAL_MEM_SIZE [%d]: \t%d\n",plList, 0, value);
	result = devices[0].getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,&value);
	if(result == CL_SUCCESS) printf("CL_PLATFORM_VERSION [%d]: CL_DEVICE_GLOBAL_MEM_CACHE_SIZE [%d]: \t%d\n",plList, 0, value);
	///////////////////AP2012////////////////

	queue = cl::CommandQueue( context, devices[ 0 ], 0, &err );
	if( err != CL_SUCCESS ){
		throw std::runtime_error( "failed to create command queue" );
	}

	std::string sourceFileName( "src/sphFluidDemo.cl" );
	std::ifstream file( sourceFileName.c_str() );
	if( !file.is_open() ){
		throw std::runtime_error( "could not open file " + sourceFileName );
	}

	std::string programSource( std::istreambuf_iterator<char>( file ), ( std::istreambuf_iterator<char>() ));
	cl::Program::Sources source( 1, std::make_pair( programSource.c_str(), programSource.length()+1 ));

	program = cl::Program( context, source );   
//#ifdef NDEBUG
	//E:\Distrib\_OpenWorm related soft\Smoothed-Particle-Hydrodynamics
/*work*/  //	err = program.build( devices, "-g -s \"E:\\Distrib\\_OpenWorm related soft\\PCISPH+elastic\\src\\sphFluidDemo.cl\"" );
/*homeS*/ // err = program.build( devices,"-g -s \"C:\\Users\\Sergey\\Desktop\\SphFluid_CLGL_myNeighborhoodSearch_12may2012\\sphFluidDemo.cl\"" );
/*homeA*/  //  err = program.build( devices, "-g -s \"D:\\_OpenWorm\\SphFluid_CLGL_original_32nearest_PCI+elastic\\src\\sphFluidDemo.cl\"" );
/*workS*///  	err = program.build( devices, "-g -s \"C:\\Users\\Serg\\Documents\\GitHub\\Smoothed-Particle-Hydrodynamics\\src\\sphFluidDemo.cl\"" );			 			 
	//D:\_OpenWorm\SphFluid_CLGL_original_32nearest_PCI
/*#else*/
	//err = program.build( devices, "-g" );
	err = program.build( devices, "" );
/*#endif*/
	if( err != CL_SUCCESS ){
		std::string compilationErrors;
		compilationErrors = program.getBuildInfo< CL_PROGRAM_BUILD_LOG >( devices[ 0 ] );
		std::cerr << "Compilation failed: " << std::endl << compilationErrors << std::endl;
		throw std::runtime_error( "failed to build program" );
	}

	return;
}


int simulation_start ( /*int argc, char **argv*/ )
{
	int err;
	positionBuffer = new float[ 8 * PARTICLE_COUNT ];
	velocityBuffer = new float[ 4 * PARTICLE_COUNT ];
	densityBuffer  = new float[ 1 * PARTICLE_COUNT ];

	neighborMapBuffer = new float[( NK * sizeof( float ) * 2 )];
	particleIndexBuffer = new unsigned int[PARTICLE_COUNT * 2];

	try{

		initializeOpenCL( context, devices, queue, program );
//AP2012
/*
#ifdef NDEBUG
		radixSort.initializeSort( context, queue, PARTICLE_COUNT, 16, true );		
#endif
*/
//AP2012

		// initialize buffers
		gridCellsX = (int)( ( XMAX - XMIN ) / h ) + 1;
		gridCellsY = (int)( ( YMAX - YMIN ) / h ) + 1;
		gridCellsZ = (int)( ( ZMAX - ZMIN ) / h ) + 1;
		gridCellCount = gridCellsX * gridCellsY * gridCellsZ;

		//caclulate number of boundary partycles by X, Y, Z axis. Distance Between two neighbor particle is equal to 
		int n = (int)( ( XMAX - XMIN ) / r0 ); //X
		int m = (int)( ( YMAX - YMIN ) / r0 ); //Y
		int k = (int)( ( ZMAX - ZMIN ) / r0 ); //Z
		int boundaryParticleCount = 0;//2 * n * m + 2 * k * n + 2 * k * m;
		//
		//28aug_Palyanov_start_block
		gridNextNonEmptyCellBuffer = new unsigned int[gridCellCount+1];
		//28aug_Palyanov_end_block

		acceleration = cl::Buffer( context, CL_MEM_READ_WRITE, ( PARTICLE_COUNT * sizeof( float ) * 4 * 2 ), NULL, &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "buffer acceleration creation failed" );
		}
		gridCellIndex = cl::Buffer( context, CL_MEM_READ_WRITE, ( ( gridCellCount + 1 ) * sizeof( unsigned int ) * 1 ), NULL, &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "buffer gridCellIndex creation failed" );
		}
		gridCellIndexFixedUp = cl::Buffer( context, CL_MEM_READ_WRITE, ( ( gridCellCount + 1 ) * sizeof( unsigned int ) * 1 ), NULL, &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "buffer gridCellIndexFixedUp creation failed" );
		}
		neighborMap = cl::Buffer( context, CL_MEM_READ_WRITE, ( NK * sizeof( float ) * 2 ), NULL, &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "buffer neighborMap creation failed" );
		}
		particleIndex = cl::Buffer( context, CL_MEM_READ_WRITE, ( PARTICLE_COUNT * sizeof( unsigned int ) * 2 ), NULL, &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "buffer particleIndex creation failed" );
		}
		particleIndexBack = cl::Buffer( context, CL_MEM_READ_WRITE, ( PARTICLE_COUNT * sizeof( unsigned int ) ), NULL, &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "buffer particleIndexBack creation failed" );
		}
		position = cl::Buffer( context, CL_MEM_READ_WRITE, ( PARTICLE_COUNT * sizeof( float ) * 4 ), NULL, &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "buffer position creation failed" );
		}
		pressure = cl::Buffer( context, CL_MEM_READ_WRITE, ( PARTICLE_COUNT * sizeof( float ) * 1 ), NULL, &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "buffer pressure creation failed" );
		}
		rho = cl::Buffer( context, CL_MEM_READ_WRITE, ( PARTICLE_COUNT * sizeof( float ) * 2 ), NULL, &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "buffer rho creation failed" );
		}
		rhoInv = cl::Buffer( context, CL_MEM_READ_WRITE, ( PARTICLE_COUNT * sizeof( float ) ), NULL, &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "buffer rhoInv creation failed" );
		}
		sortedPosition = cl::Buffer( context, CL_MEM_READ_WRITE, ( PARTICLE_COUNT * sizeof( float ) * 4 * 2 ), NULL, &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "buffer sortedPosition creation failed" );
		}
		sortedVelocity = cl::Buffer( context, CL_MEM_READ_WRITE, ( PARTICLE_COUNT * sizeof( float ) * 4 ), NULL, &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "buffer sortedVelocity creation failed" );
		}
		velocity = cl::Buffer( context, CL_MEM_READ_WRITE, ( PARTICLE_COUNT * sizeof( float ) * 4 ), NULL, &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "buffer velocity creation failed" );
		}

		// create kernels
		clearBuffers = cl::Kernel( program, "clearBuffers", &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "kernel clearBuffers creation failed" );
		}
		computeAcceleration = cl::Kernel( program, "computeAcceleration", &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "kernel computeAcceleration creation failed" );
		}
		computeDensityPressure = cl::Kernel( program, "computeDensityPressure", &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "kernel computeDensityPressure creation failed" );
		}
		findNeighbors = cl::Kernel( program, "findNeighbors", &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "kernel findNeighbors creation failed" );
		}
		hashParticles = cl::Kernel( program, "hashParticles", &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "kernel hashParticles creation failed" );
		}
		indexx = cl::Kernel( program, "indexx", &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "kernel indexx creation failed" );
		}
		integrate = cl::Kernel( program, "integrate", &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "kernel integrate creation failed" );
		}
		sortPostPass = cl::Kernel( program, "sortPostPass", &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "kernel sortPostPass creation failed" );
		}
		// additional, PCISPH
		preElasticMatterPass  = cl::Kernel( program, "preElasticMatterPass", &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "kernel preElasticMatterPass creation failed" );
		}

		pcisph_computeForcesAndInitPressure = cl::Kernel( program, "pcisph_computeForcesAndInitPressure", &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "kernel pcisph_computeForcesAndInitPressure creation failed" );
		}

		pcisph_integrate = cl::Kernel( program, "pcisph_integrate", &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "kernel pcisph_integrate creation failed" );
		}

		pcisph_predictPositions = cl::Kernel( program, "pcisph_predictPositions", &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "kernel pcisph_predictPositions creation failed" );
		}

		pcisph_predictDensity = cl::Kernel( program, "pcisph_predictDensity", &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "kernel pcisph_predictDensity creation failed" );
		}

		pcisph_correctPressure = cl::Kernel( program, "pcisph_correctPressure", &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "kernel pcisph_correctPressure creation failed" );
		}

		pcisph_computePressureForceAcceleration = cl::Kernel( program, "pcisph_computePressureForceAcceleration", &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "kernel pcisph_computePressureForceAcceleration creation failed" );
		}

		pcisph_computeDensity = cl::Kernel( program, "pcisph_computeDensity", &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "kernel pcisph_computeDensity creation failed" );
		}

		pcisph_computeElasticForces = cl::Kernel( program, "pcisph_computeElasticForces", &err );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "kernel pcisph_computeElasticForces creation failed" );
		}



/*
		for( int i = 0; i < PARTICLE_COUNT; ++i ){
			float x, y, z;
			float r;
			r = ( (float)rand() / (float)RAND_MAX );
#define SCALE( MIN, MAX, X ) ( (MIN) + (X) * ( (MAX) - (MIN) ) )
			x = SCALE( XMIN, ( XMAX / 10 ), r );
			r = ( (float)rand() / (float)RAND_MAX );
			y = SCALE( YMIN, YMAX, r );
			r = ( (float)rand() / (float)RAND_MAX );
			z = SCALE( ZMIN, ZMAX, r );
			float * positionVector = positionBuffer + 4 * i;
			positionVector[ 0 ] = x;
			positionVector[ 1 ] = y;
			positionVector[ 2 ] = z;
			positionVector[ 3 ] = 0;
			float * velocityVector = velocityBuffer + 4 * i;
			r = ( (float)rand() / (float)RAND_MAX );
			velocityVector[ 0 ] = SCALE( -1.0f, 1.0f, r );
			r = ( (float)rand() / (float)RAND_MAX );
			velocityVector[ 1 ] = SCALE( -1.0f, 1.0f, r );
			r = ( (float)rand() / (float)RAND_MAX );
			velocityVector[ 2 ] = SCALE( -1.0f, 1.0f, r );
			velocityVector[ 3 ] = 0;

		}//for
		
		/**/

		
		float x,y,z;
		float coeff = r0; // for particle mass = 0.0003
		int i = 0;
		//drop
		int status = 1;
		//Creation of Boundary Particle
		x = xmin;
		z = zmin;
		y = ymin;
		float x1, y1, z1;
		y1 = ymax;
		float speed = 1.0f;
		float normCorner = 1/sqrt(3.f);
		float normBoundary = 1/sqrt(2.f);
		bool isBoundary = false;
		for(;i <= 2 *( k * n +  n + k );i+=2)
		{
			float * positionVector = positionBuffer + 4 * i;
			positionVector[ 0 ] = x;
			positionVector[ 1 ] = y;
			positionVector[ 2 ] = z;
			positionVector[ 3 ] = 3.1;// 3 = boundary

			positionVector = positionBuffer + 4 * (i + 1);
			positionVector[ 0 ] = x;
			positionVector[ 1 ] = y1;
			positionVector[ 2 ] = z;
			positionVector[ 3 ] = 3.1;// 3 = boundary
			x+= r0;		
			if(i == 0){
				float * velocityVector = velocityBuffer + 4 * i;
				velocityVector[ 0 ] = normCorner;
				velocityVector[ 1 ] = normCorner;
				velocityVector[ 2 ] = normCorner;
				velocityVector[ 3 ] = 0;

				velocityVector = velocityBuffer + 4 * (i + 1);
				velocityVector[ 0 ] = normCorner;
				velocityVector[ 1 ] = -normCorner;
				velocityVector[ 2 ] = normCorner;
				velocityVector[ 3 ] = 0;
				isBoundary = true;
			}
			if(x >= xmax && z == zmin && !isBoundary){
				float * velocityVector = velocityBuffer + 4 * i;
				velocityVector[ 0 ] = -normCorner;
				velocityVector[ 1 ] = normCorner;
				velocityVector[ 2 ] = normCorner;
				velocityVector[ 3 ] = 0;

				velocityVector = velocityBuffer + 4 * (i + 1);
				velocityVector[ 0 ] = -normCorner;
				velocityVector[ 1 ] = -normCorner;
				velocityVector[ 2 ] = normCorner;
				velocityVector[ 3 ] = 0;
				isBoundary = true;
			}
			
			if(x >= xmax && z >= zmax - r0 && !isBoundary){
				float * velocityVector = velocityBuffer + 4 * i;
				velocityVector[ 0 ] = -normCorner;
				velocityVector[ 1 ] = normCorner;
				velocityVector[ 2 ] = -normCorner;
				velocityVector[ 3 ] = 0;

				velocityVector = velocityBuffer + 4 * (i + 1);
				velocityVector[ 0 ] = -normCorner;
				velocityVector[ 1 ] = -normCorner;
				velocityVector[ 2 ] = -normCorner;
				velocityVector[ 3 ] = 0;
				isBoundary = true;
			}
			if(x - r0 == xmin && z >= zmax - r0 && !isBoundary){
				float * velocityVector = velocityBuffer + 4 * i;
				velocityVector[ 0 ] = normCorner;
				velocityVector[ 1 ] = normCorner;
				velocityVector[ 2 ] = -normCorner;
				velocityVector[ 3 ] = 0;

				velocityVector = velocityBuffer + 4 * (i + 1);
				velocityVector[ 0 ] = normCorner;
				velocityVector[ 1 ] = -normCorner;
				velocityVector[ 2 ] = -normCorner;
				velocityVector[ 3 ] = 0;
				isBoundary = true;
			}
			
			if(x >= xmax && !isBoundary){
				float * velocityVector = velocityBuffer + 4 * i;
				velocityVector[ 0 ] = -normBoundary;
				velocityVector[ 1 ] = normBoundary;
				velocityVector[ 2 ] = 0;
				velocityVector[ 3 ] = 0;

				velocityVector = velocityBuffer + 4 * (i + 1);
				velocityVector[ 0 ] =  -normBoundary;
				velocityVector[ 1 ] = -normBoundary;
				velocityVector[ 2 ] = 0;
				velocityVector[ 3 ] = 0;
				isBoundary = true;
			}
			
			if(z == zmin && !isBoundary){
				float * velocityVector = velocityBuffer + 4 * i;
				velocityVector[ 0 ] = 0;
				velocityVector[ 1 ] = normBoundary;
				velocityVector[ 2 ] = normBoundary;
				velocityVector[ 3 ] = 0;

				velocityVector = velocityBuffer + 4 * (i + 1);
				velocityVector[ 0 ] =  0;
				velocityVector[ 1 ] = -normBoundary;
				velocityVector[ 2 ] = normBoundary;
				velocityVector[ 3 ] = 0;
				isBoundary = true;
			}
			if(x - r0 == xmin && !isBoundary){
				float * velocityVector = velocityBuffer + 4 * i;
				velocityVector[ 0 ] = normBoundary;
				velocityVector[ 1 ] = normBoundary;
				velocityVector[ 2 ] = 0;
				velocityVector[ 3 ] = 0;

				velocityVector = velocityBuffer + 4 * (i + 1);
				velocityVector[ 0 ] =  normBoundary;
				velocityVector[ 1 ] = -normBoundary;
				velocityVector[ 2 ] = 0;
				velocityVector[ 3 ] = 0;
				isBoundary = true;
				isBoundary = true;
			}
			if(z >= zmax - r0 && !isBoundary){
				float * velocityVector = velocityBuffer + 4 * i;
				velocityVector[ 0 ] = 0;
				velocityVector[ 1 ] = normBoundary;
				velocityVector[ 2 ] = -normBoundary;
				velocityVector[ 3 ] = 0;

				velocityVector = velocityBuffer + 4 * (i + 1);
				velocityVector[ 0 ] = 0;
				velocityVector[ 1 ] = -normBoundary;
				velocityVector[ 2 ] = -normBoundary;
				velocityVector[ 3 ] = 0;
				isBoundary = true;
			}
			if(isBoundary == false){
				float * velocityVector = velocityBuffer + 4 * i;
				velocityVector[ 0 ] = 0;
				velocityVector[ 1 ] = speed;
				velocityVector[ 2 ] = 0;
				velocityVector[ 3 ] = 0;

				velocityVector = velocityBuffer + 4 * (i + 1);
				velocityVector[ 0 ] = 0;
				velocityVector[ 1 ] = -speed;
				velocityVector[ 2 ] = 0;
				velocityVector[ 3 ] = 0;
			}
			isBoundary = false;
			if(x>xmax) { 
				x = xmin; 
				z += r0; 
			}
		}
		
		x = xmin;
		y = ymin + r0;
		z = zmin;

		x1 = xmax;
		isBoundary = false;
		int count = 2 *( k * ( m - 2 )  + k + m - 2) + i;
		for(;i <= count;i+=2)
		{
			float * positionVector = positionBuffer + 4 * i;
			positionVector[ 0 ] = x;
			positionVector[ 1 ] = y;
			positionVector[ 2 ] = z;
			positionVector[ 3 ] = 3.1;// 3 = boundary

			positionVector = positionBuffer + 4 * (i + 1);
			positionVector[ 0 ] = x1;
			positionVector[ 1 ] = y;
			positionVector[ 2 ] = z;
			positionVector[ 3 ] = 3.1;// 3 = boundary
			
			if(z == zmin){
				float * velocityVector = velocityBuffer + 4 * i;
				velocityVector[ 0 ] = normBoundary;
				velocityVector[ 1 ] = 0;
				velocityVector[ 2 ] = normBoundary;
				velocityVector[ 3 ] = 0;

				velocityVector = velocityBuffer + 4 * (i + 1);
				velocityVector[ 0 ] = -normBoundary;
				velocityVector[ 1 ] = 0;
				velocityVector[ 2 ] = normBoundary;
				velocityVector[ 3 ] = 0;
				isBoundary = true;
			}
			if(z >= zmax - r0 && !isBoundary){
				float * velocityVector = velocityBuffer + 4 * i;
				velocityVector[ 0 ] = normBoundary;
				velocityVector[ 1 ] = 0;
				velocityVector[ 2 ] = -normBoundary;
				velocityVector[ 3 ] = 0;

				velocityVector = velocityBuffer + 4 * (i + 1);
				velocityVector[ 0 ] = -normBoundary;
				velocityVector[ 1 ] = 0;
				velocityVector[ 2 ] = -normBoundary;
				velocityVector[ 3 ] = 0;
				isBoundary = true;
			}
			if(isBoundary == false){
				float * velocityVector = velocityBuffer + 4 * i;
				velocityVector[ 0 ] = speed;
				velocityVector[ 1 ] = 0.f;
				velocityVector[ 2 ] = 0.f;
				velocityVector[ 3 ] = 0;

				velocityVector = velocityBuffer + 4 * (i + 1);
				velocityVector[ 0 ] = -speed;
				velocityVector[ 1 ] = 0.f;
				velocityVector[ 2 ] = 0;
				velocityVector[ 3 ] = 0;
			}
			isBoundary = false;
			y+= r0;
			
			if(y > ymax - r0) { y = ymin+r0; z += r0; }
		}
		x = xmin + r0;
		y = ymin + r0;
		z = zmin;

		z1 = zmax;
		count = 2 *( ( n - 2 ) * ( m - 2 )  + n + m - 4) + i;
		for(;i <= count;i+=2)
		{
			float * positionVector = positionBuffer + 4 * i;
			positionVector[ 0 ] = x;
			positionVector[ 1 ] = y;
			positionVector[ 2 ] = z;
			positionVector[ 3 ] = 3.1;// 3 = boundary

			positionVector = positionBuffer + 4 * (i + 1);
			positionVector[ 0 ] = x;
			positionVector[ 1 ] = y;
			positionVector[ 2 ] = z1;
			positionVector[ 3 ] = 3.1;// 3 = boundary
			
			float * velocityVector = velocityBuffer + 4 * i;
			velocityVector[ 0 ] = 0.f;
			velocityVector[ 1 ] = 0.f;
			velocityVector[ 2 ] = speed;
			velocityVector[ 3 ] = 0;

			velocityVector = velocityBuffer + 4 * (i + 1);
			velocityVector[ 0 ] = 0.f;
			velocityVector[ 1 ] = 0.f;
			velocityVector[ 2 ] = -speed;
			velocityVector[ 3 ] = 0;

			y+= r0;
			
			if(y > ymax - r0) { y = ymin+r0; x += r0; }
		}
		//End

		coeff = 0.2325f; // for particle mass = 0.0003
		x = XMAX*4.0f/8.f+h*coeff;
		y = YMAX*2.0f/4.f+h*coeff; // 45*h*coeff;
		z = ZMAX*4.0f/8.f+h*coeff;

		int segmentsCount = 50;
		float alpha = 2.f*3.14159f/segmentsCount;
		float wormBodyRadius = h*coeff / sin(alpha/2);
		//float sinHalfAlpha = (h*coeff)/wormBodyRadius;
		//float alpha = asin(sinHalfAlpha)*2;
		//float sinAlpha = 2.f*sinHalfAlpha*sqrt(1.f-sinHalfAlpha*sinHalfAlpha);


		float * positionVector;
		float * velocityVector;
		/**/

		int j;


		int pCount = i;//particle counter

		// outer elastic cylinder generation
		
		//pCount++;

		//makeworm

		/*
		for( j = -45; j <= 45; ++j)
		{
			segmentsCount = (int)sqrt((float)(40*40-j*j*1*1));
			alpha = 2.f*3.14159f/segmentsCount;
			wormBodyRadius = h*coeff / sin(alpha/2);

			if(wormBodyRadius>=1.7f*h*coeff) 
			{
		
			for(int cylinderLayers = 1; cylinderLayers<= 9; cylinderLayers++ )
			{
				float pdist_coeff = 2.0;//2.0f;

				for( i = 0 ; i < segmentsCount; ++i )
				{
					positionVector = positionBuffer + 4 * pCount;
					velocityVector = velocityBuffer + 4 * pCount;

					positionVector[ 0 ] = x + wormBodyRadius*sin(alpha*i);
					positionVector[ 1 ] = y + wormBodyRadius*cos(alpha*i);
					positionVector[ 2 ] = z + pdist_coeff*h*coeff*(float)j ;
					positionVector[ 3 ] = 2.1;// 2 = elastic matter

					if(cylinderLayers>9) positionVector[ 3 ] = 1.1;// 1 = liquid

					velocityVector[ 0 ] = 0;
					velocityVector[ 1 ] = 0;
					velocityVector[ 2 ] = 0;
					velocityVector[ 3 ] = 0;

					pCount++;
				}

				if(wormBodyRadius<1.9f*2.f*h*coeff) 
				{
					break;
				}
				
				//if(cylinderLayers>4) pdist_coeff = 1.7f;

				segmentsCount *= (wormBodyRadius - pdist_coeff*h*coeff);
				segmentsCount /= wormBodyRadius;
				wormBodyRadius -= pdist_coeff*h*coeff;
				
				alpha = 2.f*3.14159f/segmentsCount;
			}

			}

		}
		/**/

		// inner elastic cylinder generation
		

		//wormBodyRadius = h*coeff / sin(alpha/2);

		// end of elastic matter setup

		/*
		for( ; (i < PARTICLE_COUNT/15.12); ++i )
		{
			float * positionVector = positionBuffer + 4 * i;
			positionVector[ 0 ] = x;
			positionVector[ 1 ] = y;
			positionVector[ 2 ] = z;
			positionVector[ 3 ] = 2.1;// 2 = elastic matter

			float * velocityVector = velocityBuffer + 4 * i;
			velocityVector[ 0 ] = 0;
			velocityVector[ 1 ] = 0.0;//initial velocity
			velocityVector[ 2 ] = 0;
			velocityVector[ 3 ] = 0;

			x+= 2*h*coeff;

			
			if(x>XMAX*4.5f/8.f) { x = XMAX*3.5f/8.f+h*coeff; z += 2*h*coeff; }
			if(z>ZMAX*4.5f/8.f) { x = XMAX*3.5f/8.f+h*coeff; z=ZMAX*3.5f/8.f+h*coeff; y += 2*h*coeff; }

		}
		/**/

		// bottom layer of water
/*
		x = 0*XMAX/4+h*coeff;
		y = h*coeff + r0;
		z = h*coeff;*/
		x = r0 * 5 + 0*XMAX/4+h*coeff;
		y = r0 * 10 + h*coeff;
		z = r0 * 5 + h*coeff;
		for( ; pCount < PARTICLE_COUNT; ++pCount )
		{
			float * positionVector = positionBuffer + 4 * pCount;
			positionVector[ 0 ] = x;
			positionVector[ 1 ] = y;
			positionVector[ 2 ] = z;
			positionVector[ 3 ] = 1.1;// 1 = liquid

			float * velocityVector = velocityBuffer + 4 * pCount;
			velocityVector[ 0 ] = 0;
			velocityVector[ 1 ] = 0;
			velocityVector[ 2 ] = 0;
			velocityVector[ 3 ] = 0;

			x+= 2*h*coeff;

			if(x>XMAX/2) { x = r0 * 5 + h*coeff; z += 2*h*coeff; }
			if(z>ZMAX/2) { x = r0 * 5 + h*coeff; z = r0 * 5 + h*coeff; y += 2*h*coeff; }
		}
		

		err = queue.enqueueWriteBuffer( position, CL_TRUE, 0, PARTICLE_COUNT * sizeof( float ) * 4, positionBuffer );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "could not enqueue position write" );
		}
		err = queue.enqueueWriteBuffer( velocity, CL_TRUE, 0, PARTICLE_COUNT * sizeof( float ) * 4, velocityBuffer );
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "could not enqueue velocity write" );
		}
		err = queue.finish();
		if( err != CL_SUCCESS ){
			throw std::runtime_error( "failed queue.finish" );
		}

	}catch( std::exception &e ){
		std::cout << "ERROR: " << e.what() << std::endl;
		exit( -1 );
	}

	printf("Entering main loop\n");

	int nIter = 0;
/*
	while(1)
	{
		nIter++;
		printf("\n[[ Step %d ]]\n",nIter);
		step();
	}
*/
	//goDX10();

	//delete [] positionBuffer;
	//delete [] velocityBuffer;

	return err;//AP2012
}

int nIter=0;

extern int frames_counter;

void simulation_step ()
{
	nIter++;
	printf("\n[[ Step %d ]]",nIter);
	//printf("\n[[ Step %d ]], OpenGL_frames: %d",nIter,frames_counter);
	step(nIter);
	//printf("\nsimulation_step:%d\n",clock() - c);
}

void simulation_stop ()
{
	delete [] positionBuffer;
	delete [] velocityBuffer;
	delete [] densityBuffer;
	delete [] neighborMapBuffer;
	delete [] particleIndexBuffer;
	delete [] gridNextNonEmptyCellBuffer;
	delete [] elasticConnectionsDataBuffer;
}
