
// sphFluidDemo.cl  A. Heirich   11/8/2010
//
// Equations referenced here are from:
// "Particle-based fluid simulation for interactive applications", Muller, Charypar & Gross,
// Eurographics/SIGGRAPH Symposium on Computer Animation (2003).

#include "sph.h"
//#define PARTICLE_COUNT ( 32 * 1024 )//( 32 * 1024 )
//#define NEIGHBOR_COUNT 32

#define NO_PARTICLE_ID -1
#define NO_CELL_ID -1
#define NO_DISTANCE -1.0f

#define POSITION_CELL_ID( i ) i.w

#define PI_CELL_ID( name ) name.x
#define PI_SERIAL_ID( name ) name.y

#define NEIGHBOR_MAP_ID( nm ) nm.x
#define NEIGHBOR_MAP_DISTANCE( nm ) nm.y

#define RHO( i ) i.x
#define RHO_INV( i ) i.y
//#define P( i ) i.z

#define DIVIDE( a, b ) native_divide( a, b )
#define SQRT( x ) native_sqrt( x )
#define DOT( a, b ) dot( a, b )


#if 1
#define SELECT( A, B, C ) select( A, B, (C) * 0xffffffff )
#else
#define SELECT( A, B, C ) C ? B : A
#endif

//#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void clearBuffers(
						   __global float2 * neighborMap
						   )
{
	int id = get_global_id( 0 );
	__global float4 * nm = (__global float4 *)neighborMap;
	int outIdx = ( id * NEIGHBOR_COUNT ) >> 1;//int4 versus int2 addressing
	float4 fdata = (float4)( -1, -1, -1, -1 );

	nm[ outIdx++ ] = fdata;
	nm[ outIdx++ ] = fdata;
	nm[ outIdx++ ] = fdata;
	nm[ outIdx++ ] = fdata;
	nm[ outIdx++ ] = fdata;
	nm[ outIdx++ ] = fdata;
	nm[ outIdx++ ] = fdata;
	nm[ outIdx++ ] = fdata;
	nm[ outIdx++ ] = fdata;
	nm[ outIdx++ ] = fdata;
	nm[ outIdx++ ] = fdata;
	nm[ outIdx++ ] = fdata;
	nm[ outIdx++ ] = fdata;
	nm[ outIdx++ ] = fdata;
	nm[ outIdx++ ] = fdata;
	nm[ outIdx++ ] = fdata;
}





// Gradient of equation 21.  Vector result.

float4 gradWspiky(
				  float r,
				  float h,
				  float gradWspikyCoefficient,
				  float4 position_i,
				  float4 position_j,
				  float simulationScale
				  )
{
	float4 rVec = position_i - position_j; rVec.w = 0.f;
	float4 scaledVec = rVec * simulationScale;
	if(r>=0.f)
	scaledVec /= r; 
	//rVec.w = 0.0f;  // what for? never used any longer
	float x = h - r; // r & h are scaled, as I see at least in basic SPH mode
	float4 result = 0.f;
	if(x>=0.f) result = (x) * (x) * scaledVec * gradWspikyCoefficient;
	return result;
}



float4 contributeGradP(
					   int id,
					   int neighborParticleId,						
					   float p_i,
					   float p_j,
					   float rho_j_inv,
					   float4 position_i,
					   __global float * pressure,
					   __global float * rho,
					   __global float4 * sortedPosition,
					   float r,
					   float mass,
					   float h,
					   float gradWspikyCoefficient,
					   float simulationScale
					   )
{
	// Following Muller Charypar and Gross ( 2003 ) Particle-Based Fluid Simulation for Interactive Applications
	// -grad p_i = - sum_j m_j ( p_i + p_j ) / ( 2 rho_j ) grad Wspiky
	// Equation 10.
	// AP2012: seem to be en error here: for correct acceleration dimension [m/s^2]  rho in denominator should be in 2nd degree
	// mass*pressure*gradW/rho^2 = (kg*Neuton/m^2)*(1/m^4)/(kg/m^3)^2 = (kg*kg*m/(m*s^2))*(1/m^4)/(kg/m^3)^2 = 1/(m*s^2)*(m^6/m^4) = m/s^2 [OK]
	// but here rho is only in 1st degree, not 2nd one
	// ===
	// Finally everything is ok here. gradP is further multiplied by rho_inv which makes acceleration dimensions correct


	float4 neighborPosition;
	neighborPosition = sortedPosition[ neighborParticleId ];
	float4 smoothingKernel = gradWspiky( r, h, gradWspikyCoefficient, position_i, neighborPosition, simulationScale );
	float4 result = mass * ( p_i + p_j ) * 0.5f * rho_j_inv * smoothingKernel;
	return result;
}


// Laplacian of equation 22.  Scalar result.

float del2Wviscosity(
					 float r,
					 float h,
					 float del2WviscosityCoefficient
					 )
{
	// equation 22
	float result = 0.f;
	if((r>=0)&&(r<h)) result = ( h - r ) * del2WviscosityCoefficient;
	return result;
}



float4 contributeDel2V(
					   int id,
					   float4 v_i,
					   int neighborParticleId,
					   __global float4 * sortedVelocity,
					   float rho_j_inv,
					   float r,
					   float mass,
					   float h,
					   float del2WviscosityCoefficient
					   )
{
	// mu del^2 v = mu sum_j m_j ( v_j - v_i ) / rho_j del^2 Wviscosity
	// Equation 14.
	float4 v_j = sortedVelocity[ neighborParticleId ];
	float4 d = v_j - v_i;
	d.w = 0.f;
	float4 result = mass * d * rho_j_inv * del2Wviscosity( r, h, del2WviscosityCoefficient );
	return result;
}



__kernel void computeAcceleration(
								  __global float2 * neighborMap,
								  __global float * pressure,
								  __global float * rho,
								  __global float * rhoInv,
								  __global float4 * sortedPosition,
								  __global float4 * sortedVelocity,
								  __global uint * particleIndexBack,
								  float CFLLimit,
								  float del2WviscosityCoefficient,
								  float gradWspikyCoefficient,
								  float h,
								  float mass,
								  float mu,
								  float simulationScale,
								  __global float4 * acceleration,
								  float rho0
								  )
{
	int id = get_global_id( 0 );
	id = particleIndexBack[id];//track selected particle (indices are not mixed anymore)

	int idk = id * NEIGHBOR_COUNT;
	float hScaled = h * simulationScale;

	float4 position_i = sortedPosition[ id ];
	float4 velocity_i = sortedVelocity[ id ];

	float p_i = pressure[ id ]; 
	float rho_i_inv = rhoInv[ id ];
	float4 result = (float4)( 0.0f, 0.0f, 0.0f, 0.0f );

	float4 gradP = (float4)( 0.0f, 0.0f, 0.0f, 0.0f );
	float4 del2V = (float4)( 0.0f, 0.0f, 0.0f, 0.0f );
	float2 nm;

	NEIGHBOR_MAP_ID( nm ) = id;
	NEIGHBOR_MAP_DISTANCE( nm ) = 0.0f;

	// basic SPH
						
	int j = 0;
	bool loop;
	do{
		nm = neighborMap[ idk + j ];
		int neighborParticleId = NEIGHBOR_MAP_ID( nm );
		bool isNeighbor = ( neighborParticleId != NO_PARTICLE_ID );
		if( isNeighbor ){
			float p_j = pressure[ neighborParticleId ];
			float rho_j_inv = rhoInv[ neighborParticleId ];
			float r = NEIGHBOR_MAP_DISTANCE( nm ); // r is scaled here
			float4 dgradP = contributeGradP( id, neighborParticleId, p_i, p_j, rho_j_inv,
				position_i, pressure, rho, sortedPosition, r, mass, hScaled,
				gradWspikyCoefficient, simulationScale );
			gradP += dgradP;
			
			float4 ddel2V = contributeDel2V( id, velocity_i, neighborParticleId,
				sortedVelocity, rho_j_inv, r, mass, hScaled, del2WviscosityCoefficient );
			del2V += ddel2V;
		}
		loop = ( ++j < NEIGHBOR_COUNT );
	}while( loop );

	result = rho_i_inv * ( mu * del2V - gradP );

	// Check CFL condition // As far as I know it is not required in case of PCISPH [proof?]
	float magnitude = result.x * result.x + result.y * result.y + result.z * result.z;
	bool tooBig = ( magnitude > CFLLimit * CFLLimit );
	float sqrtMagnitude = SQRT( magnitude );

	if(sqrtMagnitude!=0)
	{
		float scale = CFLLimit / sqrtMagnitude;
		result = SELECT( result, result * scale, (uint4)tooBig );
	}

	result.w = 0.0f;
	acceleration[ id ] = result; 
}





// Mueller et al equation 3.  Scalar result.

float Wpoly6(
			 float rSquared,
			 float hSquared,
			 float Wpoly6Coefficient
			 )
{
	float x = hSquared - rSquared;
	float result = 0.f;
	if(x>0) result = x * x * x * Wpoly6Coefficient;
	return result;
}



float densityContribution(
						  int idx,
						  int i,
						  __global float2 * neighborMap,
						  float mass,
						  float hSquared,
						  float Wpoly6Coefficient
						  )
{
	float2 nm = neighborMap[ idx + i ];
	int neighborParticleId = NEIGHBOR_MAP_ID( nm );
	float r = NEIGHBOR_MAP_DISTANCE( nm );	
	float smoothingKernel = Wpoly6( r*r, hSquared, Wpoly6Coefficient );
	float result = SELECT( smoothingKernel, 0.0f, ( neighborParticleId == NO_PARTICLE_ID ) );
	return result;
}


float densityContributionPCISPH(
						  int idx,
						  int i,
						  float4 ri,
						  __global float4 * sortedPosition,
						  __global float2 * neighborMap,
						  float mass,
						  float hSquared,//already scaled
						  float simulationScale,
						  float Wpoly6Coefficient
						  )
{
	float2 nm = neighborMap[ idx + i ];
	int neighborParticleId = NEIGHBOR_MAP_ID( nm );
	//float4 _ij = ri-sortedPosition[neighborParticleId];
	float4 rij = ri-sortedPosition[PARTICLE_COUNT+neighborParticleId];
	//float r = NEIGHBOR_MAP_DISTANCE( nm );	
	//hSquared is already scaled 
	//do it for rSquared too:
	float rSquared = (rij.x*rij.x+rij.y*rij.y+rij.z*rij.z)*simulationScale*simulationScale;
	if(rSquared>=hSquared)
	{
		return 0.f;
	}
	float smoothingKernel = Wpoly6( rSquared, hSquared, Wpoly6Coefficient );
	float result = SELECT( smoothingKernel, 0.0f, ( neighborParticleId == NO_PARTICLE_ID ) );
	return result;
}


__kernel void computeDensityPressure(
									 __global float2 * neighborMap,
									 float Wpoly6Coefficient,
									 float gradWspikyCoefficient,
									 float h,
									 float mass,
									 float rho0,
									 float simulationScale,
									 float stiffness,
									 __global float4 * sortedPosition,
									 __global float * pressure,
									 __global float * rho,
									 __global float * rhoInv,
									 __global uint * particleIndexBack,
									 float delta									 )
{
	int id = get_global_id( 0 );
	id = particleIndexBack[id];//track selected particle (indices are not shuffled anymore)

	int idx = id * NEIGHBOR_COUNT;
	float density = 0.0f;
	float hScaled = h * simulationScale;
	float hSquared = hScaled * hScaled;

	int nc=0;//neighbor counter

	while( nc<32 )// gather density contribution from all neighbors (if they exist)
	{
		if( NEIGHBOR_MAP_ID( neighborMap[ idx + nc ] ) != NO_PARTICLE_ID )
		density += densityContribution( idx,  nc, neighborMap, mass, hSquared, Wpoly6Coefficient );
		nc++;
	}


	density *= mass; // since all particles are same fluid type, factor this out to here

	rho[ id ] = density; 
	rhoInv[ id ] = SELECT( 1.0f, DIVIDE( 1.0f, density ), ( density > 0.0f ) );

	float drho = density - rho0; // rho0 is resting density
	float k = stiffness;// here k=0.75; in Chao Fang code k=2.0 (gas constant)
	float p = k * drho; // equation 12
	pressure[ id ] = p; 
}




int searchCell( 
			   int cellId,
			   int deltaX,
			   int deltaY,
			   int deltaZ,
			   int gridCellsX, 
			   int gridCellsY, 
			   int gridCellsZ,
			   int gridCellCount
			   )
{
	int dx = deltaX;
	int dy = deltaY * gridCellsX;
	int dz = deltaZ * gridCellsX * gridCellsY;
	int newCellId = cellId + dx + dy + dz;
	newCellId = SELECT( newCellId, newCellId + gridCellCount, ( newCellId < 0 ) );
	newCellId = SELECT( newCellId, newCellId - gridCellCount, ( newCellId >= gridCellCount ) );
	return newCellId;
}


#define FOUND_NO_NEIGHBOR 0
#define FOUND_ONE_NEIGHBOR 1

/*
int considerParticle(
					 int cellId,
					 int neighborParticleId,
					 float4 position_,
					 int myParticleId,
					 __global float4 * sortedPosition,
					 __global uint * gridCellIndex,
					 __global float2 * neighborMap, 
					 int myOffset,
					 float h,
					 float simulationScale
					 )
{
	float4 neighborPosition;
	neighborPosition = sortedPosition[ neighborParticleId ];
	float4 d = position_ - neighborPosition;
	d.w = 0.0f;
	float distanceSquared = DOT( d, d );
	float distance = SQRT( distanceSquared );
	bool tooFarAway = ( distance > h );
	bool neighborIsMe = ( neighborParticleId == myParticleId );
	if( tooFarAway || neighborIsMe ){
		return FOUND_NO_NEIGHBOR;
	}

	float scaledDistance = distance * simulationScale;
	float2 myMapEntry;
	NEIGHBOR_MAP_ID( myMapEntry ) = neighborParticleId;
	NEIGHBOR_MAP_DISTANCE( myMapEntry ) = scaledDistance;
	int myIdx = myParticleId * NEIGHBOR_COUNT + myOffset;
	neighborMap[ myIdx ] = myMapEntry;
	return FOUND_ONE_NEIGHBOR;
}
*/



uint myRandom( 
			  uint prior,
			  int maxParticles /*didn't use this variable*/
			  )
{
	unsigned long int m = PARTICLE_COUNT;//generator period, assume power of 2
	unsigned long int a = 1664525;
	unsigned long int c = 1013904223;
	uint result = (uint)(( a * prior + c ) % m );
	return result;
}


#define radius_segments 30


int searchForNeighbors( 
					   int searchCell_, 
					   __global uint * gridCellIndex, 
					   float4 position_, 
					   int myParticleId, 
					   __global float4 * sortedPosition,
					   __global float2 * neighborMap,
					   int spaceLeft,
					   float h,
					   float simulationScale,
					   int mode,
					   int * radius_distrib,
					   float r_thr
					   )
{
	int baseParticleId = gridCellIndex[ searchCell_ ];
	int nextParticleId = gridCellIndex[ searchCell_ + 1 ];
	int particleCountThisCell = nextParticleId - baseParticleId;
	int potentialNeighbors = particleCountThisCell;
	int foundCount = 0;
	bool loop = true;
	int i = 0,j;
	float _distance,_distanceSquared;
	float r_thr_Squared = r_thr*r_thr;
	float2 neighbor_data;
	int neighborParticleId;
	int myOffset;
	if(spaceLeft>0)
		while( i < particleCountThisCell ){

			neighborParticleId = baseParticleId + i;
			
			//if(myParticleId == neighborParticleId) continue;

			if(myParticleId != neighborParticleId)
			{
				float4 d = position_ - sortedPosition[ neighborParticleId ];
				d.w = 0.0f;
				_distanceSquared = DOT( d, d );
				if( _distanceSquared <= r_thr_Squared )
				{
					_distance = SQRT( _distanceSquared );
					j = (int)(_distance*radius_segments/h);
					if(j<radius_segments) radius_distrib[j]++; 

					// searchForNeighbors runs twice
					// first time with mode = 0, to build distribution
					// and 2nd time with mode = 1, to select 32 nearest neighbors
					if(mode)
					{
						myOffset = NEIGHBOR_COUNT - spaceLeft + foundCount;
						neighbor_data.x = neighborParticleId;
						neighbor_data.y = _distance * simulationScale; // scaled, OK
						neighborMap[ myParticleId*NEIGHBOR_COUNT + myOffset ] = neighbor_data;
						foundCount++;
					}
				}
			
			}

			i++;
			
		}//while

	return foundCount;
}


int4 cellFactors( 
				 float4 position,
				 float xmin,
				 float ymin,
				 float zmin,
				 float hashGridCellSizeInv
				 )
{
	//xmin, ymin, zmin ����� �� ��������������
	int4 result;
	result.x = (int)( position.x *  hashGridCellSizeInv );
	result.y = (int)( position.y *  hashGridCellSizeInv );
	result.z = (int)( position.z *  hashGridCellSizeInv );
	return result;
}





__kernel void findNeighbors(
							__global uint * gridCellIndexFixedUp,
							__global float4 * sortedPosition,
							int gridCellCount,
							int gridCellsX,
							int gridCellsY,
							int gridCellsZ,
							float h,
							float hashGridCellSize,
							float hashGridCellSizeInv,
							float simulationScale,
							float xmin,
							float ymin,
							float zmin,
							__global float2 * neighborMap
							)
{
	__global uint * gridCellIndex = gridCellIndexFixedUp;
	int id = get_global_id( 0 );
	float4 position_ = sortedPosition[ id ];
	int myCellId = (int)POSITION_CELL_ID( position_ ) & 0xffff;// truncate to low 16 bits
	int searchCell_;
	int foundCount = 0;
	int mode = 0;
	int distrib_sum = 0;
	int radius_distrib[radius_segments];
	int i=0,j;
	float r_thr = h;
	
	while( i<radius_segments )
	{
		radius_distrib[i]=0;
		i++;
	}
	
	while( mode<2 )
	{

	
		searchCell_ = myCellId;
		foundCount += searchForNeighbors( searchCell_, gridCellIndex, position_, 
			id, sortedPosition, neighborMap, NEIGHBOR_COUNT - foundCount, 
			h, simulationScale, mode, radius_distrib, r_thr );


		// p is the current particle position within the bounds of the hash grid
		float4 p;
		float4 p0 = (float4)( xmin, ymin, zmin, 0.0f );//� ��� ������� xmin, ymin, zmin -> ���� -> �������� ����� ������ ��� �����
		p = position_ - p0;

		// cf is the min,min,min corner of the current cell
		int4 cellFactors_ = cellFactors( position_, xmin, ymin, zmin, hashGridCellSizeInv );
		float4 cf;
		cf.x = cellFactors_.x * hashGridCellSize;
		cf.y = cellFactors_.y * hashGridCellSize;
		cf.z = cellFactors_.z * hashGridCellSize;

		// lo.A is true if the current position is in the low half of the cell for dimension A
		int4 lo;
		lo = (( p - cf ) < h );

		int4 delta;
		int4 one = (int4)( 1, 1, 1, 1 );
		delta = one + 2 * lo;

		// search up to 8 surrounding cells
		
	
		searchCell_ = searchCell( myCellId, delta.x, 0, 0, gridCellsX, gridCellsY, gridCellsZ, gridCellCount );
		foundCount += searchForNeighbors( searchCell_, gridCellIndex, position_, 
			id, sortedPosition, neighborMap, NEIGHBOR_COUNT - foundCount, 
			h, simulationScale, mode, radius_distrib, r_thr  );

		searchCell_ = searchCell( myCellId, 0, delta.y, 0, gridCellsX, gridCellsY, gridCellsZ, gridCellCount );
		foundCount += searchForNeighbors( searchCell_, gridCellIndex, position_, 
			id, sortedPosition, neighborMap, NEIGHBOR_COUNT - foundCount, 
			h, simulationScale, mode, radius_distrib, r_thr  );

		searchCell_ = searchCell( myCellId, 0, 0, delta.z, gridCellsX, gridCellsY, gridCellsZ, gridCellCount );
		foundCount += searchForNeighbors( searchCell_, gridCellIndex, position_, 
			id, sortedPosition, neighborMap, NEIGHBOR_COUNT - foundCount, 
			h, simulationScale, mode, radius_distrib, r_thr  );

		searchCell_ = searchCell( myCellId, delta.x, delta.y, 0, gridCellsX, gridCellsY, gridCellsZ, gridCellCount );
		foundCount += searchForNeighbors( searchCell_, gridCellIndex, position_, 
			id, sortedPosition, neighborMap, NEIGHBOR_COUNT - foundCount, 
			h, simulationScale, mode, radius_distrib, r_thr  );

		searchCell_ = searchCell( myCellId, delta.x, 0, delta.z, gridCellsX, gridCellsY, gridCellsZ, gridCellCount );
		foundCount += searchForNeighbors( searchCell_, gridCellIndex, position_, 
			id, sortedPosition, neighborMap, NEIGHBOR_COUNT - foundCount, 
			h, simulationScale, mode, radius_distrib, r_thr  );

		searchCell_ = searchCell( myCellId, 0, delta.y, delta.z, gridCellsX, gridCellsY, gridCellsZ, gridCellCount );
		foundCount += searchForNeighbors( searchCell_, gridCellIndex, position_, 
			id, sortedPosition, neighborMap, NEIGHBOR_COUNT - foundCount, 
			h, simulationScale, mode, radius_distrib, r_thr  );

		searchCell_ = searchCell( myCellId, delta.x, delta.y, delta.z, gridCellsX, gridCellsY, gridCellsZ, gridCellCount );
		foundCount += searchForNeighbors( searchCell_, gridCellIndex, position_, 
			id, sortedPosition, neighborMap, NEIGHBOR_COUNT - foundCount, 
			h, simulationScale, mode, radius_distrib, r_thr );

		if(mode==0)
		{
			j=0;

			
			while(j<radius_segments)
			{
				distrib_sum += radius_distrib[j];
				if(distrib_sum==NEIGHBOR_COUNT) break;
				if(distrib_sum> NEIGHBOR_COUNT) { j--; break; }
				j++;
			}

			r_thr = (j+1)*h/radius_segments;

		}

		mode++;
	}

}


int cellId( 
		   int4 cellFactors_,
		   int gridCellsX,
		   int gridCellsY,
		   int gridCellsZ//don't use
		   )
{
	int cellId_ = cellFactors_.x + cellFactors_.y * gridCellsX
		+ cellFactors_.z * gridCellsX * gridCellsY;
	return cellId_;
}



__kernel void hashParticles(
							__global float4 * position,
							int gridCellsX,
							int gridCellsY,
							int gridCellsZ,
							float hashGridCellSizeInv,
							float xmin,
							float ymin,
							float zmin,
							__global uint2 * particleIndex
							)
{
	int id = get_global_id( 0 );
	if( id >= PARTICLE_COUNT ){
		uint2 result;
		int gridCellCount = gridCellsX * gridCellsY * gridCellsZ;
		PI_CELL_ID( result ) = gridCellCount + 1;
		PI_SERIAL_ID( result ) = id;
		particleIndex[ id ] = result;
		return;
	}

	//position[id].w = 0.f;
	//position[PARTICLE_COUNT+id].w = 0.f;

	float4 _position = position[ id ];
	int4 cellFactors_ = cellFactors( _position, xmin, ymin, zmin, hashGridCellSizeInv ); 
	//int cellId_ = cellId( cellFactors_, gridCellsX, gridCellsY, gridCellsZ );
	int cellId_ = cellId( cellFactors_, gridCellsX, gridCellsY, gridCellsZ ) & 0xffff; // truncate to low 16 bits
	uint2 result;
	PI_CELL_ID( result ) = cellId_;
	PI_SERIAL_ID( result ) = id;
	particleIndex[ id ] = result;

}







__kernel void indexPostPass(
							__global uint * gridCellIndex,
							int gridCellCount,
							__global uint * gridCellIndexFixedUp
							)
{
	
	int id = get_global_id( 0 );
	if( id <= gridCellCount ){
		int idx = id;
		int cellId = NO_CELL_ID;
		bool loop;
		do{
			cellId = gridCellIndex[ idx++ ];
			//loop = cellId == NO_CELL_ID && idx <= gridCellCount;
			loop = (cellId == NO_CELL_ID) && (idx <= gridCellCount);
		}while( loop );
		gridCellIndexFixedUp[ id ] = cellId;
	}
}



__kernel void indexx(
					 __global uint2 * particleIndex,
					 int gridCellCount,
					 __global uint * gridCellIndex
					 )
{
	//fill up gridCellIndex
	int id = get_global_id( 0 );
	if( id > gridCellCount  ){
		return;
	}

	if( id == gridCellCount ){
		// add the nth+1 index value
		gridCellIndex[ id ] = PARTICLE_COUNT;
		return;
	}		
	if( id == 0 ){
		gridCellIndex[ id ] = 0;
		return;
	}

	// binary search for the starting position in sortedParticleIndex
	int low = 0;
	int high = PARTICLE_COUNT - 1;
	bool converged = false;

	int cellIndex = NO_PARTICLE_ID;
	while( !converged ){
		if( low > high ){
			converged = true;
			cellIndex = NO_PARTICLE_ID;
			continue;
		}

		int idx = ( high - low ) * 0.5f + low;
		uint2 sample = particleIndex[ idx ];
		int sampleCellId = PI_CELL_ID( sample );
		bool isHigh = ( sampleCellId > id );
		high = SELECT( high, idx - 1, isHigh );
		bool isLow = ( sampleCellId < id );
		low = SELECT( low, idx + 1, isLow );
		bool isMiddle = !( isHigh || isLow );

		uint2 zero2 = (uint2)( 0, 0 );
		uint2 sampleMinus1;
		int sampleM1CellId = 0;
		bool zeroCase = ( idx == 0 && isMiddle ); //it means that we in middle or 
		sampleMinus1 = SELECT( (uint2)particleIndex[ idx - 1 ], zero2, (uint2)zeroCase );//if we in middle this return zero2 else (uint2)particleIndex[ idx - 1 ]
		sampleM1CellId = SELECT( PI_CELL_ID( sampleMinus1 ), (uint)(-1), zeroCase );//if we in middle this return (uint)(-1) else sampleMinus1.x (index of cell)
		bool convergedCondition = isMiddle && ( zeroCase || sampleM1CellId < sampleCellId );
		converged = convergedCondition;
		cellIndex = SELECT( cellIndex, idx, convergedCondition );
		high = SELECT( high, idx - 1, ( isMiddle && !convergedCondition ) );
	}//while

	gridCellIndex[ id ] = cellIndex;//
}



void handleBoundaryConditions(
							  float4 position,
							  float4 * newVelocity,
							  float timeStep,
							  float4 * newPosition,
							  float xmin,
							  float xmax,
							  float ymin,
							  float ymax,
							  float zmin,
							  float zmax,
							  float damping
							  )
{
	if( (*newPosition).x < xmin ){
		float intersectionDistance = -position.x / (*newVelocity).x;
		float4 intersection = position + intersectionDistance * *newVelocity;
		float4 normal = (float4)( 1, 0, 0, 0 );
		float4 reflection = *newVelocity - 2.0f * DOT( *newVelocity, normal ) * normal;
		float remaining = timeStep - intersectionDistance;
		position = intersection;
		*newVelocity = reflection;
		*newPosition = intersection + remaining * damping * reflection;
	}
	else if( (*newPosition).x > xmax ){
		float intersectionDistance = ( xmax - position.x ) / (*newVelocity).x;
		float4 intersection = position + intersectionDistance * *newVelocity;
		float4 normal = (float4)( -1, 0, 0, 0 );
		float4 reflection = *newVelocity - 2.0f * DOT( *newVelocity, normal ) * normal;
		float remaining = timeStep - intersectionDistance;
		position = intersection;
		*newVelocity = reflection;
		*newPosition = intersection + remaining * damping * reflection;
	}

	if( (*newPosition).y < ymin ){
		float intersectionDistance = -position.y / (*newVelocity).y;
		float4 intersection = position + intersectionDistance * *newVelocity;
		float4 normal = (float4)( 0, 1, 0, 0 );
		float4 reflection = *newVelocity - 2.0f * DOT( *newVelocity, normal ) * normal;
		float remaining = timeStep - intersectionDistance;
		position = intersection;
		*newVelocity = reflection;
		*newPosition = intersection + remaining * damping * reflection;
	}
	else if( (*newPosition).y > ymax ){
		float intersectionDistance = ( ymax - position.y ) / (*newVelocity).y;
		float4 intersection = position + intersectionDistance * *newVelocity;
		float4 normal = (float4)( 0, -1, 0, 0 );
		float4 reflection = *newVelocity - 2.0f * DOT( *newVelocity, normal ) * normal;
		float remaining = timeStep - intersectionDistance;
		position = intersection;
		*newVelocity = reflection;
		*newPosition = intersection + remaining * damping * reflection;
	}

	if( (*newPosition).z < zmin ){
		float intersectionDistance = -position.z / (*newVelocity).z;
		float4 intersection = position + intersectionDistance * *newVelocity;
		float4 normal = (float4)( 0, 0, 1, 0 );
		float4 reflection = *newVelocity - 2.0f * DOT( *newVelocity, normal ) * normal;
		float remaining = timeStep - intersectionDistance;
		position = intersection;
		*newVelocity = reflection;
		*newPosition = intersection + remaining * damping * reflection;
	}
	else if( (*newPosition).z > zmax ){
		float intersectionDistance = ( zmax - position.z ) / (*newVelocity).z;
		float4 intersection = position + intersectionDistance * *newVelocity;
		float4 normal = (float4)( 0, 0, -1, 0 );
		float4 reflection = *newVelocity - 2.0f * DOT( *newVelocity, normal ) * normal;
		float remaining = timeStep - intersectionDistance;
		position = intersection;
		*newVelocity = reflection;
		*newPosition = intersection + remaining * damping * reflection;
	}

}



__kernel void integrate(
						__global float4 * acceleration,
						__global float4 * sortedPosition,
						__global float4 * sortedVelocity,
						__global uint2 * particleIndex,
						__global uint * particleIndexBack,
						float gravity_x,
						float gravity_y,
						float gravity_z,
						float simulationScaleInv,
						float timeStep,
						float xmin,
						float xmax,
						float ymin,
						float ymax,
						float zmin,
						float zmax,
						float damping,
						__global float4 * position,
						__global float4 * velocity
						)
{
	int id = get_global_id( 0 );
	id = particleIndexBack[id];

	int id_source_particle = PI_SERIAL_ID( particleIndex[id] );

	float4 acceleration_ = acceleration[ id ];
	float4 position_ = sortedPosition[ id ];
	float4 velocity_ = sortedVelocity[ id ];

	// apply external forces
	float4 gravity = (float4)( gravity_x, gravity_y, gravity_z, 0.f );
	acceleration_ += gravity;

	// Semi-implicit Euler integration 
	float4 newVelocity_ = velocity_ + timeStep * acceleration_; //newVelocity_.w = 0.f;
	float posTimeStep = timeStep * simulationScaleInv;			
	float4 newPosition_ = position_ + posTimeStep * newVelocity_; //newPosition_.w = 0.f;

	handleBoundaryConditions( position_, &newVelocity_, posTimeStep, &newPosition_,
		xmin, xmax, ymin, ymax, zmin, zmax, damping );

	//newPosition_.w = 0.f; // homogeneous coordinate for rendering

	velocity[ id_source_particle ] = newVelocity_;
	position[ id_source_particle ] = newPosition_;
}






__kernel void sortPostPass(
						   __global uint2 * particleIndex,
						   __global uint  * particleIndexBack,
						   __global float4 * position,
						   __global float4 * velocity,
						   __global float4 * sortedPosition,
						   __global float4 * sortedVelocity
						   )
{
	int id = get_global_id( 0 );
	uint2 spi = particleIndex[ id ];//contains id of cell and id of particle it has sorted 
	int serialId = PI_SERIAL_ID( spi );//get a particle Index
	int cellId = PI_CELL_ID( spi );//get a cell Index
	float4 position_ = position[ serialId ];//get position by serialId
	POSITION_CELL_ID( position_ ) = (float)cellId;
	float4 velocity_ = velocity[ serialId ];
	sortedVelocity[ id ] = velocity_;//put velocity to sortedVelocity for right order according to particleIndex
	sortedPosition[ id ] = position_;//put position to sortedVelocity for right order according to particleIndex

	particleIndexBack[ serialId ] = id;

/*
	int pib;

	for(int i=0;i<PARTICLE_COUNT;i++)
	{
		pib = particleIndexBuffer[2*i + 1];
		particleIndexBuffer[2*pib + 0] = i;
	}
*/

}



//=================================
// PCI SPH KERNELS BELOW
//=================================

__kernel void pcisph_computeDensity(
									 __global float2 * neighborMap,
									 float Wpoly6Coefficient,
									 float gradWspikyCoefficient,
									 float h,
									 float mass,
									 float rho0,
									 float simulationScale,
									 float stiffness,
									 __global float4 * sortedPosition,
									 __global float * pressure,
									 __global float * rho,
									 __global float * rhoInv,
									 __global uint * particleIndexBack,
									 float delta									 )
{
	int id = get_global_id( 0 );
	id = particleIndexBack[id];//track selected particle (indices are not shuffled anymore)
	int idx = id * NEIGHBOR_COUNT;
	int nc=0;//neighbor counter
	float density = 0.0f;
	float r_ij2;//squared r_ij
	float hScaled = h * simulationScale;//scaled smoothing radius
	float hScaled2 = hScaled*hScaled;//squared scaled smoothing radius
	float hScaled6 = hScaled2*hScaled2*hScaled2;
	float2 nm;
	int real_nc = 0;

	do// gather density contribution from all neighbors (if they exist)
	{
		if( NEIGHBOR_MAP_ID( neighborMap[ idx + nc ] ) != NO_PARTICLE_ID )
		{
			r_ij2= NEIGHBOR_MAP_DISTANCE( neighborMap[ idx + nc ] );	// distance is already scaled here
			r_ij2 *= r_ij2;
			density += (hScaled2-r_ij2)*(hScaled2-r_ij2)*(hScaled2-r_ij2);
			real_nc++;
		}

	}while( ++nc < NEIGHBOR_COUNT );
	
	//if(density==0.f) density = hScaled2*hScaled2*hScaled2;
	if(density<hScaled6) density = hScaled6;

	density *= mass*Wpoly6Coefficient; // since all particles are same fluid type, factor this out to here
	rho[ id ] = density; 		
	rhoInv[ id ] = real_nc; 		
}
/*
float4 calcBoundaryForceAcceleration(float4 position,
									 float4 velocity,
									 float xmin,
									 float xmax,
									 float ymin,
									 float ymax,
									 float zmin,
									 float zmax,
									 float h,
									 float simulationScale)
{
    float4 acceleration = (float4)( 0.0f, 0.0f, 0.0f, 0.0f );
	float hScaled = h*simulationScale;
    float dist_iw; //i-th particle to wall distance
    float diff;
    float boundaryStiffness = 2000.0f;
    float boundaryDampening = 256.0f;


	float value = 32; //value
    
	//-----------------------------------------------
	if ( ( diff = (position[0]-xmin)*simulationScale ) < hScaled)
    {
        float4 norm =  (float4)( 1.f, 0.f, 0.f, 0.f );
        float adj = boundartStiffness * diff - boundaryDampening * DOT(norm, velocity);
        acceleration +=  norm * adj;
    }

	if ( ( diff = (xmax-position[0])*simulationScale ) < hScaled)
    {
        float4 norm =  (float4)(-1.f, 0.f, 0.f, 0.f );
        float adj = boundartStiffness * diff - boundaryDampening * DOT(norm, velocity);
        acceleration +=  norm * adj;
    }
	//-----------------------------------------------
	if ( ( diff = (position[1]-ymin)*simulationScale ) < hScaled)
    {
        float4 norm =  (float4)( 0.f, 1.f, 0.f, 0.f );
        float adj = boundartStiffness * diff - boundaryDampening * DOT(norm, velocity);
        acceleration +=  norm * adj;
    }

	if ( ( diff = (ymax-position[1])*simulationScale ) < hScaled)
    {
        float4 norm =  (float4)( 0.f,-1.f, 0.f, 0.f );
        float adj = boundartStiffness * diff - boundaryDampening * DOT(norm, velocity);
        acceleration +=  norm * adj;
    }
	//-----------------------------------------------
	if ( ( diff = (position[2]-zmin)*simulationScale ) < hScaled)
    {
        float4 norm =  (float4)( 0.f, 0.f, 1.f, 0.f );
        float adj = boundartStiffness * diff - boundaryDampening * DOT(norm, velocity);
        acceleration +=  norm * adj;
    }

	if ( ( diff = (zmax-position[2])*simulationScale ) < hScaled)
    {
        float4 norm =  (float4)( 0.f, 0.f,-1.f, 0.f );
        float adj = boundartStiffness * diff - boundaryDampening * DOT(norm, velocity);
        acceleration +=  norm * adj;
    }
	//-----------------------------------------------

    return acceleration;
}
*/

__kernel void pcisph_computeForcesAndInitPressure(
								  __global float2 * neighborMap,
								  __global float * rho,
								  __global float  * pressure,
								  __global float4 * sortedPosition,
								  __global float4 * sortedVelocity,
								  __global float4 * acceleration,
								  __global uint * particleIndexBack,
								  float gradWspikyCoefficient,
								  float del2WviscosityCoefficient,
								  float h,
								  float mass,
								  float mu,
								  float simulationScale,
								  float gravity_x,
								  float gravity_y,
								  float gravity_z
								  )
{
	int id = get_global_id( 0 );
	id = particleIndexBack[id];//track selected particle (indices are not shuffled anymore)

	int idx = id * NEIGHBOR_COUNT;
	float hScaled = h * simulationScale;

	float4 acceleration_i;// = (float4)( 0.0f, 0.0f, 0.0f, 0.0f );
	float2 nm;
	float r_ij;
	int nc = 0;//neighbor counter
	int jd;
	float4 sum = (float4)( 0.0f, 0.0f, 0.0f, 0.0f );
	float4 vi,vj;
	float rho_i,rho_j;

	do{
		if( (jd = NEIGHBOR_MAP_ID(neighborMap[ idx + nc])) != NO_PARTICLE_ID )
		{
			r_ij = NEIGHBOR_MAP_DISTANCE( neighborMap[ idx + nc] );

			if(r_ij<hScaled)
			{
				rho_i = rho[id];
				rho_j = rho[jd];
				vi = sortedVelocity[id];
				vj = sortedVelocity[jd];
				sum += (sortedVelocity[jd]-sortedVelocity[id])*(hScaled-r_ij)/rho[jd];
			}
		}
		
	}while(  ++nc < NEIGHBOR_COUNT );

	float viscosity = 0.3f;//0.1f

	sum *= mass*viscosity*del2WviscosityCoefficient/rho[id];

	// apply external forces
	acceleration_i = sum;
	//acceleration_i = (float4)( 0.0f, 0.0f, 0.0f, 0.0f );//sum;
	

	//acceleration_i += calcBoundaryForceAcceleration(sortedPosition[id],sortedVelocity[id],xmin,xmax,ymin,ymax,zmin,zmax,h,simulationScale);
	
	acceleration_i += (float4)( gravity_x, gravity_y, gravity_z, 0.0f );

	acceleration[ id ] = acceleration_i; 
	// 1st half of 'acceleration' array is used to store acceleration corresponding to gravity, visc. force etc.
	acceleration[ PARTICLE_COUNT+id ] = (float4)(0.0f, 0.0f, 0.0f, 0.0f );
	// 2nd half of 'acceleration' array is used to store pressure force

	pressure[id] = 0.f;//initialize pressure with 0

}


__kernel void pcisph_predictPositions(
						__global float4 * acceleration,
						__global float4 * sortedPosition,
						__global float4 * sortedVelocity,
						__global uint2 * particleIndex,
						__global uint * particleIndexBack,
						float gravity_x,
						float gravity_y,
						float gravity_z,
						float simulationScaleInv,
						float timeStep,
						float xmin,
						float xmax,
						float ymin,
						float ymax,
						float zmin,
						float zmax,
						float damping,
						__global float4 * position,
						__global float4 * velocity
						)
{
	int id = get_global_id( 0 );
	id = particleIndexBack[id];

	float4 acceleration_ = acceleration[ id ] + acceleration[ PARTICLE_COUNT+id ];
	float4 position_ = sortedPosition[ id ];
	float4 velocity_ = sortedVelocity[ id ];

	// Semi-implicit Euler integration 
	float4 newVelocity_ = velocity_ + timeStep * acceleration_; //newVelocity_.w = 0.f;
	float posTimeStep = timeStep * simulationScaleInv;			
	float4 newPosition_ = position_ + posTimeStep * newVelocity_; //newPosition_.w = 0.f;


	handleBoundaryConditions( position_, &newVelocity_, posTimeStep, &newPosition_,
		xmin, xmax, ymin, ymax, zmin, zmax, damping );

	//sortedVelocity[id] = newVelocity_;// sorted position, as well as velocity, 
	sortedPosition[PARTICLE_COUNT+id] = newPosition_;// in current version sortedPosition array has double size, 
													 // PARTICLE_COUNT*2, to store both x(t) and x*(t+1)
}


__kernel void pcisph_predictDensity(
									 __global float2 * neighborMap,
									 __global uint * particleIndexBack,
									 float Wpoly6Coefficient,
									 float gradWspikyCoefficient,
									 float h,
									 float mass,
									 float rho0,
									 float simulationScale,
									 float stiffness,
									 __global float4 * sortedPosition,
									 __global float * pressure,
									 __global float * rho,
									 __global float * rhoInv,
									 float delta
									 )
{
	int id = get_global_id( 0 );
	id = particleIndexBack[id];//track selected particle (indices are not shuffled anymore)
	int idx = id * NEIGHBOR_COUNT;
	int nc=0;//neighbor counter
	float density = 0.0f;
	float4 r_ij;
	float r_ij2;//squared r_ij
	float hScaled = h * simulationScale;//scaled smoothing radius
	float hScaled2 = hScaled*hScaled;//squared scaled smoothing radius
	float hScaled6 = hScaled2*hScaled2*hScaled2;
	//float2 nm;
	int jd;

	do// gather density contribution from all neighbors (if they exist)
	{
		if( (jd = NEIGHBOR_MAP_ID( neighborMap[ idx + nc ])) != NO_PARTICLE_ID )
		{
			r_ij = sortedPosition[PARTICLE_COUNT+id]-sortedPosition[PARTICLE_COUNT+jd];
			r_ij2 = (r_ij.x*r_ij.x+r_ij.y*r_ij.y+r_ij.z*r_ij.z)*simulationScale*simulationScale;

			if(r_ij2<hScaled2)
			{
				density += (hScaled2-r_ij2)*(hScaled2-r_ij2)*(hScaled2-r_ij2);
			}
		}

	}while( ++nc < NEIGHBOR_COUNT );
	
	//if(density==0.f) 
	if(density<hScaled6)
	{
		//density += hScaled6;
		density = hScaled6;
	}


	density *= mass*Wpoly6Coefficient; // since all particles are same fluid type, factor this out to here
	rho[ PARTICLE_COUNT+id ] = density; 
}


__kernel void pcisph_correctPressure(
									 __global float2 * neighborMap,
									  __global uint * particleIndexBack,
									 float Wpoly6Coefficient,
									 float gradWspikyCoefficient,
									 float h,
									 float mass,
									 float rho0,
									 float simulationScale,
									 float stiffness,
									 __global float4 * sortedPosition,
									 __global float * pressure,
									 __global float * rho,
									 __global float * rhoInv,
									 float delta
									 )
{
	
	int id = get_global_id( 0 );
	id = particleIndexBack[id];//track selected particle (indices are not shuffled anymore)

	int idx = id * NEIGHBOR_COUNT;
	int nc = 0;// neigbor counter
	float rho_err;
	float p_corr;


	rho_err = rho[PARTICLE_COUNT+id] - rho0;
	p_corr = rho_err*delta;
	if(p_corr < 0) p_corr = 0;//non-negative pressure
	pressure[ id ] += p_corr;

	//just to view the variable value;
	//p_corr = pressure[ id ];
	//p_corr = 0.f;
}


__kernel void pcisph_computePressureForceAcceleration(
								  __global float2 * neighborMap,
								  __global float * pressure,
								  __global float * rho,
								  __global float * rhoInv,
								  __global float4 * sortedPosition,
								  __global float4 * sortedVelocity,
								  __global uint * particleIndexBack,
								  float CFLLimit,
								  float del2WviscosityCoefficient,
								  float gradWspikyCoefficient,
								  float h,
								  float mass,
								  float mu,
								  float simulationScale,
								  __global float4 * acceleration,
								  float rho0
								  )
{
	int id = get_global_id( 0 );
	id = particleIndexBack[id];//track selected particle (indices are not mixed anymore)

	int idx = id * NEIGHBOR_COUNT;
	float hScaled = h * simulationScale;

	//float4 position_i = sortedPosition[ id ];
	//float4 velocity_i = sortedVelocity[ id ];
	float pressure_i  = pressure[ id ]; 
	float rho_i		  = rho[ PARTICLE_COUNT+id ];

	float4 result = (float4)( 0.0f, 0.0f, 0.0f, 0.0f );
	//float2 nm;

	int nc=0;
	float4 gradW_ij;
	//float4 rj,rij,ri = sortedPosition[id];//x_i(t) // (not x_i*(t+1)) 
	//ri.w = 0.f;
	float r_ij,rho_err;
	float4 vr_ij;
	int jd;
	float value;
	int real_neighbors = 0;
	int total_neighbors = 0;
	
	do
	{
		if( (jd = NEIGHBOR_MAP_ID( neighborMap[ idx + nc ])) != NO_PARTICLE_ID)
		{
			r_ij = NEIGHBOR_MAP_DISTANCE( neighborMap[ idx + nc] );

			if(r_ij<hScaled)
			{
				//value = -(hScaled-r_ij)*(hScaled-r_ij)*0.5f*(pressure[id]+pressure[jd])/rho[PARTICLE_COUNT+jd];
				value = -(hScaled-r_ij)*(hScaled-r_ij)*( pressure[id]/(rho[PARTICLE_COUNT+id]*rho[PARTICLE_COUNT+id])
														+pressure[jd]/(rho[PARTICLE_COUNT+id]*rho[PARTICLE_COUNT+id]) );
				vr_ij = (sortedPosition[id]-sortedPosition[jd])*simulationScale; vr_ij.w = 0;
				result += value*vr_ij/r_ij;
				//result = result;

				// according to formula (3.3) in B. Solenthaler's dissertation "Incompressible Fluid Simulation and Advanced Surface Handling with SPH"
				// http://www.ifi.uzh.ch/pax/uploads/pdf/publication/1299/Solenthaler.pdf

				//result += -mass*(pressure[id]/(rho[PARTICLE_COUNT+id]*rho[PARTICLE_COUNT+id])
				//			   + pressure[jd]/(rho[PARTICLE_COUNT+jd]*rho[PARTICLE_COUNT+jd]))*gradW_ij;
				real_neighbors++;
			}

			total_neighbors++;
		}

	}while( ++nc < NEIGHBOR_COUNT );

	//result *= mass*gradWspikyCoefficient/rho[PARTICLE_COUNT+id];
	result *= mass*gradWspikyCoefficient;
	//
	//result = -2.f*mass*pressure[id]*sum_gradW/(rho0*rho0);
	//result.w = 0.0f;
	acceleration[ PARTICLE_COUNT+id ] = result; // pressureForceAcceleration "=" or "+=" ???

}

__kernel void pcisph_integrate(
						__global float4 * acceleration,
						__global float4 * sortedPosition,
						__global float4 * sortedVelocity,
						__global uint2 * particleIndex,
						__global uint * particleIndexBack,
						float gravity_x,
						float gravity_y,
						float gravity_z,
						float simulationScaleInv,
						float timeStep,
						float xmin,
						float xmax,
						float ymin,
						float ymax,
						float zmin,
						float zmax,
						float damping,
						__global float4 * position,
						__global float4 * velocity,
						__global float * rho
						)
{
	int id = get_global_id( 0 );
	id = particleIndexBack[id];

	int id_source_particle = PI_SERIAL_ID( particleIndex[id] );

	float4 acceleration_ = acceleration[ id ] + acceleration[ PARTICLE_COUNT+id ]; acceleration_.w = 0.f;
	float4 position_ = sortedPosition[ id ];
	float4 velocity_ = sortedVelocity[ id ];
	//float4 acceleration_Fp;
	//acceleration_Fp = acceleration[ PARTICLE_COUNT+id ]; acceleration_Fp.w = 0.f;
	//acceleration_ += acceleration[ PARTICLE_COUNT+id ]; 
	//float speedLimit = 100.f;

	// Semi-implicit Euler integration 
	float4 newVelocity_ = velocity_ + timeStep * acceleration_; //newVelocity_.w = 0.f;
	float posTimeStep = timeStep * simulationScaleInv;			
	float4 newPosition_ = position_ + posTimeStep * newVelocity_; //newPosition_.w = 0.f;


	handleBoundaryConditions( position_, &newVelocity_, posTimeStep, &newPosition_,
		xmin, xmax, ymin, ymax, zmin, zmax, damping );

	//if(mode==0)
	/*{
		//sortedVelocity[PARTICLE_COUNT+id] = newVelocity_;// sorted position, as well as velocity, in current version has double size, PARTICLE_COUNT*2, to store both x(t) and x*(t+1)
		sortedPosition[PARTICLE_COUNT+id] = newPosition_;
	}*/
	//else//if mode==1

	// in Chao Fang version here is also acceleration 'speed limit' applied

	newPosition_.w = rho[id];

	velocity[ id_source_particle ] = newVelocity_;//(velocity_+newVelocity_)*0.5f;
	position[ id_source_particle ] = newPosition_;
}

