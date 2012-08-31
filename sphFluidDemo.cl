
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
#define P( i ) i.z

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
						   __global float2 * neighborMap,
						   __global int * elasticNeighbourCount
						   )
{
	int id = get_global_id( 0 );
	__global float4 * nm = (__global float4 *)neighborMap;
	int outIdx = ( id * NEIGHBOR_COUNT ) >> 1;//int4 versus int2 addressing
	float4 fdata = (float4)( -1, -1, -1, -1 );
	if(id < LIQUID_PARTICLE_COUNT){
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
	else{
		int i = elasticNeighbourCount[id - LIQUID_PARTICLE_COUNT];
		while( i > NEIGHBOR_COUNT){
			nm[ i ] = fdata;
			i++;
		}
	}
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
	float4 rVec = position_i - position_j;
	float4 scaledVec = rVec * simulationScale;
	scaledVec /= r;
	rVec.w = 0.0f;// зачем? ведь дальше не используется
	float x = h - r;
	float4 result = x * x * scaledVec * gradWspikyCoefficient;
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
	// Following Muller Charypis and Gross ( 2003 )
	// -grad p_i = - sum_j m_j ( p_i + p_j ) / ( 2 rho_j ) grad Wspiky
	// Equation 10.
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
	float result = ( h - r ) * del2WviscosityCoefficient;
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
								  float CFLLimit,
								  float del2WviscosityCoefficient,
								  float gradWspikyCoefficient,
								  float h,
								  float mass,
								  float mu,
								  float simulationScale,
								  __global float4 * acceleration
								  )
{
	int id = get_global_id( 0 );
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

	int j = 0;
	bool loop;

	do{
		nm = neighborMap[ idk + j ];
		int neighborParticleId = NEIGHBOR_MAP_ID( nm );
		bool isNeighbor = ( neighborParticleId != NO_PARTICLE_ID );
		if( isNeighbor ){
			float p_j = pressure[ neighborParticleId ];
			float rho_j_inv = rhoInv[ neighborParticleId ];
			float r = NEIGHBOR_MAP_DISTANCE( nm );
			float4 dgradP = contributeGradP( id, neighborParticleId, p_i, p_j, rho_j_inv,
				position_i, pressure, rho, sortedPosition, r, mass, hScaled,
				gradWspikyCoefficient, simulationScale );
			float4 ddel2V = contributeDel2V( id, velocity_i, neighborParticleId,
				sortedVelocity, rho_j_inv, r, mass, hScaled, del2WviscosityCoefficient );
			gradP += dgradP;
			del2V += ddel2V;
		}
		loop = ( ++j < NEIGHBOR_COUNT );
	}while( loop );

	result = rho_i_inv * ( mu * del2V - gradP );

	// Check CFL condition
	float magnitude = result.x * result.x + result.y * result.y + result.z * result.z;
	bool tooBig = ( magnitude > CFLLimit * CFLLimit );
	float sqrtMagnitude = SQRT( magnitude );
	float scale = CFLLimit / sqrtMagnitude;
	result = SELECT( result, result * scale, (uint4)tooBig );

	result.w = 0.0f;
	acceleration[ id ] = result;
}





// Mueller et al equation 3.  Scalar result.

float Wpoly6(
			 float r,
			 float hSquared,
			 float Wpoly6Coefficient
			 )
{
	float x = hSquared - r * r;
	float result = x * x * x * Wpoly6Coefficient;
	return result;
}

float Wshape(float r,
			 float h,
			 float shapedCoeefficient){
	float x = (r+h)*M_PI/(2*h);
	float result = shapedCoeefficient * (cos(x) + 1);
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
	float smoothingKernel = Wpoly6( r, hSquared, Wpoly6Coefficient );
	float result = SELECT( smoothingKernel, 0.0f, ( neighborParticleId == NO_PARTICLE_ID ) );
	return result;
}



__kernel void computeDensityPressure(
									 __global float2 * neighborMap,
									 float Wpoly6Coefficient,
									 float h,
									 float mass,
									 float rho0,
									 float simulationScale,
									 float stiffness,
									 __global float * pressure,
									 __global float * rho,
									 __global float * rhoInv
									 )
{
	int id = get_global_id( 0 );
	int idx = id * NEIGHBOR_COUNT;
	float density = 0.0f;
	float hScaled = h * simulationScale;
	float hSquared = hScaled * hScaled;

	density = density + densityContribution( idx, 0, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 1, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 2, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 3, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 4, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 5, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 6, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 7, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 8, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 9, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 10, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 11, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 12, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 13, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 14, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 15, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 16, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 17, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 18, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 19, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 20, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 21, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 22, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 23, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 24, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 25, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 26, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 27, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 28, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 29, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 30, neighborMap, mass, hSquared, Wpoly6Coefficient );
	density = density + densityContribution( idx, 31, neighborMap, mass, hSquared, Wpoly6Coefficient );

	density *= mass; // since all particles are same fluid type, factor this out to here
	rho[ id ] = density; // my density
	rhoInv[ id ] = SELECT( 1.0f, DIVIDE( 1.0f, density ), ( density > 0.0f ) );
	float drho = density - rho0; // rho0 is resting density
	float k = stiffness;
	float p = k * drho; // equation 12
	pressure[ id ] = p; // my pressure

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
	float distance,distanceSquared;
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
				distanceSquared = DOT( d, d );
				if( distanceSquared <= r_thr_Squared )
				{
					distance = SQRT( distanceSquared );
					j = (int)(distance*radius_segments/h);
					if(j<radius_segments) 
						radius_distrib[j]++;
					if(mode)
					{
						myOffset = NEIGHBOR_COUNT - spaceLeft + foundCount;
						neighbor_data.x = neighborParticleId;
						neighbor_data.y = distance* simulationScale;
						neighborMap[ myParticleId*NEIGHBOR_COUNT + myOffset ] = neighbor_data;
						foundCount++;
					}
				}
			
			}

			i++;
			
		}//while

	return foundCount;
}

int searchNeighborForElasticParticle(int particleId, 
									 __global float8 * position,
									 __global float2 * neighborMap,
								     float h,
								     float simulationScale,
								     int mode,
								     int * radius_distrib,
								     float r_thr,int objectId){
	int particleCountThisCell = ELASTIC_PARTICLE_COUNT;
	int potentialNeighbors = particleCountThisCell;
	int foundCount = 0;
	float4 position_ = position[ particleId ].lo;
	bool loop = true;
	int i = 0,j;
	float distance,distanceSquared;
	float r_thr_Squared = r_thr*r_thr;
	float2 neighbor_data;
	int neighborParticleId;
	int myOffset;
	while( i < particleCountThisCell ){

		neighborParticleId = LIQUID_PARTICLE_COUNT + i;
		
		//if(myParticleId == neighborParticleId) continue;

		if(particleId != neighborParticleId && position[ neighborParticleId ].s4 == objectId)
		{
			float4 d = position_ - position[ neighborParticleId ].lo;
			d.w = 0.0f;
			distanceSquared = DOT( d, d );
			if( distanceSquared <= r_thr_Squared )
			{
				distance = SQRT( distanceSquared );
				j = (int)(distance*radius_segments/h);
				if(j<radius_segments) 
					radius_distrib[j]++;
				if(mode)
				{
					myOffset = foundCount;
					neighbor_data.x = neighborParticleId;
					neighbor_data.y = distance* simulationScale;
					neighborMap[ particleId*NEIGHBOR_COUNT + myOffset ] = neighbor_data;
					foundCount++;
				}
			}
		
		}

		i++;
		
	}//while

	return foundCount;
}
__kernel void findNeighborFotElasticParticle(__global float8 * position,
											 float h,
											 __global float2 * neighborMap,
											  float simulationScale,
											 __global int * elasticNeighbourCount )
{
	int posId = get_global_id(0);
	int id = posId + LIQUID_PARTICLE_COUNT;
	int objectId = (int)position[id].s4;
	float distance = 0;	
	int i = 0,j;
	int foundCount = 0;
	int mode = 0;
	int distrib_sum = 0;
	int radius_distrib[radius_segments];
	float r_thr = h;
	while( i<radius_segments )
	{
		radius_distrib[i]=0;
		i++;
	}
	while( mode<2 )
	{
		foundCount = searchNeighborForElasticParticle(id,position,neighborMap,h,simulationScale,mode,&radius_distrib,r_thr, objectId);
		if(mode==0)
		{
			j=0;

			while(j<radius_segments)
			{
				distrib_sum += radius_distrib[j];
				if(distrib_sum==32) break;
				if(distrib_sum> 32) { j--; break; }
				j++;
			}

			r_thr = (j+1)*h/radius_segments;

		}

		mode++;
	}
	elasticNeighbourCount[posId] = foundCount;
}

int4 cellFactors( 
				 float4 position,
				 float xmin,
				 float ymin,
				 float zmin,
				 float hashGridCellSizeInv
				 )
{
	//xmin, ymin, zmin здесь не иссплользуются
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
							__global float2 * neighborMap,
							__global float8 * position,
							__global int * elasticNeighbourCount
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
	bool is_elastic= false;
	if(id>=LIQUID_PARTICLE_COUNT){
		if(elasticNeighbourCount[id]>=NEIGHBOR_COUNT)
			return;
		is_elastic = true;
		foundCount = elasticNeighbourCount[id];
	}	
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
			h, simulationScale, mode, &radius_distrib, r_thr );


		// p is the current particle position within the bounds of the hash grid
		float4 p;
		float4 p0 = (float4)( xmin, ymin, zmin, 0.0f );//я так понимаю xmin, ymin, zmin -> нули -> наверное стоит убрать это тогда
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
			h, simulationScale, mode, &radius_distrib, r_thr  );

		searchCell_ = searchCell( myCellId, 0, delta.y, 0, gridCellsX, gridCellsY, gridCellsZ, gridCellCount );
		foundCount += searchForNeighbors( searchCell_, gridCellIndex, position_, 
			id, sortedPosition, neighborMap, NEIGHBOR_COUNT - foundCount, 
			h, simulationScale, mode, &radius_distrib, r_thr  );

		searchCell_ = searchCell( myCellId, 0, 0, delta.z, gridCellsX, gridCellsY, gridCellsZ, gridCellCount );
		foundCount += searchForNeighbors( searchCell_, gridCellIndex, position_, 
			id, sortedPosition, neighborMap, NEIGHBOR_COUNT - foundCount, 
			h, simulationScale, mode, &radius_distrib, r_thr  );

		searchCell_ = searchCell( myCellId, delta.x, delta.y, 0, gridCellsX, gridCellsY, gridCellsZ, gridCellCount );
		foundCount += searchForNeighbors( searchCell_, gridCellIndex, position_, 
			id, sortedPosition, neighborMap, NEIGHBOR_COUNT - foundCount, 
			h, simulationScale, mode, &radius_distrib, r_thr  );

		searchCell_ = searchCell( myCellId, delta.x, 0, delta.z, gridCellsX, gridCellsY, gridCellsZ, gridCellCount );
		foundCount += searchForNeighbors( searchCell_, gridCellIndex, position_, 
			id, sortedPosition, neighborMap, NEIGHBOR_COUNT - foundCount, 
			h, simulationScale, mode, &radius_distrib, r_thr  );

		searchCell_ = searchCell( myCellId, 0, delta.y, delta.z, gridCellsX, gridCellsY, gridCellsZ, gridCellCount );
		foundCount += searchForNeighbors( searchCell_, gridCellIndex, position_, 
			id, sortedPosition, neighborMap, NEIGHBOR_COUNT - foundCount, 
			h, simulationScale, mode, &radius_distrib, r_thr  );

		searchCell_ = searchCell( myCellId, delta.x, delta.y, delta.z, gridCellsX, gridCellsY, gridCellsZ, gridCellCount );
		foundCount += searchForNeighbors( searchCell_, gridCellIndex, position_, 
			id, sortedPosition, neighborMap, NEIGHBOR_COUNT - foundCount, 
			h, simulationScale, mode, &radius_distrib, r_thr );

		if(mode==0)
		{
			j=0;

			
			while(j<radius_segments)
			{
				distrib_sum += radius_distrib[j];
				if(distrib_sum==32) break;
				if(distrib_sum> 32) { j--; break; }
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
							__global float8 * position,
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

	float4 _position = position[ id ].lo;
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
			loop = cellId == NO_CELL_ID && idx <= gridCellCount;
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
__kernel void calculateVolume(__global float8 * position,
							  __global float * volume,
							  __global float2 * neighborMap,
							  float mass, 
							  float shapedCoeefficient,
							  float h
							  )
{
	int id = get_global_id(0);
	int i = 0;
	float density = 0;
	float result;
	while(i < NEIGHBOR_COUNT && NEIGHBOR_MAP_ID(neighborMap[i]) != -1){
		density += mass * Wshape(NEIGHBOR_MAP_DISTANCE(neighborMap[i]),h,shapedCoeefficient); 
		i++;
	}
	if(density != 0){
		result = mass / density;
	}
	volume[id] = result;
}
float4 WshapeGrad(float r,
				 float h,
				 float gradShapedCoeefficient,
				 float4 position_i,
				 float4 position_j,
				 float simulationScale
				 ){
	float4 rVec = position_i - position_j;
	float4 scaledVec = rVec * simulationScale;
	rVec.w = 0.0f;
	float x = (r + h)*M_PI/(2*h);
	float4 result = gradShapedCoeefficient * (-sin(x)) * scaledVec;
	return result;
}
void calculateStrainTensor( float4 gradDisplacement,
							float poison_ratio,
							float young_module
							)
{
	float4 nullVec = (float4)(0.f,0.f,0.f,0.f);
	//float4 stressMatrix = (float4)(nullVec, nullVec, nullVec, NullVec);
	
}


/*WORK WITH MATRIX*/
float4 multRowOnColumn(float4 row, float4 column1, float4 column2,float4 column3){
	float4 result= (float4)(0.f,0.f,0.f,0.f);
	result.x = dot(row , column1);
	result.y = dot(row , column2);
	result.z = dot(row , column3);
	return result;
}

void calcStressJacobianMult(float4 * u,
							float4 * v,
							float4 * w, 
							float poison_ratio, 
							float young_module,
							float4 * row1,
							float4 * row2,
							float4 * row3){
	float e_xx, e_yy, e_zz, e_xy, e_yz, e_zx;
	float s_xx, s_yy, s_zz, s_xy, s_yz, s_zx;
	float ratio = young_module / ((1+poison_ratio)*(1-2*poison_ratio));
	float4 result;
	//Strain Tensor
	e_xx = 2 * u->x + u->x * u->x + v->x * v->x + w->x * w->x;
	e_yy = 2 * v->y + u->y * u->y + v->y * v->y + w->y * w->y;
	e_zz = 2 * w->z + u->z * u->z + v->z * v->z + w->z * w->z;
	e_xy = u->y + v->x + u->x *u->y + v->x * v->y + w->z * w->y;
	e_yz = v->z + w->y + u->y * u->z + v->y * v->z + w->y * w->z;
	e_zx = w->y + u->z + u->z * u->x + v->z * v->x + w->z * w->x;
	//STRESS TENSOR
	s_xx = ratio * ( (1-poison_ratio) * e_xx + poison_ratio * e_yy + poison_ratio * e_zz);
	s_yy = ratio * ( poison_ratio * e_xx + (1-poison_ratio) * e_yy + poison_ratio * e_zz);
	s_zz = ratio * ( poison_ratio * e_xx + poison_ratio * e_yy + (1 - poison_ratio) * e_zz);
	s_xy = ratio * ( e_xy * ( 1 - 2 * poison_ratio ) );
	s_yz = ratio * ( e_yz * ( 1 - 2 * poison_ratio ) );
	s_zx = ratio * ( e_zx * ( 1 - 2 * poison_ratio ) );
	//This is Jacobian J = I + dU^T u v and w is rows of transposition gradientU
	u->x += 1.f;
	v->y += 1.f;
	w->z += 1.f;
	//F = -2Vj *J *s *dij
	//First ROW
	row1->x = u->x * s_xx + u->y * s_xy + u->z * s_zx;
	row1->y = u->x * s_xy + u->y * s_yy + u->z * s_yz;
	row1->z = u->x * s_zx + u->y * s_yz + u->z * s_zz;
	row1->w = 0;
	//Second Row
	row2->x = v->x * s_xx + v->y * s_xy + v->z * s_zx;
	row2->y = v->x * s_xy + v->y * s_yy + v->z * s_yz;
	row2->z = v->x * s_zx + v->y * s_yz + v->z * s_zz;
	row2->w = 0;
	//3th Row
	row3->x = w->x * s_xx + w->y * s_xy + w->z * s_zx;
	row3->y = w->x * s_xy + w->y * s_yy + w->z * s_yz;
	row3->z = w->x * s_zx + w->y * s_yz + w->z * s_zz;
	row3->w = 0;

	//
	
}

__kernel void calculateElasticForces(__global float8 * position,
									 __global float * volume,
									 __global float4 * displacement,
									 float shapedCoeefficient,
									 float gradShapedCoeefficient,
									 float h,
									 float simulationScale,
									 float poisson_ratio,
									 float young_module,
									 __global float2 * neighborMap,
									 __global float4 * acceleration,
									 float mass)
{
	int id = get_global_id(0);
	int particleId = id + LIQUID_PARTICLE_COUNT;
	float4 d_ij;
	int j = 0;
	float4 u_ji;
	float4 nullVec = (float4)(0.f,0.f,0.f,0.f);
	int neighborId;
	float neighborDistance;
	float4 gradVec;
	float4 gradDisplacement1 = nullVec;
	float4 gradDisplacement2 = nullVec;
	float4 gradDisplacement3 = nullVec;
	float4 rowVec = nullVec;
	float4 column1 = nullVec;
	float4 column2 = nullVec;
	float4 column3 = nullVec;
	float4 u;
	float4 v;
	float4 w;
	float4 row1;
	float4 row2;
	float4 row3;
	float4 force;
	//float4 jacobianMatrix = (float4)(nullVec, nullVec, nullVec, nullVec);
	//float4 strainMatrix = (float4)(nullVec, nullVec, nullVec, nullVec);
	while(j < NEIGHBOR_COUNT && NEIGHBOR_MAP_ID(neighborMap[NEIGHBOR_COUNT * particleId + j]) != -1){
		neighborId = NEIGHBOR_MAP_ID(neighborMap[NEIGHBOR_COUNT * particleId + j]);
		if(neighborId >= LIQUID_PARTICLE_COUNT){
			neighborDistance = NEIGHBOR_MAP_DISTANCE(neighborMap[NEIGHBOR_COUNT * particleId + j]);
			gradVec = WshapeGrad(neighborDistance,h,gradShapedCoeefficient,position[particleId].lo,position[neighborId].lo,simulationScale);
			u_ji = displacement[neighborId - LIQUID_PARTICLE_COUNT] - displacement[particleId - LIQUID_PARTICLE_COUNT];
			/**/
			rowVec.x = gradVec.x;
			column1.x = u_ji.x;
			column2.x = u_ji.y;
			column3.x = u_ji.z;
			gradDisplacement1 += volume[neighborId - LIQUID_PARTICLE_COUNT] * multRowOnColumn(rowVec,column1,column2,column3);
			rowVec = nullVec;
			rowVec.y = gradVec.y;
			gradDisplacement2 += volume[neighborId - LIQUID_PARTICLE_COUNT] * multRowOnColumn(rowVec,column1,column2,column3);
			rowVec = nullVec;
			rowVec.z = gradVec.z;
			gradDisplacement3 += volume[neighborId - LIQUID_PARTICLE_COUNT] * multRowOnColumn(rowVec,column1,column2,column3);
			
			/**/
		}
		j++;
	}
	u.x = gradDisplacement1.x;
	u.y = gradDisplacement2.x;
	u.z = gradDisplacement3.x;
	u.w = 0;
	v.x = gradDisplacement1.y;
	v.y = gradDisplacement2.y;
	v.z = gradDisplacement3.y;
	v.w = 0;
	w.x = gradDisplacement1.z;
	w.y = gradDisplacement2.z;
	w.z = gradDisplacement3.z;
	w.w = 0;
	calcStressJacobianMult(&u,&v,&w,poisson_ratio,young_module,&row1,&row2,&row3);
	while(j < NEIGHBOR_COUNT && NEIGHBOR_MAP_ID(neighborMap[NEIGHBOR_COUNT * particleId + j]) != -1){
		neighborId = NEIGHBOR_MAP_ID(neighborMap[NEIGHBOR_COUNT * particleId + j]);
		if(neighborId >= LIQUID_PARTICLE_COUNT){
			d_ij = volume[neighborId - LIQUID_PARTICLE_COUNT];
			force.x += -2 * volume[particleId] * ( row1.x * d_ij.x + row1.y * d_ij.y + row1.z * d_ij.z );
			force.y += -2 * volume[particleId] * ( row2.x * d_ij.x + row2.y * d_ij.y + row2.z * d_ij.z );
			force.z += -2 * volume[particleId] * ( row3.x * d_ij.x + row3.y * d_ij.y + row3.z * d_ij.z );
			force.w = 0;
		}
	}
	acceleration[particleId] = force / mass;
	//return result;
}

__kernel void integrate(
						__global float4 * acceleration,
						__global float4 * sortedPosition,
						__global float4 * sortedVelocity,
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
						__global float8 * position,
						__global float4 * velocity,
						__global uint * elasticParticleCurrentIndex,
						__global float4 * displacement,
						__global float4 * previousAcceleration
						)
{
	int id = get_global_id( 0 );
	float4 acceleration_ = acceleration[ id ];
	float4 position_;
	float4 velocity_;
	int realId;
	if(id>LIQUID_PARTICLE_COUNT){
		realId = elasticParticleCurrentIndex[id - LIQUID_PARTICLE_COUNT];
	}
	position_ = sortedPosition[ id ];
	velocity_ = sortedVelocity[ id ];
	// apply external forces
	float4 gravity = (float4)( gravity_x, gravity_y, gravity_z, 0.0f );
	acceleration_ += gravity;

	if(id < LIQUID_PARTICLE_COUNT){
		
		// Semi-implicit Euler integration 
		float4 newVelocity_ = velocity_ + timeStep * acceleration_;
		float posTimeStep = timeStep * simulationScaleInv; 
		float4 newPosition_ = position_ + posTimeStep * newVelocity_;

		handleBoundaryConditions( position_, &newVelocity_, posTimeStep, &newPosition_,
			xmin, xmax, ymin, ymax, zmin, zmax, damping );

		newPosition_.w = 1.0f; // homogeneous coordinate for rendering
		velocity[ id ] = newVelocity_;
		position[ id ].x = newPosition_.x;
		position[ id ].y = newPosition_.y;
		position[ id ].z = newPosition_.z;
	}
	else{//If elastic matter
		//ElasticForceContribution
		//Leapfrog integration http://en.wikipedia.org/wiki/Leapfrog_integration
		float4 prevAcceleration_ = previousAcceleration[realId];
		
		/*float4 newVelocity_ = velocity_ + timeStep * (acceleration_ + prevAcceleration_)/2;
		float posTimeStep = timeStep * simulationScaleInv; 
		float4 newPosition_ = position_ + posTimeStep * posTimeStep * prevAcceleration_ / 2;
		handleBoundaryConditions( position_, &newVelocity_, posTimeStep, &newPosition_,
			xmin, xmax, ymin, ymax, zmin, zmax, damping );
		velocity[ id ] = newVelocity_;
		position[ id ].x = newPosition_.x;
		position[ id ].y = newPosition_.y;
		position[ id ].z = newPosition_.z;
		previousAcceleration[realId] = acceleration_;
		/**/
	}
}





__kernel void sortPostPass(
						   __global uint2 * particleIndex,
						   __global float8 * position,
						   __global float4 * velocity,
						   __global float4 * sortedPosition,
						   __global float4 * sortedVelocity,
						   __global uint * elasticParticleCurrentIndex
						   )
{
	int id = get_global_id( 0 );
	if(id >= LIQUID_PARTICLE_COUNT){
		
		elasticParticleCurrentIndex[PI_SERIAL_ID(particleIndex[id]) - LIQUID_PARTICLE_COUNT] = id;
	}
	uint2 spi = particleIndex[ id ];//contains id of cell and id of particle it has sorted 
	int serialId = PI_SERIAL_ID( spi );//get a particle Index
	int cellId = PI_CELL_ID( spi );//get a cell Index
	float4 position_ = position[ serialId ].lo;
	POSITION_CELL_ID( position_ ) = (float)cellId;
	float4 velocity_ = velocity[ serialId ];
	sortedVelocity[ id ] = velocity_;//put velocity to sortedVelocity for right order according to particleIndex
	sortedPosition[ id ] = position_;//put position to sortedVelocity for right order according to particleIndex
}