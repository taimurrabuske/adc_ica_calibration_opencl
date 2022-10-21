typedef struct Params
{
  float amp_coeff[ORDER];  //padding
  float lr_coeff[ORDER];
  float delta;
  float lr;
} Params;

inline void AtomicAdd(volatile __global float *source, const float operand) {
  union {
    unsigned int intVal;
    float floatVal;
  } newVal;
  union {
    unsigned int intVal;
    float floatVal;
  } prevVal;
  do {
    prevVal.floatVal = *source;
    newVal.floatVal = prevVal.floatVal + operand;
  } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}


inline void LAtomicAdd(volatile __local float *source, const float operand) {
  union {
    unsigned int intVal;
    float floatVal;
  } newVal;
  union {
    unsigned int intVal;
    float floatVal;
  } prevVal;
  do {
    prevVal.floatVal = *source;
    newVal.floatVal = prevVal.floatVal + operand;
  } while (atomic_cmpxchg((volatile __local unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}


__kernel void run (__global const float *y,
		       __global const unsigned char *Td,
		       __global float *out,
		       __global float *cor,
		       __global float *b,
		       __global float *error,
		       __constant struct Params* params
		       )
{
  const int gid = get_global_id(0);
  const int gsz = get_global_size(0);
  const int lid = get_local_id(0);
  const int lsz = get_local_size(0);
  __private int i;
  __private float Vo;
  __private float Vx;
  __private float Vy;
  __private float T;
  float x;
  __local float tmp_cor[ORDER];


  // if you're the first to arrive, clean up the space!
  if(gid==0)
    {
      //clear correlations
      for(i=ORDER-1;i>=0;i--)
	{
	  cor[i]=0;
	}
    }
  barrier(CLK_GLOBAL_MEM_FENCE);

  if(lid==0)
    {
      //clear correlations
      for(i=ORDER-1;i>=0;i--)
	{
	  tmp_cor[i]=0;
	}
    }
  barrier(CLK_LOCAL_MEM_FENCE);


  T=2*(float)Td[gid]-1.0;

  //convert
  out[gid]=0;
  Vo=0;
  for(i=ORDER-1;i>=0;i--)
    Vo=Vo+params->amp_coeff[i]*pown((y[gid]+T*params->delta),i);

  Vx=0;
  for(i=ORDER-1;i>=0;i--)
    Vx=Vx+b[i]*pown(Vo,i);

  // subtract PN
  Vy = Vx-T*params->delta;
  out[gid]=Vy;

  for(i=ORDER-1;i>=0;i--)
    {
      x=pown(Vy,(i))*T;
      //atomic_xchg(&cor[i],cor[i]+x);
      LAtomicAdd(&tmp_cor[i],x);
    }
  barrier(CLK_LOCAL_MEM_FENCE);

  if(lid==(lsz-1))
    for(i=ORDER-1;i>=0;i--)
      AtomicAdd(&cor[i],tmp_cor[i]);
  barrier(CLK_GLOBAL_MEM_FENCE);

  if(gid==(gsz-1))
    {
      for(i=ORDER-1;i>=0;i--)
	{
	  b[i]=b[i]-cor[i]*params->lr*params->lr_coeff[i];
	  error[i]=params->amp_coeff[i]-b[i];
	}
    }
  barrier(CLK_GLOBAL_MEM_FENCE);
 
}


