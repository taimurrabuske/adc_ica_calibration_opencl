
typedef struct Params
{
  float c[RAW_BITS];  //padding
  float Ctotal;
  float Vref;
  float vptb;
  float w_delta;
  float delta;
  float lr;
  float Cu;
} Params;

__kernel void convert (__global const float *y,
		       __global const unsigned char *Td,
		       __global float *out,
		       volatile __global int *cor,
		       volatile __global int *cor_delta,
		       __global float *w,
		       __global float *error,
		       __global float *w_delta,
		       __constant struct Params* params
		       )
{
  const int gid = get_global_id(0);
  const int gsz = get_global_size(0);
  const int lid = get_local_id(0);
  const int lsz = get_local_size(0);
  int i,j;
  int x[RAW_BITS];
  unsigned int d_raw;
  unsigned int d_hat;
  float Cpn = params->Ctotal;
  float Vpn;
  float res;
  __local int tmp_cor[RAW_BITS];

  if(gid == 0)
    for(i=RAW_BITS-1;i>=0;i--)
      {
	cor[i]=0;
      }
  barrier(CLK_GLOBAL_MEM_FENCE);

  if(lid == 0)
    for(i=RAW_BITS-1;i>=0;i--)
      {
	tmp_cor[i]=0;
      }
  barrier(CLK_LOCAL_MEM_FENCE);  
  
  //convert
  d_raw=0;
  Vpn=-(y[gid]+(float)(2*Td[gid]-1)*params->vptb);
  for(i=RAW_BITS-1; i>=0; i--)
    {
      Vpn=Vpn+params->Vref*params->c[i]/params->Ctotal;
      if (Vpn>0.0)
	{
	  Vpn=Vpn-params->Vref*params->c[i]/params->Ctotal;
	}
      else
	{
	  d_raw+=1<<i;
	}
    }

  //raw2bin 
  res=0;
  for(i=RAW_BITS-1;i>=0;i--)
    {
      if((unsigned int)d_raw&(unsigned int)1<<i)
	res=res+w[i];
      else
	res=res-w[i];
    }


  // subtract PN
  res=res-(float)(2*Td[gid]-1)*w_delta[0]*params->delta;
  out[gid] = res;

  //requantize
  d_hat=0;
  res=res-w[RAW_BITS-1];
  for(i=RAW_BITS-2; i>=0; i--)
    res=res+w[i];
  res=res+w[0];
  for(i=RAW_BITS-1; i>=0; i--)
    {
      if (res>0.0)
	{
	  d_hat+=(unsigned int)1<<i;
	  res=res-2*w[i-1];
	}
      else
	{
	  res=res+2*w[i]-2*w[i-1];
	}
    }

  // compute partial correlations
  for(i=RAW_BITS-1;i>=0;i--)
    {
      x[i]=2*(((d_hat&(unsigned int)1<<i)>>i)^((unsigned int)Td[gid]))-1;
      atomic_add(&tmp_cor[i],-x[i]);
    }
  barrier(CLK_LOCAL_MEM_FENCE);

  // last in group? add partial correlation to the global one.
  if(lid==(lsz-1))
    {
      for(i=RAW_BITS-1;i>=4;i--)
	{
	  atomic_add(&cor[i],tmp_cor[i]);
	  atomic_add(&cor_delta[0],tmp_cor[i]*w[i]);
	}
    }
  barrier(CLK_GLOBAL_MEM_FENCE);
  
  // last? update coefficients.
  if(gid==(gsz-1))
    {
      for(i=RAW_BITS-1;i>=4;i--)
	{
	  w[i]=w[i]-cor[i]*params->lr;
	  
	  // Update of delta_weight disabled, as it
	  // appears to bring convergence problems!
	  //w_delta[0]=w_delta[0]-cor_delta[0]*0.00000000001*params->lr;
	  
	  error[i]=(params->c[i]/(w[i]*params->Cu))-1;
	}
    }
  barrier(CLK_GLOBAL_MEM_FENCE);
  
}


