//
// model.cuh
//
// The result - a single number - is stored in datm[imd].
//

#include <cuda_runtime.h>

__device__ float planck(float T, float nu);
__device__ float y1_func(float theta0, float lambda1, float T0,
                         float psi, float nu);
__device__ float Sv_func(float theta0, float lambda1, float T0, float psi,
                         float r0, float ymax, float nu, float d);



static constexpr float ak = 0.212f;
static constexpr float myu = 1.67e-24f; // * u.g
static constexpr float h = 6.626e-27f;  // erg-sec, Planck constant
static constexpr float k = 1.38e-16f;   // erg/deg, Boltzmann constant
static constexpr float c = 3e10f;       // cm/sec,  Speed of light
static constexpr float c2 = 9e20f;       // cm^2/sec^2,  Speed of light squared
static constexpr float dist = 5.49e21f;

 
//

__device__ int model(CCalcModel *mc, int id, int ipt, int imd, int ipass) {

  /* Arguments:
   *   id: index into mc->dat. 
   *   ipt: index of the first parameter in the set of model parameters 
   *        for ibeta == threadIdx.x and iseq == blockIdx.x. 
   *        mc->pcur[ipt] is the first model parameter, and 
   *        mc->pcur[ipt+mc->nptot-1] is the last parameter.
   *        If i is a parameter index, from 0 to mc->nptot-1,
   *        then mc->pcur[ipt+i] is the i-th parameter, or
   *        pcur[ibeta,iseq,i] in Python.
   *   imd: "through index" into datm[ibeta,isec,id]. Only used 
   *        to save the result before return.
   *   ipass: pass number, starting from 0. The model() function can be
   *          called multiple times, or in many passes, so pass is to know
   *          which time model() is called now.  
   *
   *
   * All member arrays and parameters of the class CCalcModel are accessible
   * here. Use the expressions like mc->coor, mc->idat, mc->ncoor, 
   * mc->nidat etc. See the class CCalcModel definition in mcmcjob.cuh and
   * mcmcjob.cu.
   * 
   * The result must be saved in
   *   mc->datm[imd]
   */

    //
    // Place your code here. 
    // Remember: it should only calculate one value
    // and save it in mc->datm[imd]. This value will further be compared
    // with its respective value in mc->dat[id] to be used in the chi^2
    // calculation.
    //

    float nu, spec, spec2, Sv, Sv2;
    float theta0 =  mc->pcur[ipt++];
    float lambda1 = mc->pcur[ipt++];
    float T0 =      mc->pcur[ipt++];
    float psi =     mc->pcur[ipt++];
    float r0 =      mc->pcur[ipt++];
    float ymax =    mc->pcur[ipt];

    //
    // Better?:
    //
    // float theta0, lambda1, T0, psi, r0, ymax;
    // float *pcur = &mc->pcur[ipt];
    // theta0 = *pcur++;
    // lambda1 = *pcur++;
    // T0 = *pcur++;
    // psi = *pcur++;
    // r0 = *pcur++;
    // ymax = *pcur;
    //


    
    nu = mc->coor[id];    // A frequency

    Sv = Sv_func(theta0, lambda1, T0, psi, r0, ymax, nu, dist);
    Sv2 = Sv * Sv;   // Sv^2

    spec = mc->dat[id];    // A spectrum
    spec2 = spec * spec;
    
    mc->datm[imd] = fabsf(Sv2 - spec2);

    if (imd < 100) 
        printf("\nid=%d, ipt=%d, imd=%d, ipass=%d\n"                    \
               "theta0=%g, lambda1=%g, T0=%g, psi=%g, r0=%g, ymax=%g, nu=%g, " \
               "dist=%g\n", id, ipt, imd, ipass,
               theta0, lambda1, T0, psi, r0, ymax, nu, dist);
        
   return 1;
}
    
// Bv - распределение Планка
// Spectral radiance of a body for frequency ν at absolute temperature T 
__device__ float planck(float T, float nu) { 
        return 2*h*nu*nu*nu/c2/(expf(h*nu/k/T) - 1.f);
}


__device__ float y1_func(float theta0, float lambda1, float T0,
                         float psi, float nu) {
    float tanth = tanf(theta0);
    float y1;

    y1 = ak/(16*myu*myu) * lambda1*lambda1 * powf(T0,-1.35f) * powf(nu,-2.1f);
    y1 *= (1.f + tanth*tanth) / (tanth*tanth);
    y1 *= (sinf(2.f*theta0) * cosf(2.f*psi) + 2.f*theta0) / (2*theta0*theta0);
    y1 = powf(y1, 0.33333333f);
    return y1;
}


__device__ float Sv_func(float theta0, float lambda1, float T0, float psi,
                         float r0, float ymax, float nu, float d) {
    float Sv, y0, y1, thetam, Br;
    float d2 = d*d;
    float y00, y11, y111;
    
    y0 = r0*cosf(psi);
    y1 = y1_func(theta0, lambda1, T0, psi, nu);
    thetam = atan2f(tanf(theta0), cosf(psi));
    
    y00 = y0*y0;
    y11 = y1*y1;
    y111 = y11*y1;
    
    Br = planck(T0, nu);

    if (y1 < y0)
        Sv = (2.f*thetam*Br)/d2*(y111*(1.f/y0 - 1.f/ymax));
    else if (y0 <= y1 && y1 < ymax)
        Sv = 2.f*(thetam*Br)/d2*(y11 - y00)/2.f + y111*(1.f/y1 - 1.f/ymax);
    else // if (y1 >= ymax)
        Sv = (thetam*Br)/d2*(ymax*ymax - y00);
       
    return Sv/1e-23f; // to Jy
}

