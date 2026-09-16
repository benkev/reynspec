//
// model.cuh
//
// This is a template model file.
//
// The result - a single number - is stored in datm[imd].
//

#include <cuda_runtime.h>

//
// For convenience and brevity, define the variable and coefficient names
// (BE CAREFUL not to mix with the names in src/gpu_mcmc.cu !!!
// To avoid the mess, always undefine the names after use!):
//
// For example:
//
#define THETA0  (mc->pcur[ipt++])
#define LAMBDA1 (mc->pcur[ipt++])
#define T0  (mc->pcur[ipt++])
#define PSI (mc->pcur[ipt++])
#define R0  (mc->pcur[ipt++])
#define YMAX (mc->pcur[ipt])

// #define V2 (mc->pcur[ipt])
// #define C1 (mc->coor[0])
// #define C2 (mc->coor[1])
// #define K (mc->idat[0])
// #define K1 (mc->idat[1])

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
    
   return 1;
}


static constexpr float ak = 0.212f;
static constexpr float myu = 1.67e-24f; // * u.g
static constexpr float h = 6.626e-27f;  // erg-sec, Planck constant
static constexpr float k = 1.38e-16f;   // erg/deg, Boltzmann constant
static constexpr float c = 3e10f;       // cm/sec,  Speed of light
static constexpr float c2 = 9e20f;       // cm^2/sec^2,  Speed of light squared

    
// Bv - распределение Планка
// Spectral radiance of a body for frequency ν at absolute temperature T 
__device__ float planck(float T, float nu) { 
        return 2*h*nu*nu*nu/c2/(expf(h*nu/k/T) - 1.f);
}


__device__ float y1_func(float theta0, float lambda1, float T0,
                         float psi, float nu) {
    float tanth = tanf(theta0);
    float tanth = tanf(theta0);
    y1 = ak/(16*myu*myu) * lambda1*lambda1 * powf(T0,-1.35f) * powf(nu,-2.1f);
    y1 *= (1.f + tanth*tanth) / (tanth*tanth);
    y1 *= (sinf(2.f*theta0) * cosf(2.f*psi) + 2.f*theta0) / (2*theta0*theta0);
    y1 = powf(y1, 0.33333333f);
    return y1;



__device__ float Sv_func(float theta0, float lambda1, float T0, float psi,
                         float r0, float ymax, float nu, float d) {
    float d2 = d*d;
    float y111 = y1*y1*y1;
    float y11 = y1*y1, y00 = y0*y0;
    
    y0 = r0*cosf(psi);
    y1 = y1_func(theta0, lambda1, T0, psi, nu);
    thetam = atan2f(tanf(theta0), cosf(psi));
   
    Br = planck(T0, nu);

    if (y1 < y0)
        Sv = (2.f*thetam*Br)/d2*(y111*(1.f/y0 - 1.f/ymax));
    else if (y0 <= y1 && y1 < ymax)
        Sv = 2.f*(thetam*Br)/d2*(y11 - y00)/2.f + y111*(1.f/y1 - 1.f/ymax);
    else // if (y1 >= ymax)
        Sv = (thetam*Br)/d2*(ymax*ymax - y00);
       
    return Sv/1e-23f; // to Jy
}


 nu = freqs;

 // [numparam, numrealisation,freq]
__device__ float Sv_func_p(float p[], float nu, float dis):
    return Sv_func(p[0], p[1], p[2], p[3], p[4], p[5], nu, d);

__device__ float log_likelihood(float p, float nu, float d):
    p=p.copy()
    p[logvar] = 10**p[logvar]
    Svs = Sv_func_p(p.T[:,None], nu[None], d)
    
    ressq = abs((Svs**2 - spec**2))
    res = -0.5*(ressq.sum())/(np.std(spec))**2
    return res

def log_prior(p,variables):
    linpar = p.copy()
    linpar[logvar] = 10**(linpar[logvar])
    if (linpar <=0).any() :
        return -np.inf
    theta = variables[:,1]
    sigma = abs(variables[:,2]-variables[:,0])
    r = (p-theta)
    L = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*r**2/sigma**2)
    return np.log(L.sum())

def log_probability(p,nu,d, variables):
    lp = log_prior(p,variables)
    if not np.isfinite(lp) or np.isnan(lp):
        return -np.inf
    ll = log_likelihood(p,nu, d)
    ret = lp + ll
    if np.isnan(ret):
        return -np.inf
    print(lp, ll)
    return ret


        


//
// Always undefine the names after use!  
//
#undef THETA0
#undef LAMBDA1
#undef T0
#undef PSI
#undef R0
#undef YMAX
 
