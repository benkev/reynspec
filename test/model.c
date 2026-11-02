//
// model.cc
//
// The result - a single number - is stored in datm[imd].
//

#include <stdio.h>
#include <math.h>


float planck(float T, float nu);
float y1_func(float theta0, float lambda1, float T0,
                         float psi, float nu);
float Sv_func(float theta0, float lambda1, float T0, float psi,
                         float r0, float ymax, float nu, float d);

static constexpr float ak = 0.212f;
static constexpr float myu = 1.67e-24f; // * u.g
static constexpr float h = 6.626e-27f;  // erg-sec, Planck constant
static constexpr float k = 1.38e-16f;   // erg/deg, Boltzmann constant
static constexpr float c = 3e10f;       // cm/sec,  Speed of light
static constexpr float c2 = 9e20f;       // cm^2/sec^2,  Speed of light squared
static constexpr float dist = 5.49e21f;

int main(int argc, char *argv[]) {

    float nu, spect, Sv;
    float theta0;
    float lambda1;
    float T0;
    float psi;
    float r0;
    float ymax;
    float nuspect[620];

    // Cast the 1D pointer to a pointer to an array of 'cols' elements
    // This supports dynamic (runtime) dimensions in C99 and later VLA rules
    int cols = 2;
    float (*fs)[cols] = (float (*)[cols]) nuspect; // fs[i][j] == nuspect[k]
    // int (*array2D)[cols] = (int (*)[cols])array1D;
    
    FILE *file = fopen("nuspect.bin", "rb");
    fread(nuspect, sizeof(float), 620, file);

    for (int i=0; i<20; i++)
        printf("nu=%10f, spect=%g\n", fs[i][0], fs[i][1]);
    printf("\n\n");
    
    theta0=0.851893;
    lambda1=13.2958;
    T0=4.9185;
    psi=0.845629;
    r0=15.0202;
    ymax=15.9441;
    nu=3.95189e+09;
    //    dist=5.49e+21;

    Sv = Sv_func(theta0, lambda1, T0, psi, r0, ymax, nu, dist);

    // Sv2 = Sv * Sv;   // Sv^2
    // spect = mc->dat[id];    // A spectrum
    // spect2 = spect * spect;
    
    // mc->datm[imd] = fabsf(Sv2 - spect2);
    
    // mc->datm[imd] = Sv;

    printf("nu=%g, Sv=%g \n" \
           "theta0=%g, lambda1=%g, T0=%g, psi=%g, r0=%g, ymax=%g, dist=%g\n",
           nu, Sv, theta0, lambda1, T0, psi, r0, ymax, dist);
}
// Bv - распределение Планка
// Spectral radiance of a body for frequency ν at absolute temperature T 
float planck(float T, float nu) { 
        return 2*h*nu*nu*nu/c2/(expf(h*nu/k/T) - 1.f);
}


float y1_func(float theta0, float lambda1, float T0,
                         float psi, float nu) {
    float tanth = tanf(theta0);
    float y1;

    y1 = ak/(16*myu*myu) * lambda1*lambda1 * powf(T0,-1.35f) * powf(nu,-2.1f);
    y1 *= (1.f + tanth*tanth) / (tanth*tanth);
    y1 *= (sinf(2.f*theta0) * cosf(2.f*psi) + 2.f*theta0) / (2*theta0*theta0);
    y1 = powf(y1, 0.33333333f);
    return y1;
}


float Sv_func(float theta0, float lambda1, float T0, float psi,
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

