#include "thrmlLib.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

// uncomment to debug conductionQ
//#define VERBOSE 1

void conductionQ_Ttdep(int nz, double z[], double dt, double Qn, double Qnp1,
				 double T[], double ti[], double rhoc[], double emiss,
				 double Fgeotherm, double *Fsurf)  {
/************************************************************************
   conductionQ:  program to calculate the diffusion of temperature
                 into the ground and thermal emission at the surface
                 with variable thermal properties on irregular grid
   Crank-Nicolson scheme, flux conservative
                          uses Samar's radiation formula
   Eqn: rhoc*T_t = (k*T_z)_z
   BC (z=0): Q(t) + kT_z = em*sig*T^4
   BC (z=L): heat flux = Fgeotherm

   nz = number of grid points
   z[1..nz] = depths below surface
   dt = time step
   ti[1..nz] = thermal inertia [J m^-2 K^-1 s^-1/2]
   rhoc[1..nz] = rho*c  where rho=density [kg m^-3] and
                              c=specific heat [J K^-1 kg^-1]
   ti and rhoc are not allowed to vary in the layers immediately adjacent
               to the surface or the bottom
   T[0..nz] = vertical temperature profile [K] (in- and output)
   Qn,Qnp1 = net solar insolation at time steps n and n+1 [Watts/m^2]
   emiss = emissivity
   Fgeotherm = geothermal heat flux at bottom boundary [W/m^2]
   Fsurf = heat flux at surface [W/m^2]  (output)

   Grid: surface is at z=0
         z[0]=0, z[2]=3*z[1], i.e., the top layer has half the width
         T[0] is at z[0]=0; ...; T[i] is at z[i]
         rhoc[i], ti[i] are midway between z[i-1] and z[i]

   input arrays go from 1...nz (as in the Fortran version),
         zeroth elements are not used, except that
         T[0] is the surface temperature

   originally written by Samar Khatiwala, 2001
   extended to variable thermal properties
         and irregular grid by Norbert Schorghofer
   converted from Fortran to C in 2019
************************************************************************/

	// TODO: using variable-length arrays below... should either pass
	// in a workspace or allocate on the heap (preferably the former)
	int i, iter;
	const double sigSB = 5.6704e-8;
	double k1, Tr;
	double arad, brad, ann, annp1, bn, buf, dz, beta;

	double *a = malloc((nz + 1)*sizeof(double));
	double *b = malloc((nz + 1)*sizeof(double));
	double *c = malloc((nz + 1)*sizeof(double));
	double *r = malloc((nz + 1)*sizeof(double));
	double *k = malloc((nz + 1)*sizeof(double));
	double *alpha = malloc((nz + 1)*sizeof(double));
	double *gamma = malloc((nz + 1)*sizeof(double));
	double *Told = malloc((nz + 1)*sizeof(double));

    double dT300, cp;
	double *rho = malloc((nz + 1)*sizeof(double));
	double *Kc = malloc((nz + 1)*sizeof(double));
    const double Kd = 5.7E-3; // Wm^-1K^-1
    const double Ks = 4.E-4; // Wm^-1K^-1
    const double H = 0.06; // meters
    const double rho_d = 1700.; // 1.7 g/cm3 = *10^6/10^3 kg/m3
    const double rho_s = 1000.; // 1 g/cm3
    const double B = 4.2E-11; // Wm^-1K^-4

	/* set some values */
//    printf("nz=%d, rhoc_old=960000 k_old=%f \n", nz, 120 * 120 / 960000.);
    for(int i = 0; i < nz; i++) {
        rho[i] = rho_d - (rho_d-rho_s)*exp(-z[i]/H);  // eq 9 Hayne15
        Kc[i] = Kd - (Kd-Ks)*(rho_d - rho[i])/(rho_d-rho_s); // eq 10 Hayne15

        dT300 = (T[i] - 300.)/300.;
        cp = 4184 * (0.1812+0.1191*dT300+0.0176*pow(dT300,2)+0.2721*pow(dT300,3)+0.1869*pow(dT300,4)); // Eq 6 Ledlow92 [4184 cal g^-1 K^-1 --> J kg^-1 K^-1]
        k[i] = Kc[i] + B*pow(T[i],3);  //  eq 11 Hayne15
        rhoc[i] = rho[i]*cp;
//        k[i] *= 3.;
//        rhoc[i] *= 1.7;
//        k[i] = 120.;
//        rhoc[i] = 960000.;
//        printf("%d rhoc_new=%f k=%f \n", i, rhoc[i], k[i]);
    }
    rhoc[0] = rhoc[1];
    rhoc[nz] = rhoc[nz-1];
//    exit(1);

	dz = 2.*z[1];
	beta = dt / rhoc[1] / (2.*dz*dz);  // assumes rhoc[0]=rhoc[1]
	alpha[1] = beta * k[2];
	gamma[1] = beta * k[1];
	for (i=2; i<nz; i++) {
		buf = dt / (z[i+1] - z[i-1]);
		alpha[i] = 2 * k[i+1] * buf / (rhoc[i] + rhoc[i+1]) / (z[i+1] - z[i]);
		gamma[i] = 2 * k[i] * buf / (rhoc[i] + rhoc[i+1]) / (z[i] - z[i-1]);
	}
	buf = dt / (z[nz] - z[nz-1]) / (z[nz] - z[nz-1]);
	gamma[nz] = k[nz] * buf / (2 * rhoc[nz]);  // assumes rhoc(nz+1)=rhoc(nz)

	k1 = k[1] / dz;

	/* elements of tridiagonal matrix */
	a[0] = NAN;
	b[0] = NAN;
	c[0] = NAN;
	for (i=1; i<=nz; i++) {
		a[i] = -gamma[i];  //  a[1] is not used
		b[i] = 1. + alpha[i] + gamma[i];  //  b[1] has to be reset at ever
		c[i] = -alpha[i];  //  c[nz] is not used
	}
	c[nz] = NAN;
	b[nz] = 1. + gamma[nz];

	Tr = T[0];    //   'reference' temperature
	iter = 0;
	for (i=1; i<=nz; i++) Told[i] = T[i];

  lbpredcorr:
    // update with new T
    for(int i = 0; i < nz; i++) {
        dT300 = (T[i] - 300.)/300.;
        cp = 4184 * (0.1812+0.1191*dT300+0.0176*pow(dT300,2)+0.2721*pow(dT300,3)+0.1869*pow(dT300,4)); // Eq 6 Ledlow92 [4184 cal g^-1 K^-1 --> J kg^-1 K^-1]

        k[i] = Kc[i] + B*pow(T[i],3);  //  eq 11 Hayne15
        rhoc[i] = rho[i]*cp;
//        k[i] = 120.;
//        rhoc[i] = 960000.;
//        k[i] *= 3.;
//        rhoc[i] *= 1.7;
//        printf("%d rhoc_new_lbpredcorr@%d=%f k=%f \n", i, iter, rhoc[i], k[i]);
//        exit(1);
        }
    rhoc[0] = rhoc[1];
    rhoc[nz] = rhoc[nz-1];

#ifdef VERBOSE
	printf("iter = %d\n", iter);
#endif

	/* Emission */
	arad = -3 * emiss * sigSB * Tr * Tr * Tr * Tr;
	brad = 2 * emiss * sigSB * Tr * Tr * Tr;
	ann = (Qn - arad) / (k1 + brad);
	annp1 = (Qnp1 - arad) / (k1 + brad);
	bn = (k1 - brad) / (k1 + brad);
	b[1] = 1. + alpha[1] + gamma[1] - gamma[1] * bn;

	/* Set RHS */
	r[0] = NAN;
	r[1] = gamma[1] * (annp1 + ann) +
		(1. - alpha[1] - gamma[1] + gamma[1] * bn) * T[1] + alpha[1] * T[2];
	for (i=2; i<nz; i++) {
		r[i] = gamma[i] * T[i-1] + (1 - alpha[i] - gamma[i]) * T[i] + alpha[i] * T[i+1];
	}

#ifdef VERBOSE
	printf("rhoc[nz] = rhoc[%d] = %g\n", nz, rhoc[nz]);
#endif

	r[nz] = gamma[nz] * T[nz-1] + (1. - gamma[nz]) * T[nz] +
		dt / rhoc[nz] * Fgeotherm / (z[nz] - z[nz-1]);   // assumes rhoc[nz+1]=rhoc[nz]

#ifdef VERBOSE
	for (size_t j = 0; j <= nz; ++j)
		printf("T[%lu] = %g\n", j, T[j]);

	for (size_t j = 0; j <= nz; ++j)
		printf("a[%lu] = %g\n", j, a[j]);

	for (size_t j = 0; j <= nz; ++j)
		printf("b[%lu] = %g\n", j, b[j]);

	for (size_t j = 0; j <= nz; ++j)
		printf("c[%lu] = %g\n", j, c[j]);

	for (size_t j = 0; j <= nz; ++j)
		printf("r[%lu] = %g\n", j, r[j]);
#endif

	/*  Solve for T at n+1 */
	tridag(a, b, c, r, T, (unsigned long)nz);  // update by tridiagonal inversion

#ifdef VERBOSE
	for (size_t j = 0; j <= nz; ++j)
		printf("T[%lu] = %g\n", j, T[j]);
#endif

	T[0] = 0.5 * (annp1 + bn * T[1] + T[1]);

#ifdef VERBOSE
	printf("T[0] = %g\n", T[0]);
#endif

	/* iterative predictor-corrector */
	if ((T[0] > 1.2*Tr || T[0] < 0.8*Tr) && iter<10) {  // linearization error expected
		/* redo until Tr is within 20% of new surface temperature
		   (under most circumstances, the 20% threshold is never exceeded)
		*/
		iter++;
		Tr = sqrt(Tr * T[0]);  // linearize around an intermediate temperature
		for (i=1; i<=nz; i++) T[i] = Told[i];
		goto lbpredcorr;
	}
//    if (iter>=9) printf("consider taking shorter time steps %d\n",iter);

	*Fsurf = -k[1] * (T[1] - T[0]) / z[1];  // heat flux into surface

} /* conductionQ */

