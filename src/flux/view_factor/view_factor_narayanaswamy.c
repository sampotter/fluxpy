#include "view_factor.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <math.h>

static double const PI = 3.1415926535897932;
static double const PI_SQUARED = 9.8696044010893584;
static double const TWO_PI = 6.2831853071795864;
static double const FOUR_PI = 12.566370614359173;
static double const ALMOST_ZERO = 1e-13;
static double const HALF_TOL = 0.5e-13; // == ALMOST_ZERO/2;

static void sub(double const u[3], double const v[3], double w[3]) {
	w[0] = u[0] - v[0];
	w[1] = u[1] - v[1];
	w[2] = u[2] - v[2];
}

static double normalize(double u[3]) {
	double norm = sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
	u[0] /= norm;
	u[1] /= norm;
	u[2] /= norm;
	return norm;
}

void cross(double const u[3], double const v[3], double w[3]) {
  w[0] = u[1]*v[2] - u[2]*v[1];
  w[1] = u[2]*v[0] - u[0]*v[2];
  w[2] = u[0]*v[1] - u[1]*v[0];
}

static double dot(double const u[3], double const v[3]) {
	return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
}

static void saxpy(double a, double const x[3], double const y[3], double z[3]) {
	z[0] = a*x[0] + y[0];
	z[1] = a*x[1] + y[1];
	z[2] = a*x[2] + y[2];
}

static double dist(double const u[3], double const v[3]) {
	double uv[3] = {v[0] - u[0], v[1] - u[1], v[2] - u[2]};
	return sqrt(uv[0]*uv[0] + uv[1]*uv[1] + uv[2]*uv[2]);
}

// function ClausenIntegral = Cl(theta) % Eq.(26) from paper
static double Cl(double theta) {

//     theta = mod(theta, 2*pi);
	theta = fmod(theta, TWO_PI);

//     chebArg = theta/pi - 1;
	double chebArg = theta/PI - 1;

//     b = [1.865555351433979e-1, 6.269948963579612e-2, 3.139559104552675e-4, ...
//          3.916780537368088e-6, 6.499672439854756e-8, 1.238143696612060e-9, ...
//          5.586505893753557e-13];
	double b[] = {
		1.865555351433979e-1, 6.269948963579612e-2, 3.139559104552675e-4,
		3.916780537368088e-6, 6.499672439854756e-8, 1.238143696612060e-9,
		5.586505893753557e-13
	};

//     % Chebyshev polynomials of degrees 2*n+1 (n=1:6) found using the sym command:
//     % >> chebyshevT((2*(0:6)+1), sym(chebArg));
//     T = [chebArg, 4*chebArg^3 - 3*chebArg, ...
//          16*chebArg^5 - 20*chebArg^3 + 5*chebArg, ...
//          64*chebArg^7 - 112*chebArg^5 + 56*chebArg^3 - 7*chebArg, ...
//          256*chebArg^9 - 576*chebArg^7 + 432*chebArg^5 - 120*chebArg^3 + 9*chebArg, ...
//          1024*chebArg^11 - 2816*chebArg^9 + 2816*chebArg^7 - 1232*chebArg^5 + 220*chebArg^3 - 11*chebArg, ...
//          4096*chebArg^13 - 13312*chebArg^11 + 16640*chebArg^9 - 9984*chebArg^7 + 2912*chebArg^5 - 364*chebArg^3 + 13*chebArg];
	double chebArg2 = chebArg*chebArg;
	double chebArg3 = chebArg2*chebArg;
	double chebArg5 = chebArg3*chebArg2;
	double chebArg7 = chebArg5*chebArg2;
	double chebArg9 = chebArg7*chebArg2;
	double chebArg11 = chebArg9*chebArg2;
	double chebArg13 = chebArg11*chebArg2;
	double T[] = {
		chebArg,
		4*chebArg3 - 3*chebArg,
		16*chebArg5 - 20*chebArg3 + 5*chebArg,
		64*chebArg7 - 112*chebArg5 + 56*chebArg3 - 7*chebArg,
		256*chebArg9 - 576*chebArg7 + 432*chebArg5 - 120*chebArg3 + 9*chebArg,
		1024*chebArg11 - 2816*chebArg9 + 2816*chebArg7 - 1232*chebArg5 +
		  220*chebArg3 - 11*chebArg,
		4096*chebArg13 - 13312*chebArg11 + 16640*chebArg9 - 9984*chebArg7 +
		  2912*chebArg5 - 364*chebArg3 + 13*chebArg
	};

	double b_dot_T = 0;
	for (size_t i = 0; i < 7; ++i) {
		b_dot_T += b[i]*T[i];
	}

//      ClausenIntegral = (theta - pi)*(2 + log((pi^2)/2)) + (2*pi - theta)*log((2*pi - theta)*(1 - ALMOST_ZERO) + ALMOST_ZERO) ...
//          - theta*log(theta*(1 - ALMOST_ZERO) + ALMOST_ZERO) + sum(b.*T);
	return (theta - PI)*(2 + log(PI_SQUARED/2)) +
		(TWO_PI - theta)*log((TWO_PI - theta)*(1 - ALMOST_ZERO) + ALMOST_ZERO) -
		theta*log(theta*(1 - ALMOST_ZERO) + ALMOST_ZERO) + b_dot_T;
// end
}

// function imaginaryPart = imagLi_2(mag, angle) % Eq.(24) from paper
double imagLi_2(double mag, double angle) {
//     if mag > ALMOST_ZERO
	if (mag > ALMOST_ZERO) {
//         omega = atan2(mag*sin(angle), (1 - mag*cos(angle)));
		double omega = atan2(mag*sin(angle), (1 - mag*cos(angle)));
//         imaginaryPart = 0.5*Cl(2*angle) + 0.5*Cl(2*omega) - 0.5*Cl(2*omega + 2*angle) + log(mag)*omega;
		return 0.5*Cl(2*angle) + 0.5*Cl(2*omega) - 0.5*Cl(2*omega + 2*angle) + log(mag)*omega;
//     else
	} else {
//         imaginaryPart = mag*sin(angle);
		return mag*sin(angle);
//     end
	}
// end
}

// function F = f(s, l, alpha, cosAlpha, sinAlpha, d) % Eq.(22b) from paper
double f(double s, double l, double alpha, double cosAlpha, double sinAlpha, double d) {
//     s2 = s^2;
	double s2 = s*s;

//     l2 = l^2;
	double l2 = l*l;

//     d2 = d^2;
	double d2 = d*d;

//     sinAlpha2 = sinAlpha^2;
	double sinAlpha2 = sinAlpha*sinAlpha;

//     wsqrt = sqrt(s2 + d2/sinAlpha2);
	double wsqrt = sqrt(s2 + d2/sinAlpha2);

//     psqrt = sqrt(l2 + d2/sinAlpha2);
	double psqrt = sqrt(l2 + d2/sinAlpha2);

//     if abs(s + wsqrt) > 0
//     wdim = s + wsqrt;
//     else
//     wdim = ALMOST_ZERO;
//     end
	double wdim = fabs(s + wsqrt) > 0 ? s + wsqrt : ALMOST_ZERO; // TODO: weird

//     if abs(l + psqrt) > 0
//     pdim = l + psqrt;
//     else
//     pdim = ALMOST_ZERO;
//     end
	double pdim = fabs(l + psqrt) > 0 ? l + psqrt : ALMOST_ZERO; // TODO: weird

//     F = (0.5*cosAlpha*(s2 + l2) - s*l)*log(s2 + l2 - 2*s*l*cosAlpha + d2) ...
//       + s*sinAlpha*wsqrt*atan2(sqrt(s2*sinAlpha2 + d2), (l - s*cosAlpha)) ...
//       + l*sinAlpha*psqrt*atan2(sqrt(l2*sinAlpha2 + d2), (s - l*cosAlpha)) + s*l ...
//       + 0.5*(d2/sinAlpha)*(imagLi_2((wdim/pdim), alpha) + imagLi_2((pdim/wdim), alpha) - 2*imagLi_2((wdim - 2*s)/pdim, (pi - alpha)));
     return (0.5*cosAlpha*(s2 + l2) - s*l)*log(s2 + l2 - 2*s*l*cosAlpha + d2) +
		 s*sinAlpha*wsqrt*atan2(sqrt(s2*sinAlpha2 + d2), (l - s*cosAlpha)) +
		 l*sinAlpha*psqrt*atan2(sqrt(l2*sinAlpha2 + d2), (s - l*cosAlpha)) +
		 s*l +
		 0.5*(d2/sinAlpha)*(imagLi_2((wdim/pdim), alpha) +
							imagLi_2((pdim/wdim), alpha) -
							2*imagLi_2((wdim - 2*s)/pdim, (PI - alpha)));
// end
}

// function F = fParallel(s, l, d) % Eq.(23) from paper
double fParallel(double s, double l, double d) {
//     if d == 0
//         d = ALMOST_ZERO;
//     end
	if (d == 0) d = ALMOST_ZERO;

//     sMinusl = s - l;
	double sMinusl = s - l;

//     sMinusl2 = sMinusl^2;
	double sMinusl2 = sMinusl*sMinusl;

//     s2 = s^2;
	double s2 = s*s;

//     l2 = l^2;
	double l2 = l*l;

//     d2 = d^2;
	double d2 = d*d;

	double acos_arg = fmin(1.0, sMinusl/sqrt(s2 + l2 - 2*s*l + d2));
	acos_arg = fmax(-1.0, acos_arg);

//     F = 0.5*(sMinusl2 - d2)*log(sMinusl2 + d2) ...
//       - 2*sMinusl*d*acos(sMinusl/sqrt(s2 + l2 - 2*s*l + d2)) + s*l;
	return 0.5*(sMinusl2 - d2)*log(sMinusl2 + d2) -
		2*sMinusl*d*acos(acos_arg) + s*l;
// end
}


void edgePairParameters(double const Po[3], double const Pf[3],
						double const Qo[3], double const Qf[3],
						double *D,
						double sOrigin[3], double sHat[3],
						double lHat[3], double lOrigin[3],
						bool *skew)
{

// function [D, sOrigin, sHat, lHat, lOrigin, skew] = edgePairParameters(Po, Pf, Qo, Qf)
//     % http://geomalgorithms.com/a07-_distance.html
//     % find shortest distance D between line Po+s*u and Qo+t*v for initial
//     %  points Po and Qo, parameters s and t, and vectors u and v

//     u = Pf - Po;
	double u[3];
	sub(Pf, Po, u);

//     v = Qf - Qo;
	double v[3];
	sub(Qf, Qo, v);

//     w = Po - Qo;
	double w[3];
	sub(Po, Qo, w);

//     Plength = norm(u);
//     u = u/Plength;  % make these unit vectors
	normalize(u);

//     Qlength = norm(v);
//     v = v/Qlength;
	normalize(v);

//     a = 1; % dot(u, u)
	double a = 1; // dot(u, u)

//     b = dot(u, v);
	double b = dot(u, v);

//     c = 1; % dot(v, v)
	double c = 1; // dot(v, v)

//     d = dot(u, w);
	double d = dot(u, w);

//     e = dot(v, w);
	double e = dot(v, w);

//     den = a*c - b^2;
	double den = a*c - b*b;

//     % calculate shortest distance between edge rays
//     if den > ALMOST_ZERO
	double s, l;
	if (den > ALMOST_ZERO) {
//         skew = true;
		*skew = true;
//         s = (b*e - c*d)/den;
		s = (b*e - c*d)/den;
//         l = (a*e - b*d)/den;
		l = (a*e - b*d)/den;
//         D = norm(w + s*u - l*v);
		*D = 0;
		for (size_t i = 0; i < 3; ++i) {
			double term = w[i] + s*u[i] - l*v[i];
			*D += term*term;
		}
		*D = sqrt(fmax(0, *D));
//     else % origin is arbitrary if lines are parallel
	} else {
//         skew = false;
		*skew = false;
//     %     s = 1.5*Plength;
//     %     l = 1.5*Qlength;
//         s = 0;
		s = 0;
//         l = e/c;
		l = e/c;
//         D = norm(w - (e/c)*v);
		*D = 0;
		for (size_t i = 0; i < 3; ++i) {
			double term = w[i] - (e/c)*v[i];
			*D += term*term;
		}
		*D = sqrt(fmax(0, *D));
//     end
	}

//     % see Fig 5 in this paper:
//     %   Narayanaswamy, Arvind. "An analytic expression for radiation view
//     %   factor between two arbitrarily oriented planar polygons." International
//     %   Journal of Heat and Mass Transfer 91 (2015): 841-847.
//     % for description of why these values are calculated in this way.

//     % parameter origin is location on edge ray where distance between edges has
//     %  its smallest value
//     sOrigin = Po + u*s;
	saxpy(s, u, Po, sOrigin);
//     lOrigin = Qo + v*l;
	saxpy(l, v, Qo, lOrigin);

//     s_toEnd = norm(Pf - sOrigin);
	double s_toEnd = dist(Pf, sOrigin);
//     l_toEnd = norm(Qf - lOrigin);
	double l_toEnd = dist(Qf, lOrigin);

//     % unit vectors point from parameter origin to furthest of the two vertices
//     if abs(s) < s_toEnd
	if (fabs(s) < s_toEnd)
//         sHat = (Pf - sOrigin)/norm(Pf - sOrigin);
		sub(Pf, sOrigin, sHat);
//     else
	else
//         sHat = (Po - sOrigin)/norm(Po - sOrigin);
		sub(Po, sOrigin, sHat);
//     end
	normalize(sHat);

//     if abs(l) < l_toEnd
	if (fabs(l) < l_toEnd)
//         lHat = (Qf - lOrigin)/norm(Qf - lOrigin);
		sub(Qf, lOrigin, lHat);
//     else
	else
//         lHat = (Qo - lOrigin)/norm(Qo - lOrigin);
		sub(Qo, lOrigin, lHat);
//     end
	normalize(lHat);
// end
}

// function [F_AB] = viewFactor_sfp(tri1, tri2)
double ff_narayanaswamy_impl(double const tri1[3][3], double const tri2[3][3]) {
//     N = dim_A(1);
//     M = dim_B(1);

//     % VIEW FACTOR ANALYICAL CALCULATION

	double p1[3];
	sub(tri1[1], tri1[0], p1);

	double p2[3];
	sub(tri1[2], tri1[0], p2);

	double n1[3];
	cross(p1, p2, n1);

	double area_A = normalize(n1)/2;

//     sumTerms = zeros(N,M);  % terms to sum to yield conductance
	double sumTerms[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

//     skewPairs = zeros(N,M); % tracks which terms come from parallel edges (for debugging)
//	bool skewPairs[3][3] = {{false, false, false}, {false, false, false}, {false, false, false}};

//     for p = 1:M      % loop through vertices of polygon B
	for (size_t p = 0; p < 3; ++p) {

//             r_p = tri2(p,:);
		double r_p[3] = {tri2[p][0], tri2[p][1], tri2[p][2]};

//             if p < M
//                 r_q = tri2(p+1,:);
//             else % loop
//                 r_q = tri2(1,:);
//             end
		size_t q = (p + 1) % 3;
		double r_q[3] = {tri2[q][0], tri2[q][1], tri2[q][2]};

//         for i = 1:N  % loop through vertices of polygon A
		for (size_t i = 0; i < 3; ++i) {

//             r_i = tri1(i,:);
			double const *r_i = tri1[i];

//             % loop pairings of vertices to cycle through edges
//             if i < N
//                 r_j = tri1(i+1,:);
//             else % loop
//                 r_j = tri1(1,:);
//             end
			size_t j = (i + 1) % 3;
			double const *r_j = tri1[j];

//             % check for coincident vertices - nudge polygon B vertices if found
//             if norm(r_i - r_p) < HALF_TOL || norm(r_j - r_p) < HALF_TOL
//                 r_p = r_p + ALMOST_ZERO;
//             elseif norm(r_i - r_q) < HALF_TOL || norm(r_j - r_q) < HALF_TOL
//                 r_q = r_q + ALMOST_ZERO;
//             end
			if (dist(r_i, r_p) < HALF_TOL || dist(r_j, r_p) < HALF_TOL) {
				for (size_t k = 0; k < 3; ++k)
					r_p[k] += ALMOST_ZERO;
			} else if (dist(r_i, r_q) < HALF_TOL || dist(r_j, r_q) < HALF_TOL) {
				for (size_t k = 0; k < 3; ++k)
					r_q[k] += ALMOST_ZERO;
			}

//             % determine parameterized coordinates for each edge, and minimum
//             %  distance between edge rays (edges extended infinitely into space)
//             [dMin, sOrigin, sHat, lHat, lOrigin, skew] = edgePairParameters(r_i, r_j, r_p, r_q);
			double dMin, sOrigin[3], sHat[3], lHat[3], lOrigin[3];
			bool skew;
			edgePairParameters(r_i, r_j, r_p, r_q, &dMin, sOrigin, sHat, lHat, lOrigin, &skew);

//             if skew  % if these edges are NOT parallel...
			if (skew) {
//                 % locate each vertex in the parameterized coordinate system
//                 s_i = dot((r_i - sOrigin), sHat);
//                 s_j = dot((r_j - sOrigin), sHat);
//                 l_p = dot((r_p - lOrigin), lHat);
//                 l_q = dot((r_q - lOrigin), lHat);
				double s_i = dot(r_i, sHat) - dot(sOrigin, sHat);
				double s_j = dot(r_j, sHat) - dot(sOrigin, sHat);
				double l_p = dot(r_p, lHat) - dot(lOrigin, lHat);
				double l_q = dot(r_q, lHat) - dot(lOrigin, lHat);

//                 skewPairs(i,p) = 1;
				// skewPairs[i][p] = true;

//                 cosAlpha = dot(sHat, lHat);
				double cosAlpha = dot(sHat, lHat);

//                 alpha = acos(cosAlpha);
				double alpha = acos(cosAlpha);

//                 sinAlpha = sin(alpha);
				double sinAlpha = sin(alpha);

//                 % Eq.(22a) from paper - calculate final terms that yield the
//                 %  view factor when summed and divided by (4*pi*area)
//                 sumTerms(i,p) = cosAlpha*(f(s_j, l_q, alpha, cosAlpha, sinAlpha, dMin) ...
				sumTerms[i][p] = cosAlpha*(
					  f(s_j, l_q, alpha, cosAlpha, sinAlpha, dMin)
//                     - f(s_i, l_q, alpha, cosAlpha, sinAlpha, dMin) ...
					- f(s_i, l_q, alpha, cosAlpha, sinAlpha, dMin)
//                     - f(s_j, l_p, alpha, cosAlpha, sinAlpha, dMin) ...
					- f(s_j, l_p, alpha, cosAlpha, sinAlpha, dMin)
//                     + f(s_i, l_p, alpha, cosAlpha, sinAlpha, dMin));
					+ f(s_i, l_p, alpha, cosAlpha, sinAlpha, dMin));
//             else     % alternate expression for when alpha approaches zero
			} else {
//                 lHat = sHat; % this is important for the parallel case
				for (size_t k = 0; k < 3; ++k) lHat[k] = sHat[k];
//                 % locate each vertex in the parameterized coordinate system
//                 s_i = dot((r_i - sOrigin), sHat);
//                 s_j = dot((r_j - sOrigin), sHat);
//                 l_p = dot((r_p - lOrigin), lHat);
//                 l_q = dot((r_q - lOrigin), lHat);
				double s_i = dot(r_i, sHat) - dot(sOrigin, sHat);
				double s_j = dot(r_j, sHat) - dot(sOrigin, sHat);
				double l_p = dot(r_p, lHat) - dot(lOrigin, lHat);
				double l_q = dot(r_q, lHat) - dot(lOrigin, lHat);

//                 skewPairs(i,p) = 0;
				// skewPairs[i][p] = false;

//                 sumTerms(i,p) = dot(sHat, lHat)*(fParallel(s_j, l_q, dMin) ...
				sumTerms[i][p] = dot(sHat, lHat)*(
					fParallel(s_j, l_q, dMin)
//                     - fParallel(s_i, l_q, dMin) - fParallel(s_j, l_p, dMin) ...
					- fParallel(s_i, l_q, dMin) - fParallel(s_j, l_p, dMin)
//                     + fParallel(s_i, l_p, dMin));
					+ fParallel(s_i, l_p, dMin));
//             end
			}
//         end
		}
//     end
	}

//     % "radiation conductance" : radUA = area_A*F_AB = area_B*F_BA
//     radUA = abs(sum(sumTerms, 'all'))/(4*pi);
	double radUA = 0;
	for (size_t p = 0; p < 3; ++p)
		for (size_t i = 0; i < 3; ++i)
			radUA += sumTerms[i][p];
	radUA = fabs(radUA);

//     % FINAL CALCULATION
//     if isnan(radUA)
//         error('Unknown error occured.');
//     else
//         F_AB = radUA/area_A;
//     end
	assert(!isnan(radUA));
	return radUA/(FOUR_PI*area_A);
// end
}
