from math import *


def generalorbit(edays,a,ecc,omega,eps):
    #==============================================================
    # Function that returns distance, longitude, and declination of 
    # the sun in planetocentric coordinates from orbital elements
    #
    # INPUTS: 
    # edays = time since perihelion (earth days)
    # a = semimajor axis (AU)
    # ecc = eccentricity
    # omega = Ls of perihelion, relative to equinox (radians)
    # eps = obliquity (radians)
    #
    # OUTPUTS:
    # Ls = areocentric longitude (radians)
    # dec = planetocentric solar declination (radians)
    # r = heliocentric distance (AU)
    #==============================================================

    # T ... orbital period (days)
    T = sqrt(4*pi**2/(6.674e-11*1.989e30)*(a*149.598e9)**3)/86400.

    # M ... mean anomaly (radians)
    M = 2*pi*edays/T  # M=0 at perihelion

    # E ... eccentric anomaly 
    # solve M = E - ecc*sin(E) by Newton method
    E = M
    for j in range(0,10):
        Eold = E
        E = E - (E - ecc*sin(E) - M)/(1.-ecc*cos(E))
        if abs(E-Eold)<1.e-8:
            break
        
    # nu ... true anomaly
    #nu = acos(cos(E)-ecc/(1.-ecc*cos(E)))
    #nu = sqrt(1-ecc^2)*sin(E)/(1.-ecc*cos(E))
    #nu = atan(sqrt(1-ecc^2)*sin(E)/(1-cos(E)))
    nu = 2*atan( sqrt((1.+ecc)/(1.-ecc)) * tan(E/2.) )

    #r = a*(1.-ecc**2)/(1.+ecc*cos(nu))
    r = a*(1-ecc*cos(E))
    Ls = fmod(nu + omega, 2*pi)   
    dec = asin(sin(eps)*sin(Ls))

    return Ls, dec, r



def equatorial2sundir(decl, HA):
    #=========================================================================
    # converts declination and hour angle to Cartesian direction in body frame
    #     decl: solar declination [radians]
    #     HA: hour angle [radians from noon, clockwise]
    #=========================================================================

    x = cos(decl)*cos(HA)
    y = cos(decl)*sin(HA)
    z = sin(decl)

    #norm = sqrt( x**2 + y**2 + z**2 )
    #print(norm)
    #assert fabs(norm-1.)<1e-6

    # return direction of sun
    return x, y, z

