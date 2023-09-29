from cloud_model import CloudModel 
from scipy import interpolate
from scipy import optimize
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
from cmath import sqrt
from pathlib import Path

class RelVector(CloudModel):
    """
    Relativistic (alpha~1) model for vector bosons

    All in inputs are in units where black hole mass=1
    alpha is boson mass (times black hole mass)
    abh is black hole dimensionless spin
    Mcloud is boson cloud mass (as fraction of black hole mass) 
    m is azimuthal number of cloud
    """
    def __init__(self):
        """Caching tabulated and fit data"""

        """ 
        m=1 modes: 
        """
        m1_data = np.load(Path(__file__).parent.joinpath('data/m1_pr_mds.npz'))
        m1_wr = m1_data['wr'].flatten()
        m1_wi = m1_data['wi'].flatten()
        m1_a = m1_data['a'].flatten()
        m1_y = m1_data['y'].flatten()
        m1_dwr = m1_data['dwr'].flatten()
        self._f1wr = interpolate.LinearNDInterpolator(list(zip(m1_y,m1_a)),m1_wr)
        self._f1dwr = interpolate.LinearNDInterpolator(list(zip(m1_y,m1_a)),m1_dwr)
        self._f1wi = interpolate.LinearNDInterpolator(list(zip(m1_y,m1_a)),m1_wi)

        """
        m=2 modes
        """
        m2_data = np.load(Path(__file__).parent.joinpath('data/m2_pr_mds.npz'))
        m2_wr = m2_data['wr'].flatten()
        m2_wi = m2_data['wi'].flatten()
        m2_a = m2_data['a'].flatten()
        m2_y = m2_data['y'].flatten()
        m2_dwr = m2_data['dwr'].flatten()
        self._f2wr = interpolate.LinearNDInterpolator(list(zip(m2_y,m2_a)),m2_wr)
        self._f2dwr = interpolate.LinearNDInterpolator(list(zip(m2_y,m2_a)),m2_dwr)
        self._f2wi = interpolate.LinearNDInterpolator(list(zip(m2_y,m2_a)),m2_wi)

        """
        Fit coefficients
        """
        fit_data = np.load(Path(__file__).parent.joinpath('data/pr_fits.npz'))
        self._amat1 = fit_data['amat1']
        self._bmat1 = fit_data['bmat1']
        self._cmat1 = fit_data['cmat1']
        self._amat2 = fit_data['amat2']
        self._bmat2 = fit_data['bmat2']
        self._cmat2 = fit_data['cmat2']

        """
        Radiation data
        """
        sat_flux_data = np.load(Path(__file__).parent.joinpath('data/pr_sat_gw.npz'))
        m1_flux = sat_flux_data['m1_flux']
        m1_mu = sat_flux_data['m1_mu']
        m1_Z2r = sat_flux_data['m1_z2r']
        m1_Z2i = sat_flux_data['m1_z2i']
        m1_Z3r = sat_flux_data['m1_z3r']
        m1_Z3i = sat_flux_data['m1_z3i']
        m2_flux = sat_flux_data['m2_flux']
        m2_mu = sat_flux_data['m2_mu']
        m2_Z4r = sat_flux_data['m2_z4r']
        m2_Z4i = sat_flux_data['m2_z4i']
        m2_Z5r = sat_flux_data['m2_z5r']
        m2_Z5i = sat_flux_data['m2_z5i']

        self._pwm1 = interpolate.interp1d(m1_mu, m1_flux, kind='cubic')
        self._pwm2 = interpolate.interp1d(m2_mu, m2_flux, kind='cubic')
        self._z2rm1 = interpolate.interp1d(m1_mu, m1_Z2r, kind='cubic')
        self._z2im1 = interpolate.interp1d(m1_mu, m1_Z2i, kind='cubic')
        self._z3rm1 = interpolate.interp1d(m1_mu, m1_Z3r, kind='cubic')
        self._z3im1 = interpolate.interp1d(m1_mu, m1_Z3i, kind='cubic')
        self._z4rm2 = interpolate.interp1d(m2_mu, m2_Z4r, kind='cubic')
        self._z4im2 = interpolate.interp1d(m2_mu, m2_Z4i, kind='cubic')
        self._z5rm2 = interpolate.interp1d(m2_mu, m2_Z5r, kind='cubic')
        self._z5im2 = interpolate.interp1d(m2_mu, m2_Z5i, kind='cubic')

    def max_azi_num(self):
        """Maximum azimuthal number the model is defined for"""
        return 2
    def max_spin(self):
        """Maximum spin the model is defined for"""
        return 0.995
    def omega_real(self, m, alpha, abh, Mcloud):
        """Returns real frequency of cloud oscillation"""
        yl, alphamax, Oh = self._y(m, alpha, abh)
        dwr = self._deltaomega(m, alpha, abh)
        if (m==1):
            if (alpha>=0.05 and abh>=self._aswitch(m)):
                wr = alpha*self._f1wr(yl, abh)
            else:
                # Ensures output fails sup. condition if alpha to large
                if (not(self._maxalphaInterp(m, alpha))):
                    return 1e10
                wr = alpha*(1.0-alpha**2/2.0-35.0*alpha**4/24.0+8.0*alpha**5*abh/3.0)
                wr += alpha*0.9934826313642444*alpha**5*(abh*np.sqrt(1-abh**2)-abh)
                for p in range(6, 9):
                    for q in range(0, 4):
                        wr += alpha**(p+1)*(1.0-abh**2)**(q/2.0)*self._amat1[p-6,q]
            return wr+Mcloud*dwr
        elif (m==2):
            if (alpha>=0.25 and abh>=self._aswitch(m)):
                wr = alpha*self._f2wr(yl, abh)
            else:
                # Ensures output fails sup. condition if alpha to large
                if (not(self._maxalphaInterp(m, alpha))):
                    return 1e10
                wr = alpha*(1.0-alpha**2/8.0-143.0*alpha**4/1920.0+alpha**5*abh/15.0)
                wr += alpha*0.03298155184748759*alpha**5*(abh*np.sqrt(1-abh**2)-abh)
                for p in range(6, 9):
                    for q in range(0, 4):
                        wr += alpha**(p+1)*(1.0-abh**2)**(q/2.0)*self._amat2[p-6,q]
            return wr+Mcloud*dwr
        else:
            raise ValueError("Azimuthal index too large")
    def domegar_dmc(self, m, alpha, abh, Mcloud):
        """Returns derivative of real frequency of cloud w.r.t. cloud mass"""
        return self._deltaomega(m, alpha, abh)
    def omega_imag(self, m, alpha, abh):
        """Returns imaginary frequency, 
           i.e. growth rate of superradiant instability"""
        yl, alphamax, Oh = self._y(m, alpha, abh)
        wr = self.omega_real(m, alpha, abh, 0)
        if (m==1):
            if (alpha>=0.05 and abh>=self._aswitch(m)):
                return -np.exp(self._f1wi(yl, abh))*(wr-m*Oh)
            else:
                wi = 1.0
                fm = 1-abh**2+(abh*m-2.0*wr*(1.0+np.sqrt(1.0-abh**2)))**2
                for p in range(1, 11):
                    for q in range(0, 2):
                        wi += alpha**p*(abh**(q+1)*self._bmat1[q,p-1]+self._cmat1[q,p-1]*(1.0-abh**2)**(q/2.0))
                wi *= -4.0*(1.0+np.sqrt(1.0-abh**2))*fm*alpha**7*(wr-m*Oh)
                return wi
        elif (m==2):
            if (alpha>=0.25 and abh>=self._aswitch(m)):
                return -np.exp(self._f2wi(yl, abh))*(wr-m*Oh)
            else:
                wi = 1.0
                for p in range(5, 11):
                    for q in range(0, 2):
                        wi += alpha**p*(abh**(q+1)*self._bmat2[q,p-5]+self._cmat2[q,p-5]*(1.0-abh**2)**(q/2.0))
                fm = 1.0-abh**2+(abh*m-2.0*wr*(1.0+np.sqrt(1.0-abh**2)))**2
                fm *= 4.0*(1.0-abh**2)+(abh*m-2.0*wr*(1.0+np.sqrt(1.0-abh**2)))**2
                wi *= fm*(1.0+np.sqrt(1.0-abh**2))*alpha**11*(-wr+m*Oh)/864.0
                return wi
        else:
            raise ValueError("Azimuthal index too large")
    def power_gw(self, m, alpha, abh):
        """Returns gravitational wave power, scaled to Mcloud=1 
        Using: omega_R(Mcloud=0), valid only at saturation"""
        if (m==1):
            if (alpha<0.17):
                a10 = 23.6070141940415
                a11 = -115.398015735204
                a12 = 222.729191099650
                return a10*alpha**10+a11*alpha**11+a12*alpha**12
            else:
                return self._pwm1(alpha)
        elif (m==2):
            if (alpha<0.45):
                a14 = 0.00174831340006483
                a15 = -0.00676143664936868
                a16 = 0.00696518854011634
                return a14*alpha**14+a15*alpha**15+a16*alpha**16
            else:
                return self._pwm2(alpha) 
        else:
            raise ValueError("Azimuthal index too large")
    def strain_sph_harm(self, m, alpha, abh):
        """Returns e^{iwt}R h^{\ell 2m} (-2-weighted spherical harmonic components)"""
        spsat = self._spinsat(m, alpha)
        wr = 2.0*self.omega_real(m, alpha, spsat, 0)
        if (m==1):
            if (alpha<0.17):
                z2abs = 2.0*np.pi*np.sqrt(wr**2*self.power_gw(m, alpha, spsat))
                return 2.0*np.array([z2abs,0])/(np.sqrt(2.0*np.pi)*wr**2)
            else:
                z2 = self._z2rm1(alpha)+1j*self._z2im1(alpha)
                z3 = self._z3rm1(alpha)+1j*self._z3im1(alpha)
                return -2.0*np.array([z2, z3])/(np.sqrt(2.0*np.pi)*wr**2)
        elif (m==2):
            if (alpha<0.45):
                z4abs = 2.0*np.pi*np.sqrt(wr**2*self.power_gw(m, alpha, spsat))
                return 2.0*np.array([z4abs,0])/(np.sqrt(2.0*np.pi)*wr**2)
            else:
                z4 = self._z4rm2(alpha)+1j*self._z4im2(alpha)
                z5 = self._z5rm2(alpha)+1j*self._z5im2(alpha)
                return -2.0*np.array([z4, z5])/(np.sqrt(2.0*np.pi)*wr**2)
        else:
            raise ValueError("Azimuthal index too large")
    def _aswitch(self, m):
        return 0.6
    def _maxalphaInterp(self, m, alpha):
        """Returns True if alpha is below the saturation-alpha for each m at abh=aswitch(m)"""
        if (m==1):
            return alpha<0.18
        elif (m==2):
            return alpha<0.35
        else:
            raise ValueError("Azimuthal index too large")
    def _y(self, m, alpha, abh, beta=0.91):
        """Returns utility parameter y, approximate maximal alpha, horizon frequency"""
        if (m==1):
            alpha0 = 0.05
        else:
            alpha0 = 0.25
        Oh = 0.5*abh/(1.0+np.sqrt(1.0-abh**2))
        temp = 9.0*m**3*Oh*beta**2+sqrt(3*m**6*beta**4*(27.0*Oh**2-8.0*beta**2))
        alphamaxC = (-3j+sqrt(3.0))*m**2*beta/(3.0**(5.0/6.0)*temp**(1.0/3.0))\
                        +(1.0+sqrt(3.0)*1j)*temp**(1.0/3.0)/(2.0*3.0**(2.0/3.0)*beta)
        alphamax = alphamaxC.real
        yl = (alpha-alpha0)/(alphamax-alpha0)
        return yl, alphamax, Oh
    def _deltaomega(self,m,alpha,abh):
        """Returns the cloud mass-independent frequency shift due to self-gravity"""
        yl, alphamax, Oh = self._y(m, alpha, abh)
        #Factor of 2 below is to go from gravitational potential energy to frequency shift
        if (m==1):
            if (alpha>=0.05 and abh>=self._aswitch(m)):
                return -2.0*alpha*self._f1dwr(yl, abh)
            else:
                dwr = -5.0*alpha**3/16.0
                dwr += -alpha*(-0.011413831834689237149*alpha**3+0.41745609670046540662*alpha**4-2.4755344267838781391*alpha**5+2.5945305106065412737*alpha**6)
                return 2.0*dwr
        elif (m==2):
            if (alpha>=0.25 and abh>=self._aswitch(m)):
                return -2.0*alpha*self._f2dwr(yl, abh)
            else:
                dwr = -93*alpha**3/1024.0
                dwr += -alpha*(0.0031473331860228763446*alpha**3-0.0062807699451029961463*alpha**4-0.011951847200246173628*alpha**5)
                return 2.0*dwr
        else:
            raise ValueError("Azimuthal index too large")
    def _alphasat(self, m, abh):
        """Bisection to find saturation point, returns alpha at saturation"""
        yh, amaxh, Ohh = self._y(m, 0, abh)
        yl, amaxl, Ohl = self._y(m, 0, abh, 1.0)
        def _sat(al):
            return self.omega_real(m, al, abh, 0)-m*0.5*abh/(1.0+np.sqrt(1.0-abh**2))
        return optimize.bisect(_sat, amaxl, amaxh) 
    def _spinsat(self, m, al):
        """Bisection to find saturation point, returns spin at saturation"""
        def _sat(abh):
            return self.omega_real(m, al, abh, 0)-m*0.5*abh/(1.0+np.sqrt(1.0-abh**2))
        return optimize.bisect(_sat, self.max_spin(), 0)
