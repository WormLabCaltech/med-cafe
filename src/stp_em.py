"""
An EM class.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

class EM_lsq():
    """
    An EM object to do least squares multiple regression on one dataset
    See:
    http://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf
    and
    https://www.sciencedirect.com/science/article/pii/0167947389900431
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def params(self, t1, b1, b2, m1, m2, sigma21, sigma22):
        self.t1 = t1
        self.b1 = b1
        self.b2 = b2
        self.m1 = m1
        self.m2 = m2
        self.sigma21 = sigma21
        self.sigma22 = sigma22
        self.delta = np.inf

    def weights(self, p_z1J):

        if len(p_z1J) != len(self.X):
            raise ValueError('p_z1J must be the same length as the data')

        # set values to initial guess
        self.p_z1J = p_z1J

    def p(self, x, y, n):
        """P(Y| X, N(mu, sigma), mu = model i, sigma)."""

        def model(n):
            """A line"""
            if n == 1:
                return np.abs(self.m1)*x
            if n == 2:
                return np.max(self.Y) - np.abs(self.m2)*x

        if n == 1:
            mu = model(1)
            s = self.sigma21
            return 1/np.sqrt(s)*np.exp(-0.5*(y - mu)**2/s)
        else:
            mu = model(2)
            s = self.sigma22
            return 1/np.sqrt(s)*np.exp(-0.5*(y - mu)**2/s)

    def E_step(self):
        """E-step for an EM algorithm"""

        def weights(x_j, y_j):
            """Calculate probability of P(model 1) for gene j, p_z1J."""
            t2 = 1-self.t1
            # partition function:
            Z = (self.t1*self.p(x_j, y_j, 1) + t2*self.p(x_j, y_j, 2))

            if Z < 10**-100:
                return 0.5

            # weight calculation:
            p_z1j = self.t1*self.p(x_j, y_j, 1)/Z

            if np.isnan(p_z1j):
                return 0.5

            return p_z1j

        # re-calculate the weights for the EM algorithm:
        for j, x_j in enumerate(self.X):
            y_j = self.Y[j]
            self.p_z1J[j] = weights(x_j, y_j)

    def M_step(self):
        """M-step of the EM algorithm."""

        if np.sum(self.p_z1J) == len(self.X):
            # terminate with a delta of zero if p_z1J is 1
            delta = 0
            return delta

        def lin(b, m):
            """A line"""
            if m >= 0:
                return np.abs(m)*self.X
            if m < 0:
                return np.max(self.Y) - np.abs(m)*self.X

        def find_params(Tij, bi, mi):
            """
            MLE estimates for multiple linear regression. See De Veaux, 1989,
            Mixtures of Linear Regressions.
            """
            wY = np.sum(Tij*self.Y)/np.sum(Tij)
            wX = np.sum(Tij*self.X)/np.sum(Tij)
            wX2 = np.sum(Tij*self.X)**2/np.sum(Tij)

            # Estimate the intercept, slope and variance:
            bi = wY - mi*wX
            mi = (np.sum(Tij*self.X*self.Y) - np.sum(Tij*self.Y)*wX)
            mi = mi/(np.sum(Tij*self.X**2)-wX2)
            sigma2i = np.sum(Tij*(self.Y - lin(bi, mi))**2)/np.sum(Tij)

            # guarantee intercept is within acceptable bounds
    #         bi = np.min([bi, np.max(Y)])
    #         bi = np.max([0, bi])

            return bi, mi, sigma2i

        def logL():
            """Expectation of Log Likelihood. Used to terminate"""
            p_z2j = 1 - self.p_z1J
            t2 = 1 - self.t1
            logL1 = (np.log(self.t1) - 1/2*np.log(self.sigma21) - 1/2*(self.Y
                     - lin(self.b1, self.m1))**2/self.sigma21)
            logL1 = np.sum(self.p_z1J*logL1)
            logL2 = (np.log(t2) - 1/2*np.log(self.sigma22) - 1/2*(self.Y
                     - lin(self.b2, self.m2)**2//self.sigma22))
            logL2 = np.sum(p_z2j*logL2)
            logLikelihood = logL1 + logL2
            return logLikelihood

        # log like at beginning:
        Lbefore = logL()

        # recalculate the mixing parameter, tau:
        self.t1 = 1/len(self.p_z1J)*np.sum(self.p_z1J)

        # re-calculate slopes:
        self.b1, self.m1, self.sigma21 = find_params(self.p_z1J, self.b1,
                                                     self.m1)
        self.b2, self.m2, self.sigma22 = find_params(1-self.p_z1J, self.b2,
                                                     self.m2)

        if self.m1 < 0:
            self.m1 = 0
    #     if m1 > 1:
    #         m1 = 1
        if self.m2 > 0:
            self.m2 = 0
    #     if m2 < -1:
    #         m2 = -1

        # log like at end:
        Lafter = logL()

        self.delta = Lafter - Lbefore

    def EM(self, T=10**-5):
        """Runs the EM algorithm"""
        c = True
        self.i = 0
        while c:
            self.M_step()
            self.E_step()

            if np.abs(self.delta) < T:
                c = False
                break

            self.i += 1

            if self.i > 500:
                m = 'EM didn\'t converge in {0} iterations, delta ={1:.2g}'
                print(m.format(self.i, self.delta))
                break

        m = 'EM algorithm converged in {0} iterations, delta = {1:.2g}'
        print(m.format(self.i, self.delta))

    def plot_results(self):
        plt.scatter(self.X, self.Y, c=(self.p_z1J-0.5 > 0), cmap='PiYG',
                    alpha=0.6)

        if np.sum(self.p_z1J) > 0:
            plt.plot(self.X, np.abs(self.m1)*self.X, label='model 1')
        if np.sum(1-self.p_z1J) > 0:
            plt.plot(self.X, np.max(self.Y) - np.abs(self.m2)*self.X,
                     label='model 2')
        plt.legend()


class EM_tlsq():
    """
    Trimmed lsq EM.
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def params(self, t1, b1, b2, m1, m2, sigma21, sigma22):
        self.t1 = t1
        self.b1 = b1
        self.b2 = b2
        self.m1 = m1
        self.m2 = m2
        self.sigma21 = sigma21
        self.sigma22 = sigma22
        self.delta = np.inf

    def weights(self, p_z1J):

        if len(p_z1J) != len(self.X):
            raise ValueError('p_z1J must be the same length as the data')

        # set values to initial guess
        self.p_z1J = p_z1J
        self.T = 1*(self.p_z1J > 0.5)

    def p(self, x, y, n):
        """P(Y| X, N(mu, sigma), mu = model i, sigma)."""

        def model(n):
            """A line"""
            if n == 1:
                return self.b1 + np.abs(self.m1)*x
            if n == 2:
                return self.b2 - np.abs(self.m2)*x

        if n == 1:
            mu = model(1)
            s = self.sigma21
            return 1/np.sqrt(s)*np.exp(-0.5*(y - mu)**2/s)
        else:
            mu = model(2)
            s = self.sigma22
            return 1/np.sqrt(s)*np.exp(-0.5*(y - mu)**2/s)

    def E_step(self):
        """E-step for an EM algorithm"""

        def weights(x_j, y_j):
            """Calculate probability of P(model 1) for gene j, p_z1J."""
            t2 = 1-self.t1
            # partition function:
            Z = (self.t1*self.p(x_j, y_j, 1) + t2*self.p(x_j, y_j, 2))

            if Z < 10**-100:
                return 0.5

            # weight calculation:
            p_z1j = self.t1*self.p(x_j, y_j, 1)/Z

            if np.isnan(p_z1j):
                return 0.5

            return p_z1j

        # re-calculate the weights for the EM algorithm:
        for j, x_j in enumerate(self.X):
            y_j = self.Y[j]
            self.p_z1J[j] = weights(x_j, y_j)
            self.T[j] = 1*(self.p_z1J[j] > 0.5)

    def M_step(self):
        """M-step of the EM algorithm."""

        if np.sum(self.p_z1J) == len(self.X):
            # terminate with a delta of zero if p_z1J is 1
            delta = 0
            return delta

        def lin(b, m, n):
            """A line"""
            if n == 1:
                return b + np.abs(m)*self.X
            else:
                return b - np.abs(m)*self.X

        def find_params(Tij, bi, mi, n):
            """
            MLE estimates for multiple linear regression. See De Veaux, 1989,
            Mixtures of Linear Regressions.
            """
            if np.sum(Tij) == 0:
                0, 0, 1

            wY = np.sum(Tij*self.Y)/np.sum(Tij)
            wX = np.sum(Tij*self.X)/np.sum(Tij)
            wX2 = np.sum(Tij*self.X)**2/np.sum(Tij)

            # Estimate the intercept, slope and variance:
            bi = wY - mi*wX
            mi = (np.sum(Tij*self.X*self.Y) - np.sum(Tij*self.Y)*wX)
            mi = mi/(np.sum(Tij*self.X**2)-wX2)
            sigma2i = np.sum(Tij*(self.Y - lin(bi, mi, n))**2)/np.sum(Tij)

            # guarantee intercept is within acceptable bounds
            bi = np.min([bi, np.max(self.Y)])
            bi = np.max([0, bi])

            return bi, mi, sigma2i

        def logL():
            """Expectation of Log Likelihood. Used to terminate"""
            T_2 = 1 - self.T
            t2 = 1 - self.t1
            logL1 = (np.log(self.t1) - 1/2*np.log(self.sigma21) - 1/2*(self.Y
                     - lin(self.b1, self.m1, n=1))**2/self.sigma21)
            logL1 = np.sum(self.T*logL1)
            logL2 = (np.log(t2) - 1/2*np.log(self.sigma22) - 1/2*(self.Y
                     - lin(self.b2, self.m2, n=2)**2//self.sigma22))
            logL2 = np.sum(T_2*logL2)
            logLikelihood = logL1 + logL2
            return logLikelihood

        # log like at beginning:
        Lbefore = logL()

        # recalculate the mixing parameter, tau:
        self.t1 = 1/len(self.p_z1J)*np.sum(self.p_z1J)

        # re-calculate slopes:
        self.b1, self.m1, self.sigma21 = find_params(self.T, self.b1,
                                                     self.m1, n=1)
        self.b2, self.m2, self.sigma22 = find_params(1-self.T, self.b2,
                                                     self.m2, n=2)

        if self.m1 < 0:
            self.m1 = 0
        if self.m2 > 0:
            self.m2 = 0

        # log like at end:
        Lafter = logL()

        self.delta = Lafter - Lbefore

    def EM(self, T=10**-5):
        """Runs the EM algorithm"""
        c = True
        self.i = 0
        while c:
            self.M_step()
            self.E_step()

            if np.abs(self.delta) < T:
                c = False
                break

            self.i += 1

            if self.i >= 500:
                m = 'EM didn\'t converge in {0} iterations, delta ={1:.2g}'
                print(m.format(self.i, self.delta))
                break

        if self.i < 500:
            m = 'EM algorithm converged in {0} iterations, delta = {1:.2g}'
            print(m.format(self.i, self.delta))

    def plot_results(self, **kwargs):
        plt.scatter(self.X, self.Y, c=(self.p_z1J-0.5 > 0), cmap='PiYG',
                    **kwargs)

        if np.sum(self.p_z1J) > 0:
            plt.plot(self.X, self.b1 + np.abs(self.m1)*self.X, label='model 1',
                     c='#276419')
        if np.sum(1-self.p_z1J) > 0:
            plt.plot(self.X, self.b2 - np.abs(self.m2)*self.X,
                     label='model 2', c='#8e0152')
        plt.legend()
