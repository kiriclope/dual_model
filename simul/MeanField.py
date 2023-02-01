from __future__ import division
import numpy as np
import pylab as plt
from scipy.optimize import fsolve, brenth, root
from scipy.integrate import quad, simps
from scipy.special import erfc, erfcinv
from multiprocessing import Pool
from functools import partial 
import sys, ipdb

def MyErfc(z):
    # normal CDF
    return 0.5 * erfc(z / np.sqrt(2.0))

def MyErfc3(z):
    # normal CDF sqrd
    return (0.5 * erfc(z / np.sqrt(2.0))) **2


def MyErfc2(z):
    # normal CDF sqrd
    return (MyErfc(z)) **2

def QFuncInv(z):
    return np.sqrt(2.0) * erfcinv(z * 2.0)

def alphaF(EorI):
    # out = np.dot(Jab**2, mFourier0 + p * mFourier1 * np.cos(2 * phi))
    out = np.dot(Jab**2, mFourier0) + Ea**2  * mExtZero
    return np.abs(out[EorI])

def MeanInput(uintial, *args):
    uE0, uE1, uI0, uI1 = uintial
    mFourier0, mFourier1 = args
    funcE0 = lambda phi: ((1.0 / (np.pi)) * MyErfc((1.0 - uE0 - uE1 * np.cos(2 * phi)) / np.sqrt(alphaF(0))))
    funcE1 = lambda phi: ((2.0 / (np.pi)) * MyErfc((1.0 - uE0 - uE1 * np.cos(2 * phi)) / np.sqrt(alphaF(0))) * np.cos(2 * phi))
    funcI0 = lambda phi: ((1.0 / (np.pi)) * MyErfc((1.0 - uI0 - uI1 * np.cos(2 * phi)) / np.sqrt(alphaF(1))))
    funcI1 = lambda phi: ((2.0 / (np.pi)) * MyErfc((1.0 - uI0 - uI1 * np.cos(2 * phi)) / np.sqrt(alphaF(1))) * np.cos(2 * phi))
    a = mFourier0[0] - quad(funcE0, 0, np.pi, epsabs = 1e-8, limit = int(1e8))[0]
    b = mFourier1[0] - quad(funcE1, 0, np.pi, epsabs = 1e-8, limit = int(1e8))[0]
    c = mFourier0[1] - quad(funcI0, 0, np.pi, epsabs = 1e-8, limit = int(1e8))[0]
    d = mFourier1[1] - quad(funcI1, 0, np.pi, epsabs = 1e-8, limit = int(1e8))[0]
    return (a, b, c, d)

# def meanInput(uintial, *args):
#     u0, u1 = uintial
#     mFourier0, mFourier1 = args
#     func0 = lambda phi: ((1.0 / (np.pi)) * MyErfc((1.0 - u0 - u1 * np.cos(2 * phi)) / np.sqrt(alphaF(EorI))))
#     func1 = lambda phi: ((2.0 / (np.pi)) * MyErfc((1.0 - u0 - u1 * np.cos(2 * phi)) / np.sqrt(alphaF(EorI))) * np.cos(2 * phi))
#     a = mFourier0[EorI] - quad(func0, 0, np.pi, epsabs = 1e-8, limit = int(1e8))[0]
#     b = mFourier1[EorI] - quad(func1, 0, np.pi, epsabs = 1e-8, limit = int(1e8))[0]
#     return (a, b)

def FixedPointIterateAux(uintial):
    uE0, uE1, uI0, uI1 = uintial
    funcE0 = lambda phi: ((1.0 / (np.pi)) * MyErfc((1.0 - uE0 - uE1 * np.cos(2 * phi)) / np.sqrt(alphaF(0))))
    funcE1 = lambda phi: ((2.0 / (np.pi)) * MyErfc((1.0 - uE0 - uE1 * np.cos(2 * phi)) / np.sqrt(alphaF(0))) * np.cos(2 * phi))
    funcI0 = lambda phi: ((1.0 / (np.pi)) * MyErfc((1.0 - uI0 - uI1 * np.cos(2 * phi)) / np.sqrt(alphaF(1))))
    funcI1 = lambda phi: ((2.0 / (np.pi)) * MyErfc((1.0 - uI0 - uI1 * np.cos(2 * phi)) / np.sqrt(alphaF(1))) * np.cos(2 * phi))
    a = quad(funcE0, 0, np.pi, epsabs = 1e-8, limit = int(1e8))[0]
    b = quad(funcE1, 0, np.pi, epsabs = 1e-8, limit = int(1e8))[0]
    c = quad(funcI0, 0, np.pi, epsabs = 1e-8, limit = int(1e8))[0]
    d = quad(funcI1, 0, np.pi, epsabs = 1e-8, limit = int(1e8))[0]
    return (a, b, c, d)
    
def FixedPointIterate(maxIterations = 1000, tolerance = 1e-6):
    alphaA = np.dot(Jab**2, mFourier0) + (Ea * mExtZero * mExtOne)**2
    uEInitialGuess = [1 -np.sqrt(alphaA[0]) * QFuncInv(mFourier0[0]), kappa * Jab[0, 0] * mFourier1[0]]
    uIInitialGuess = [1 -np.sqrt(alphaA[1]) * QFuncInv(mFourier0[1]), 1e-6]    
    uInital = uEInitialGuess + uIInitialGuess # CONCATENATED LIST
    mE0, mE1, mI0, mI1 = FixedPointIterateAux(uInital)
    print mE0, mE1, mI0, mI1
    print 'M'*25
    counter = 0
    mE0Old = mE0
    mE1Old = mE1
    mI0Old = mI0
    mI1Old = mI1    
    while(counter < maxIterations):
        alphaA = np.dot(Jab**2, np.array([mE0, mFourier0[1]])) + (Ea * mExtZero * mExtOne)**2
        uEInitialGuess = [1 -np.sqrt(alphaA[0]) * QFuncInv(mE0), kappa * Jab[0, 0] * mE1]
        uEInitialGuess = [1 -np.sqrt(alphaA[1]) * QFuncInv(mI0), 1e-20]
        uInital = uEInitialGuess + uIInitialGuess # CONCATENATED LIST
        mE0, mE1, mI0, mI1 = FixedPointIterateAux(uInital)
        if(np.abs(mE0 - mE0Old) < tolerance and (np.abs(mE1 - mE1Old) < tolerance)):
            break;
        else:
            counter += 1
            mE0Old = mE0
            mE1Old = mE1
            mI0Old = mI0
            mI1Old = mI1    
        # print counter, mE0, mE1, mI0, mI1
    print counter, mE0, mE1, mI0, mI1
    return mE0, mE1

def ME1vsKappa(kappaList):
    alphaA = np.dot(Jab**2, mFourier0) + (Ea * mExtZero * mExtOne)**2
    mE1 = []
    for kappa in kappaList:
        uEInitialGuess = [1 -np.sqrt(alphaA[0]) * QFuncInv(mFourier0[0]), kappa * Jab[0, 0] * mFourier1[0]]
        uIInitialGuess = [1 -np.sqrt(alphaA[1]) * QFuncInv(mFourier0[1]), 1e-12]    
        uInital = uEInitialGuess + uIInitialGuess # CONCATENATED LIST
        uE0, uE1, uI0, uI1 =  fsolve(MeanInput, uInital, args =  (mFourier0, mFourier1))
        # func0 = lambda phi: ((1.0 / (np.pi)) * MyErfc((1.0 - uE0 - uE0 * np.cos(2 * phi)) / np.sqrt(alphaF(EorI = 0))))
        func1 = lambda phi: ((2.0 / (np.pi)) * MyErfc((1.0 - uE0 - uE1 * np.cos(2 * phi)) / np.sqrt(alphaF(EorI = 1))) * np.cos(2 * phi))
        mE1.append(quad(func1, 0, np.pi)[0])
    plt.plot(kappaList, mE1, 'ko-')
    plt.show()
    return mE1

    

if __name__ == "__main__":
    Jab = np.array([[1.0, -1.5],
                    [1.0, -1.00]])
    m0 = 0.075
    mExtZero = m0
    mExtOne = m0
    Ea = np.array([2.0, 1.0])
    mFourier0 = -1.0 * np.dot(np.linalg.inv(Jab), Ea) * mExtZero
    p = 5.0 #6.28936
    kappa = p
    mFourier1 = np.array([0.36 * m0, 1e-10])    
    mExtOne = mFourier1[0]
    alphaA = np.dot(Jab**2, mFourier0) + (Ea * mExtZero * mExtOne)**2
    # uEInitialGuess = [1 -np.sqrt(alphaA[0]) * QFuncInv(mFourier0[0]), kappa * Jab[0, 0] * mExtOne]
    # uIInitialGuess = [1 -np.sqrt(alphaA[1]) * QFuncInv(mFourier0[1]), 0]
    uEInitialGuess = [1 -np.sqrt(alphaA[0]) * QFuncInv(mFourier0[0]), kappa * Jab[0, 0] * mFourier1[0]]
    uIInitialGuess = [1 -np.sqrt(alphaA[1]) * QFuncInv(mFourier0[1]), 1e-6]    
    # ipdb.set_trace()
    uInital = uEInitialGuess + uIInitialGuess # CONCATENATED LIST
    print '*'*25
    print '     INITIAL u(theta) '
    print uInital
    print '*'*25    
    uE0, uE1, uI0, uI1 =  fsolve(MeanInput, uInital, args =  (mFourier0, mFourier1))

    print '*'*50
    print 'Solutions uA'
    print uE0, uI0
    print '*'*50    

    
    # u0I, u1I =  fsolve(MeanInput, uIInitialGuess, args = (mFourier0[0], mFourier1[0]))    

    # u1E = kappa * Jab[0, 0] * .05
    func0 = lambda phi: ((1.0 / (np.pi)) * MyErfc((1.0 - uE0 - uE0 * np.cos(2 * phi)) / np.sqrt(alphaF(EorI = 0))))
    func1 = lambda phi: ((2.0 / (np.pi)) * MyErfc((1.0 - uE0 - uE1 * np.cos(2 * phi)) / np.sqrt(alphaF(EorI = 0))) * np.cos(2 * phi))    
    print quad(func0, 0, np.pi)[0]
    print 'mE0 = ', quad(func0, 0, np.pi)[0]
    print 'mE1 = ', quad(func1, 0, np.pi)[0]
    # print 'mE1 = ', u1E / (kappa * Jab[0, 0] * u1E)
    func0 = lambda phi: ((1.0 / (np.pi)) * MyErfc((1.0 - uI0 - uI0 * np.cos(2 * phi)) / np.sqrt(alphaF(EorI = 1))))
    func1 = lambda phi: ((2.0 / (np.pi)) * MyErfc((1.0 - uI0 - uI1 * np.cos(2 * phi)) / np.sqrt(alphaF(EorI = 1))) * np.cos(2 * phi))        
    print 'mI0 = ',  quad(func0, 0, np.pi)[0]
    print 'mI1 = ',  quad(func1, 0, np.pi)[0]    

    print '*'*50
    print mFourier0

    # print '*'*50
    # print ' FIXED POINT ITERATION '
    # FixedPointIterate()

#    ME1vsKappa(np.linspace(1e-3, 8, 10))
