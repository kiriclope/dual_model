from __future__ import division
import numpy as np
import pylab as plt
from scipy.optimize import fsolve, brenth, root
from scipy.integrate import quad, simps
from scipy.special import erfc, erfcinv
from multiprocessing import Pool
from functools import partial 
import sys, ipdb

hermiteDeg = 30
dim = 2
const = np.pi**(-0.5 * dim)
hermiteX, hermiteWeights = np.polynomial.hermite.hermgauss(hermiteDeg)
hermiteX = hermiteX #* np.sqrt(2.0)
hermiteWeights = hermiteWeights #/ np.sqrt(np.pi)
hermiteXn = np.array(list(itertools.product(*(hermiteX,) * dim)))
hermiteWn = np.prod(np.array(list(itertools.product(*(hermiteWeights, ) * dim))), 1)


def MyErfc(z):
    # normal CDF
    return 0.5 * erfc(z / np.sqrt(2.0))

def MyErfc2(z):
    # normal CDF sqrd
    return (MyErfc(z)) **2

def QFuncInv(z):
    return np.sqrt(2.0) * erfcinv(z * 2.0)

def alphaF(EorI):
    # out = np.dot(Jab**2, mFourier0 + p * mFourier1 * np.cos(2 * phi))
    out = np.dot(Jab**2, mFourier0) + Ea**2  * mExtZero
    return np.abs(out[EorI])

def alphaFunc(EorI, mA0):
    # out = np.dot(Jab**2, mFourier0 + p * mFourier1 * np.cos(2 * phi))
    out = np.dot(Jab**2, mA0) + Ea**2  * mExtZero
    return np.abs(out[EorI])

def BetaOfPhi(delta, q0, q1, EorI):
    qExt0 = mExtZero**2 + 0.500 * mExtOne**2 * np.cos(4.0 * delta)
    out = np.dot(Jab**2, q0) + Ea**2 *  qExt0
    return np.abs(out[EorI])

def BetaOfPhiUnTuned(q0, EorI):
    qExt0 = mExtZero**2 # + 0.500 * mExtOne**2 * np.cos(4.0 * delta)
    out = np.dot(Jab**2, q0) + Ea**2 *  qExt0
    return np.abs(out[EorI])

        
def QuenchedDisorder(qinitial, *args):
    q0E, q0I = qinitial
    [mFourier0, mFourier1, u0E, u0I, mE1, u1I, delta, kappa] = args
    q0 = np.array([q0E, q0I])
    q1 = np.array([0, 0])
    # hermiteDeg = 30
    # hermiteX, hermiteWeights = np.polynomial.hermite.hermgauss(hermiteDeg)
    # hermiteX = hermiteX * np.sqrt(2.0)
    # hermiteWeights = hermiteWeights / np.sqrt(np.pi)
    sqrt2 = np.sqrt(2)
    betaE = lambda delta: BetaOfPhi(delta, q0, q1, EorI = 0)
    betaI = lambda delta: BetaOfPhi(delta, q0, q1, EorI = 1)
    betaEext1 = 0.500 * Ea[0]**2 * mExtOne**2
    betaIext1 = 0.500 * Ea[1]**2 * mExtOne**2
    uBarE = lambda x: uE0 - 1 + np.sqrt(betaE(0) - betaEext1) * x
    uBarI = lambda x: uI0 - 1 + np.sqrt(betaI(0) - betaIext1) * x
    alphaTE = lambda delta: np.abs(alphaA[0] - betaE(delta))
    alphaTI = lambda delta: np.abs(alphaA[1] - betaI(delta))
    print q0E, q0I 
    if mExtOne == 0:
        factorE = lambda phi, delta: 0
        factorI = lambda phi, delta: 0
    else:
        factorE = lambda phi, delta: betaEext1 / (alphaTE(0) + betaEext1 * np.cos(2 * (phi + delta) )**2)
        factorI = lambda phi, delta: betaIext1 / (alphaTI(0) + betaIext1 * np.cos(2 * (phi + delta) )**2)    
    AE = lambda phi, delta: np.sqrt(factorE(phi, delta))
    AI = lambda phi, delta: np.sqrt(factorI(phi, delta))    
    BE = lambda x, phi, delta: np.exp(uBarE(x)**2 * (1 +  factorE(phi, delta) * np.cos(2 * (phi + delta))**2 ) / (2 * alphaTE(0)))
    BI = lambda x, phi, delta: np.exp(uBarI(x)**2 * (1 +  factorI(phi, delta) * np.cos(2 * (phi + delta))**2 ) / (2 * alphaTI(0)))
    if mExtOne == 0:
        CE = lambda x, phi, delta: 0
        CI = lambda x, phi, delta: 0
    else:
        CE = lambda x, phi, delta: (MyErfc((-factorE(phi, delta) * uBarE(x) * np.cos(2 * (phi + delta))) / np.sqrt(betaEext1)) - 1.0 ) * np.cos(2 * (phi + delta))
        CI = lambda x, phi, delta: (MyErfc((-factorI(phi, delta) * uBarI(x) * np.cos(2 * (phi + delta))) / np.sqrt(betaIext1)) - 1.0 ) * np.cos(2 * (phi + delta))

    miE = lambda x, phi, delta: MyErfc((-uBarE(x))/ np.sqrt(alphaTE(0)) ) +  AE(phi, delta) * BE(x, phi, delta) * CE(x, phi, delta)
    miI = lambda x, phi, delta: MyErfc((-uBarI(x))/ np.sqrt(alphaTI(0)) ) +  AI(phi, delta) * BI(x, phi, delta) * CI(x, phi, delta)
    
    funcE0 = lambda phi: (1.0 / np.pi) * np.dot(miE(hermiteX, phi, delta) * miE(hermiteX, phi, -delta), hermiteWeights)
    funcE1 = lambda phi: (1.0 / np.pi) * np.dot(miE(hermiteX, phi, delta) * miE(hermiteX, phi, -delta) * np.cos(2 * phi), hermiteWeights)
    funcI0 = lambda phi: (1.0 / np.pi) * np.dot(miI(hermiteX, phi, delta) * miI(hermiteX, phi, -delta), hermiteWeights)
    funcI1 = lambda phi: (1.0 / np.pi) * np.dot(miI(hermiteX, phi, delta) * miI(hermiteX, phi, -delta) * np.cos(2 * phi), hermiteWeights)
    aE = np.abs(q0E) - np.abs(quad(funcE0, 0, np.pi)[0])
    aI = np.abs(q0I) - np.abs(quad(funcI0, 0, np.pi)[0])
    return (aE, aI)


    
def QuenchedDisorderEval(qinitial, args):
    q0E, q0I = qinitial
    [mFourier0, mFourier1, u0E, u0I, mE1, u1I, delta, kappa] = args
    q0 = np.array([q0E, q0I])
    q1 = np.array([0, 0])
    hermiteDeg = 30
    hermiteX, hermiteWeights = np.polynomial.hermite.hermgauss(hermiteDeg)
    hermiteX = hermiteX * np.sqrt(2.0)
    hermiteWeights = hermiteWeights / np.sqrt(np.pi)
    sqrt2 = np.sqrt(2)
    betaE = lambda delta: BetaOfPhi(delta, q0, q1, EorI = 0)
    betaI = lambda delta: BetaOfPhi(delta, q0, q1, EorI = 1)
    betaEext1 = 0.500 * Ea[0]**2 * mExtOne**2
    betaIext1 = 0.500 * Ea[1]**2 * mExtOne**2
    uBarE = lambda x: uE0 - 1 + np.sqrt(betaE(0) - betaEext1) * x
    uBarI = lambda x: uI0 - 1 + np.sqrt(betaI(0) - betaIext1) * x
    alphaTE = lambda delta: np.abs(alphaA[0] - betaE(delta))
    alphaTI = lambda delta: np.abs(alphaA[1] - betaI(delta))
    if mExtOne == 0:
        factorE = lambda phi, delta: 0
        factorI = lambda phi, delta: 0
    else:
        factorE = lambda phi, delta: betaEext1 / (alphaTE(0) + betaEext1 * np.cos(2 * (phi + delta) )**2)
        factorI = lambda phi, delta: betaIext1 / (alphaTI(0) + betaIext1 * np.cos(2 * (phi + delta) )**2)    
    AE = lambda phi, delta: np.sqrt(factorE(phi, delta))
    AI = lambda phi, delta: np.sqrt(factorI(phi, delta))    
    BE = lambda x, phi, delta: np.exp(uBarE(x)**2 * (1 +  factorE(phi, delta) * np.cos(2 * (phi + delta))**2 ) / (2 * alphaTE(0)))
    BI = lambda x, phi, delta: np.exp(uBarI(x)**2 * (1 +  factorI(phi, delta) * np.cos(2 * (phi + delta))**2 ) / (2 * alphaTI(0)))
    if mExtOne == 0:
        CE = lambda x, phi, delta: 0
        CI = lambda x, phi, delta: 0
    else:
        CE = lambda x, phi, delta: (MyErfc((-factorE(phi, delta) * uBarE(x) * np.cos(2 * (phi + delta))) / np.sqrt(betaEext1)) - 1.0 ) * np.cos(2 * (phi + delta))
        CI = lambda x, phi, delta: (MyErfc((-factorI(phi, delta) * uBarI(x) * np.cos(2 * (phi + delta))) / np.sqrt(betaIext1)) - 1.0 ) * np.cos(2 * (phi + delta))
    miE = lambda x, phi, delta: MyErfc((-uBarE(x))/ np.sqrt(alphaTE(0)) ) +  AE(phi, delta) * BE(x, phi, delta) * CE(x, phi, delta)
    miI = lambda x, phi, delta: MyErfc((-uBarI(x))/ np.sqrt(alphaTI(0)) ) +  AI(phi, delta) * BI(x, phi, delta) * CI(x, phi, delta)
    funcE0 = lambda phi: (1.0 / np.pi) * np.dot(miE(hermiteX, phi, delta) * miE(hermiteX, phi, -delta), hermiteWeights)
    funcE1 = lambda phi: (1.0 / np.pi) * np.dot(miE(hermiteX, phi, delta) * miE(hermiteX, phi, -delta) * np.cos(2 * phi), hermiteWeights)
    funcI0 = lambda phi: (1.0 / np.pi) * np.dot(miI(hermiteX, phi, delta) * miI(hermiteX, phi, -delta), hermiteWeights)
    funcI1 = lambda phi: (1.0 / np.pi) * np.dot(miI(hermiteX, phi, delta) * miI(hermiteX, phi, -delta) * np.cos(2 * phi), hermiteWeights)
    aE = np.abs(quad(funcE0, 0, np.pi)[0])
    aI = np.abs(quad(funcI0, 0, np.pi)[0])
    return (aE, aI)

    
def QuenchedDisorder_OLD(qinitial, *args):
    q0E, q0I = np.abs(qinitial)
    [mFourier0, mFourier1, u0E, u0I, mE1, u1I, delta, kappa] = args
    q0 = np.array([q0E, q0I])
    q1 = np.array([0, 0])
    hermiteDeg = 30
    hermiteX, hermiteWeights = np.polynomial.hermite.hermgauss(hermiteDeg)
    hermiteX = hermiteX * np.sqrt(2.0)
    hermiteWeights = hermiteWeights / np.sqrt(np.pi)
    sqrt2 = np.sqrt(2)
    uFuncE = lambda phi: 1 - uE0 - (sqrt2 * Ea[0] * mExtOne + Jab[0, 0] * kappa * mE1) * np.cos(2 * phi)
    uFuncI = lambda phi: 1 - uI0 - (sqrt2 * Ea[1] * mExtOne) * np.cos(2 * phi)
    denomE = lambda delta: np.sqrt(np.abs(alphaF(EorI = 0) - BetaOfPhi( delta, q0, q1, EorI = 0)))
    denomI = lambda delta: np.sqrt(np.abs(alphaF(EorI = 0) - BetaOfPhi( delta, q0, q1, EorI = 1)))
    betaE = lambda delta: BetaOfPhi(delta, q0, q1, EorI = 0)
    betaI = lambda delta: BetaOfPhi(delta, q0, q1, EorI = 1)
    # betaEext1 = lambda delta: 0.500 * Ea[0]**2 mExtOne**2 * np.cos(4.0 * delta)
    # betaIext1 = lambda delta: 0.500 * Ea[1]**2 mExtOne**2 * np.cos(4.0 * delta)
    betaEext1 = 0.500 * Ea[0]**2 * mExtOne**2
    betaIext1 = 0.500 * Ea[1]**2 * mExtOne**2
    uBarE = lambda x: uE0 - 1 + np.sqrt(betaE(0) - betaEext1) * x
    uBarI = lambda x: uI0 - 1 + np.sqrt(betaI(0) - betaIext1) * x
    alphaTE = lambda delta: alphaA[0] - betaE(delta)
    alphaTI = lambda delta: alphaA[1] - betaI(delta)
    if mExtOne == 0:
        factorE = lambda phi, delta: 0
        factorI = lambda phi, delta: 0
    else:
        factorE = lambda phi, delta: betaEext1 / (alphaTE(0) + betaEext1 * np.cos(2 * phi )**2)
        factorI = lambda phi, delta: betaIext1 / (alphaTI(0) + betaIext1 * np.cos(2 * phi )**2)    
    AE = lambda phi, delta: np.sqrt(factorE(phi, delta))
    AI = lambda phi, delta: np.sqrt(factorI(phi, delta))    
    BE = lambda x, phi, delta: np.exp(uBarE(x)**2 * (1 +  factorE(phi, delta) * np.cos(2 * phi)**2 ) / (2 * alphaTE(0)))
    BI = lambda x, phi, delta: np.exp(uBarI(x)**2 * (1 +  factorI(phi, delta) * np.cos(2 * phi)**2 ) / (2 * alphaTI(0)))
    if mExtOne == 0:
        CE = lambda x, phi, delta: 0
        CI = lambda x, phi, delta: 0
    else:
        CE = lambda x, phi, delta: (MyErfc((-factorE(phi, delta) * uBarE(x) * np.cos(2 * phi)) / np.sqrt(betaEext1)) - 1.0 ) * np.cos(2 * phi)
        CI = lambda x, phi, delta: (MyErfc((-factorI(phi, delta) * uBarI(x) * np.cos(2 * phi)) / np.sqrt(betaIext1)) - 1.0 ) * np.cos(2 * phi)
    miE = lambda x, phi, delta: MyErfc((-uBarE(x))/ np.sqrt(alphaTE(0)) ) +  AE(phi, delta) * BE(x, phi, delta) * CE(x, phi, delta)
    miI = lambda x, phi, delta: MyErfc((-uBarI(x))/ np.sqrt(alphaTI(0)) ) +  AI(phi, delta) * BI(x, phi, delta) * CI(x, phi, delta)    
    funcE0 = lambda phi: (1.0 / np.pi) * np.dot(miE(hermiteX, phi, delta) * miE(hermiteX, phi, -delta), hermiteWeights)
    funcE1 = lambda phi: (1.0 / np.pi) * np.dot(miE(hermiteX, phi, delta) * miE(hermiteX, phi, -delta) * np.cos(2 * phi), hermiteWeights)
    funcI0 = lambda phi: (1.0 / np.pi) * np.dot(miI(hermiteX, phi, delta) * miI(hermiteX, phi, -delta), hermiteWeights)
    funcI1 = lambda phi: (1.0 / np.pi) * np.dot(miI(hermiteX, phi, delta) * miI(hermiteX, phi, -delta) * np.cos(2 * phi), hermiteWeights)
    aE = quad(funcE0, 0, np.pi)[0] - q0E
    aI = quad(funcI0, 0, np.pi)[0] - q0I
    return (aE, aI)
    
def QuenchedDisorderEval_Old(qinitial, args):
    q0E, q0I = qinitial
    [mFourier0, mFourier1, u0E, u0I, mE1, u1I, delta, kappa] = args
    q0 = np.array([q0E, q0I])
    q1 = np.array([0, 0])
    hermiteDeg = 30
    hermiteX, hermiteWeights = np.polynomial.hermite.hermgauss(hermiteDeg)
    hermiteX = hermiteX * np.sqrt(2.0)
    hermiteWeights = hermiteWeights / np.sqrt(np.pi)
    sqrt2 = np.sqrt(2)
    uFuncE = lambda phi: 1 - uE0 - (sqrt2 * Ea[0] * mExtOne + Jab[0, 0] * kappa * mE1) * np.cos(2 * phi)
    uFuncI = lambda phi: 1 - uI0 - (sqrt2 * Ea[1] * mExtOne) * np.cos(2 * phi)
    denomE = lambda delta: np.sqrt(np.abs(alphaF(EorI = 0) - BetaOfPhi( delta, q0, q1, EorI = 0)))
    denomI = lambda delta: np.sqrt(np.abs(alphaF(EorI = 0) - BetaOfPhi( delta, q0, q1, EorI = 1)))
    betaE = lambda delta: BetaOfPhi(delta, q0, q1, EorI = 0)
    betaI = lambda delta: BetaOfPhi(delta, q0, q1, EorI = 1)
    betaEext1 = 0.500 * Ea[0]**2 * mExtOne**2
    betaIext1 = 0.500 * Ea[1]**2 * mExtOne**2
    uBarE = lambda x: uE0 - 1 + np.sqrt(betaE(0) - betaEext1) * x
    uBarI = lambda x: uI0 - 1 + np.sqrt(betaI(0) - betaIext1) * x
    alphaTE = lambda delta: alphaA[0] - betaE(delta)
    alphaTI = lambda delta: alphaA[1] - betaI(delta)
    if mExtOne == 0:
        factorE = lambda phi, delta: 0
        factorI = lambda phi, delta: 0
    else:
        factorE = lambda phi, delta: betaEext1 / (alphaTE(0) + betaEext1 * np.cos(2 * phi )**2)
        factorI = lambda phi, delta: betaIext1 / (alphaTI(0) + betaIext1 * np.cos(2 * phi )**2)    
    AE = lambda phi, delta: np.sqrt(factorE(phi, delta))
    AI = lambda phi, delta: np.sqrt(factorI(phi, delta))    
    BE = lambda x, phi, delta: np.exp(uBarE(x)**2 * (1 +  factorE(phi, delta) * np.cos(2 * phi)**2 ) / (2 * alphaTE(0)))
    BI = lambda x, phi, delta: np.exp(uBarI(x)**2 * (1 +  factorI(phi, delta) * np.cos(2 * phi)**2 ) / (2 * alphaTI(0)))
    if mExtOne == 0:
        CE = lambda x, phi, delta: 0
        CI = lambda x, phi, delta: 0
    else:
        CE = lambda x, phi, delta: (MyErfc((-factorE(phi, delta) * uBarE(x) * np.cos(2 * phi)) / np.sqrt(betaEext1)) - 1.0 ) * np.cos(2 * phi)
        CI = lambda x, phi, delta: (MyErfc((-factorI(phi, delta) * uBarI(x) * np.cos(2 * phi)) / np.sqrt(betaIext1)) - 1.0 ) * np.cos(2 * phi)
    miE = lambda x, phi, delta: MyErfc((-uBarE(x))/ np.sqrt(alphaTE(0)) ) +  AE(phi, delta) * BE(x, phi, delta) * CE(x, phi, delta)
    miI = lambda x, phi, delta: MyErfc((-uBarI(x))/ np.sqrt(alphaTI(0)) ) +  AI(phi, delta) * BI(x, phi, delta) * CI(x, phi, delta)    
    funcE0 = lambda phi: (1.0 / np.pi) * np.dot(miE(hermiteX, phi, delta) * miE(hermiteX, phi, -delta), hermiteWeights)
    funcE1 = lambda phi: (1.0 / np.pi) * np.dot(miE(hermiteX, phi, delta) * miE(hermiteX, phi, -delta) * np.cos(2 * phi), hermiteWeights)
    funcI0 = lambda phi: (1.0 / np.pi) * np.dot(miI(hermiteX, phi, delta) * miI(hermiteX, phi, -delta), hermiteWeights)
    funcI1 = lambda phi: (1.0 / np.pi) * np.dot(miI(hermiteX, phi, delta) * miI(hermiteX, phi, -delta) * np.cos(2 * phi), hermiteWeights)
    aE = quad(funcE0, 0, np.pi)[0]
    aI = quad(funcI0, 0, np.pi)[0]
    return (aE, aI)

def FPIterateQD(nloops = 10):
    args =  [mFourier0, mFourier1, uE0, uI0, mE1, mI1, delta, kappa]
    for i in range(nloops):
        print i, ':     ', 
        qGuess = [np.random.rand() * 0.001, np.random.rand() * 0.001]
        qEList = []; qIList = []
        for j in range(25):
            qE0, qI0 = QuenchedDisorderEval(qGuess, args)
            qGuess = [qE0, qI0]
            qEList.append(qE0); qIList.append(qI0)
        plt.plot(qEList, 'k-'); plt.plot(qIList, 'r-')
        print qE0, qI0
        # print BetaOfPhi(0, [qE0, qI0], [0, 0], 0), BetaOfPhi(0, [qE0, qI0], [0, 0], 1)    
    return qE0, qI0

def MeanInput(uintial, *args):
    uE0, mE1, uI0, mI1 = uintial    
    mFourier0, mFourier1 = args
    sqrt2 = np.sqrt(1 / 2.0)
    uEofTheta = lambda phi: 1 - uE0 - (sqrt2 * Ea[0] * mExtOne + Jab[0, 0] * kappa * mE1) * np.cos(2 * phi)
    uIofTheta = lambda phi: 1 - uI0 - (sqrt2 * Ea[1] * mExtOne) * np.cos(2 * phi)
    funcE0 = lambda phi: ((1.0 / (np.pi)) * MyErfc(uEofTheta(phi) / np.sqrt(alphaF(0))))
    funcE1 = lambda phi: ((2.0 / (np.pi)) * MyErfc(uEofTheta(phi) / np.sqrt(alphaF(0))) * np.cos(2 * phi))
    funcI0 = lambda phi: ((1.0 / (np.pi)) * MyErfc(uIofTheta(phi) / np.sqrt(alphaF(1))))
    funcI1 = lambda phi: ((2.0 / (np.pi)) * MyErfc(uIofTheta(phi) / np.sqrt(alphaF(1))) * np.cos(2 * phi))
    a = mFourier0[0] - quad(funcE0, 0, np.pi, epsabs = 1e-16, limit = int(1e8))[0]
    b = mE1 - quad(funcE1, 0, np.pi, epsabs = 1e-16, limit = int(1e8))[0]
    c = mFourier0[1] - quad(funcI0, 0, np.pi, epsabs = 1e-16, limit = int(1e8))[0]
    d = mI1 - quad(funcI1, 0, np.pi, epsabs = 1e-16, limit = int(1e8))[0]
    return (a, b, c, d)

def MeanInputAux(uintial):
    uE0, mE1, uI0, mI1 = uintial    
    sqrt2 = np.sqrt(1 / 2.0)
    uEofTheta = lambda phi: 1 - uE0 - (sqrt2 * Ea[0] * mExtOne + Jab[0, 0] * kappa * mE1) * np.cos(2 * phi)
    uIofTheta = lambda phi: 1 - uI0 - (sqrt2 * Ea[1] * mExtOne) * np.cos(2 * phi)    
    funcE0 = lambda phi: ((1.0 / (np.pi)) * MyErfc(uEofTheta(phi) / np.sqrt(alphaF(0))))
    funcE1 = lambda phi: ((2.0 / (np.pi)) * MyErfc(uEofTheta(phi) / np.sqrt(alphaF(0))) * np.cos(2 * phi))
    funcI0 = lambda phi: ((1.0 / (np.pi)) * MyErfc(uIofTheta(phi) / np.sqrt(alphaF(1))))
    funcI1 = lambda phi: ((2.0 / (np.pi)) * MyErfc(uIofTheta(phi) / np.sqrt(alphaF(1))) * np.cos(2 * phi))
    a = quad(funcE0, 0, np.pi, epsabs = 1e-16, limit = int(1e8))[0]
    b = quad(funcE1, 0, np.pi, epsabs = 1e-16, limit = int(1e8))[0]
    c = quad(funcI0, 0, np.pi, epsabs = 1e-16, limit = int(1e8))[0]
    d = quad(funcI1, 0, np.pi, epsabs = 1e-16, limit = int(1e8))[0]
    return (a, b, c, d)

def FixedPointIterateAux(uintial):
    uE0, mE1, uI0, mI1 = uintial
    sqrt2 = np.sqrt(1 / 2.0)
    uEofTheta = lambda phi: 1 - uE0 - (sqrt2 * Ea[0] * mExtOne + Jab[0, 0] * kappa * mE1) * np.cos(2 * phi)
    uIofTheta = lambda phi: 1 - uI0 - (sqrt2 * Ea[1] * mExtOne) * np.cos(2 * phi)    
    funcE0 = lambda phi: ((1.0 / (np.pi)) * MyErfc(uEofTheta(phi) / np.sqrt(alphaF(0))))
    funcE1 = lambda phi: ((2.0 / (np.pi)) * MyErfc(uEofTheta(phi) / np.sqrt(alphaF(0))) * np.cos(2 * phi))
    funcI0 = lambda phi: ((1.0 / (np.pi)) * MyErfc(uIofTheta(phi) / np.sqrt(alphaF(1))))
    funcI1 = lambda phi: ((2.0 / (np.pi)) * MyErfc(uIofTheta(phi) / np.sqrt(alphaF(1))) * np.cos(2 * phi))
    a = quad(funcE0, 0, np.pi, epsabs = 1e-16, limit = int(1e8))[0]
    b = quad(funcE1, 0, np.pi, epsabs = 1e-16, limit = int(1e8))[0]
    c = quad(funcI0, 0, np.pi, epsabs = 1e-16, limit = int(1e8))[0]
    d = quad(funcI1, 0, np.pi, epsabs = 1e-16, limit = int(1e8))[0]
    return (a, b, c, d)
    
def FixedPointIterate(uInital, maxIterations = 1000, nInitialConditions = 10, tolerance = 1e-20):
    alphaA = np.dot(Jab**2, mFourier0) + Ea**2 * mExtZero
    mE0, mE1, mI0, mI1 = FixedPointIterateAux(uInital)
    print mE0, mE1, mI0, mI1
    print 'M'*25
    mE0Old = mE0
    mE1Old = mE1
    mI0Old = mI0
    mI1Old = mI1
    for i in range(nInitialConditions):
        mE0List = []
        mI0List = []
        counter = 0
        uInital = np.random.rand(4, 1) * .1
        print 'uinitial = ', FixedPointIterateAux(uInital)
        while(counter < maxIterations):
            mE0, mE1, mI0, mI1 = FixedPointIterateAux(uInital)
            mE0List.append(mE0); mI0List.append(mI0);
            uEInitialGuess = [1 -np.sqrt(alphaA[0]) * QFuncInv(mE0), mE1]
            uIInitialGuess = [1 -np.sqrt(alphaA[1]) * QFuncInv(mI0), mI1]
            uInital = uEInitialGuess + uIInitialGuess # CONCATENATED LIST
            counter += 1            
            # if(np.abs(mE0 - mE0Old) < tolerance and (np.abs(mE1 - mE1Old) < tolerance)):
            #     break;
            # else:

            #     mE0Old = mE0
            #     mE1Old = mE1
            #     mI0Old = mI0
            #     mI1Old = mI1    
            # print counter, mE0, mE1, mI0, mI1
        plt.plot(mE0List, 'ko-'); plt.plot(mI0List, 'ro-');
        print counter, mE0, mE1, mI0, mI1
    plt.show()
    return mE0, mE1, mI0, mI1

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

def SolveAtKappa(kappa, initialGuess = '', IF_VERBOSE = False):
    if initialGuess == '':
        if mExtOne == 0:
            uEInitialGuess = [1 -np.sqrt(alphaA[0]) * QFuncInv(mFourier0[0]), 1e-6] #mFourier1[0]]
            uIInitialGuess = [1 -np.sqrt(alphaA[1]) * QFuncInv(mFourier0[1]), 1e-6] #1e-6]
        else:
            uEInitialGuess = [1 -np.sqrt(alphaA[0]) * QFuncInv(mFourier0[0]), np.random.rand() * 0.1]
            uIInitialGuess = [1 -np.sqrt(alphaA[1]) * QFuncInv(mFourier0[1]), np.random.rand() * 0.1]
        uInital = uEInitialGuess + uIInitialGuess # CONCATENATED LIST
    else:
        uInital = initialGuess
    out = fsolve(MeanInput, uInital, args =  (mFourier0, mFourier1), full_output = True)
    if out[-2] == 1:
        uE0, mE1, uI0, mI1 = out[0]
    else:
        print out[-1]
    if IF_VERBOSE:
        PrintError(out, 'uE0', 0)
        PrintError(out, 'mE1', 1)
        PrintError(out, 'uI0', 2)
        PrintError(out, 'mI1', 3)    
    return out[0]

def PrintError(outStruct, varName, idx):
    print "-----------------------------------"
    print "Variable        :",  varName
    print "Solution        :", outStruct[0][idx]
    print "error           :", outStruct[1]['fvec'][idx]
    print "Solution Status :", outStruct[-1]
    print "-----------------------------------"    

def SolveAtKappaList(kappaList):
    # vpS = np.vectorize(SolveAtKappa)
    # out = vpS(kappaList)
    # ipdb.set_trace()
    mE1List = []
    uE0List = []
    uI0List = []
    initialGuess = ''
    for kappa in kappaList:
        print 'kappa = ', kappa
        uE0, mE1, uI0, mI1 = SolveAtKappa(kappa, initialGuess)
        initialGuess = [uE0, mE1, uI0, mI1]
        mE1List.append(mE1)
        uE0List.append(uE0)
        uI0List.append(uI0)
    np.save('./data/kappa_vs_mE1_m0%s'%(int(m0 * 1e3)), [uE0List, uI0List, mE1List])
    plt.plot(kappaList, mE1List, 'k.-')
    plt.savefig('./figs/kappa_vs_me1')
    return mE1List

def GenerateTuningCurve(qinitial, *args):
    q0E, q0I = qinitial
    [mFourier0, mFourier1, u0E, u0I, mE1, u1I, delta, kappa] = args
    q0 = np.array([q0E, q0I])
    q1 = np.array([0, 0])
    sqrt2 = np.sqrt(2)
    betaE = lambda delta: BetaOfPhi(delta, q0, q1, EorI = 0)
    betaI = lambda delta: BetaOfPhi(delta, q0, q1, EorI = 1)
    betaEext1 = 0.500 * Ea[0]**2 * mExtOne**2
    betaIext1 = 0.500 * Ea[1]**2 * mExtOne**2
    uBarE = lambda x: uE0 - 1 + np.sqrt(betaE(0) - betaEext1) * x
    uBarI = lambda x: uI0 - 1 + np.sqrt(betaI(0) - betaIext1) * x
    alphaTE = lambda delta: np.abs(alphaA[0] - betaE(delta))
    alphaTI = lambda delta: np.abs(alphaA[1] - betaI(delta))
    print q0E, q0I 
    if mExtOne == 0:
        factorE = lambda phi, delta: 0
        factorI = lambda phi, delta: 0
    else:
        factorE = lambda phi, delta: betaEext1 / (alphaTE(0) + betaEext1 * np.cos(2 * (phi + delta) )**2)
        factorI = lambda phi, delta: betaIext1 / (alphaTI(0) + betaIext1 * np.cos(2 * (phi + delta) )**2)    
    AE = lambda phi, delta: np.sqrt(factorE(phi, delta))
    AI = lambda phi, delta: np.sqrt(factorI(phi, delta))    
    BE = lambda x, phi, delta: np.exp(uBarE(x)**2 * (1 +  factorE(phi, delta) * np.cos(2 * (phi + delta))**2 ) / (2 * alphaTE(0)))
    BI = lambda x, phi, delta: np.exp(uBarI(x)**2 * (1 +  factorI(phi, delta) * np.cos(2 * (phi + delta))**2 ) / (2 * alphaTI(0)))
    if mExtOne == 0:
        CE = lambda x, phi, delta: 0
        CI = lambda x, phi, delta: 0
    else:
        CE = lambda x, phi, delta: (MyErfc((-factorE(phi, delta) * uBarE(x) * np.cos(2 * (phi + delta))) / np.sqrt(betaEext1)) - 1.0 ) * np.cos(2 * (phi + delta))
        CI = lambda x, phi, delta: (MyErfc((-factorI(phi, delta) * uBarI(x) * np.cos(2 * (phi + delta))) / np.sqrt(betaIext1)) - 1.0 ) * np.cos(2 * (phi + delta))
    miE = lambda x, phi, delta: MyErfc((-uBarE(x))/ np.sqrt(alphaTE(0)) ) +  AE(phi, delta) * BE(x, phi, delta) * CE(x, phi, delta)
    # miI = lambda x, phi, delta: MyErfc((-uBarI(x))/ np.sqrt(alphaTI(0)) ) +  AI(phi, delta) * BI(x, phi, delta) * CI(x, phi, delta)
    return 

if __name__ == "__main__":
    Jab = np.array([[1.0, -1.5],
                    [1.0, -1.00]])
    m0 = 0.0750
    mExtZero = m0
    mExtOne = 00.0 #1.0 * m0
    Ea = np.array([2.0, 1.0])
    mFourier0 = -1.0 * np.dot(np.linalg.inv(Jab), Ea) * mExtZero
    p = 0.0 #6.28936
    kappa = p
    mFourier1 = np.array([0.01 * m0, 0.01 * m0])    
    alphaA = np.dot(Jab**2, mFourier0) + Ea**2 * mExtZero**2
    uE0, mE1, uI0, mI1 = SolveAtKappa(kappa)
    print '*'*50
    print 'solution check: mE0, mE1, mI0, mI1'
    print MeanInputAux([uE0, mE1, uI0, mI1])
    print '*'*50
    print '\n\n'
    # SolveAtKappaList(np.arange(12, 0, -1e-6 )) #-1e-6), )

    # ipdb.set_trace()
    delta = 0

    qGuess = [0.01, 0.01]
    out = fsolve(QuenchedDisorder, qGuess, args =  (mFourier0, mFourier1, uE0, uI0, mE1, mI1, delta, kappa), full_output = True)
    qE0, qI0 = np.abs(out[0])


    
    
    print '-'*50
    print out[-1]
    print 'beta = ', BetaOfPhi(0, [qE0, qI0], [0, 0], 0), BetaOfPhi(0, [qE0, qI0], [0, 0], 1)
    print '-'*50

    # ipdb.set_trace()

    qE0, qI0 = FPIterateQD(nloops = 4)
    print 'beta = ', BetaOfPhi(0, [qE0, qI0], [0, 0], 0), BetaOfPhi(0, [qE0, qI0], [0, 0], 1)
    print qE0
    
    plt.show()
    # args =  [mFourier0, mFourier1, uE0, uI0, mE1, mI1, delta, kappa]    
