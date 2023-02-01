import sys, ipdb
from scipy.optimize import fsolve
import numpy as np
from scipy.integrate import quad
from scipy.special import erfc
import pylab as plt

import sys
basefolder = "/homecentral/srao/Documents/code/mypybox"
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf

def alphaE(mExtZero, meanFieldRates):
    return JE0**2 * mExtZero + JEE**2 * meanFieldRates[0] + JEI**2 * meanFieldRates[1]
    # return JEE**2 * meanFieldRates[0] + JEI**2 * meanFieldRates[1]    

def MyErfc(z):
    # normal CDF
    return 0.5 * erfc(z / np.sqrt(2.0))

def PrintError(outStruct, varName):
    print "-----------------------------------"
    print "Variable        :",  varName
    print "Solution        :", outStruct[0][0]
    print "error           :", outStruct[1]['fvec']
    print "Solution Status :", outStruct[-1]
    print "-----------------------------------"    
    
def CheckBalConditions(JEE, JEI, JIE, JII, JE0, JI0):
    ipdb.set_trace()            # 
    JE = -JEI/JEE
    JI = -JII/JIE
    E = JE0
    I = JI0
    if not ( (E/I > JE/JI) and (JE/JI > 1) and JE > 1):
    # if((JE < JI) or (E/I < JE/JI) or (JE < 1) or (E/I < 1) or (JE/JE < 1)):
        print "NOT IN BALANCED REGIME!!!!!! "
        raise SystemExit

def Pcritical(uE0, alpha, JEE):
    b = (1 - uE0) / np.sqrt(alpha)
    C0 = np.exp(-b**2 / 2.0) / np.sqrt(2.0 * np.pi)
    num = np.sqrt(alpha)
    denom = C0 * JEE
    return num / denom

def UE0GivenUE1(u0Guess, *args):
    # solve for u_E(0) given U_E(1)
    u0 = u0Guess
    jee, alpha, u1, meanFieldRate = args
    funcME0 = lambda phi: (1.0 / np.pi) * MyErfc((1.0 - u0 - u1 * np.cos(2.0 * phi)) / np.sqrt(alpha))
    return meanFieldRate - quad(funcME0, 0, np.pi)[0]

def Eq0(u0Guess, *args):
    # solve for u_E(0)
    u0 = u0Guess
    jee, alpha, me0 = args
    return me0 - MyErfc((1.0 - u0) / np.sqrt(alpha))
    
def SolveForPCritical(m0List):
    nPoints = 100
    uE1List = np.linspace(-2.0, 1.0, nPoints)
    for l, lM0 in enumerate(m0List):
        meanFieldRates = -1.0 * np.dot(np.linalg.inv(Jab), Ea) * lM0
        alpha_E = alphaE(lM0, meanFieldRates)
        #============================================ 
        uE0Solution = fsolve(Eq0, [0.1], args = (JEE, alphaE(lM0, meanFieldRates), meanFieldRates[0]), full_output = 1)    
        PrintError(uE0Solution, 'u_E(0)')
        print  'mA0 = ', meanFieldRates
        print "alpha = ", alphaE(lM0, meanFieldRates)

        uE0 = uE0Solution[0]
        pCAnalytic = Pcritical(uE0, alpha_E, JEE)
        #------------------------------------------
        uE0List = []
        mE1List = []
        pList = []
        for i, iUE1 in enumerate(uE1List):
            uE0Sol = fsolve(UE0GivenUE1, [0.1], args = (JEE, alpha_E, iUE1, meanFieldRates[0]), full_output = True, xtol = 1e-6)
            if(uE0Sol[-2] == 1): # .i.e IF SOLUTION CONVERGED
                FuncME0 = lambda phi: (1.0 / np.pi) * MyErfc((1.0 - uE0Sol[0][0] - iUE1 * np.cos(2.0 * phi)) / np.sqrt(alpha_E))
                # PrintError(uE0Sol, 'U_E(0)[U_E(1)]')            
                print uE0Sol[0][0], iUE1, quad(FuncME0, 0, np.pi)[0], uE0Sol[-1], uE0Sol[-2]
                uE0List.append((uE0Sol[0][0], iUE1))
                FuncME1 = lambda phi: (2.0 / np.pi) * np.cos(2.0 * phi) * MyErfc((1.0 - uE0Sol[0][0] - iUE1 * np.cos(2.0 * phi)) / np.sqrt(alpha_E))
                me1 = quad(FuncME1, 0, np.pi)[0]
                if(me1 <= meanFieldRates[0] and iUE1 > 0):
                    pList.append(iUE1 / (JEE * me1))
                    mE1List.append(me1)

        if(lM0 == m0List[-1]):
            plt.plot(pList, mE1List, 'bo-', markersize = 4, label = 'numeric')
            plt.vlines(pCAnalytic, 0, meanFieldRates[0], 'k', label = 'analytic')
            plt.legend(loc = 0, frameon = False, numpoints = 1)
            plt.xlabel('p')
            plt.ylabel(r'$m_E^{(1)}$')
            plt.xlim(pCAnalytic[0] - 1, pList[-1] + .5)
            plt.ylim(0, max(mE1List) + 0.05)
        else:
            plt.plot(pList, mE1List, 'bo-', markersize = 4)
            plt.vlines(pCAnalytic, 0, meanFieldRates[0], 'k')
        if(l == 0):
            plt.text(pList[-1] + 0.01, mE1List[-1], r'$m_0 = %s$'%(lM0))
        else:
            plt.text(pList[-1] + 0.01, mE1List[-1], '%s'%(lM0))
    print "--" * 25
    print "m0 = ", m0List[0], "p_critical = ", pCAnalytic[0]
    print "--" * 24    
    # np.save('./data/p_vs_mE1_m0%s'%(int(lM0 * 1e3)), [pList, mE1List, np.ones((len(pList), )) * meanFieldRates[0], np.ones((len(pList), )) * meanFieldRates[1]])
    np.save('./data/p_vs_mE1_m0%s'%(int(lM0 * 1e3)), [pList, mE1List]) #, np.ones((len(pList), )) * meanFieldRates[0], np.ones((len(pList), )) * meanFieldRates[1]])            
    

if __name__ == "__main__":
    JEE = 1.0 
    JIE = 1.0 
    JEI = -1.5 
    JII = -1.0 
    # m0 = float(sys.argv[1])
    cFF = 1.0 #0.2
    JE0 = 2.0 
    JI0 = 1.0
    Jab = np.array([[JEE, JEI],
                    [JIE, JII]])
    Ea = np.array([JE0, JI0])

    CheckBalConditions(JEE, JEI, JIE, JII, JE0, JI0)
    gamma = 0.0

    m0List = [0.075, 0.1, 0.15, 0.175, .2, .3]
    m0List = [0.075]
    mExtOne = 0
    SolveForPCritical(m0List)
    figFolder = './figs/' 
    figname = './p_vs_mE1' # '/p%s'%(p)
    paperSize = [5, 4.0]
    figFormat = 'png'
    axPosition = [0.17, 0.15, .8, 0.8]
    print figFolder, figname
    Print2Pdf(plt.gcf(),  figFolder + figname,  paperSize, figFormat=figFormat, labelFontsize = 12, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
    plt.show()



   
    # meanFieldRates = -1.0 * np.dot(np.linalg.inv(Jab), Ea) * m0
    # alpha_E = alphaE()        
    # uE0Solution = fsolve(Eq0, [0.1], args = (JEE, alphaE()), full_output = 1)    
    # PrintError(uE0Solution, 'u_E(0)')
    # uE0 = uE0Solution[0]
    # pCAnalytic = Pcritical(uE0, alpha_E, JEE)
    # GRAPHICAL SOLUTION
    # SolveGraphically()
