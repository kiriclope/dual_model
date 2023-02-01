import numpy as np
import pylab as plt
import os, sys
import ipdb

rootFolder = "" #"~/Documents/lab"
basefolder = rootFolder + "/homecentral/srao/Documents/code/mypybox"
print basefolder
sys.path.append(rootFolder + '/homecentral/srao/Documents/code/mypybox/utils')
from Print2Pdf import Print2Pdf


def LoadFr(p, gamma, phi, mExt, mExtOne, trNo = 0, T = 1000, N = 10000, K = 1000, nPop = 2, IF_VERBOSE = False):
    baseFldr = '/homecentral/srao/Documents/code/binary/tmp/' #twopop/data/N1000K1000/m0150/mExtOne60/p0gamma0/T1/tr0/'
    if nPop == 1:
        baseFldr = baseFldr + 'onepop/data/N%sK%s/m0%s/mExtOne%s/p%sgamma%s/T%s/tr%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(gamma * 10), int(T*1e-3), trNo)
    if nPop == 2:
        baseFldr = baseFldr + 'twopop/data/N%sK%s/m0%s/mExtOne%s/p%sgamma%s/T%s/tr%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(gamma * 10), int(T*1e-3), trNo)
    if IF_VERBOSE:
	print baseFldr
    filename = 'meanrates_theta%.6f_tr%s.txt'%(phi, trNo)
    return np.loadtxt(baseFldr + filename)
 
def LoadSpkTimes(p, gamma, phi, mExt, mExtOne, trNo = 0, T = 1000, N = 10000, K = 1000, nPop = 2):
    baseFldr = rootFolder + '/homecentral/srao/Documents/code/binary/c/'
    if nPop == 1:
	baseFldr = baseFldr + 'onepop/data/N%sK%s/m0%s/mExtOne%s/p%sgamma%s/T%s/tr%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(gamma * 10), int(T*1e-3), trNo)
    if nPop == 2:
	baseFldr = baseFldr + 'twopop/data/N%sK%s/m0%s/mExtOne%s/p%sgamma%s/T%s/tr%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(gamma * 10), int(T*1e-3), trNo)
    print 'loading from fldr: ', baseFldr
    return np.loadtxt(baseFldr + 'spktimes_theta%.6f_tr%s.txt'%(phi, trNo), delimiter = ';')

def SpikesInInterval(st, spkStart, spkEnd = -1):
    if(spkEnd == -1):
        spkEnd = np.max(st[:, 1])
        idx = st[:, 1] >= spkStart
    else:
        idx = np.logical_and(st[:, 1] >= spkStart, st[:, 1] <= spkEnd)
    return st[idx, :]

def PlotRasterBasic(st):
    plt.ion()
    plt.plot(st[:, 1], st[:, 0], '|k')
    plt.draw()

def PlotRaster(st, spkStart, spkStop, NE, NI, IF_COLOR = False):
    plt.ion()
    st = SpikesInInterval(st, spkStart, spkStop)
    if IF_COLOR:
        idxNE = st[:, 0] < NE
        idxNI = st[:, 0] > NE
        stNE = st[idxNE, :]
        stNI = st[idxNI, :]
        plt.plot(stNE[:, 1], stNE[:, 0], '|k')
        plt.plot(stNI[:, 1], stNI[:, 0], '|', color = [0.7, 0.1, 0.1])
        plt.draw()
        print 'hello'
    else:
        plt.plot(st[:, 1], st[:, 0], '.k')
        plt.draw()
    plt.xlabel('Time')
    plt.ylabel('Neuron #')

def RasterPlot(st, spkStart = 0, spkStop = -1, ):
    st = SpikesInInterval(st, spkStart, spkStop)
    print st.shape
    neuronIdx = np.unique(st[:, 0])
    nNeurons = np.size(neuronIdx)
    nSpks, _ = st.shape
    totalvLength = 200.0
    vLenght = 100.0
    vLineOffset = (totalvLength - vLenght) * 0.5
    plt.figure()
    plt.ion()
    x = np.array([])
    y = np.array([])
    for idx, iNeuron in enumerate(neuronIdx):
        iSpkTimes = st[st[:, 1] == iNeuron, 0]
        x = np.r_[x, iSpkTimes]
        y = np.r_[y, totalvLength * iNeuron * np.ones((np.size(iSpkTimes),)) + vLineOffset - vLenght]
    plt.vlines(x, y, y + vLenght)
    plt.ylim(y[0] - 1 - vLenght, y[-1] + 1 + vLenght)
    plt.yticks(neuronIdx * totalvLength)
    plt.gca().set_yticklabels(neuronIdx)
    plt.draw()
    plt.show()

def AutoCorr(x, corrLength = "same"):
    # x : spike Times in ms
    N = len(x)
    nPointFFT = int(np.power(2, np.ceil(np.log2(len(x)))))
    fftX = np.fft.fft(x, nPointFFT)
    return np.abs(np.abs(np.fft.ifft(np.multiply(fftX, np.conj(fftX)))))

def AvgAutoCorrInInterval(starray, neuronsList, spkTimeStart, spkTimeEnd, minSpks = 100, maxTimeLag = 100, NE = 10000, NI = 10000, NFF = 10000, fileTag = 'E', theta = 0, TAU_FF = 4, TAU_E = 4, TAU_I = 2):
    N = len(neuronsList)
    simDT = (TAU_FF * TAU_E * TAU_I) / (NE * TAU_FF * TAU_I + NI * TAU_FF * TAU_E + NFF * TAU_E * TAU_I)
    pcDone = 0
    simDuration = spkTimeEnd - spkTimeStart
    nTimeLagBins = int(2 * maxTimeLag) # works if downsample bin size = 1ms otherwise divide by value
    avgCorr = np.zeros((int(np.power(2, np.ceil(np.log2(simDuration + simDT)))), ))
    binFlag = 0;
    nValidNeurons = 0;
    downSampleBinSize = 1
    spkBins = np.arange(spkTimeStart, spkTimeEnd + simDT, downSampleBinSize)
    nSpkBins = len(spkBins)
    avgRate = 0
    popMeanRate = [];
    for i, kNeuron in enumerate(neuronsList):
        spkTimes = starray[starray[:, 0] == kNeuron, 1]
        spksTimes = spkTimes[spkTimes > spkTimeStart]
        meanRate = float(spkTimes.size) / float(simDuration)
        nValidNeurons += 1
        avgRate += meanRate
        # print spkTimes.size, spkTimes.size >= minSpks
        if(spkTimes.size >= minSpks):
            # print meanRate
            st = np.histogram(np.squeeze(spkTimes), spkBins)
            tmpCorr = AutoCorr(st[0])
            avgCorr += tmpCorr / ((downSampleBinSize) **2 * nSpkBins * meanRate)
    avgCorr = avgCorr / nValidNeurons
    bins = np.array(downSampleBinSize)
    avgCorr[np.argmax(avgCorr)] = 0.0
    avgCorr = avgCorr[: maxTimeLag]
    print 'avgRate = ', avgRate / N
    if(len(avgCorr) > 0):
        return avgCorr
    else :
        return 0

def PlotAC(st, nNeurons, spkTimeStart, spkTimeEnd, minSpks = 100, maxTimeLag = 100, NE = 10000, NI = 10000, NFF = 10000, fileTag = 'E', theta = 0, TAU_FF = 4, TAU_E = 4, TAU_I = 2):
    nENeurons = np.arange(NE)
    nINeurons = np.arange(NE, NE + NI)
    if nNeurons < NE:
        nENeurons = np.random.randint(0, NE, nNeurons)
    if nNeurons < NI:
        nINeurons = np.random.randint(NE, NE + NI, nNeurons)
    acE = AvgAutoCorrInInterval(st, nENeurons, spkTimeStart, spkTimeEnd, NE = NE, NI = NI, NFF = NFF, TAU_E = TAU_E, TAU_I = TAU_I, TAU_FF = TAU_FF, theta = theta, maxTimeLag = maxTimeLag)
    acI = AvgAutoCorrInInterval(st, nINeurons, spkTimeStart, spkTimeEnd, NE = NE, NI = NI, NFF = NFF, TAU_E = TAU_E, TAU_I = TAU_I, TAU_FF = TAU_FF, theta = theta, maxTimeLag = maxTimeLag, minSpks = minSpks)
    plt.figure()
    plt.plot(acE, 'k', label = 'E')
    plt.plot(acI, color = [0.7, 0.1, 0.1], label = 'I')
    plt.legend(loc = 0, frameon = False, numpoints = 1, prop = {'size': 10})
    plt.xlabel('Time')
    # plt.ylabel('')
    plt.title('AC')
    return acE, acI
    
def CV(spkTimes):
    out = np.array([np.nan, np.nan])
    if(spkTimes.size > 3):
        isi = np.diff(spkTimes)
        cv = isi.std() / isi.mean()
        cv2 = CV2(isi)
        out = np.array([cv, cv2])
    return out

def CV2(isi):
    denom = isi[1:] + isi[:-1]
    cv2 = 2.0 * np.abs(np.diff(isi)) / denom
    return np.nanmean(cv2)

def CVofList(starray, listOfneurons, spkTimeStart = 200):
    out = np.empty((len(listOfneurons), 2))
    out[:] = np.nan
    for i, iNeuron in enumerate(listOfneurons):
        spkTimes = starray[starray[:, 0] == iNeuron, 1]
        spksTimes = spkTimes[spkTimes > spkTimeStart]
        out[i, :] = CV(spkTimes)
    return out #out[:, 0] is CV, out[:, 1] is CV2

def PlotCVDistr(starray, NE, NI, nNeurons, spkTimeStart = 500, nBins = 10):
    plt.figure()
    print 'computeing CV, E ... ',
    sys.stdout.flush()
    cvE = CVofList(starray, np.random.randint(0, NE, nNeurons), spkTimeStart)
    print 'done'
    print 'computeing CV, I ... ',    
    cvI = CVofList(starray, np.random.randint(NE, NE + NI, nNeurons), spkTimeStart)
    print 'done'
    avgCVE = np.nanmean(cvE[:, 0])
    avgCVI = np.nanmean(cvI[:, 0])
    avgCV2E = np.nanmean(cvE[:, 1])
    avgCV2I = np.nanmean(cvI[:, 1])
    print 'avg CVE = ', avgCVE, 'avg CVI = ', avgCVI
    print 'avg CV2E = ', avgCV2E, 'avg CV2I = ', avgCV2I    
    plt.hist(cvE[~np.isnan(cvE[:, 0]), 0], nBins, histtype = 'step', normed = 1, label = 'E', color = 'k')
    plt.hist(cvI[~np.isnan(cvI[:, 0]), 0], nBins, histtype = 'step', normed = 1, label = 'I', color = [0.7, 0.1, 0.1])
    _, ymax = plt.ylim()
    plt.vlines(avgCVE, 0, ymax, color = 'k', lw = 2)
    plt.vlines(avgCVI, 0, ymax, color = [0.7, 0.1, .1], lw = 2)    
    plt.legend(loc = 0, frameon = False, numpoints = 1, prop = {'size': 10})
    plt.xlabel('CV')
    plt.ylabel('Density')
    plt.figure()
    plt.hist(cvE[~np.isnan(cvE[:, 1]), 1], nBins, histtype = 'step', normed = 1, label = 'E', color = 'k')
    plt.hist(cvI[~np.isnan(cvI[:, 1]), 1], nBins, histtype = 'step', normed = 1, label = 'I', color = [0.7, 0.1, 0.1])
    _, ymax = plt.ylim()    
    plt.vlines(avgCV2E, 0, ymax, color = 'k', lw = 2)
    plt.vlines(avgCV2I, 0, ymax, color = [0.7, 0.1, .1], lw = 2)    
    plt.legend(loc = 0, frameon = False, numpoints = 1, prop = {'size': 10})
    plt.xlabel(r'$CV_{2}$')
    plt.ylabel('Density')
        
def PlotInstantRates(stArray, spkTimeStart, spkTimeEnd, NE, NI, NFF, windowSize, IF_NEW_FIG = True, TAU_FF = 4.0, TAU_E = 4.0, TAU_I = 2.0):
    st =  SpikesInInterval(stArray, spkTimeStart, spkTimeEnd)
    dt = (TAU_FF * TAU_E * TAU_I) / (NE * TAU_FF * TAU_I + NI * TAU_FF * TAU_E + NFF * TAU_E * TAU_I)
    windowSize *= dt
    print dt
    binEdges = np.arange(spkTimeStart, spkTimeEnd + windowSize, windowSize)
    if IF_NEW_FIG:
        plt.figure()
    spkTimesE = st[st[:, 0] < NE, 1]
    spkTimesI = st[st[:, 0] >= NE, 1]
    cntE, binsE = np.histogram(spkTimesE, binEdges)
    cntI, binsI = np.histogram(spkTimesI, binEdges)
    print binsE.shape, binsE[-1], binsE[0]
    print cntE
    print cntI
    plt.plot(binsE[:-1], cntE / (float(windowSize) * NE), color = 'k')
    plt.plot(binsI[:-1], cntI / (float(windowSize) * NI), color = [0.7, .1, .1])
    plt.grid()
    
def OSI(firingRate, atTheta):
    out = np.nan
    zk = 2.0 * np.dot(firingRate, np.exp(2j * atTheta * np.pi / 180))
    if(firingRate.size > 0):
        out = np.absolute(zk) / firingRate.size
    return out

def OSIOfPop(firingRates, atThetas):
    # thetas in degrees
    nNeurons, nThetas = firingRates.shape
    out = np.zeros((nNeurons, ))
    for i in range(nNeurons):
        out[i] = OSI(firingRates[i , :], atThetas)
    return out

def PltOSIHist(p, gamma, nPhis, mExt, mExtOne, trNo = 0, N = 10000, K = 1000, nPop = 2, T = 1000, IF_NEW_FIG = True):
    NE = N
    NI = N
    tc = np.zeros((NE + NI, nPhis))
    phis = np.linspace(0, 180, nPhis, endpoint = False)
    if IF_NEW_FIG:
	plt.figure()
    for i, iPhi in enumerate(phis):
        print 'loading from fldr: ', 
	fr = LoadFr(p, gamma, iPhi, mExt, mExtOne, trNo, T, NE, K, nPop, IF_VERBOSE = True)
        if(len(fr) == 1):
            if(np.isnan(fr)):
                print 'file not found!'
	tc[:, i] = fr
    osi = OSIOfPop(tc[:NE, :], phis)
    # plt.xlabel(r"$\mathrm{OSI} \,\,\,\,  (m_{E, i}^{(1)})$")
    plt.xlabel('OSI')    
    plt.ylabel('Density')
    plt.hist(osi[~np.isnan(osi)], 27, normed = 1, histtype = 'step', label = r'$p = %s,\, \gamma = %s, \, m_0^{(1)} = %s$'%(p, gamma, mExtOne))
    # plt.title(r'$N = %s,\, K = %s,\, m_0^{(0)} = %s,\, m_0^{(1)} = %s$'%(NE, K, mExt, mExtOne))
    plt.title(r'$N = %s,\, K = %s,\, m_0^{(0)} = %s$'%(NE, K, mExt,))
    print "MEAN OSI = ", np.nanmean(osi)
    return osi

def CompareOSIHist(pList, gList, nPhis, mExt, mExtOneList, trNo, N = 10000, K = 1000, nPop = 2, T = 1000):
    for mExtOne in mExtOneList:
	for p in pList:
	    for gamma in gList:
		try:
		    PltOSIHist(p, gamma, nPhis, mExt, mExtOne, trNo = trNo, IF_NEW_FIG = False)
		except IOError:
		    print "p = ", p, " gamma = ", gamma, " trial# ", trNo, " file not found"
    plt.legend(loc = 0, frameon = False, numpoints = 1, prop = {'size': 10})
    
def ComputeTuningForNeuron(p, gamma, nPhis, mExt, mExtOne, neuronIdx, trNo = 0, N = 10000, K = 1000, nPop = 2, T = 1000):
    NE = N
    NI = N
    tc = np.zeros((nPhis))
    phis = np.linspace(0, 180, nPhis, endpoint = False)
    for i, iPhi in enumerate(phis):
	fr = LoadFr(p, gamma, iPhi, mExt, mExtOne, trNo, T, NE, K, nPop)
	# print fr.shape
	tc[i] = fr[neuronIdx]
    plt.figure()
    plt.ion()
    plt.plot(phis, tc, 'ks-')
    osi = OSI(tc, phis)
    plt.title('neuron# %s, osi = %s'%(i, osi))
    return tc
    
def ComputeTuning(p, gamma, nPhis, mExt, mExtOne, trNo = 0, N = 10000, K = 1000, nPop = 2, T = 1000):
    NE = N
    NI = N
    tc = np.zeros((NE + NI, nPhis))
    phis = np.linspace(0, 180, nPhis, endpoint = False)
    for i, iPhi in enumerate(phis):
	fr = LoadFr(p, gamma, iPhi, mExt, mExtOne, trNo, T, NE, K, nPop)
	# print fr.shape
	tc[:, i] = fr
    plt.ion()
    for i in np.random.randint(0, NE, 101):
	plt.plot(phis, tc[i, :], 'ks-')
        osi = OSI(tc[i, :], phis)
        plt.title('neuron# %s, osi = %s'%(i, osi))
	plt.waitforbuttonpress()
	plt.clf()

def PopAvgTuning(p, gamma, nPhis, mExt, mExtOne, trNo = 0, N = 10000, K = 1000, nPop = 2, T = 1000):
    NE = N
    NI = N
    tc = np.zeros((NE + NI, nPhis))
    phis = np.linspace(0, 180, nPhis, endpoint = False)
    for i, iPhi in enumerate(phis):
	fr = LoadFr(p, gamma, iPhi, mExt, mExtOne, trNo, T, NE, K, nPop)
	print fr.shape
	tc[:, i] = fr
    prefferedOri = np.argmax(tc, 1)    
    tcMat = np.empty((NE + NI, nPhis))
    for kNeuron in np.arange(NE + NI):
	tcMat[kNeuron, :] = np.roll(tc[kNeuron, :], -1 * prefferedOri[kNeuron])
    meanE = np.nanmean(tcMat[:NE, :], 0)
    meanI = np.nanmean(tcMat[NE:, :], 0)
    rotateMeanBy = 4
    meanE = np.roll(meanE, rotateMeanBy)
    meanI = np.roll(meanI, rotateMeanBy)
    theta = np.linspace(-90, 90, nPhis, endpoint = False)
    print "mE = ", np.nanmean(meanE), "MI = ", np.nanmean(meanI)
    plt.figure()
    plt.plot(theta, meanE, 'ko-', label = 'E') 
    plt.plot(theta, meanI, 'ro-', label = 'I')
    plt.title(r"$m_0 = %s, m_0^{(1)} = %s, \, p = %s, \, \gamma = %s$"%(mExt, mExtOne, p, gamma), fontsize = 16)    
    plt.legend(loc = 0, frameon = False, numpoints = 1, prop = {'size': 10})    
    

def M1Component(x):
    out = np.nan
    if len(x) > 0:
	dPhi = np.pi / len(x)
	out = 2.0 * np.absolute(np.dot(x, np.exp(-2.0j * np.arange(len(x)) * dPhi))) / len(x)
    # print 'm1 out: ', out
    return out

def SmoothedMeanM1(p, gamma, phis, mExt, mExtOne, NE = 10000, NI = 10000, K = 1000, nTr = 1, nPop = 2, T = 1000, neuronType = 'E', IF_M1_SQRD = False, IF_SMOOTHED = False):
    m1 = []
    winSize = int(float(NE) / 10.0)
    N = NE
    if neuronType == 'I':
        winSize = int(float(NI) / 10.0)
        N = NI
    nValidTrials = np.zeros((len(phis), ))
    print "--" * 25
    print "N = %s, K = %s, p = %s, m0 = %s, m0One = %s" %(N, K, p, mExt, mExtOne)
    print "trial#: ", 
    sys.stdout.flush()
    # phis = np.arange(0, 180.0, 22.5)
    # ipdb.set_trace()
    for n in range(nTr):
        print n
        m1OfPhi = []
        print 'PHI_0: ', 
        for phiIdx, atPhi in enumerate(phis):
            try:
                # print "LoadFr() ", p, gamma, atPhi, mExt, mExtOne, n, T, NE, K, nPop
                # print '#populations:', nPop
                # raise SystemExit

                m = LoadFr(p, gamma, atPhi, mExt, mExtOne, n, T, NE, K, nPop)
                # ipdb.set_trace()
                # print m
                if(neuronType == 'E'):
                    m = m[:N]
                else:
                    m = m[N:]
                if IF_SMOOTHED:
                    smoothedM = CirConvolve(m, winSize)
                    m1OfPhi.append(M1Component(smoothedM))
                else:
                    m1OfPhi.append(M1Component(m))                    
                # print m1OfPhi
                nValidTrials[phiIdx] += 1
                print atPhi, 
            except IOError:
                print 'x', #file not found'
            sys.stdout.flush()
        m1.append(np.nanmean(m1OfPhi))
    print "\n" + "--" * 25
    
    print 'mean m1 = ', np.mean(m1), 'SEM = ', np.nanstd(m1) / np.sqrt(float(nValidTrials[0]))
    if IF_M1_SQRD:
	print "sqrd"
	return np.nanmean(np.array(m1)**2), np.std(np.array(m1)**2), np.std(np.array(m1)**2) / np.sqrt(float(nValidTrials[0]))
    else:
	return np.nanmean(m1), np.nanstd(m1), np.nanstd(m1) / np.sqrt(float(nValidTrials[0]))

    
def M1vsp_Smoothed(pList, gamma, nPhis, mExt, mExtOne, NList = [10000], KList =[1000], trNo = 0, nPop = 2, T = 1000, nTrials = 1, IF_NONSMOOTHED = False):
    m1 = []
    # plt.ion()
    # plt.figure()
    IF_LEGEND = False
    nTr = nTrials
    # ipdb.set_trace()
    phis = np.linspace(0, 180.0, nPhis, endpoint = False)
    for k, kK in enumerate(KList):
        for n, nNE in enumerate(NList):
            m1 = []
            semM1 = []
            m1Sqrd = []
            semM1Sqrd = []
            for iip, p in enumerate(pList):
		if( not IF_NONSMOOTHED):
		    tmpM1, dummy, tmpsem = SmoothedMeanM1(p, gamma, phis, mExt, mExtOne, nNE, nNE, kK, nTr, nPop, T)
		    # tmpM1Sqrd, dummySqrd, tmpsemSqrd = SmoothedMeanM1(nNE, kK, p, mExt, nTr, T, IF_M1_SQRD = True)
		else:
		    tmpM1, dummy, tmpsem = NonSmoothedMeanM1(nNE, kK, p, mExt, nTr, T)
                m1.append(tmpM1)
                semM1.append(tmpsem)
                # print m1
                # m1Sqrd.append(tmpM1Sqrd)
                # semM1Sqrd.append(tmpsemSqrd)
                print '==' * 28; print '||' * 28; print '==' * 28
            m1 = np.array(m1)
	    # m1Sqrd = np.array(m1Sqrd)
            validPIdx = ~np.isnan(m1)
            semM1 = np.array(semM1)
            # semM1Sqrd = np.array(semM1Sqrd)
            # ipdb.set_trace()
            print m1[validPIdx].shape, np.sum(validPIdx)
            print pList[validPIdx]
            print semM1[validPIdx]
            if(np.sum(np.array(validPIdx)) > 0):
                IF_LEGEND = True
                (_, caps, _) = plt.errorbar(pList[validPIdx], m1[validPIdx], fmt = 'o-', markersize = 3, yerr = semM1[validPIdx], lw = 0.8, elinewidth=0.8, label = r'$N = %s, K = %s, \gamma = %s$'%(nNE, kK, gamma))
                for cap in caps:
                    # cap.set_color('red')
                    cap.set_markeredgewidth(0.8)
                outArray = np.empty((sum(validPIdx), 3))
                outArray[:] = np.nan
                outArray[:, 0] = pList[validPIdx]
                outArray[:, 1] = m1[validPIdx]
                outArray[:, 2] = semM1[validPIdx]                
                print outArray
                analysisDataFldr = '/homecentral/srao/Documents/code/binary/c/analysis/data/'
                if nPop == 1:
                    analysisDataFldr = analysisDataFldr + 'onepop/'
                if nPop == 2:
                    analysisDataFldr = analysisDataFldr + 'twopop/'
                np.save(analysisDataFldr +  "sim_mE1_vs_p_m0%s_N%s_K%s"%(int(mExt * 1e3), nNE, kK), outArray)
                # outArray2 = np.empty((sum(validPIdx), 3))                
		# outArray2[:, 0] = outArray[:, 0]
		# outArray2[:, 1] = m1Sqrd[validPIdx]
		# outArray2[:, 2] = semM1Sqrd[validPIdx]                
                # np.save("./data/analysisData/sim_mE1_sqrd_vs_p_m0%s_N%s_K%s"%(int(mExt * 1e3), nNE, kK), outArray2)		
            

    plt.xlabel(r'$p$', fontsize = 16)    
    plt.ylabel(r'$\left[ m^{(1)} \right]_{\phi}$', fontsize = 16) # + ' avg over 10 realizations')
    if IF_LEGEND:
        plt.legend(loc = 0, frameon = False, numpoints = 1)
    plt.xlim(0, pList[-1] + .5)
    # plt.ylim(0, mExt + 0.1)
    if IF_NONSMOOTHED:
	plt.title(r"$m_0 = %s, m_0^{(1)} = %s$"%(mExt, mExtOne), fontsize = 16)
        figname = 'non_smoothed_m1_m0%s'%(int(mExt * 1e3))	
    else:
	# plt.title(r"$m_0 = %s, smoothened$"%(mExt), fontsize = 16)
	plt.title(r"$m_0 = %s,\, m_0^{(1)} = %s, \,$"%(mExt, mExtOne) + 'smoothened', fontsize = 16)
        figname = 'smoothed_m1_m0%s'%(int(mExt * 1e3))	
    print 'saving as', figname
    figFolder = '/homecentral/srao/Documents/code/binary/c/analysis/figs/'
    if nPop == 1:
        figFolder = figFolder + 'onepop/'
    if nPop == 2:
        figFolder = figFolder + 'twopop/'
    #-------------------------------------#
    plt.savefig(figFolder + figname + '.png')
    plt.show()
    if IF_LEGEND:
        plt.legend(loc = 0, frameon = False, numpoints = 1, prop = {'size': 10})
    plt.xlim(0, pList[-1] + .5)
    #plt.ylim(0, mExt + 0.1)
    ylims = plt.ylim()
    plt.yticks([0, ylims[1]])
    paperSize = [2.5, 2.0]
    figFormat = 'pdf'
    axPosition = [0.28, 0.23, .65, 0.65]
    print figFolder, figname
    Print2Pdf(plt.gcf(),  figFolder + figname,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)


def CirConvolve(signal, windowSize):
    ker = np.concatenate((np.ones((windowSize, )), np.zeros((signal.size - windowSize, ))))
    return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker))) * (1.0 / float(windowSize))
    
