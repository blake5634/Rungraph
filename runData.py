##
#
#

# IMPORTS
import math as m
import numpy as np       # operations on numerical arrays
import datetime as dt
from   dateutil import parser 

import csv               # file I/O
# CLASSES

class run:
    def __init__(self,date='',pace=0, dur=0, cmt=''):
        self.date = date
        self.pace = pace
        self.dur = float(dur)
        self.comment = cmt
        self.tempDegF = 0.00
        
    #distance
    def dist(self):
        return float(self.dur)/self.pace
    
    def __repr__(self):
        t = float(self.dur)/60
        return '{:} {:4.1f}km {:6.1f}sec/km {:8.1f}min {:30.30} {:6.1f}F'.format(str(self.date)[0:10],self.dist(),self.pace,t,self.comment,self.tempDegF )
        
    
class route:
    def __init__(self,string,rnum):
        self.hmin = 4*60 + 30
        self.hmax = 5*60 + 30
        self.name= string
        self.distance = 0.0
        self.rnum = rnum   # route number (col 2)
        self.n = 0
        self.tot_secp = 0
        self.tot_secp2 = 0
        self.min_secp = 9999999999
        self.max_secp = 0
        self.hist = np.zeros(self.hmax-self.hmin)
        self.times = []
        self.dates = []
        self.plusgain = 0   # elevation gains
        self.minusgain = 0

    def add(self, sec, date):  # add a run record to a route
        self.n += 1
        #print 't = ', int(sec), minsec(sec)
        #accumulate histogram of paces
        indx = int(sec)-self.hmin
        self.times.append(sec)
        self.dates.append(dt.datetime.strptime(date,'%Y-%m-%d'))
        if(indx > 0 and indx < (len(self.hist)-1)):
            self.hist[int(sec)-self.hmin] += 1
        # build mean and SD values
        self.tot_secp += sec
        self.tot_secp2 += sec*sec
        if (sec > self.max_secp):
            self.max_secp = sec
        if (sec < self.min_secp):
            self.min_secp = sec

    def avg(self):    # compute mean and sd of route
        self.avg_pace = self.tot_secp/self.n   # mean
        self.sd_pace  = m.sqrt(                # standard Dev.
            (self.n*self.tot_secp2-self.tot_secp*self.tot_secp) /
            (self.n*(self.n-1))
            )
        
class runLists:
    def __init__(self):
        self.nruns = 0
        self.allruns = []
        self.runs = []     # "valid" runs
        self.runs3k = []
        self.runs5k = []
        self.routed = {}   # routes by id #
        self.nvr = 0 # number of valid runs
        self.nrt = 0 # number of routes
        self.nruns = 0 # all runs

        # DATASET READER
    def reader(self):
        with open('ActivityLog.csv','rt') as f:
            data = csv.reader(f,delimiter=',',quotechar='"')
            for row in data:
                self.nruns += 1
                #print row[0], '---> ' , row[3]
                #######################   Run Data
                stdate  = row[0].strip()
                stactiv = row[1].strip()
                stroutenum = row[2].strip()
                stroute = row[3].strip()
                stsec  = row[4].strip()
                stdur  = row[5].strip()   #  hh:mm:ss
                stdist = row[6].strip()
                stpace = row[7].strip()
                stcomnt = row[10].strip()
                
                valid = True
                if(stactiv != '' and stactiv != 'Run'):
                    valid = False
                if(stdate == 'Date'):  # e.g. header
                    valid = False

                if stsec == '' and stdist == '': # we know NOTHING!
                    valid == False
                    
                if(valid): # this list (allruns) includes runs with only time
                    # or only distance
                    pace = 300
                    
                    if stdist != '' and stsec != '':   # if there is BOTH time and dist
                        d = float(stdist)
                        t = int(stsec)
                        pace = t/d

                    if stdist == '' and stsec != '':   # if there is time and NOT dist
                        t = int(stsec)
                        d = t/pace
                        
                    if stdist != '' and stsec != '':   # if there is NOT time and dist 
                        d = float(stdist)
                        t = int(d*pace)
                    
                    date = parser.parse(stdate)
                    self.allruns.append(run(date, pace, t, stcomnt))  # nominal pace

                if(stsec == ''):
                    valid = False
                if(stdist == ''):
                    valid = False
                if(stroute == ''):
                    valid = False

                if(valid):  # this list (runs) is the best data
                    self.nvr += 1
                    dist = float(stdist)
                    secpace = int(stsec)/dist
                    pacemin = int(secpace)/60
                    pacesec = int(secpace - 60*pacemin)
                    date = parser.parse(stdate)
                    self.runs.append(run(date,secpace,float(stsec),stcomnt))  # store the overall runs
                    #
                    # break runs down by "3k" and "5k" lengths
                    #
                    d = float(stdist)
                    if (2.8 < d and d < 4.0):
                        self.runs3k.append(secpace)
                    if (4.0 < d and d < 7.5):
                        self.runs5k.append(secpace)

                    if not (stroute in self.routed):
                        self.nrt += 1
                        #print "creating class instance"
                        r = route(stroute,stroutenum)  # create the new route
                        r.distance = float(stdist)
                        self.routed[stroute] = r

                    self.routed[stroute].add(secpace,stdate)   # count this run


##  Functions:

def minutes(sec):
    return int(int(sec)/60)

def seconds(s):
    return int(s - minutes(s)*60)

def minsec(s):
    return '{:d}:{:02d}'.format(minutes(s),seconds(s))

    
    #
    #  make a horizontal bar for mean and += 1 SD
    #
def plotHbar(x1,x2,y,plotobj):
    #  were going for |------|------| here:
    x21 = (x1+x2)/2.0
    xl = [x1,x1,x1,x21,x21,x21,x21, x2,x2,x2]
    #for j in range(0,len(xl)):
        #xl[j] -= 300   # subtract off 5:00 pace
        # partway up the Y-axis
    T = y/10.0  # length of vertical 
    yl = [y-T/2,y+T/2,y,   y, y+0.55*T,y-0.55*T,y ,  y  ,y-T/2,y+T/2]
#    for i in [0,2,4,6]:
#        x = [xl[i], xl[i+1]]
#        y = [yl[i], yl[i+1]]
#        plt.plot(x,y, linewidth= 2.0, alpha= 1.0,color='blue')
    plotobj.plot(xl,yl,linewidth=2.0, alpha=1.0,color='blue')


#//  http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x,window_len=11,window='hanning'):
    l = x.size
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    y1 = np.zeros(l)
    for i in range(0,l-1):
        y1[i] = y[i]
    return y1
