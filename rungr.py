#!/usr/bin/env python
import numpy as np       # operations on numerical arrays
import csv               # file I/O
import math as m
import operator          # for sorting list of class instances
import numpy as np
from scipy import stats
import datetime as dt
from   dateutil import parser

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

routes = []

####################################
#
#     Make a heatmap plot of run pace vs N runs per week
#
#

def plot_freq(r,N):
    b1 = 0
    flag = False
    freqs = []
    times = []
    rbuf = []
    for j in range(0,N):
        rbuf.append(parser.parse('01 January 2010'))
    
    for r in r:  # run object instances 
        
        #   update freq data buffer
        for j in range(0,N-1):
            rbuf[j] = rbuf[j+1]
        rbuf[N-1] = r.date
        
        
        #   compute frequency 
    
        deltaT = r.date - rbuf[0] #  dt of last N runs
        f = 60*60*24*7*N/deltaT.total_seconds()   # seconds
        
        #print '>>', f, r.pace
        
        freqs.append(f)
        times.append(r.pace)
        
        #   store freq and pace
        
    # compute linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(freqs, times)
    
    print ' Regression model: '
    print 'Slope:    ', slope
    print 'intercept:', intercept
    print 'r_value:  ', r_value
    print 'p_value:  ', p_value
    print 'std_err:  ', std_err
    
    #   plot the graph
    
    
    
    ###############################################   Plot Pace vs Frequency
    # 
    ###############################################    Build/Plot Heatmap
    lims = [0, 4.0, 240, 340]  # xmin, xmax, ymin ymax
    fmin = lims[0]
    fmax = lims[1]
    tmin = lims[2]
    tmax = lims[3]
    df = 0.1
    dt = 1.0
    
    nbins = 25
        
    map = np.zeros((nbins+1,nbins+1))
    for i in range(0,len(times)):
        r = -1 + int((times[i] - tmin)/(tmax-tmin) * nbins)
        c = -1 + int((freqs[i] - fmin)/(fmax-fmin) * nbins)
        if (r < nbins and c < nbins): 
            map[r,c] += 1

    [nr,nc] = map.shape 
    
    r = np.linspace(tmin, tmax, nr-1)
    c = np.linspace(fmin, fmax, nc-1)
    
    xx, yy = np.meshgrid(c,r)
    
    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    map = map[:-1, :-1]

    miz = np.amin(map)
    mxz = np.amax(map)
    levels = MaxNLocator(nbins=15).tick_values(miz, mxz)


    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('PiYG')
    cmap = plt.get_cmap('summer')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, (ax0) = plt.subplots(nrows=1)

    print 'Shapes: '
    print 'xx:     ', xx.shape
    print 'yy:     ', yy.shape
    print 'map:    ', map.shape
    

    im = ax0.pcolormesh(xx, yy, map, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax0)
    ax0.set_title('Run Pace vs Runs/week')
    ax0.set_xlabel('Runs per week ')
    ax0.set_ylabel('Pace (sec/km)')
    plt.show()
              
#
#######################################################################
#
#     Plot some stats over all runs in database
#

        
def plot_global_stats(r_in, allruns):       
    data = []
    topRnames = []
    Nruns = []
    rElevs = []
    #make data into an array
    i = 0
    #for r in r_in:        
        #for j in range(0,len(r.times)):
            #r.times[j] -= 300  # subtract 5 min from all stats
    for r in r_in:    # eg. a list of routes
        i += 1
        dtmp = []
        for j in range(0,len(r.times)):# subtract 5 min from all stats
            dtmp.append(r.times[j]-300)  # get all the run times in this route
        data.append(dtmp)
        topRnames.append(r.name)
        # elevation gain of this route
        echange = r.plusgain + -1*r.minusgain  # abs value
        rElevs.append(echange)
        Nruns.append('n = ' + str(r.n))
        if i >= max: 
            break

    PLOTS=True
    if(PLOTS):
        ###########################################################
        #  graph all the data as boxplots
        #
        #plt.figure(10)    #  boxplot for each route
        fig, ax1 = plt.subplots(figsize=(14,6))
        plt.subplots_adjust(left=.25)
        rect = fig.patch
        rect.set_facecolor('white')
        ax1.xaxis.grid(True,linestyle='-', which='major', color='lightgrey',alpha=0.5)

        # make boxplots for all the routes    
        bp = plt.boxplot(data, notch=True,vert=False ,patch_artist=True)
        for b in bp['boxes']:
            b.set_facecolor('lightblue')
            
        plt.title('Route Pace statistics')
        #plt.ylabel('Route')
        plt.xlabel('sec/km (relative to 5:00)')

        #  add the names of the routes to left side of plot
        plt.yticks(range(1,max+1), topRnames)

        for j in range(0,max):
            ax1.text(47, j+1 , Nruns[j], size='small')
        ##   add n figures at right edge

        plt.show()
        #ax = fig.add_axes()
        #ax.xaxis.set_ticks_position('left')
        #ax.set_yticks(range(1,10))
        #ax.set_yticklabels(topRnames)

    if(PLOTS):
        ###########################################################
        #  graph routes as boxplots according to elevation gain
        # 
        # now sort and reparse according to route elevation gain (up + -dn)
        r3 = sorted(r_in,key=lambda x: x.plusgain , reverse=True)      
        data = []
        topRnames = []
        Nruns = []
        rElevs = []
        #make data into an array
        i = 0
        NMIN = 10     # only plot if run at least NMIN times
        for r in r3:
            if(r.n >= NMIN):
                i += 1
                
                data.append(r.times)
                #topRnames.append(str(r.plusgain))
                topRnames.append(r.name)
                echange = r.plusgain   # abs value
                rElevs.append(echange)
                Nruns.append('n = ' + str(r.n))
                if i >= max: 
                    break
        fig, ax1 = plt.subplots(figsize=(14,6))
        plt.subplots_adjust(left=.25)
        rect = fig.patch
        rect.set_facecolor('white')
        ax1.xaxis.grid(True,linestyle='-', which='major', color='lightgrey',alpha=0.5)

        # make boxplots for all the routes    
        bp = plt.boxplot(data, notch=True,vert=False ,patch_artist=True)
        for b in bp['boxes']:
            b.set_facecolor('lightblue')
            
        plt.title('Route Pace vs. Elevation Gain')
        #plt.ylabel('Route')
        plt.xlabel('sec/km')

        #  add the names of the routes to left side of plot
        #plt.yticks(range(1,max+1), topRnames) 
        #  add the elevations of each route on left side of plot
        estrings = []
        for re in rElevs:
            estrings.append('{:6d} '.format(int(re)))
        for j in range(0,len(estrings)):
            t = estrings[j]
            estrings[j] = topRnames[j] + t.ljust(5)
        plt.yticks(range(1,max+1), estrings) 

        # add the run count to the right side of the plot
        for j in range(0,len(estrings)):
            plt.text(350, j+1 , Nruns[j], size='small') 
        
        plt.show()
        #ax = fig.add_axes()
        #ax.xaxis.set_ticks_position('left')
        #ax.set_yticks(range(1,10))
        #ax.set_yticklabels(topRnames)

    if(PLOTS):
        ##############################################################
        plt.figure(12)    # histogram of ALL run paces 
        paces = []
        for r in allruns:
            paces.append(r.pace)
        n, bins, patches = plt.hist(paces, 50, normed=0, facecolor='blue', alpha=0.5)
        plt.xlabel('time (sec)')
        plt.xlim([270,350])
        plt.title('All Runs')

        ##############################################################
        fig, ax2 = plt.subplots()   #  boxplots of 3k vs 5k runs
        data = []
        data.append(runs3k)
        data.append(runs5k)
        bp = plt.boxplot(data, notch=True, vert=True)
        plt.xlabel(' Distance (km) ')
        plt.ylabel('pace (sec)')
        plt.title('3K vs 5K pace')
        plt.xticks([1, 2], ['3K', '5K'])
        ax2.yaxis.grid(True,linestyle='-', which='major', color='lightgrey',alpha=0.5)

        plt.show()


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
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


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



class run:
    def __init__(self,date='',pace=0):
        self.date = date
        self.pace = pace

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
        self.avg_pace = self.tot_secp/self.n 
        self.sd_pace  = m.sqrt(
            (self.n*self.tot_secp2-self.tot_secp*self.tot_secp) /
            (self.n*(self.n-1)) 
            )
        

def minutes(sec):
    return int(sec)/60   

def seconds(s):
    return int(s - minutes(s)*60) 

def minsec(s):
    return '{:d}:{:02d}'.format(minutes(s),seconds(s))


rd = {}

nv = 0
nrn = 0
nrt = 0
runs = []
runs3k = []
runs5k = []
with open('ActivityLog.csv','rb') as f:
    data = csv.reader(f,delimiter=',',quotechar='"')
    for row in data:
        nrn += 1
        #print row[0], '---> ' , row[3]
        #######################   Run Data 
        stdate  = row[0]
        stactiv = row[1]
        stroutenum = row[2]
        stroute = row[3]
        stsec  = row[4]
        stdist = row[6]
        stpace = row[7]
        
        
        valid = 1
        if(stactiv == 'Bike'):
            valid = 0
        if(stroute == ''): 
            valid = 0
        if(stsec == ''):
            valid = 0
        if(stdist == ''):
            valid = 0
        
        if(valid and stdate != "Date"):
            nv += 1
            dist = float(stdist)
            secpace = int(stsec)/dist
            pacemin = int(secpace)/60
            pacesec = int(secpace - 60*pacemin)
            runs.append(run(parser.parse(stdate),secpace))  # store the overall runs
            d = float(stdist)
            if (2.8 < d and d < 4.0):
                runs3k.append(secpace)
            if (4.0 < d and d < 7.5):
                runs5k.append(secpace)
                
            if not (stroute in rd):
                nrt += 1
                #print "creating class instance" 
                r = route(stroute,stroutenum)  # create the new route
                r.distance = float(stdist)
                rd[stroute] = r
                routes.append(r)
            
            rd[stroute].add(secpace,stdate)   # count this run
            
            #print stdate, stroute, "{}:{:02d}".format(pacemin, pacesec)
            
# add elevation gains to routes
print 'Elevation gains:'
with open('elev_gain.csv','rb') as f:
    data = csv.reader(f,delimiter=',',quotechar='"')
    for row in data:
        if len(row) == 3:
            print row
            rn = row[0]
            eplus  = int(row[1])
            eminus = int(row[2])
            found = False
            for route in routes:
                if route.rnum == rn:
                    found = True
                    route.plusgain = eplus
                    route.minusgain = eminus

print '\n\n'
print nrn , ' runs'
print nv  , ' valid runs'
print nrt , ' routes'

# sort by number of runs on each route
 
#r2 = sorted(routes,key=operator.attrgetter('tot_secp'))
r2 = sorted(routes,key=lambda x: x.n,reverse=True)
#print r2

#
#  Tabular Data Display
#
print '                                                   Pace '
print '  i     Route                               N     min  avg   max    sd'
print '--------------------------------------------------------------------------'
max = 16
i = 0 
for r in r2:
    if(i > max):
        break
    r.avg()
    print '{:3d} {:40s}{:3d}   {:4s}  {:4s}  {:4s}  {:4.1f}'.format(i,r.name,int(r.n), minsec(r.min_secp), minsec(r.avg_pace), minsec(r.max_secp), r.sd_pace)  
    i += 1
 

PLOTS = False

##############################################################

print '\n'    #    Get user input

while (True):
    i = int(input("Select a route to graph: (-1 to quit, 80 = freq, 99 for global plots) "))
    if(i<0):
        quit()
    if(i==80):   #plot pace vs run frequency (runs/wk)
        WINDOW = 10  # runs
        r3 = sorted(runs,key=lambda x: x.date, reverse=False)
        plot_freq(r3,WINDOW)
            
    if(i==99):
        plot_global_stats(r2, runs)
        continue
    if(i+1 > len(r2)):
        print "Selected Invalid run number: ", i
        continue
        
    #
    #   Route Histogram
    #
    r = r2[i]    # get the route object

    #fig, ax2 = plt.subplot()
    plt.figure(1)
    #fig, ax1 = plt.subplots(figsize=(6,6))

    #total = 0
    l = len(r.times) # times have been converted already to pace-300
    
    colors = ['green', 'red']
    onecolor = ['green']
    pctile = 0.15 # this fraction of most recent runs will be in red
    BIG_NEG_FLAG = -1000000
    recent_mean = BIG_NEG_FLAG  # absurd flag value 
    # change color for most recent pctile% of runs
    if(int(pctile*l) > 1):
        #  NOTE: runs are listed most RECENT first
        n1 = int((pctile)*l)  # first 1-p % runs
        d1 = (r.times[:n1])  # most recent runs
        d2 = (r.times[n1:])  # rest
        print "size l,d1,d2: ", l, np.size(d1), np.size(d2)
    # plot the histogram
        for j in range(0, len(d1)):  # shift times down so relative to 5:00 
            d1[j] -= 300  # 300 seconds
        for j in range(0, len(d2)):
            d2[j] -= 300 
        n, bins, patches = plt.hist([d2,d1], 50, normed=0,color=colors,stacked=True,alpha=0.5)
        recent_mean = np.float(np.sum(d1))/n1
        print "Sum: ", np.sum(d1)
        plt.title(r.name + " (recent runs in RED)")
    else: # plain old boring histogram
        d1 = (r.times[:])
        for j in range(0, len(d1)):  # shift times down so relative to 5:00 
            d1[j] -= 300  # 300 seconds
        n, bins, patches = plt.hist(d1, 50, normed=0,color=onecolor,alpha=0.5)
        plt.title(r.name)
    #plt.xlabel('time (sec)')
   
    #  make a horizontal bar for mean and += 1 SD
    xl = [r.avg_pace - r.sd_pace,r.avg_pace - r.sd_pace, r.avg_pace, r.avg_pace, r.avg_pace + r.sd_pace, r.avg_pace + r.sd_pace, r.avg_pace - r.sd_pace, r.avg_pace + r.sd_pace]
    for j in range(0,len(xl)):
        xl[j] -= 300   # subtract off 5:00 pace
    b1 = 6.0
    tick = 0.25
    b2 = b1+tick
    b3 = b2+tick/2
    b4 = b3+tick
    b = (b1+b2)/2
    yl = [ b1,b2,b1,b2,b1,b2,b,b]
    for i in [0,2,4,6]:
        x = [xl[i], xl[i+1]]
        y = [yl[i], yl[i+1]]
        plt.plot(x,y, linewidth= 2.0, alpha= 1.0,color='blue')
    # plot a tick mark for mean of most recent runs (if applicable)
    if(recent_mean > BIG_NEG_FLAG):
        x = recent_mean
        plt.plot([x,x],[b3, b4],linewidth=2.0, color='red')
        
    # plot the normal distribution 
    y = 100*mlab.normpdf(bins, r.avg_pace-300, r.sd_pace)
    plt.plot(bins, y, 'r')

    plt.xlim([-30,30])
    plt.ylim([0, 13])
    plt.ylabel('N runs')
    plt.xlabel('Pace per km (sec)')  



    
    ##############################################   Plot Pace vs Time
    plt.figure(2) 
    plt.plot(r.dates, r.times)
    if len(r.times) > 20:
        sm = smooth(np.asarray(r.times), 15, 'flat')
        #print "Data is smoothed ", sm.shape
        plt.plot(r.dates, sm.T)
        plt.title('Pace History with 15 run moving avg.: '+r.name)
    else:
        plt.title('Pace History: '+ r.name)
    plt.grid([1,1])
    plt.ylim([250,350])
    plt.show()
