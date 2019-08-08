# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:34:55 2019

@author: 3722270
"""

import numpy as np
import pandas as pd
from matplotlib import ticker
from sympy import Symbol, sympify
import sympy as sp
import scipy.optimize
from symfit import Fit,parameters,variables,GreaterThan,LessThan,Equality,Parameter
import matplotlib.pyplot as plt
#import seaborn as sns
from matplotlib.patches import Polygon
from scipy.integrate import quad
#sns.set(style="darkgrid")
class Fourier :
    """
    Fits data with a Fourier Series
    """
    def __init__(self,file_name,y_label=None,revenue_goal=None,freq=None,number_of_terms=5) :
        if isinstance(file_name,str) :
            df = pd.read_excel(file_name)
            ydata = df['{}'.format(y_label)].values
        elif isinstance(file_name,list) :
            ydata = np.asarray(file_name)
            
        if len(ydata) == 3 :
            self.x = np.linspace(0,1,1000)
            y = np.full_like(self.x,ydata[0])
            y[self.x>0.333] = ydata[1]; y[self.x>0.666] = ydata[2]
            self.y = y
        else :
            self.y = ydata
            self.x = np.linspace(0,1,len(ydata))
        '''
        insAvg = [(a + b) / 2 for a, b in zip(ydata[::2], ydata[1::2])]
        ins = np.arange(1,len(ydata),2)
        for i,j in zip(ins,insAvg) :
            ydata.insert(i,j)
        self.x = np.linspace(0,1,len(ydata))
        '''
        self.label = y_label
        self.revenueGoal = revenue_goal
        self.n = number_of_terms
        if not freq == None :
            self.w = freq
        else :
            self.w = Parameter('w',1*2*np.pi)
    
    def fourier(self) :
        n=self.n
        w = self.w
        lst = range(n+1)
        self.a_n = parameters(','.join(['a{}'.format(i) for i in lst]))
        self.b_n = parameters(','.join(['b{}'.format(i) for i in lst]))
        self.coeff = self.a_n + self.b_n
        self.eqn = sum([i * sp.cos(k * w * Symbol('x')) + j * sp.sin(k * w * Symbol('x')) for k,(i,j) in enumerate(zip(self.a_n,self.b_n))])
        return self.eqn

    def fit(self) :
        x, y = variables('x, y')
        model_dict = {y: self.fourier()}
        self.ffit = Fit(model_dict, x=self.x, y=self.y)
        self.fit_result = self.ffit.execute()
        self.orderedDict = self.fit_result.params
        return self.fit_result.params

    def fitFunc(self) :
        self.fiteqn = self.eqn
        for k,v in self.orderedDict.items() :
            self.fiteqn = self.fiteqn.subs(Parameter('{}'.format(k)),self.orderedDict[k])
        return self.fiteqn
        
    def fFunc(self,x) :
        """Function for plugging into distConst to get constant c"""
        return self.fiteqn.subs(Symbol('x'),x)

    def adjustFunc(self) :
        integral = quad(self.fFunc,0,1)
        c = self.revenueGoal/integral[0]
        self.orderedDict.update((k, v*c) for k, v in self.orderedDict.items())
        #print(self.eqn)
        self.adjeqn = self.eqn
        for k,v in self.orderedDict.items() :
            self.adjeqn = self.adjeqn.subs(Parameter('{}'.format(k)),self.orderedDict[k])
        print(self.orderedDict)
        print(self.adjeqn)
        return self.adjeqn
    
    def adjFunc(self,x) :
        """Function for plugging into AttainmentCalc"""
        return self.adjeqn.subs(Symbol('x'),x)
    
    def fitPlot(self,plot_data=True) :
        if plot_data == True :
            plt.plot(self.x, self.y,lw=3,alpha=0.7, label=self.label) # plots line that is being fit
        plt.plot(self.x, self.ffit.model(self.x, **self.fit_result.params).y, color='red', ls='--')
        #ax = plt.gca()
        #ax.grid()
        #ax.set_facecolor('g')

class AttainmentCalc :
    "Predicts revenue for the given time period. Predicts Attainment for given\
    actual revenue for given time period."
    def __init__(self, distributed_function,start,stop,color='r',actual_revenue=None) :
        self.dfunction = distributed_function
        self.a = start
        self.b = stop
        self.actRevenue = actual_revenue
        self.x = np.linspace(0,1,1000)
    
    def integrateFunction(self) :
        self.predRev = quad(self.dfunction,self.a,self.b)[0] # pred revenue to date
        self.calcGoal = quad(self.dfunction, 0, 1)[0]
        return self.predRev, self.calcGoal
    
    def calcAttainment(self) :
        return self.actRevenue/self.predRev   # actual/goal

    def plotIntegral(self,function, color='r') :
        function = function
        y = np.array([function(i) for i in self.x])
        fig, ax = plt.subplots()
        plt.plot(self.x, y, color, linewidth=2)
        #plt.ylim(ymin=0)
        #if not limits == None :
        # Make the shaded region
        ix = np.linspace(self.a, self.b) # integral limits
        iy = np.array([function(i) for i in ix])
        verts = [(self.a, 0), *zip(ix, iy), (self.b, 0)]
        #poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
        #poly = Polygon(verts, facecolor=[0.5,0,1], edgecolor=[0,0,0],alpha=0.2)
        poly = Polygon(verts, facecolor=[1,0.5,0], edgecolor=[0,0,0],alpha=0.6)
        ax.add_patch(poly)

        plt.figtext(0.01, 0.98, r'Quarterly Goal: ${:,.2f}'.format(self.calcGoal),horizontalalignment='left',
                    verticalalignment='center', fontsize=10,transform=ax.transAxes)
        plt.figtext(0.01, 0.94, r'Goal To Date: ${:,.2f}'.format(self.predRev), horizontalalignment='left',
                 verticalalignment='center', fontsize=10,transform=ax.transAxes)
        plt.figtext(0.01, 0.90, r'Attainment to Date: {:,.2f}%'.format(self.calcAttainment()*100), horizontalalignment='left',
                    verticalalignment='center', fontsize=10, transform=ax.transAxes)

        plt.figtext(0.9, 0.05, '$x$')
        plt.figtext(0.1, 0.9, '$y$')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        formatter = ticker.StrMethodFormatter('{x:,.0f}')
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xticks((self.a, self.b))
        ax.set_xticklabels(('${}$'.format(self.a), '${}$'.format(self.b)))
        #ax.set_yticks([])
        #plt.show()
        #plt.close()

ff = Fourier(file_name = 'fs_data1.xlsx',y_label='net_rev', number_of_terms=25,freq=6,revenue_goal = 15000)
fit_result = ff.fit()
ff.fitFunc()
ff.fitPlot()
ff.adjustFunc()
ff.fitPlot()
ffa = AttainmentCalc(ff.adjFunc,start=0.0,stop=0.5,actual_revenue=8000)
I = ffa.integrateFunction()
attain = ffa.calcAttainment()*100
ffa.plotIntegral(ff.adjFunc,color='red')
plt.legend(('2016','Revenue to Date'),loc='upper right')
plt.ylabel('Revenue ($)')
plt.xlabel('Time (Months)')
plt.xticks(np.arange(0,1,step=0.25), ('','April', 'May', 'June',''))
plt.savefig('fourier_attain2.png',dpi=300,bbox_inches='tight')
plt.show()

"""       
ff = Fourier(file_name = [10,2,6], number_of_terms=15)
fit_result = ff.fit()
ff.fitFunc()
ff.fitPlot() # Plot the result
plt.xlabel('hey')
plt.xticks([])
"""

"""
#ff = Fourier(file_name = 'python_data/Magnon_INTL_FXF_PAYER_PYQ_FY19Q3.xlsx',y_column='PYQ_REVENUE',revenue_goal = 9565, number_of_terms=15)
ff = Fourier(file_name = 'python_data/fs_data2.xlsx',y_column='net_rev',revenue_goal = 1, number_of_terms=15)
fit_result = ff.fit()
ff.fitFunc()
ff.fitPlot() # Plot the result
ff.adjustFunc()
ff.fitPlot() # Plot the result
ffa = AttainmentCalc(ff.adjFunc,start=0.0,stop=0.766,actual_revenue=7019)
I = ffa.integrateFunction()
attain = ffa.calcAttainment()*100
ffa.plotIntegral(ff.adjFunc,color='purple')
print('Quarterly Goal: ', '${:,.2f}'.format(ff.revenueGoal))
print('Goal To date: ', '${:,.2f}'.format(I))
print('Attainment to date: ', '{:,.2f}%'.format(attain))
#print(ff.ffit.model(x=xdata, **ff.fit_result.params).y)
"""