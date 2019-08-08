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
    
    def fitPlot(self,plot_data=True,color='red') :
        if plot_data == True :
            plt.plot(self.x, self.y,lw=3,alpha=0.7, label=self.label,color=color) # plots line that is being fit
        plt.plot(self.x, self.ffit.model(self.x, **self.fit_result.params).y, color='red', ls='--',label='_nolegend_')
        formatter = ticker.StrMethodFormatter('{x:,.0f}')
        plt.gca().yaxis.set_major_formatter(formatter)
        #ax = plt.gca()
        #ax.grid()
        #ax.set_facecolor('g')


"""
ff = Fourier(file_name = [1,3,2], number_of_terms=20)
fit_result = ff.fit()
ff.fitFunc()
ff.fitPlot()
plt.ylabel('y')
plt.xlabel('x')
plt.savefig('fourier_fit20.png',dpi=300,bbox_inches='tight')
plt.show()
"""
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