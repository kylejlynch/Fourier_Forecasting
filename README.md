# Fourier Series Forecasting
Time series forecasting with Fourier series

This repository is devoted to my latest project. I started this project with the aim of developing a forecasting model that could compete with Holt-Winters methods. Please see below for a description and my progess.

**Background** : A Fourier series is a summation of sine and cosine terms each multiplied by a unique constant value. For example, A Fourier series with 4 terms (n=2) would look like
  
f(x)= a<sub>1</sub>&middot;cos⁡πx+ b<sub>1</sub>&middot;sin⁡πx+ a<sub>2</sub>&middot;cos⁡2πx+ b<sub>2</sub>&middot;sin⁡2πx 
  
where the coefficients a and b would need to be solved for the particular case. Fourier series can be used to approximate functions making it a powerful tool for analysis. As the number of terms (n) increases, the approximation becomes more precise. Figure 1 shows how the fit becomes better as n increases.
![](https://i.imgur.com/HThNoUw.png "Figure1")
After writing the code to produce the fits shown in Figure 1 above, I then had to develop a way to use the resulting equations to forecast time series data. I decided to try doing simple linear regressions on all of the coefficients respectively (e.g. all of the a1 terms, all of the a2 terms, etc.). 
![Figure2](https://i.imgur.com/ce3PRBH.png "Figure2")
Figure 3 shows an example of how I used this to forecast the next line in a series of lines with increasing slopes.
![Figure3](https://i.imgur.com/8lH1E0L.png "Figure3")
After confirming with simple examples such as the one above, I tried my code on more complex data. Figure 4 shows the result of applying the Fourier series forecast to a more realistic data set.
![Figure4](https://i.imgur.com/2HOTIrR.png "Figure4")
After obtaining an equation for the forecasted curve, other analysis can be done. Figure 4 below displays application of my code for measuring attainment (see the next section for details)
![Figure5](https://i.imgur.com/0hCVz85.png "Figure5")

![](https://i.imgur.com/gUEadBe.png "Figure6")

