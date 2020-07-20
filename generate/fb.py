
#Forced Burgers Equations
#using Flux-difference + (Euler or RK3)
#fb.py

import numpy as np
import math
import time as walltime
import ast
import sys
#import matplotlib.pyplot as plt

from fbpdata import *
from fbsolndata import *
from fbstatdata import *

#from IPython import get_ipython
#get_ipython().magic('reset -sf') 


# -----------------------------------
def PrintHelp():
   "Print Help"

   print("-------------------------------")
   print("==> Forced Burgers equation Simulation")
   print("Long Equilibrium Simulation \n")
   print("Version August 2018")
   print("-------------------------------")

   print("Parameters in the file fb_par.txt \n")

   print("L          = Domain\n")

   
   print("dim        = dimension of the problem")
   print("Tend       = final time of simulation")
   print("dt         = time-step of integration")
   print("seed       = seed for the random number generator")   
   print("Tskip      = skip to schieve stationarity")
   print("dtsubs     = time-step for computing stat (int multiple of dt) \n")


   print("dtmes      = How often to print message about progress")
   print("MesFile    = True - print into file mes.out (for long simulations)")
   print("             False - print to terminal \n")

   print("TstartOut  = Time to Start Soln Output")
   print("TstopOut   = Time to Stop Soln Output")
   print("dtout      = time-step to Soln Output (int multiple of dt) \n") 
   
   print("cflen      = length of array for correlation (<= 100)")
   print("cflag      = CF lag (int multiple of sub-sampling step dtsubs)")
   print("cfdim      = compute CF for u[0:cfdim]")

   print("pdflen     = length for Marginal PDF")
   print("pdfstep    = size of the bin for copmuting PDF using bin-counting")
   print("pdfmean    = pdf will be centered to this value")
   print("pdfdim     = compute PDF for u[0:pdfdim]\n")


   print("-------------------------------")
   print("-- Params for Slow Variables")
   print("averwindow = Aver Window for Fast")
   print("dimslow    = Dimension for the slow variables/averages")

   print("-------------------------------")
   print("Output Files \n")

   print("fbsolnxu.dat  = Slow variables \n")
   print("statxu.dat    = Mean, Variance, 3rd and 4th Moment of xu[k] \n")
   print("cftxu.dat     = CF in time <xu[k](t) xu[k](t+s)> k=0:cfdim \n")
   print("specxu.dat    = Energy spectra for xu \n")
   print("solnfile1xu.dat  = Time series for xu[k] k=0:n")
   print("-------------------------------")

      

# -----------------------------------
def printfile_soln_slow(xu,pdat):
   "Output Soln into a file"
   
   if pdat.output_condition():
      
      solnfilexu.write(str(pdat.time))
      
      for k in xu:
         solnfilexu.write(" ")         
         solnfilexu.write(str(k))
      solnfilexu.write("\n")
      

#-------------
def PrintSlow(filename1,pdat):
    
    with open(filename1, 'w') as handle:
       for i in range(pdat.n):
          for j in range(int(pdat.Tend/pdat.dtsubs)):
             handle.write(str(solndat.Uout[i][j]))
             handle.write(" ")
          handle.write("\n")

#-----------

def lrelu(y,scale=0.2):
    out = np.zeros(len(y), dtype=np.float64)
    for i in range(len(y)):
        if y[i] >= 0:
            out[i] = y[i]
        else:
            out[i] = scale*y[i]
            
    return out

#---------
def compute_addPoly(xu):
    
    
    xujm1 = xu[pdat.dimslow-1]
    xujp1 = xu[0]
    
    xxp1 = np.zeros((pdat.dimslow, 2), dtype = np.float64)

    for i in range(pdat.dimslow - 1):
        xxp1[i][0] = xu[i]
        xxp1[i][1] = xu[i+1]
        
    for i in [pdat.dimslow - 1]:
        xxp1[i][0] = xu[i]
        xxp1[i][1] = xu[0]

    addPoly = np.zeros((pdat.dimslow,2), dtype = np.float64)
    
    for i in range(pdat.dimslow -1):
        input_h = np.append(Z[Z_counter,i], xxp1[i])
        h_1 = np.matmul(input_h, G_W1) + G_b1
        G_h1 = lrelu(h_1)
        h_2 = np.matmul(G_h1, G_W2) + G_b2
        G_h2 = lrelu(h_2)
        h_3 = np.matmul(G_h2, G_W3) + G_b3
        G_h3 = lrelu(h_3)
        addPoly[i] = np.matmul(G_h3, G_W4) + G_b4
    

    return addPoly.T


#----------
def compute_rhs_slow1(xu):
    

    xujm1 = xu[pdat.dimslow-1]
    xujp1 = xu[0]

    for j in range(1,pdat.dimslow-1):
       solndat.netfluxxu[j] =   -(xu[j+1]**2+xu[j+1]*xu[j]-xu[j]*xu[j-1]-xu[j-1]**2)/pdat.ndx6+pdat.vndx22*(xu[j+1]-2*xu[j]+xu[j-1]) 
       

    for j in [0]:
       solndat.netfluxxu[j] = -(xu[j+1]**2+xu[j+1]*xu[j]-xu[j]*xujm1-xujm1**2)/pdat.ndx6+pdat.vndx22*(xu[j+1]-2*xu[j]+xujm1) 


    for j in [pdat.dimslow-1]:
       solndat.netfluxxu[j] = -(xujp1**2+xujp1*xu[j]-xu[j]*xu[j-1]-xu[j-1]**2)/pdat.ndx6+pdat.vndx22*(xujp1-2*xu[j]+xu[j-1]) 
   
    solndat.netfluxxu= solndat.netfluxxu + solndat.xforce 
    
#----------    
def force_term():
    
    global ran_counter
    
    for i in range(0,3):
        solndat.alpha[i]=ran[ran_counter + 2*i]
        solndat.phi[i]=ran[ran_counter + 2*i + 1]
    
    ran_counter +=  6
    
    for j in range(0,pdat.dimslow):
       solndat.xforce[j]=pdat.Adt*(solndat.alpha[0]*np.cos(pdat.tpi*(j*pdat.dxl))+ solndat.phi[0]*np.sin(pdat.tpi*(j*pdat.dxl)) +\
                   solndat.alpha[1]/np.sqrt(2)*np.cos(pdat.tpi*(2*j*pdat.dxl))+ solndat.phi[1]/np.sqrt(2)*np.sin(pdat.tpi*(2*j*pdat.dxl)) +\
                   solndat.alpha[2]/np.sqrt(3)*np.cos(pdat.tpi*(3*j*pdat.dxl)) + solndat.phi[2]/np.sqrt(3)*np.sin(pdat.tpi*(3*j*pdat.dxl)))
       for i in range(pdat.averwindow):
          indx = i + j*pdat.averwindow
          solndat.force[indx] = solndat.xforce[j]


   

#----------
def make_one_step_RK3_average():
    
   global addP
    
   if (int(pdat.time*10) % 1 == 0):
        
       addPflux = compute_addPoly(solndat.xu)
        
       for i in range(pdat.dimslow):
           addP[0][i] = addPflux[0][i] -addPflux[0][i -1]
           addP[1][i] = addPflux[1][i] -addPflux[1][i -1]
   else:
       addP = np.zeros((2, pdat.dimslow), dtype = np.float64)

   global Z_counter
    
   compute_rhs_slow1(solndat.xu)
   
   k1=pdat.dt*solndat.netfluxxu
   
   compute_rhs_slow1(solndat.xu+0.5*k1)
   
   k2=pdat.dt*solndat.netfluxxu
   
   compute_rhs_slow1(solndat.xu - k1 + 2.0*k2)
   
   k3=pdat.dt*solndat.netfluxxu
   

   solndat.xu = solndat.xu + (k1+4.0*k2+k3)/6.0 + pdat.dt*(-addP[0]/pdat.ndx6 + pdat.vndx22*addP[1])
       
   solndat.fftxu = np.fft.rfft(solndat.xu)/pdat.dimslow

   
   Z_counter += 1
   
   


# -----------------------------------
# MAIN PROGRAM
# -----------------------------------


if len(sys.argv) >= 2: 
   if (sys.argv[1] == '-h')or(sys.argv[1]=='-help'):
      PrintHelp()
      sys.exit()

pdat = PData()

pdat.update_from_file('fb_par.txt')
pdat.init_params()


#initializaing stochastic terms and stochastic forcing
np.random.seed(pdat.seed)
ran = np.random.normal(0,1,int(6*pdat.Tend/pdat.dt))
Z = np.random.uniform(-1.,1.,[int(pdat.Tend/pdat.dt),pdat.dimslow,1])
ran_counter = 0
Z_counter = 0

# initializing arrays for solution and integration
solndat = SolnData(pdat)
statdat = StatData(pdat)

#output files for solution on short interval
solnfilexu = open('fbsolnxhu.dat', 'w')
solnfile1xu = open('solnfile1xu.dat', 'w')

cl1 = walltime.clock()
pdat.clockstart = cl1
print("-------------------")
print("Start Computations")
print("Flux-difference scheme + RK3 time-stepping")
print("-------------------")

epoch_num = 100

G_W1 = np.genfromtxt('G_W1_%d.txt'%epoch_num,dtype=np.float64, delimiter=' ')
G_W2 = np.genfromtxt('G_W2_%d.txt'%epoch_num,dtype=np.float64, delimiter=' ')
G_W3 = np.genfromtxt('G_W3_%d.txt'%epoch_num,dtype=np.float64, delimiter=' ')
G_W4 = np.genfromtxt('G_W4_%d.txt'%epoch_num,dtype=np.float64, delimiter=' ')
G_b1 = np.genfromtxt('G_b1_%d.txt'%epoch_num,dtype=np.float64, delimiter=' ')
G_b2 = np.genfromtxt('G_b2_%d.txt'%epoch_num,dtype=np.float64, delimiter=' ')
G_b3 = np.genfromtxt('G_b3_%d.txt'%epoch_num,dtype=np.float64, delimiter=' ')
G_b4 = np.genfromtxt('G_b4_%d.txt'%epoch_num,dtype=np.float64, delimiter=' ')


addP = np.zeros((2, pdat.dimslow), dtype = np.float64)

t=0
# Skip up to time Tskip
for j in range(pdat.nums_skip):
    
   for k in range(pdat.nums_subs):
       
      force_term()
      make_one_step_RK3_average()
      pdat.advance_time() 
   pdat.PrintMesProgress(solndat.xu, pdat.dtsubs)

   for i in range(pdat.n):
      solndat.Uout[i][t] = solndat.xu[i]
      
   t=t+1
   printfile_soln_slow(solndat.xu,pdat)

   pdat.reset_timeout

print("Time After Skip = ", pdat.time)

compute_stat = True

if compute_stat:

    statdat.compute_specall(solndat.fftu, solndat.fftU, solndat.fftxu)
    statdat.compute_all_onep_statslow(solndat.xu,pdat)
    statdat.record_cfsolnslow(solndat.xu, pdat)
    
# ---------------
# Main Loop
# After Tskip
# ---------------
for j in range(pdat.nums):
   for k in range(pdat.nums_subs):
      force_term()
      make_one_step_RK3_average()
      pdat.advance_time()      
      
   for i in range(pdat.n):
      solndat.Uout[i][t] = solndat.xu[i]

   t=t+1
   printfile_soln_slow(solndat.xu, pdat)
   pdat.reset_timeout
   
      
   pdat.PrintMesProgress(solndat.xu, pdat.dtsubs)
   
   if compute_stat:

      # compute spectra 
      statdat.compute_specall(solndat.fftu, solndat.fftU, solndat.fftxu)
      
      # compute one parameter statistics
      statdat.compute_all_onep_statslow(solndat.xu, pdat)
      
      # compute CF
      statdat.compute_cftslow(solndat.xu,pdat)
      
      # Update timecf
      statdat.update_timecfxu(pdat.dt * pdat.nums_subs)
      
      

print("-------------------")
print("End Computations")
print("-------------------")


# ---------------
# End of Main Loop
# ---------------

solnfilexu.close()
solnfile1xu.close()
pdat.enerfile.close()

if compute_stat:
   
   statdat.output_specslow("specxu.dat", "meanfftxu.dat")
   
   print("-------------------")
   print("Output Stat for Full Dynamics")
   print("-------------------")
   statdat.output_all_statslow(pdat)

cl2 = walltime.clock()
print("-------------------")
print("End of Computations, walltime = ", (cl2-cl1), " sec")
days = int((cl2-cl1)/60/60/24)
hrs  = int((cl2-cl1)/60/60 - days*24)
mnts = int((cl2-cl1)/60 - hrs*60.0 - days*60.0*24.0)
print("    days =", days, "hrs =", hrs, "min =", mnts)
print("-------------------")

PrintSlow('Useries.dat',pdat)
