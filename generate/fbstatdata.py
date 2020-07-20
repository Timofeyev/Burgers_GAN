

import numpy as np
import math
#import time as walltime
import pickle
import ast
#import sys


class StatData:
   "Data for Statistics"

   def __init__(self, pdat):
      "Allocate Arrays for computing Stat"
       # one-point stat

      self.meanxu = np.zeros(pdat.dimslow, dtype=np.float64)
      self.varxu  = np.zeros(pdat.dimslow, dtype=np.float64)
      self.ex3xu = np.zeros(pdat.dimslow, dtype=np.float64)
      self.ex4xu = np.zeros(pdat.dimslow, dtype=np.float64)
      
      
      self.stat_counter = 0.0
      self.stat_counter_param = 0.0

      self.cfxu     = np.zeros((pdat.cfdimslow, pdat.cflen), dtype=np.float64)
      self.cfsolnxu = np.zeros((pdat.cfdimslow, pdat.cflen), dtype=np.float64)
      
      
      self.cfxu_soln_filled = False
      self.stat_counter_cfxu = 0.0
      self.cfxu_index = 0
      self.timecfxu = 0.0
      
      self.meanfftu = np.zeros(pdat.dim21, dtype=np.float64)
      self.specu    = np.zeros(pdat.dim21, dtype=np.float64)
      
      self.meanfftU = np.zeros(pdat.dimslow21, dtype=np.float64)
      self.specU    = np.zeros(pdat.dimslow21, dtype=np.float64)
      
      self.meanfftxu = np.zeros(pdat.dimslow21, dtype=np.float64)
      self.specxu    = np.zeros(pdat.dimslow21, dtype=np.float64)
      
      self.spec_counter = 0.0
      
      
      

   # -----------------------------------
   def compute_specall(self, fftu, fftU, fftxu):
      "Compute Averaged Spectra"

      self.spec_counter += 1.0
  
      self.meanfftu = self.meanfftu + fftu
      self.specu    = self.specu + np.absolute(fftu)**2
      self.meanfftU = self.meanfftU + fftU
      self.specU    = self.specU + np.absolute(fftU)**2   
      self.meanfftxu = self.meanfftxu + fftxu      
      self.specxu    = self.specxu + np.absolute(fftxu)**2   


   
   # -----------------------------------
   def output_specslow(self, filenameu, filenameu2):
      "Output Averaged Spectra xu"


      with open(filenameu, 'w') as handle:
         for k in self.specxu:
            tmp3 = k / self.spec_counter
            handle.write(str(tmp3))
            handle.write(" ")

      with open(filenameu2, 'w') as handle:
         for k in self.meanfftxu:
            tmp4 = k / self.spec_counter
            handle.write(str(tmp4))
            handle.write(" ")

   
   # -----------------------------------
   # One Point Stat Slow = mean and var of slow variables
   # -----------------------------------
   def compute_onep_statslow(self, xu):
      "Compute One-Point Stat"

      self.meanxu = self.meanxu + xu
      self.varxu = self.varxu + np.square(xu)
      self.ex3xu = self.ex3xu + np.power(xu,3)
      self.ex4xu = self.ex4xu + np.power(xu,4)
      
      self.stat_counter += 1.0  

   def print_onep_statslow(self, filenameu, pdat):
      "Print One Point Stat"
      print("-- Stat Counter    = ", self.stat_counter)   

      
      print("Mean(xu) = ", self.meanxu/self.stat_counter)
      print("Var(xu)  = ", self.varxu/self.stat_counter - (self.meanxu/self.stat_counter)**2)
      print("E[xu^3] = ", self.ex3xu/self.stat_counter)
      print("E[xu^4] = ", self.ex4xu/self.stat_counter)
          
      with open(filenameu, 'w') as handle:
         for idim in range(pdat.dimslow):
            handle.write(str(idim))
            handle.write(" ")
         handle.write("\n")

         for idim in range(pdat.dimslow):
            tmp = self.meanxu[idim]/self.stat_counter
            handle.write(str(tmp))
            handle.write(" ")
         handle.write("\n")

         for idim in range(pdat.dimslow):
            tmp = self.varxu[idim]/self.stat_counter - (self.meanxu[idim]/self.stat_counter)**2
            handle.write(str(tmp))
            handle.write(" ")
         handle.write("\n")
         
         for idim in range(pdat.dimslow):
            tmp = self.ex3xu[idim]/self.stat_counter
            handle.write(str(tmp))
            handle.write(" ")
         handle.write("\n")
         
         for idim in range(pdat.dimslow):
            tmp = self.ex4xu[idim]/self.stat_counter
            handle.write(str(tmp))
            handle.write(" ")
         handle.write("\n")
 
       
   # -----------------------------------
   def compute_all_onep_statslow(self, xu, pdat):
      "WRAPPER: Compute all onep stat"

      self.compute_onep_statslow(xu)
      
      
    # -----------------------------------
   def output_all_statslow(self, pdat):
      "WRAPPER: Output all stat"

      self.print_onep_statslow("statxu.dat", pdat)
      self.print_cftslow("cftxu.dat", pdat)

   

   

   # -----------------------------------
   # CF Time Slow
   # -----------------------------------
   def record_cfsolnslow(self, xu, pdat):
      "Record soln into cf_soln"
      
      self.cfsolnxu[:,self.cfxu_index] = xu[0:pdat.cfdimslow]
      self.cfxu_index += 1
      self.timecfxu = 0.0

      
   # -----------------------------------
   def update_timecfxu(self, dt):
      "Update timecf"
      self.timecfxu += dt


   # -----------------------------------
   def compute_cftslow(self, xu, pdat):
      "Compute CF Time"

      if self.cfxu_soln_filled:
         if (self.timecfxu >= pdat.cflag - pdat.dt05):
            #print("Computing CF, time = ", pdat.time)

            # compute CF
            for ilen in range(pdat.cflen):
               self.cfxu[:,ilen]  = self.cfxu[:,ilen]  + self.cfsolnxu[:,0] * self.cfsolnxu[:,ilen]
            self.stat_counter_cfxu += 1.0
            self.timecfxu = 0.0

            # shift soln
            for ilen in range(pdat.cflen-1):
               self.cfsolnxu[:,ilen] = self.cfsolnxu[:,ilen+1]

            # add one more value of v to cfsoln
            self.cfsolnxu[:,pdat.cflen-1] = xu[0:pdat.cfdimslow]
      else:
         if (self.timecfxu >= pdat.cflag - pdat.dt05)and(self.cfxu_index < pdat.cflen):
            self.record_cfsolnslow(xu, pdat)
            if (self.cfxu_index == pdat.cflen):
               self.cfxu_soln_filled = True

      
               
   # -----------------------------------
   def print_cftslow(self, filenamecfu, pdat):
      "Write CF Time into File"

      print("-- Stat Counter CF =", self.stat_counter_cfxu)

      with open(filenamecfu, 'w') as handle:
         for ilen in range(pdat.cflen):
            handle.write(str(ilen*pdat.cflag))
            handle.write(" ")         
      
         handle.write("\n")


         for idim in range(pdat.cfdimslow):
            for ilen in range(pdat.cflen):
               tmp = self.cfxu[idim,ilen]/self.stat_counter_cfxu - (self.meanxu[idim]/self.stat_counter)**2
               handle.write(str(tmp))
               handle.write(" ")
            handle.write("\n")
     

   
# -----------------------------------
# End class StatData
# -----------------------------------
