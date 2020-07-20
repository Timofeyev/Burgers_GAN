
import numpy as np


class SolnData():
   "Data for the Solution"

   def __init__(self, pdat):

      self.u = np.zeros(pdat.dim, dtype=np.float64)
      
      self.U = np.zeros(pdat.dimslow, dtype=np.float64)
      self.Uout = np.zeros((pdat.n,int(pdat.Tend/pdat.dtsubs)), dtype=np.float64)
      self.uoutMC = np.zeros((10,int(pdat.Tend/pdat.dtsubs)), dtype=np.float64)
      self.xu = np.zeros(pdat.dimslow, dtype=np.float64)
   
      for kk in range(pdat.dim):
         self.u[kk] = 0
       
      for k in range(pdat.dimslow):
          averu = 0.0
          for j in range(pdat.averwindow):
              indx = j + k*pdat.averwindow
              averu += self.u[indx]
          self.xu[k] = averu/pdat.averwindow
      
      self.U=self.xu
      
      print("Initial Momentum = ", np.sum(self.xu))
      print("IC for u = ", self.u, "\n\n")
      
      print("IC for U, xu = ", self.U, "\n\n")
      
      self.fftu    = np.zeros(pdat.dim21, dtype=np.float64)
      self.fftxu    = np.zeros(pdat.dimslow21, dtype=np.float64)
      self.fftU    = np.zeros(pdat.dimslow21, dtype=np.float64)
      
      self.fftu  = np.fft.rfft(self.u) / pdat.dim
      self.fftxu  = np.fft.rfft(self.xu) / pdat.dimslow
      self.fftU  = np.fft.rfft(self.U) / pdat.dimslow
      
      self.netfluxxu    = np.zeros(pdat.dimslow, dtype=np.float64)
      
      self.force    = np.zeros(pdat.dim, dtype=np.float64)
      self.alpha = np.zeros(3, dtype=np.float64)
      self.phi = np.zeros(3, dtype=np.float64)
     
      self.xforce    = np.zeros(pdat.dimslow, dtype=np.float64)
      
      
          

      
# -----------------------------------
# End class SolnData
# -----------------------------------
