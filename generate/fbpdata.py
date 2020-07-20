
#
# This is the class which keeps all the data about integration,
# computing statistics, etc
#


import numpy as np
import math
import time as walltime
import pickle
import ast
#import sys

class PData:
   "All Integrator Data for the Program"

   # -----------------------------------
   # pdict = dictionary
   def __init__(self):
      "Init Class for Parameters"
      self.pdict = dict()
     

   # -----------------------------------
   # d = another dictionary
   def update_from_dict(self, d):
      "Update Parameter dict from another Dict"
      self.pdict.update(d)

   # -----------------------------------
   # filename = string
   def update_from_file(self, filename):
      "Update Parameter dict from file"
      dd = dict()
      with open(filename,'r') as tmpfile:
         for line in tmpfile:
            (key, val) = line.split()
            dd[key] = ast.literal_eval(val)
            
      self.pdict.update(dd)

   # -----------------------------------
   def init_params(self):
      "Init Parameters"
      self.dim = self.pdict['dim']
      self.dim21 = int(self.dim/2 + 1)
      self.L = self.pdict['L']
      self.dx = self.L / (self.dim - 1)
      self.dt = self.pdict['dt']
      self.A = self.pdict['A']
      self.vis = self.pdict['vis']
      self.seed = self.pdict['seed']
      self.averwindow = self.pdict['averwindow']
      self.n = self.pdict['n']
      
      
      self.dtdx = self.dt/self.dx
      self.dxdt2   = self.dx/(2*self.dt)
      self.dtdx2   = self.dt/(2*self.dx)
      self.x0 = (2*np.sqrt(2)-1)**2
      self.dxl = (self.dx*self.averwindow)/self.L  
      self.tpi=2*np.pi
      self.Adt=self.A/np.sqrt(self.dt)
      
      self.Tend = self.pdict['Tend']

      self.TstartOut = self.pdict['TstartOut']
      self.TstopOut = self.pdict['TstopOut']
      self.dtout = self.pdict['dtout']

      self.Tskip = self.pdict['Tskip']
      self.dtsubs = self.pdict['dtsubs']

      self.time = 0.0
      self.timeout = 0.0
      self.dt05   = 0.5*self.dt
      
      self.cflen   = self.pdict['cflen']
      self.cflag   = self.pdict['cflag']
      self.cfdim   = self.pdict['cfdim']
      
      self.pdflen  = self.pdict['pdflen']
      self.pdfstep = self.pdict['pdfstep']
      self.pdfmean = self.pdict['pdfmean']
      self.pdfdim  = self.pdict['pdfdim']
      

      # messaging
      self.dtmes = self.pdict['dtmes']
      self.mes_into_file = self.pdict['MesFile']
      self.timemes = 0.0
     
      
      # compute how many steps between various things -
      
      # skipping, sub-sampling, etc.
      self.nums_subs = int(self.dtsubs/self.dt)
      self.nums      = int((self.Tend - self.Tskip)/self.dtsubs)
      self.nums_skip = int(self.Tskip/self.dtsubs)

      self.dimslow    = int(self.dim /self.averwindow)
      self.dimslow21 = int(self.dimslow/2 + 1)
      
      self.n2dx=self.averwindow*2.0*self.dx
      self.ndx2=self.averwindow*((self.dx)**2)
      self.dtndx=self.dt/(self.dx*self.averwindow)
      
      self.cfdimslow = int(self.cfdim/self.averwindow)
      
      self.vdx2=self.vis/(self.dx**2)
      self.dx6=self.dx*6.0
      self.vndx2=self.vis/((self.dx*self.averwindow)**2)
      self.vndx22=self.vis/(self.averwindow*(self.dx**2))
      self.ndx6=self.dx*6.0*self.averwindow
      
      # -------------------------------
      # I also want to compute the time-interval since last message
      # and estimate the running time of the program.
      self.clockprev = walltime.clock()
      self.enerprev = 0.0

      self.clockstart = 0.0

      # -------------------------------
      # file to output time-series of energy
      self.enerfile = open('ener.dat','w')

   # -----------------------------------
   def compute_energy(self, xu):
      "Compute energy in Solution"

      ener = 0.0
      for i in range(int(self.dimslow-1)):
         ener += xu[i]**2
      return ener/self.dimslow*self.L
  
   # -----------------------------------
   def PrintMesProgress(self, u, dt):
      "Print Message about Progress"
      "assume this function is called outside of the sub-samplig loop"
      " thus, we advance time by dt * nums_subs"
      self.timemes += dt

      if (self.timemes >= self.dtmes - self.dt05):
         self.timemes = 0.0         
         if self.mes_into_file:

            clockcurrent = walltime.clock()
            ener = self.compute_energy(u)

            # write energy into file
            self.enerfile.write(str(self.time))
            self.enerfile.write(" ")
            self.enerfile.write(str(ener))
            self.enerfile.write("\n")

            
            # write message
            tmpfile = open('mes.out','w')
            tmpfile.write("Computing at Time = ")
            tmpfile.write(str(self.time))
            tmpfile.write("   ")
            tmpfile.write("Energy = ")
            tmpfile.write(str(ener))
            tmpfile.write("\n")
            
            tmpfile.write("Prev Energy = ")
            tmpfile.write(str(self.enerprev))
            tmpfile.write("\n")
            self.enerprev = ener
            
            tmpfile.write("Momentum = ")
            tmpfile.write(str(np.sum(u)))
            tmpfile.write("\n")

            tmp = clockcurrent - self.clockprev
            tmpfile.write("Wallcklock time since last mes = ")
            tmpfile.write(str(tmp))
            tmpfile.write(" sec\n")
            self.clockprev = clockcurrent

            tmpfile.write("Wallclock Time Remaining\n")

            tmp = (clockcurrent - self.clockstart) * (self.Tend - self.time) / self.time
            days = int(tmp/60/60/24)
            hrs  = int(tmp/60/60 - days*24)
            mnts = int(tmp/60 - hrs*60.0 - days*60.0*24.0)
            tmpfile.write("-- days = ")
            tmpfile.write(str(days))
            tmpfile.write(",  hrs = ")
            tmpfile.write(str(hrs))
            tmpfile.write(",  min = ")
            tmpfile.write(str(mnts))
      
            tmpfile.close()
         else:
            print("Computing at Time = ", self.time, "   Energy = ", self.compute_energy(u))

   # -----------------------------------
   def reset_timeout(self):
      "Reset timout"
      
      if self.output_condition():
          self.timeout = 0.0

   # -----------------------------------
   def advance_time(self):
      " Advance All Time Vaiables after 1 time-step"
      self.time    += self.dt
      self.timeout += self.dt

   # -----------------------------------
   def output_condition(self):
      "Return Condition to Output Soln into File"
      cond1 = (self.TstartOut - self.dt05) <= self.time <= (self.TstopOut + self.dt05)
      cond2 = self.timeout >= self.dtout - self.dt05
      return (cond1 and cond2)