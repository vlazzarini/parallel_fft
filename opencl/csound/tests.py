import ctcsound as csnd
import pylab as pl
import time

devs = 3
dur = 100.0
tests = []

gpu = []
ps = [9,11,13,15]
cs = csnd.Csound()
lens = [16,17,18,19,20,21,22]
for M in ps:
 runs = []
 for N in lens:
  repeats = 1
  print("TESTING N=%d s", N)
  aver = []
  for dev in range(0,devs):
   aver.append(0.0)
   for i in range(0,repeats):
    cs.compile_("csound", "tests.csd", "--opcode-lib=./libclconv.dylib", "-n", "-dm0")
    cs.readScore("i1 0 %d %d %d %d" % (dur, M, N, dev))
    cs.performKsmps()
    ti = time.clock()
    while cs.scoreTime() < dur:
     cs.performKsmps()
    aver[dev] += time.clock() - ti
    cs.reset()
   aver[dev] /= repeats
   print(aver[dev])
   t = dur/aver[dev]
   aver[dev] = t
   if dev == 1: gpu.append(t)
  runs.append(aver)
 tests.append(runs)

pl.rcParams["figure.figsize"] = (8,8)
fig, axs = pl.subplots(len(tests)//2, len(tests)//2)


fmt = ['k:', 'k--', 'k']


    
n = 0
for test in tests:  
 for i in range(0,devs):
    d = pl.zeros(len(test))
    k = 0 
    for j in test:
        d[k] = j[i]
        k += 1
    axs[n%2,n//2].set(xticks=lens)
    axs[n%2,n//2].set(xlim=(lens[0], lens[len(lens)-1]))
    axs[n%2,n//2].set_title("$M=$%d" % 2**ps[n])
    axs[n%2,n//2].set(ylabel='RT ratio')
    axs[n%2,n//2].set(xlabel='$\log_2L$')
    axs[n%2,n//2].plot(lens, d, fmt[i])
    axs[n%2,n//2].grid()

 n += 1



fig.tight_layout()
pl.savefig("plot.eps")
#pl.show()

table = '''{\\bf 512} & %.1f & %.1f & %.1f & %.1f & %.1f  & %.1f  & %.1f  \\\\ \hline
{\\bf 2048} & %.1f & %.1f & %.1f & %.1f & %.1f  & %.1f  & %.1f \\\\ \hline
{\\bf 8192} & %.1f & %.1f & %.1f & %.1f & %.1f  & %.1f  & %.1f \\\\ \hline
{\\bf 32768} & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f  & %.1f \\\\ \hline''' % tuple(gpu)
f = open("table.tex", "w")
f.write(table)
print(table)
