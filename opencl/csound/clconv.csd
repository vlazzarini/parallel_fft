<CsoundSynthesizer>
<CsOptions>
--opcode-lib=./libcl_conv.dylib
</CsOptions>
<CsInstruments>
ksmps = 64
0dbfs = 1

gift ftgen 0, 0, 0, 1, "pianoc2.wav", 0, 0, 1

instr 1
 ipsize = p5
 idev =  p6  
 ain1 diskin "fox.wav", 1, 0, 1
  asig ftconv ain1, gift, ipsize
 out asig*linenr(p4,0.1,0.5,0.01)
endin

instr 2
 ipsize = p5
 idev =  p6
 icsize = filelen("beats.wav")*sr
 ain1 diskin "fox.wav", 1, 0, 1
 ain2 diskin "beats.wav", 1, 0, 1
  asig tvconv ain1, ain2, 1, 1, ipsize, icsize
  out asig*linenr(p4,0.1,0.5,0.01)
endin


</CsInstruments>
<CsScore>
;i1 0 10 0.005 2048 0
i2 0 10 0.005 2048 0
</CsScore>
</CsoundSynthesizer>

