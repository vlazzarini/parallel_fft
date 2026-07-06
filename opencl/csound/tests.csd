<CsoundSynthesizer>
<CsOptions>
</CsOptions>
<CsInstruments>
ksmps = 64
0dbfs = 1

instr 1
 iM = pow(2,p4)
 iL = pow(2,p5)
 idev = p6
 ain1 = diskin:a("beats.wav", 1, 0, 1)
 ain2 = diskin:a("fox.wav", 1, 0, 1)
 if idev < 3 then
  asig = cltvconv(ain1,ain2,1,1,iM,iL,idev)
 else
  asig = tvconv(ain1,ain2,1,1,iM,iL)
 endif
  out(asig/900)
endin

</CsInstruments>
<CsScore>
</CsScore>
</CsoundSynthesizer>

