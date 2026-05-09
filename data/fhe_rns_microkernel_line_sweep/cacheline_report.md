# FHE RNS Cache Line Sweep

Config: L1+L2, trials=3, repeats=1, real-sized RNS operands (`N=16384`, 4 towers, centroid_dim=384).

## Cache Line 32 B
- `simInsts`: mean=3.99027e+08, stdev=0, min=3.99027e+08, max=3.99027e+08, n=3
- `system.cpu.numCycles`: mean=8.00517e+08, stdev=0, min=8.00517e+08, max=8.00517e+08, n=3
- `system.cpu.cpi`: mean=2.00617, stdev=0, min=2.00617, max=2.00617, n=3
- `system.cpu.ipc`: mean=0.498462, stdev=0, min=0.498462, max=0.498462, n=3
- `system.cpu.dcache.overallMissRate::total`: mean=0.006121, stdev=0, min=0.006121, max=0.006121, n=3
- `system.l2.overallAccesses::total`: mean=1.45438e+06, stdev=0, min=1.45438e+06, max=1.45438e+06, n=3
- `system.l2.overallMisses::total`: mean=1.13338e+06, stdev=0, min=1.13338e+06, max=1.13338e+06, n=3
- `system.l2.overallMissRate::total`: mean=0.779289, stdev=0, min=0.779289, max=0.779289, n=3
- `hostSeconds`: mean=337.487, stdev=3.95133, min=333.93, max=341.74, n=3

## Cache Line 64 B
- `simInsts`: mean=3.99027e+08, stdev=0, min=3.99027e+08, max=3.99027e+08, n=3
- `system.cpu.numCycles`: mean=8.00517e+08, stdev=0, min=8.00517e+08, max=8.00517e+08, n=3
- `system.cpu.cpi`: mean=2.00617, stdev=0, min=2.00617, max=2.00617, n=3
- `system.cpu.ipc`: mean=0.498462, stdev=0, min=0.498462, max=0.498462, n=3
- `system.cpu.dcache.overallMissRate::total`: mean=0.003107, stdev=0, min=0.003107, max=0.003107, n=3
- `system.l2.overallAccesses::total`: mean=738212, stdev=0, min=738212, max=738212, n=3
- `system.l2.overallMisses::total`: mean=566732, stdev=0, min=566732, max=566732, n=3
- `system.l2.overallMissRate::total`: mean=0.767709, stdev=0, min=0.767709, max=0.767709, n=3
- `hostSeconds`: mean=328.727, stdev=5.75728, min=322.09, max=332.38, n=3

## Cache Line 128 B
- `simInsts`: mean=3.99027e+08, stdev=0, min=3.99027e+08, max=3.99027e+08, n=3
- `system.cpu.numCycles`: mean=8.00517e+08, stdev=0, min=8.00517e+08, max=8.00517e+08, n=3
- `system.cpu.cpi`: mean=2.00617, stdev=0, min=2.00617, max=2.00617, n=3
- `system.cpu.ipc`: mean=0.498462, stdev=0, min=0.498462, max=0.498462, n=3
- `system.cpu.dcache.overallMissRate::total`: mean=0.001614, stdev=0, min=0.001614, max=0.001614, n=3
- `system.l2.overallAccesses::total`: mean=383647, stdev=0, min=383647, max=383647, n=3
- `system.l2.overallMisses::total`: mean=283424, stdev=0, min=283424, max=283424, n=3
- `system.l2.overallMissRate::total`: mean=0.738762, stdev=0, min=0.738762, max=0.738762, n=3
- `hostSeconds`: mean=328.503, stdev=5.35183, min=324.34, max=334.54, n=3
