import pyabf

for fname in [
    "data/raw/26629000.abf",
    "data/raw/26629001.abf",
    "data/raw/26629002.abf",
    "data/raw/26629003.abf",
    "data/raw/26629004.abf",
]:
    abf = pyabf.ABF(fname)
    print(f"\n{fname}:")
    print(f"  sweepCount:     {abf.sweepCount}")
    print(f"  sweepLengthSec: {abf.sweepLengthSec:.1f} сек")
    print(f"  dataRate:       {abf.dataRate} Гц")
    print(f"  channelCount:   {abf.channelCount}")
    print(f"  protocol:       {abf.protocol}")