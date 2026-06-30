import pyabf
import numpy as np

for fname in [
    "data/raw/26629000.abf",
    "data/raw/26629001.abf",
    "data/raw/26629002.abf",
    "data/raw/26629003.abf",
    "data/raw/26629004.abf",
]:
    abf = pyabf.ABF(fname)
    abf.setSweep(0)
    
    print(f"\n{'='*50}")
    print(f"Файл: {fname}")
    
    # Вариант 1: командный канал
    print(f"  channelCount: {abf.channelCount}")
    print(f"  channelList:  {abf.channelList}")
    
    # Вариант 2: DAC команды
    print(f"  dacUnits:     {abf.dacUnits}")
    
    # Вариант 3: теговые события (epoch)
    print(f"  sweepEpochs:  {abf.sweepEpochs}")
    
    # Вариант 4: временные метки событий
    print(f"  tagComments:  {abf.tagComments}")
    print(f"  tagTimesSec:  {abf.tagTimesSec}")