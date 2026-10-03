#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eeg_comp.submission import save_candidate

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Validate and write a local candidate, without uploading.')
    p.add_argument('--competition',required=True,choices=['within_subject','cross_subject'])
    p.add_argument('--category',required=True,choices=['ML','DL'])
    p.add_argument('--predictions',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(save_candidate(a.predictions,Path('artifacts')/a.competition/'metadata.csv',a.competition,a.category,a.output),indent=2))
