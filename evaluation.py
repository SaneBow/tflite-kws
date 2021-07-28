import time
from pathlib import Path
import argparse
import soundfile as sf
import logging
from kws import TFLiteKWS, SILENCE, NOT_KW, VERBOSE
import numpy as np
from matplotlib import pyplot as plt

# logging.basicConfig(level=logging.INFO)

g_SCORES = {
    'neg': {},
    'kw': {}
}


def load_kw_index(idxfile):
    kw_index = {}
    with open(idxfile, 'r') as f:
        for line in f.read().splitlines():
            ts, kw = line.split()
            ts = float(ts)
            kw_index[ts] = kw
    return kw_index


def process_file(infile, block_len_ms, contain_kw=False, gkws=None):
    print(f"Processing file: {infile}")
    infile = str(infile)
    score_file = Path(infile.rsplit('.', maxsplit=1)[0] + '.scores')

    if gkws:
        score_fp = open(score_file, 'w+')
        info = sf.info(infile)
        assert info.channels == 1
        bs = block_len_ms * info.samplerate // 1000
        blocks = sf.blocks(infile, blocksize=bs)
        dur = info.duration
        tot_blocks = dur * 1000 // block_len_ms
        st = time.time()
        for i, b in enumerate(blocks):
            if not b.shape[0] == bs:
                # logging.warning(f"last block? length={b.shape[0]}")
                break
            gkws.process(b, dump_fp=score_fp)
            if i > 0 and i % 1024 == 0:
                dt = time.time() - st
                print(f"{i/tot_blocks: >4.0%}\tspeed: {dt/1024*1000: >3.2f}ms/block\tremain: {(tot_blocks-i)*dt/1024/60: >2.0f}min", end='\r')
                st = time.time()
        score_fp.close()
        
    scores = np.loadtxt(score_file, delimiter=', ')
    retdic = {'scores': scores}
    if contain_kw:
        idx_file = Path(infile.rsplit('.', maxsplit=1)[0] + '.index')
        kw_idx = load_kw_index(idx_file)
        retdic['kw_idx'] = kw_idx

    if contain_kw:
        g_SCORES['kw'][infile] = retdic
    else:
        g_SCORES['neg'][infile] = retdic

    return retdic


def process_folder(inpath, block_len_ms, contain_kw=False, gkws=None):
    for wavp in inpath.glob('*.wav'):
        process_file(wavp, block_len_ms, contain_kw=contain_kw, gkws=gkws)


def process_path(inpath, bs_ms, contain_kw=False, gkws=None):
    if inpath.is_dir():
        process_folder(inpath, bs_ms, contain_kw=contain_kw, gkws=gkws)
    elif inpath.is_file():
        process_file(inpath, bs_ms, contain_kw=contain_kw, gkws=gkws)
    else:
        print(f"{inpath} doens't exist")


def eval_gkws(gkws, scores_all):
    detected = {}
    for blk, scores in enumerate(scores_all):
        kw, _ = gkws._any_kw_triggered(scores)
        if len(kw) > 0:
            t = blk * gkws.block_ms / 1000
            detected[t] = kw
    return detected


def compute_acc(gkws, tolerate=1.0):
    ptot = ttot = 0
    for infile, kw_dic in g_SCORES['kw'].items():
        scores_all = kw_dic['scores']
        truth = kw_dic['kw_idx']
        detected = eval_gkws(gkws, scores_all)
        pcnt = 0
        for tt in truth:
            for td in detected:
                if abs(td - tt) < tolerate and truth[tt] in detected[td]:
                    pcnt += 1
                    break
            else:
                logging.debug(f"missed at {tt}")
        logging.info("%s: %s TP out of %s", infile, pcnt, len(truth))
        ptot += pcnt
        ttot += len(truth)
    return ptot / ttot


def compute_wer(gkws):
    fp_tot = dur_tot = 0
    for infile, neg_dic in g_SCORES['neg'].items():
        scores_all = neg_dic['scores']        
        detected = eval_gkws(gkws, scores_all)
        dur_hrs = len(scores_all) * gkws.block_ms / 1000 / 60 / 60
        logging.info("%s: %s FP in %.2f hours", infile, len(detected), dur_hrs)
        fp_tot += len(detected) 
        dur_tot += dur_hrs
    return fp_tot / dur_tot



parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', type=str, required=True,
    help='tflite model')
parser.add_argument(
    '-ik', '--infile-kw', type=str, required=True,
    help='file or folder containing keywords')
parser.add_argument(
    '-in', '--infile-neg', type=str, required=True,
    help='file or folder with negative recording')
parser.add_argument(
    '-b', '--block-len-ms', type=int, default=20,
    help='input block (window stride) length (ms)')
parser.add_argument(
    '-s', '--score-strategy', choices=['smoothed_confidence', 'hit_ratio'], default='hit_ratio',
    help='score strategy, choose between "smoothed_confidence" (default) or "hit_ratio" (default)'),
parser.add_argument(
    '--add-softmax', action='store_true',
    help='add softmax layer to output')
parser.add_argument(
    '-e', '--eval-only', action='store_true',
    help='evaluate based on scores files (do not process wav files)')
parser.add_argument(
    '--silence-on', action='store_true',
    help='turn on silence detection')

args = parser.parse_args()

if args.score_strategy == 'hit_ratio':
    threshold = 0.01
else:
    threshold = 0.8

def gen_gkws(ht):
    return TFLiteKWS(args.model, [SILENCE, NOT_KW, 'gjr'], add_softmax=args.add_softmax, silence_off=not args.silence_on,
        score_strategy=args.score_strategy, score_threshold=threshold, hit_threshold=ht)

gkws = None
if not args.eval_only:
    gkws = gen_gkws(7)

bs_ms = args.block_len_ms

neg_path = Path(args.infile_neg)
kw_path = Path(args.infile_kw)

process_path(neg_path, bs_ms, gkws=gkws)
process_path(kw_path, bs_ms, contain_kw=True, gkws=gkws)

ht_series = np.linspace(5, 15, 21)
acc_series = []
wer_series = []
for ht in ht_series:
    gkws = gen_gkws(ht) 
    acc = compute_acc(gkws)
    wer = compute_wer(gkws)
    acc_series.append(acc)
    wer_series.append(wer)

print("ht:", ht_series)
print("ACC:", acc_series)
print("WER:", wer_series)

min_ht_idx = next(iter([i for i, wer in enumerate(wer_series) if wer < 0.1]))
print("best ht:", ht_series[min_ht_idx])

plt.plot(acc_series, wer_series, marker='x')
plt.plot(acc_series[min_ht_idx], wer_series[min_ht_idx], marker='o')
for x,y,z in zip(acc_series, wer_series, ht_series):
    label = "{:.2f}".format(z)
    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 fontsize=5,
                 ha='center') # horizontal alignment can be left, right or center

plt.show()