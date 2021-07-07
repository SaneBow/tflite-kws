import time
import argparse
import threading
import collections
import sounddevice as sd
import numpy as np
import logging
from kws import TFLiteKWS, SILENCE, NOT_KW, VERBOSE


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    '-m', '--model', type=str, required=True,
    help='tflite model')
parser.add_argument(
    '-i', '--input-device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-c', '--channel', type=int, default=None,
    help='specify the channel index of input device (start from 0)')
parser.add_argument(
    '-r', '--sample-rate', type=int, default=16000,
    help='input sample rate')
parser.add_argument(
    '-b', '--block-len-ms', type=int, default=20,
    help='input block (window stride) length (ms)')
parser.add_argument(
    '-s', '--score-strategy', choices=['smoothed_confidence', 'hit_ratio'], default='hit_ratio',
    help='score strategy, choose between "smoothed_confidence" or "hit_ratio" (default)'),
parser.add_argument(
    '-t', '--threshold', type=float,
    help='score threshold, if not specified, this is automatically determined by strategy and softmax options')
parser.add_argument(
    '--hit-threshold', type=float, default=7,
    help='hit threshold')
parser.add_argument(
    '--add-softmax', action='store_true',
    help='do not add softmax layer to output')
parser.add_argument(
    '--silence-on', action='store_true',
    help='turn on silence detection')
parser.add_argument(
    '--delay-trigger', action='store_true',
    help='only trigger after uttrance end')
parser.add_argument(
    '--max-kw', type=int, default=1,
    help='max number of kw in one utterance')
parser.add_argument(
    '--measure', action='store_true',
    help='measure and report processing time')
parser.add_argument(
    '-v', '--verbose', type=int, default=1,
    help='verbose level: 0 - quiet, 1 - info, 2 - debug, 3 - verbose'
)

args = parser.parse_args(remaining)

if args.verbose > 0:
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    if args.verbose == 2:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose == 3:
        logging.getLogger().setLevel(VERBOSE)

if not args.threshold:
    if args.score_strategy == 'hit_ratio':
        threshold = 0.01
    else:
        threshold = 0.8
else:
    threshold = args.threshold


gkws = TFLiteKWS(args.model, [SILENCE, NOT_KW, 'keyword'], add_softmax=args.add_softmax, silence_off=not args.silence_on,
    score_strategy=args.score_strategy, score_threshold=threshold, hit_threshold=args.hit_threshold, 
    immediate_trigger=not args.delay_trigger, max_kw_cnt=args.max_kw)

t_ring = collections.deque(maxlen=128)

def callback(indata, frames, buf_time, status):
    global gkws, t_ring, args
    if args.measure:
        start_time = time.time()

    if status:
        logging.warning(status)
    if args.channel is not None:
        indata = indata[:, [args.channel]]

    gkws.process(indata)

    if args.measure:
        t_ring.append(time.time() - start_time)


def open_stream():
    block_shift = int(np.round(args.sample_rate * (args.block_len_ms / 1000)))
    with sd.InputStream(device=args.input_device, samplerate=args.sample_rate,
                blocksize=block_shift,
                dtype=np.float32, channels=1 if args.channel is None else None,
                callback=callback):
        print('#' * 80)
        print('Ctrl-C to exit')
        print('#' * 80)
        if args.measure:
            while True:
                time.sleep(1)
                print('Processing time: {:.2f} ms'.format( 1000 * np.average(t_ring) ), end='\r')
        else:
            threading.Event().wait()


try:
    open_stream()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))

