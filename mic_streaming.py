import time
import argparse
import threading
import collections
import sounddevice as sd
import numpy as np
import logging
from kws import TFLiteKWS, SILENCE, NOT_KW

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


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
    '-m', '--model', type=str,
    help='tflite model')
parser.add_argument(
    '-i', '--input-device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-c', '--channel', type=int, default=None,
    help='use specific channel of input device')
parser.add_argument(
    '-r', '--sample-rate', type=int, default=16000,
    help='input sample rate')
parser.add_argument(
    '-b', '--block-len-ms', type=int, default=20,
    help='input block (window stride) length (ms)')
parser.add_argument(
    '--measure', action='store_true',
    help='measure and report processing time')

args = parser.parse_args(remaining)


gkws = TFLiteKWS(args.model, [SILENCE, NOT_KW, 'keyword'])

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
    
