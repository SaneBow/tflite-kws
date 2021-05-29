#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to process realtime audio with a trained DTLN model. 
This script supports ALSA audio devices. The model expects 16kHz single channel audio input/output.

Example call:
    $python rt_dtln_ns.py -i capture -o playback

Author: sanebow (sanebow@gmail.com)
Version: 23.05.2021

This code is licensed under the terms of the MIT-license.
"""


import numpy as np
import sounddevice as sd
# import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite
import argparse
import collections
import time
import daemon
import threading
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


SILENCE = 0
NOT_KW = 1

class TFLiteKWS(object):
    def __init__(self, model_path, labels, sensitivity=3.0, neg_ms=200, min_kw_ms=100, block_ms=20, tflite_threads=1):
        """ 
        TensorFlow Lite KWS model processor class

        :param model_path: path of .tflite model file
        :param labels: classification labels, exmaple: [SILENCE, NOT_KW, 'keyword1', 'keyword2']
        :param sensitivity: score threshold of kw hit for each block
        :param neg_ms: utterance end after how long of silence or not_kw (default 200 ms)
        :param min_kw_ms: minimum kw duration (default 100 ms)
        :param block_ms: block duration (default 20 ms), must match the model
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.labels = labels
        keywords = [kw for kw in self.labels if kw not in [SILENCE, NOT_KW]]
        self._utterance_blocks = 0  # total of blocks in an utterance
        self._utterance_hits = {k:0 for k in keywords}  # total kw hit count map in one utterance
        self._consecutive_neg = 0   # number of consecutive silence or not_kw 
        self.block_ms = 20
        self._neg_threshold = int(neg_ms / self.block_ms)
        self.sensitivity = sensitivity
        self.neg_ms = neg_ms
        self.min_kw_ms = min_kw_ms

        ring_len = int(4000 / block_ms)     # 4000 ms of records
        self.ring = deque([SILENCE]*ring_len, maxlen=ring_len)  # initialize with SILENCE

        # init interpreter
        self.interpreter = tflite.Interpreter(model_path=model_path, num_threads=tflite_threads)
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        if len(self.input_details) != len(self.output_details):
            raise ValueError('Number of inputs should be equal to the number of outputs'
                             'for the case of streaming model with external state')
        # init states 
        self.input_states = []
        for s in range(len(self.input_details)):
            self.input_states.append(np.zeros(self.input_details[s]['shape'], dtype=np.float32))

    def process(self, pcm, raw=False):
        """
        Process an audio block

        :param pcm: input audio block data
        :param raw: is input data in raw format? (default to False for numpy array)
        :returns: keword string when hit or None when not hit
        """
        if raw:
            indata = np.ndarray(pcm)
        else:
            indata = pcm

        # set input audio data (by default input data at index 0)
        indata = np.reshape(indata, (1, -1)).astype('float32')
        self.interpreter.set_tensor(self.input_details[0]['index'], indata)
        
        # set input states (index 1...)
        for s in range(1, len(self.input_details)):
            self.interpreter.set_tensor(self.input_details[s]['index'], self.input_states[s])

        # run calculation 
        self.interpreter.invoke()

        # get the output of the first block
        out = self.interpreter.get_tensor(self.output_details[0]['index'])

        # get output states and set it back to input states
        for s in range(1, len(self.input_details)):
            self.input_states[s] = self.interpreter.get_tensor(self.output_details[s]['index'])

        return self._any_kw_hit(out)

    def _reset_states(self):
        self._utterance_blocks = 0
        for k in self._utterance_hits:
            self._utterance_hits[k] = 0
        self._consecutive_neg = 0 

    def _debug(self, preds):
        self.logger.debug("%s\t%s\tneg=%s\ttot=%s",
            '|'.join(map(lambda s: '{:6.2f}'.format(s), preds)), 
            self._utterance_hits, self._consecutive_neg, self._utterance_blocks)

    def _any_kw_hit(self, tfout):
        ring = self.ring
        label = self.labels[np.argmax(tfout[0])]
        mask = label if label in [SILENCE, NOT_KW] else 8  # all kw -> 8
        ring.append(mask)
        score = max(tfout[0])

        if label == SILENCE:
            if self._utterance_blocks == 0:
                return None # not in utterance, skip
            self._utterance_blocks += 1
            self._consecutive_neg += 1
            if self._consecutive_neg > self._neg_threshold: # end of utterance
                trimed_utter_blocks = self._utterance_blocks - self._consecutive_neg
                utterance_ms = trimed_utter_blocks * self.block_ms
                self.logger.debug("End of utterance, duration: %s ms", utterance_ms)
                hits = self._utterance_hits
                kw = max(hits, key=hits.get)
                if utterance_ms > self.min_kw_ms and hits[kw] / trimed_utter_blocks > 0.6:
                    self.logger.debug("[!] Hit keyword: [%s]", kw)
                else:
                    kw = None
                self._reset_states()
        elif label == NOT_KW:
            if self._utterance_blocks == 0:
                return None # not in utterance, skip

        else:   # label is kw
            self._consecutive_neg = 0
            self._utterance_blocks += 1        
            if score > self.sensitivity:
                self._utterance_hits[label] += 1

        self._debug(tfout[0])   
        return kw


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
    '-w', '--window-stride-ms', type=int, default=20,
    help='window stride length (ms)')
parser.add_argument(
    '-t', '--threads', type=int, default=1,
    help='number of threads for tflite interpreters')
parser.add_argument(
    '-D', '--daemonize', action='store_true',
    help='run as a daemon')
parser.add_argument(
    '--measure', action='store_true',
    help='measure and report processing time')

args = parser.parse_args(remaining)

gkws = TFLiteKWS(args.model, [SILENCE, NOT_KW, 'raspberry'])
t_ring = collections.deque(maxlen=128)

def callback(indata, frames, buf_time, status):
    global gkws, t_ring, args
    if args.measure:
        start_time = time.time()

    if status:
        print("[Warning] status")
    if args.channel is not None:
        indata = indata[:, [args.channel]] 

    gkws.process(indata)

    if args.measure:
        t_ring.append(time.time() - start_time)
    

def open_stream():
    block_shift = int(np.round(args.sample_rate * (args.window_stride_ms / 1000)))
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
    if args.daemonize:
        with daemon.DaemonContext():
            open_stream()
    else:
        open_stream()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
    
