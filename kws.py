#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API wrapper for realtime streaming keyword spotting with trained TFLite models. 

Author: sanebow (sanebow@gmail.com)
Version: 29.05.2021

This code is licensed under the terms of the MIT-license.
"""

import numpy as np
# tflite_runtime may not work, e.g., models with flex delegates
# import tflite_runtime.interpreter as tflite 
import tensorflow.lite as tflite
import collections
import logging


SILENCE = 0
NOT_KW = 1
IS_KW = 2

class TFLiteKWS(object):
    def __init__(self, model_path, labels, win_smooth=30, win_max=100, add_softmax=True, sensitivity=0.8, tailroom_ms=100, min_kw_ms=100, block_ms=20, silence_off=False, headroom_ms=40):
        """ 
        TensorFlow Lite KWS model processor class

        :param model_path: path of .tflite model file
        :param labels: classification labels, exmaple: [SILENCE, NOT_KW, 'keyword1', 'keyword2']
        :param sensitivity: score threshold of kw hit for each block
        :param tailroom_ms: utterance end after how long of silence (default 400 ms)
        :param min_kw_ms: minimum kw duration (default 100 ms)
        :param block_ms: block duration (default 20 ms), must match the model
        :param silence_off: treat SILENCE as NOT_KW, turn silence detection off
        :param headroom_ms: required silence before start of kw (default 100 ms), only effective when SILENCE label presents and silence_off=False
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.labels = labels
        self.win_smooth = win_smooth
        self.win_max = win_max
        self.add_softmax = add_softmax
        self.sensitivity = sensitivity
        self.headroom_ms = headroom_ms
        assert tailroom_ms > 2 * block_ms, "tailroom cannot be too small"
        self.tailroom_ms = tailroom_ms
        self.min_kw_ms = min_kw_ms
        self.block_ms = 20
        if SILENCE not in self.labels:
            silence_off = True  # set silence_off flag when SILENCE not present in labels
        self.silence_off = silence_off

        self._utterance_blocks = 0  # total of blocks in an utterance
        keywords = [kw for kw in self.labels if kw not in [SILENCE, NOT_KW]]
        self._utterance_hits = {k:0 for k in keywords}  # total kw hit count map in one utterance
        self._tail_threshold = int(tailroom_ms / self.block_ms)
        self._head_threshold = int(headroom_ms / self.block_ms)

        ring_len = int(4000 / block_ms)     # 4000 ms of records
        self.ring = collections.deque([SILENCE]*ring_len, maxlen=ring_len)  # initialize with SILENCE
        self.smooth_window = collections.deque(maxlen=win_smooth)
        self.score_window = collections.deque(maxlen=win_max)

        # init interpreter
        self.interpreter = tflite.Interpreter(model_path=model_path)
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
            indata = np.array(pcm)
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
        scores = out[0]
        if self.add_softmax:
            scores = self._softmax(scores)

        # get output states and set it back to input states
        for s in range(1, len(self.input_details)):
            self.input_states[s] = self.interpreter.get_tensor(self.output_details[s]['index'])

        # kw = self._any_kw_hit(scores)
        # self._debug(scores)
        label, confidence = self._confidence_score(scores)
        self.logger.info("{}: {:.2f}".format(label, confidence))
        if label not in [SILENCE, NOT_KW] and confidence > self.sensitivity:
            kw = label
        else:
            kw = None
        return kw

    def _softmax(self, x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x

    def _confidence_score(self, scores):
        # label at current frame
        label = self.labels[np.argmax(scores)]
        kw_idx = [i for i, l in enumerate(self.labels) if l not in [SILENCE, NOT_KW]]
        # discard non-kw scores
        scores = [s for i, s in enumerate(scores) if i in kw_idx]
        self.smooth_window.append(scores)
        # smoothed posterior of output probability at current frame
        pp = np.mean(self.smooth_window, axis=0)
        self.score_window.append(pp)
        # confidence score
        confidence = np.power(
            np.prod(np.max(self.score_window, axis=0)), 1./len(scores))
        return label, confidence

    def _reset_states(self):
        self._utterance_blocks = 0
        for k in self._utterance_hits:
            self._utterance_hits[k] = 0

    def _debug(self, preds):
        self.logger.debug("%s\t%s\t%s\ttot=%s",
            '|'.join(map(lambda s: '{:6.2f}'.format(s), preds)), 
            self._utterance_hits, list(self.ring)[-10:], self._utterance_blocks)

    def _end_cond(self, v):
        if not self.silence_off:
            return v == SILENCE     # kw in sentence not accepted
        else:
            return v in [SILENCE, NOT_KW]   # accept kw in sentence

    def _any_kw_hit(self, scores):
        label = self.labels[np.argmax(scores)]        
        ilabel = label if label in [SILENCE, NOT_KW] else IS_KW  # all kw -> IS_KW
        self.ring.append(ilabel)
        lring = list(self.ring)
        score = max(scores)
        kw = None

        if self._utterance_blocks == 0:     # not in utterance state
            if label in [SILENCE, NOT_KW]:
                return None
            else:   # IS_KW
                if not self.silence_off:
                    head = lring[-1 - self._head_threshold : -1]
                    if len(head) > 0 and not all(v==SILENCE for v in head):
                        return None     # ignore kw prefixed by !kw if haven't enter utterance state

        # below only run in utterance state
        self._utterance_blocks += 1

        # label is kw
        if lring[-1] == IS_KW:   
            if score > self.sensitivity:
                self._utterance_hits[label] += 1
            return None

        # end of utterance
        if all(self._end_cond(v) for v in lring[-self._tail_threshold:]):   
            trimed_utter_blocks = self._utterance_blocks - self._tail_threshold
            utterance_ms = trimed_utter_blocks * self.block_ms
            hits = self._utterance_hits
            kw = max(hits, key=hits.get)
            hit_ratio = hits[kw] / trimed_utter_blocks
            self.logger.info("End of utterance, duration: %s ms, hit_ratio: %.2f", utterance_ms, hit_ratio)
            if utterance_ms > self.min_kw_ms and hit_ratio > 0.3:
                self.logger.info('[!] Hit keyword: "%s"', kw)
            else:
                kw = None   # not enough kw hits, discard it
            self._reset_states()

        return kw

