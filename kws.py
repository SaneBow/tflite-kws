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

VERBOSE = 5
logging.addLevelName(5,"VERBOSE")


SILENCE = 0
NOT_KW = 1
IS_KW = 2

class TFLiteKWS(object):
    def __init__(self, model_path, labels, score_strategy='smoothed_confidence', add_softmax=True, score_threshold=0.8, hit_threshold=0.99, tailroom_ms=100, min_kw_ms=100, block_ms=20, silence_off=True, headroom_ms=40):
        """
        TensorFlow Lite KWS model processor class

        :param model_path: path of .tflite model file
        :param labels: classification labels, exmaple: [SILENCE, NOT_KW, 'keyword1', 'keyword2']
        :param add_softmax: whether add softmax layer to output
        :param score_strategy: can be one of the following,
            'smoothed_confidence': the score smoothing method used in Google DDN paper (default)
            'hit_ratio': count frame scores over threshold and
        :param score_threshold: threshold for kw hit ratio, or threshold for smoothed confidence (default 0.8)
        :param hit_threshold: block score threshold to trigger kw hit, only used in hit_ratio, (default 0.99)
        :param tailroom_ms: utterance end after how long of silence (default 100 ms)
        :param min_kw_ms: minimum kw duration (default 100 ms)
        :param block_ms: block duration (default 20 ms), must match the model
        :param silence_off: treat SILENCE as NOT_KW, turn silence detection off
        :param headroom_ms: required silence before start of kw (default 40 ms), only effective when SILENCE label presents and silence_off=False
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.labels = labels
        assert score_strategy in ['smoothed_confidence', 'hit_ratio'], "unknown score strategy"
        self.score_strategy = score_strategy
        self.add_softmax = add_softmax
        self.score_threshold = score_threshold
        self.hit_threshold = hit_threshold
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
        self._utterance_scores = {k:0 for k in keywords}  # total kw frame hit count map in one utterance
        self._tail_threshold = int(tailroom_ms / self.block_ms)
        self._head_threshold = int(headroom_ms / self.block_ms)
        self._already_triggered = []

        ring_len = int(4000 / block_ms)     # 4s of records
        self.label_ring = collections.deque([SILENCE]*ring_len, maxlen=ring_len)  # initialize with SILENCE
        self.smooth_window = collections.deque(maxlen=30)   # win_smooth = 30
        self.score_window = collections.deque(maxlen=100)   # win_max = 100 (follow Google DNN paper)

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
            indata = np.array(pcm) / 32768
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
        # self._debug(scores)

        # get output states and set it back to input states
        for s in range(1, len(self.input_details)):
            self.input_states[s] = self.interpreter.get_tensor(self.output_details[s]['index'])

        kw = self._any_kw_triggered(scores)

        return kw

    def _softmax(self, x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x

    def _confidence_score(self, scores):
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
        return confidence

    def _reset_states(self):
        self._utterance_blocks = 0
        for k in self._utterance_scores:
            self._utterance_scores[k] = 0
        self._already_triggered = []

    def _debug(self, preds):
        self.logger.debug("%s\t%s\t%s\ttot=%s",
            '|'.join(map(lambda s: '{:6.2f}'.format(s), preds)),
            self._utterance_scores, list(self.label_ring)[-10:], self._utterance_blocks)

    def _met_enter_cond(self):
        """ met condition to enter/keep utterance state """
        if self._utterance_blocks == 0:     # not in utterance state, need to met enter condition
            lring = list(self.label_ring)
            if not lring[-1] == IS_KW:
                return False
            else:   # IS_KW
                if not self.silence_off:
                    head = lring[-1 - self._head_threshold : -1]
                    if len(head) > 0 and not all(v==SILENCE for v in head):
                        return False     # ignore kw prefixed by !kw if haven't enter utterance state
        # can enter just stays in utterance state
        return True

    def _met_end_cond(self):
        """ met utterance end condition """
        lring = list(self.label_ring)
        if not self.silence_off:
            _cond = lambda x: x == SILENCE  # kw in sentence not accepted
        else:
            _cond = lambda x: x in [SILENCE, NOT_KW]    # accept kw in sentence
        utter_end = all(_cond(v) for v in lring[-self._tail_threshold:]) 
        return utter_end

    def _any_kw_triggered(self, scores):
        #TODO: multiple kw hits in one utterance
        label = self.labels[np.argmax(scores)]
        ilabel = label if label in [SILENCE, NOT_KW] else IS_KW  # all kw -> IS_KW
        self.label_ring.append(ilabel)
        lring = list(self.label_ring)
        if self.score_strategy == 'smoothed_confidence':
            score = self._confidence_score(scores)
            self.logger.log(VERBOSE, "{}: {:.2f}".format(label, score))
        elif self.score_strategy == 'hit_ratio':
            score = max(scores)
            self.logger.log(VERBOSE, "%s\tlabel=%s\t%s\ttot=%s",
                '|'.join(map(lambda s: '{:6.2f}'.format(s), scores)),
                label, self._utterance_scores, self._utterance_blocks)

        if not self._met_enter_cond():
            return None

        # below only run in utterance state
        self._utterance_blocks += 1
        kw = None

        # label is kw, record score
        if lring[-1] == IS_KW:
            if self.score_strategy == 'hit_ratio' and score > self.score_threshold:
                self._utterance_scores[label] += 1
            if self.score_strategy == 'smoothed_confidence':
                self._utterance_scores[label] = score   # update to latest confidence

        # early kw trigger (to support multiple kw in one utterance)
        if self.score_strategy == 'smoothed_confidence' and lring[-1] == IS_KW and score > self.score_threshold:
            if label not in self._already_triggered:
                self.logger.debug("Early trigger of kw: %s", label)
                self._already_triggered.append(label)

        # end of utterance
        if self._met_end_cond():
            utter_blocks = self._utterance_blocks - self._tail_threshold
            utterance_ms = utter_blocks * self.block_ms
            kwranks = self._utterance_scores
            kw = max(kwranks, key=kwranks.get)
            if utterance_ms < self.min_kw_ms:
                self.logger.debug("End of utterance: %s, duration: %s ms, too short!", kw, utterance_ms)
                self._reset_states()
                return None
            
            if self.score_strategy == 'hit_ratio':
                hit_ratio = kwranks[kw] / utter_blocks
                self.logger.debug("End of utterance: %s, duration: %s ms, hit_ratio: %.2f", kw, utterance_ms, hit_ratio)
                if not hit_ratio > self.score_threshold:  # hard code this, only leave sensitivity as the hyperparameter
                    kw = None   # low hit ratio, discard it
                    
            if self.score_strategy == 'smoothed_confidence':
                confidence = kwranks[kw]
                self.logger.debug("End of utterance: %s, duration: %s ms, confidence: %.2f", kw, utterance_ms, confidence)
                if len(self._already_triggered) == 1:
                    kw = self._already_triggered[0]
                elif len(self._already_triggered) > 1:
                    raise NotImplementedError("multiple kw in one utterance not implemented yet")
                else:
                    kw = None   # end after silence tail, indicating low confidence, discard it

            self._reset_states()

        if kw:
            self.logger.info('[!] Hit keyword: "%s"', kw)

        return kw

