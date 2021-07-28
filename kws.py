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

class TFLiteKWS(object):
    def __init__(self, model_path, labels, score_strategy='hit_ratio', add_softmax=False, 
        score_threshold=0.01, hit_threshold=7, tailroom_ms=100, min_kw_ms=100, block_ms=20, 
        lookahead_ms=0, silence_off=True, immediate_trigger=True, max_kw_cnt=1):
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
        :param lookahead_ms: silence lookahead window to prevent kw in sentence, 0 to turn off
        :param silence_off: treat SILENCE as NOT_KW, turn silence detection off
        :param immediate_trigger: trigger immediately once score reach threshold, don't wait for utterance end
        :param max_kw_cnt: max keyword in one utterance
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.labels = labels
        assert score_strategy in ['smoothed_confidence', 'hit_ratio'], "unknown score strategy"
        self.score_strategy = score_strategy
        self.add_softmax = add_softmax
        self.score_threshold = score_threshold
        self.hit_threshold = hit_threshold
        assert tailroom_ms > 2 * block_ms, "tailroom cannot be too small"
        self.tailroom_ms = tailroom_ms
        self.min_kw_ms = min_kw_ms
        self.block_ms = 20
        if SILENCE not in self.labels:
            silence_off = True  # set silence_off flag when SILENCE not present in labels
        self.silence_off = silence_off
        self.immediate_trigger = immediate_trigger
        self.max_kw_cnt = max_kw_cnt
        self.lookahead = int(lookahead_ms / block_ms)

        self._utterance_blocks = 0  # total of blocks in an utterance
        keywords = [kw for kw in self.labels if kw not in [SILENCE, NOT_KW]]
        self._utterance_scores = {k:0 for k in keywords} 
        self._utterance_hits = {k:0 for k in keywords}
        self._tail_threshold = int(tailroom_ms / self.block_ms)
        self._already_triggered = []

        ring_len = int(4000 / block_ms)     # 4s of records
        self.label_ring = collections.deque([SILENCE]*ring_len, maxlen=ring_len)  # initialize with SILENCE
        self.smooth_window = collections.deque([[0.0]*len(keywords)]*30, maxlen=30)   # win_smooth = 30
        self.score_window = collections.deque([[0.0]*len(keywords)]* 100, maxlen=100)   # win_max = 100 (follow Google DNN paper)

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

    def process(self, pcm, raw=False, dump_fp=None):
        """
        Process an audio block

        :param pcm: input audio block data
        :param raw: is input data in raw format? (default to False for numpy array)
        :param dump_fp: dump raw scores to a file
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

        # get output states and set it back to input states
        for s in range(1, len(self.input_details)):
            self.input_states[s] = self.interpreter.get_tensor(self.output_details[s]['index'])

        kw, info = self._any_kw_triggered(scores)

        if dump_fp:
            dump_fp.write(', '.join([str(x) for x in info['raw_score']]) + '\n')

        return kw

    def _softmax(self, x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x

    def _is_kw(self, label):
        return label not in [SILENCE, NOT_KW]

    def _last_kw_ms(self):
        lring = list(self.label_ring)
        ll = lring[-1]
        cnt = 0
        for l in lring[::-1]:
            if l == ll:
                cnt += 1
            else:
                return cnt * self.block_ms

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
            self._utterance_hits[k] = 0
        self._already_triggered = []
        self.smooth_window.extend([[0.0]*len(self._utterance_scores)]*30)
        self.score_window.extend([[0.0]*len(self._utterance_scores)]*100)

    def _met_enter_cond(self):
        """ met condition to enter/keep utterance state """
        if self._utterance_blocks == 0:     # not in utterance state, need to met enter condition
            lring = list(self.label_ring)
            if not self._is_kw(lring[-1]):
                return False
            else:   # is kw
                if not self.silence_off:
                    head = lring[-1 - self.lookahead : -1]
                    if len(head) > 0 and len([v for v in head if v==SILENCE]) / self.lookahead < 0.5:
                        return False     # ignore kw prefixed by !kw if haven't enter utterance state
        # can enter just stays in utterance state
        return True

    def _met_end_cond(self):
        """ met utterance end condition, return (is_end, end_status) """
        lring = list(self.label_ring)
        # no kw after a certain time
        if not self.silence_off:
            _cond = lambda x: x == SILENCE  # kw in sentence not accepted
        else:
            _cond = lambda x: x in [SILENCE, NOT_KW]    # accept kw in sentence
        utter_end = all(_cond(v) for v in lring[-self._tail_threshold:]) 
        return utter_end

    def _any_kw_triggered(self, scores):
        #TODO: multiple kw hits in one utterance
        ilabel = np.argmax(scores)
        label = self.labels[ilabel]
        self.label_ring.append(ilabel)
        if self.score_strategy == 'smoothed_confidence':
            score = self._confidence_score(scores)
            self.logger.log(VERBOSE, "{}: {:.2f}".format(label, score))
        elif self.score_strategy == 'hit_ratio':
            score = max(scores)
            self.logger.log(VERBOSE, "%s\tlabel=%s\t%s\ttot=%s",
                '|'.join(map(lambda s: '{:6.2f}'.format(s), scores)),
                label, self._utterance_scores, self._utterance_blocks)
        
        info = {'label': label, 'raw_score': scores}

        if not self._met_enter_cond():
            return [], info

        # below only run in utterance state
        self._utterance_blocks += 1
        info['utblk'] = self._utterance_blocks
        kw = []

        # label is kw, record score
        if self._is_kw(label):
            # calculate score
            if self.score_strategy == 'hit_ratio':
                if score > self.hit_threshold:
                    self._utterance_hits[label] += 1
                nmin = self.min_kw_ms // self.block_ms
                self._utterance_scores[label] = self._utterance_hits[label] / max(nmin, self._last_kw_ms() // self.block_ms)
            if self.score_strategy == 'smoothed_confidence':
                self._utterance_scores[label] = score   # update to latest confidence

            calc_score = self._utterance_scores[label]
            kw_ms = self._last_kw_ms()
            info['score'] = calc_score
            info['kw_ms'] = kw_ms

            # kw trigger before utterance end
            if calc_score > self.score_threshold:
                if kw_ms > self.min_kw_ms:
                    if label not in self._already_triggered:
                        self.logger.debug("Early trigger of kw: %s, dur: %s, score: %s", label, kw_ms, calc_score)
                        self._already_triggered.append(label)
        
                        if self.immediate_trigger and len(self._already_triggered) == self.max_kw_cnt:
                            kw = self._already_triggered
                            self.logger.debug("[!] Return keywords before utterance end: %s", kw)
                            return kw, info

        # end of utterance
        if self._met_end_cond():
            utter_ms = self._utterance_blocks * self.block_ms - self.tailroom_ms
            if len(self._already_triggered) == 0 or \
                (self.immediate_trigger and len(self._already_triggered) == self.max_kw_cnt):
                kw = []
            else:
                kw = self._already_triggered
            self.logger.debug("End of utterance: %s, dur: %s, scores: %s", kw, utter_ms, self._utterance_scores)

            self._reset_states()

        if len(kw):
            self.logger.info('[!] Return keywords after utterance end: "%s"', kw)

        return kw, info

