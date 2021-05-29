# TFLiteKWS

Keyword Spotting (KWS) API for TFLite streaming models.

## Model

You need to train your own model. One open source streaming KWS model in tensorflow is the [kws_streaming](https://github.com/google-research/google-research/tree/master/kws_streaming) by Google Research. You can refer to [g-kws](https://github.com/StuartIanNaylor/g-kws) as a nice setup guide, and use [Dataset-builder](https://github.com/StuartIanNaylor/Dataset-builder) to prepare your own dataset.

## Testing

Simply run `python3 mic_streaming.py -m /path/to/your/model.tflite`, then speak to your mic to test.

## API

Simple example with two keywords:
```python
gkws = TFLiteKWS(args.model, [SILENCE, NOT_KW, 'keyword1', 'keyword2'])
while True:
    keyword = gkws.process(get_next_audio_frame())
    if keyword:
        # following up actions
```

Please refer to `kws.py` for detailed API usage and tunning parameters.