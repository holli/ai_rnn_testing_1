# ai_rnn_testing_1
Dump of various machine learning practices and stuff, mostly rnn
([./ai_rnn_testing_1/tree/master/char_rnn_fastai](https://github.com/holli/ai_rnn_testing_1/tree/master/char_rnn_fastai))
and/or related to one kaggle text competition
([kaggle.com/c/text-normalization-challenge-english-language](kaggle.com/c/text-normalization-challenge-english-language))

### Notes / Keypoints

- Test minor codes separately, so many times I’d let it run overnight to notice later that model was using only single characteres instead of letters
  - in the otherhand, nice to see how well character based rnn:s are working
- Attention models in encoder-decoder are simple to understand and good to use
  - although ensure with graphs that they are working as supposed
  - parameters related to attention might need a bit different learning rate than the rest of the model
- With pytorch its easy to test custom settings in training (e.g. teacher forcing)
  - but you easily end up doing some custom cpu stuff which results better learning per iteration but very slow iterations
  - so start with simple model, benchmark it
  - always run "watch -n 1 nvidia-smi” and “top” on console to see gpu usage
- Batch size matters, combined with learning rate
- Variying data causes problems
  - training with 5 length data and suddenly having 200 length word will easily cause gradient explosion
- Log everything
  - nice to see what were the couple last samples before gradient explosion (variying data)


