# Tacotron2 with Japanese hougen intonation (Kumamoto, Osaka)
It is a TTS model synthesis japanese audio in three intonation(Neutral, Kumamoto and Osaka) based on [Tacotron2](https://github.com/NVIDIA/tacotron2).

This model extracts intonation embedding(64dim)/gender embedding(64dim) and concat the 128dim embeddings with text embeddings. The concated embeddings are used as tacotron2 encoder input data. By controlling the intonation embeddings that are extracted from mel-spectrograms having intonation feature, I tried to traind the intonation cotrollable TTS model(Japanese).

The concept is from [Controllable cross-speaker emotion transfer for end-to-end speech synthesis](https://arxiv.org/abs/2109.06733), but the structures are very different.
Dut to the limitation of computing resources(trained using Colab Pro) and datasets, the model size is minimized and the performance is not that good. Just for experimental test perpose.

I use two japanese corpus, [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) and [JMD](https://sites.google.com/site/shinnosuketakamichi/research-topics/jmd_corpus).

## Samples
- text : 1週間して、そのニュースは本当になった。
- [sample folder](https://github.com/Nhandsome/tacotron2_hougen/tree/main/sample)
- These samples are just made by the griffin-lim algorithm, so the qualities are bad.
- The quality could be improved by training Vocoder.(just by using [pre-trained waveglow model](https://drive.google.com/file/d/1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF/view)), It is getting better.)

## How to use
1. Clone this repo.
2. Download [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) and [JMD](https://sites.google.com/site/shinnosuketakamichi/research-topics/jmd_corpus) corpus.
3. Extract downloaded corpus in downloads directory.
4. Follow steps of this [colab notebook](https://colab.research.google.com/drive/1Nbnam7jG4OiQuIoKfJCo40XiA8KCHnwh?usp=sharing).
