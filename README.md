<p align="center" style="margin-bottom: 0">
  <img src="./assets/svg/logo-text.svg#gh-light-mode-only" alt="Texo Logo" width="300"/>
  <img src="./assets/svg/logo-text-dark.svg#gh-dark-mode-only" alt="Texo Logo" width="300"/>
</p>
<p align="center" style="font-size: 0.9em">
Texo is pronounced as /ˈtɛːkoʊ/
</p>
<p align="center">
  A minimalist free and open-source SOTA LaTeX OCR model which contains only 20M parameters.
</p>

## Features
- Free and open-source.
- Fast and lightweight inference.
- Trainable on consumer's-level GPU.
- Well organized code as a tutorial.
- Running in browser!

## Configure environment
```sh
git clone https://github.com/alephpi/TexoWebUIBasedGradio
uv sync
```
> For those who don't use uv, it worths to try it. For those who insist not to use, I guess you know how to adapt.

## Download model
```sh
# model only
python scripts/python/hf_hub.py pull
```
```sh
# for those who want to train from useful checkpoints
python scripts/python/hf_hub.py pull --with_useful_ckpts
```

## Inference
Check [`demo.ipynb`](./demo.ipynb)

## License
[AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html)

Copyright (C) 2025-present Sicheng Mao <maosicheng98@gmail.com>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=alephpi/Texo&type=date&legend=top-left)](https://www.star-history.com/#alephpi/Texo&type=date&legend=top-left)