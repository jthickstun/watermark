# Robust Distortion-free Watermarks for Language Models

Implementation of the methods described in [Robust Distortion-free Watermarks for Language Models](https://arxiv.org/abs/).

by [__Rohith Kuditipudi__](https://web.stanford.edu/~rohithk/), [__John Thickstun__](https://johnthickstun.com/), [__Tatsunori Hashimoto__](https://thashim.github.io/), and [__Percy Liang__](https://cs.stanford.edu/~pliang/).

-------------------------------------------------------------------------------------

This repository provides code that implements the watermarks described in [Robust Distortion-free Watermarks for Language Models](https://arxiv.org/abs/). See also the [blog post](https://crfm.stanford.edu/2023/07/28/watermarking.html), which includes an in-browser demo of the watermark detector.

We provide standalone Python code for generating and detecting text with a watermark, using our recommended instantiation of the watermarking strategies discussed in the paper in `generate.py` and `detect.py`. We also provide the Javascript implementation of the detector `detect.js` used for the [in-browser demo](https://crfm.stanford.edu/2023/07/28/watermarking.html).

To generate `m` tokens of text from a model (e.g., `facebook/opt-1.3b`) with watermark key `42`, run:

```
python generate.py --model facebook/opt-1.3b --m 80 --key 42 > doc.txt
```

Checking for the watermark requires a watermark key (in this case, `42`) and the model tokenizer, but crucially it does not require access to the model itself. To test for a watermark in a given text document `doc.txt`, run

```
python detect.py doc.txt --tokenizer facebook/opt-1.3b --key 42
```

Alternatively, you can use the javascript detector implemented `detect.js` which runs much faster (this is also the detector used for the [web demo](https://crfm.stanford.edu/2023/07/28/watermarking.html)).

See the [experiments](experiments) directory (forthcoming) for details on reproducing all the experimental results reported in the paper.

-------------------------------------------------------------------------------------

```bib
@article{kuditipudi2023robust,
  title={Robust Distortion-free Watermarks for Language Models},
  author={Kuditipudi, Rohith and Thickstun, John and Hashimoto, Tatsunori and Liang, Percy},
  journal={arXiv preprint},
  year={2023}
}
```
