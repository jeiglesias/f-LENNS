# Third-party notices

f-LENNS incorporates code from the following projects. Their original license
terms are reproduced below and continue to apply to the corresponding files.

## NAFNet (`flenns/NAFNet.py`)

`flenns/NAFNet.py` is a volumetric (3D) adaptation of the NAFNet architecture
from [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet),
released under the MIT License.

> MIT License
>
> Copyright (c) 2022 megvii-model
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

Please cite the original work if you use this architecture:

```bibtex
@inproceedings{chen2022simple,
  title     = {Simple Baselines for Image Restoration},
  author    = {Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  booktitle = {European Conference on Computer Vision (ECCV)},
  pages     = {17--33},
  year      = {2022},
  publisher = {Springer}
}
```

## lab2im (`flenns/utils.py`)

Several I/O and resampling helpers in `flenns/utils.py` are taken from the
`lab2im` package by Benjamin Billot
([BBillot/lab2im](https://github.com/BBillot/lab2im)), released under the
Apache License 2.0.
