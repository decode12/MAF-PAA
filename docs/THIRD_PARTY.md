# Third-party assets and notices

The repository license covers original MAF release code only. Dependencies, pretrained weights, and datasets retain their upstream terms. No raw DGM4 or VisualNews content is included in this Git repository.

| Component | Official source | License or terms checked | Release handling |
|---|---|---|---|
| DGM4 dataset | [Official download](https://huggingface.co/datasets/rshaojimmy/DGM4), [project](https://github.com/rshaojimmy/MultiModal-DeepFake) | The dataset card identifies Apache-2.0; rights in underlying news material remain with its holders. | Download links, pinned hashes, and deterministic preparation code; no republished media or captions. Sanitized prediction identifiers and derived scores are included. |
| DGM4/HAMMER implementation | [Official license](https://github.com/rshaojimmy/MultiModal-DeepFake/blob/main/LICENSE) | S-Lab License 1.0 permits non-commercial redistribution subject to its conditions. This is separate from the dataset card. | The minimal MAF data preparation script does not vendor HAMMER code. Baseline source has its own terms. |
| VisualNews | [Official repository](https://github.com/FuxiaoLiu/VisualNews-Repository), [official download directory](https://www.cs.rice.edu/~vo9/visualnews/) | No explicit news-content redistribution license was found in the inspected repository/download index. | Original access instructions and a verified equivalent source-map reconstruction; no articles or images redistributed. |
| OpenAI CLIP | [Official repository](https://github.com/openai/CLIP), [MIT license](https://github.com/openai/CLIP/blob/main/LICENSE), [model card](https://huggingface.co/openai/clip-vit-base-patch16) | Upstream CLIP uses the MIT license. The model card describes intended research use and limitations; its Hugging Face metadata does not add a separate license tag. | Pinned download URLs and SHA-256 checksums. Preserve upstream notices with redistributed CLIP-derived model artifacts. |
| Hugging Face Transformers | [Repository and license](https://github.com/huggingface/transformers/blob/main/LICENSE) | Apache License 2.0. | Installed as a dependency; not vendored. |
| PyTorch and torchvision | [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE), [torchvision license](https://github.com/pytorch/vision/blob/main/LICENSE) | Upstream BSD-style licenses and bundled third-party notices. | Installed as dependencies; not vendored. |

The exact source/version and download fingerprints for required pretrained assets are in `datasets/pretrained_assets.json`. Model evaluation and retraining should use those pinned assets rather than an unrelated CLIP variant. The public safetensors conversion is explicitly distinguished from the original main-branch PyTorch serialization.

These notices document the inspected upstream sources and the scope of this release; they do not grant rights beyond the relevant upstream terms.

## OpenAI CLIP MIT notice

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
