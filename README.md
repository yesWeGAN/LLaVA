# coucap: LLaVA for Fashion Product Captioning

**Goal**: <br>Generation of precise yet upbeat product descriptions from images. <br>
Fork from haotian-liu/LLaVA, All credits for this go to the original authors (below).<br>
Low-rank adaptation fine-tuning of multi-modal Large Language Model LLaVA. Product-images scraped from web using own scraper, aimed at 2M+ samples.<br><br>
Training and inference on limited hardware (3090/24G).


This repo contains the logs during development. Find the train runs at: <br>
[[wandb-training-page](https://wandb.ai/yeswegan/LLaVA-LoRA?workspace=user-yeswegan)]


version | LLaVA Base| Dataset size | Train method | global batch size | BLEU-1 score
:-------: |:-----------: | :---------:| :---------:| :---------:| :---------:
v1 | v1.5-7b | 5k | LoRA | 64| N/A
v2 | v1.5-7b | 14k | LoRA | 64| N/A
v3 | v1.5-7b | 70k | LoRA | 64| N/A
v4 | v1.5-7b | 125k | LoRA | 128| 0.09
v4.5 | v1.5-7b | 32k crop | LoRA | 128| 0.08
v4.6 | v1.5-7b | 60k crop | 4-bit qLoRA | 128| 0.13
v4.7 | v1.5-7b | 165k crop | 4-bit qLoRA | 128| tbd
v5 | 13b ? | ? | qLoRA | ?| ?

<br>

### Evaluation

Crafted a custom BLEU-1-score, **fashion-BLEU1**, ignoring filler words like (with, are, is, and, you, ..). <br>Very conservative metric, to capture performance on naming item properties (cut, length, details, appeal). BLEU-1 is not an ideal metric, but quick.<br><br>
Score is calculated as the mean BLEU-1 per sample, as each description needs to fit the specific item. Further evaluations/details see [Eval Details](###Detailed-Evaluation-Results). <br><br>
Some item-types / classes (accessoire, lingerie, sunglasses, underwear) are underrepresented and ignored in the following table (Top-5 results). Increasing sample-size in validation beyond 100 per class does not significantly influence performance.

Model| native-BLEU-1 | fashion-BLEU-1 | samples for validation
:-------: |:-----------: | :---------:| :---------:
[v4.6](###v4.6) | 0.13 | 0.10 | 768 
[v4.5](###v4.5) | 0.08 | 0.03 | 354
[v4](###v4) | 0.09 | 0.04 | 490


## Model Version Changelog

### *v5* - Roadmap to v5, tbds
- goals:
    - ready to be published (servable)
    - reduced hallucinations (item/vendor names, back descriptions)
    - type inference
    - maybe adding keywords into prompt
    - gender parity in training data, 250k train samples
- necessary changes in data
    - filter vendor names from descriptions
    - potentially filter item names (replace with "this ITEMCLASS")
    - improve / finegrain type inference, add more classes
    - image-classification: front, back, side, detail
    - image-classification: unisex, kids, women,  men
    - 
    - optional: 
        - dataset viewer in gradio to catch wrong cropping, labels, etc! 
        - MongoDB
        - store description as prompt
        - assemble RLHF dataset with preferred options
- necessary changes in model / pipeline
    - inference pipeline with type inference and cropping before forward()
    - multi-image prompt or image tiling for back
    - deployment options in huggingface
    - optional:
        - batched inference for faster evaluation / serving


### *v4.7*
- **165k samples of cropped images, 4-bit qLORA (resume training from v4.6)**
- why does loss decrease slower with qlora? (note qlora train run in 3 installments, slightly overlapping due to poor checkpoint choice)

<img src="images/lora_vs_qlora.png" width="70%">


### *v4.6*
- **60k samples of cropped images, qLoRA-train (4-bit)**
- slightly quicker training by fitting optimizer onto GPU memory (ZERO3-offload not compatible with --bits 4)
- step batch size ect. same as with regular LoRA, maybe the official qlora script offers more features
- overall performance improved significantly in terms of BLEU, even though train loss is higher

### *v4.5*
- **32k samples of cropped images, LoRA-train (no qlora)**
- improved dataset regex filters
- wrap around predict.py to predict samples
- merge base model with LoRA to decrease model setup time
- adapt BLEU score to ignore filler-words (with, are, is, and, you, ..)
- calculate adapted BLEU score for each category
- use segmentations to extract cropped image version on ROI

### *v4* - 20.01.24
**125k samples of full images, LoRA-train (no qlora), improved dataset quality**
- finished after 22.6 h of training.
- BLEU-1 score 0.19 on 10 samples, 3 candidate generations each, top-p 0.7, temp 0.2, max 512 tokens.
- had to increase swap file size to avoid running out of RAM during step-n-saves. Stores all of the optimizers, requiring 20G each.
- improved regex filtering, added new fragmentation rules
- dataset increased to 125k samples
- changed global batch size to 128 (was 64)

- #### **Next steps:**
- automated evaluation: probably best to have some kind of BLEU with weighted words. Would require a mapping of keywords/features that mean the same thing.
- def need to segment images and crop to the described item. Fine-grained details otherwise not perceivable in 336px vision tower.


### *v3* - 18.01.2024

- train run on newly filtered 70k dataset (train-time 13h) shows greatly improved generalization compared to v2. <br>
- Inference without product names seems to generate more accurate, diverse results.
- with max_length 2048, observed frequent allocator cache flushes. Reducing to 1024 caused some samples to overflow max_length

### *dataset quality improvements* - 16.01.24

- next LoRA training will run on 70k samples, greatly improved description filtering in dataset. 
Added filters for many size-, washing-, care-, design-, material-, model-related information.
- Furthermore changed up image map to include the store-first image instead of the glob-first.
- Set-up backside-related snippet filtering to maybe enable backside descriptions in next step.


### *v2* -  15.01.24

- LoRA training, 1 epoch on 14k samples, still reveals some amount of overfitting: 
many elements of the description are repeated, but this time not perfectly.
also some genuinely new descriptions come in.
- some categories are mapped wrong.
- some captions are too short.
- still too many useless snippets in dataset like please note: colors may differ, fabric, and model sizes.
**re-iterate over caption filtering process?**
- or download FACAD https://drive.google.com/drive/folders/1cgdHt8AlBukmPhuSzUTPszYPXAYmg6gy
    note 31.01.24: FACAD is useless, 256x256 images and no mapping image - caption. also no commercial use.
- How to solve the image-picking problem. Maybe use image-number 1 from JSON?



### *v1*  -  14.01.24
- LoRA training (5 epochs on 5k samples) shows heavy overfitting: description was learnt by heart.
- next train with only 1 epoch.
- improve caption filtering.
- Multi-image input is supported technically, but apparently results are poor. 
Maybe this means I have to split the captioning tasks into portions about front, back and side. 





### Detailed Evaluation Results

Crafted a custom BLEU-1-score, **fashion-BLEU1**, ignoring filler words like (with, are, is, and, you, ..). <br>Very conservative metric, to capture performance on naming item properties (cut, length, details, appeal). <br><br>
Score is calculated once on the whole corpus of all predictions in a category, and as a mean per-sample (**?-BLEU-1-mean**), details see below. <br><br>



**v4.6** - 60k cropped product images, 4-bit qLoRA training, train/val sampling (0.03), top_p 0.7, temp 0.3
Category| native-BLEU-1 | fashion-BLEU-1 | native-BLEU-1-mean | fashion-BLEU-1-mean | samples for validation
:-------: |:-----------: | :---------:| :---------:| :---------:| :---------:
dress | 0.44 | 0.445 | 0.137 | 0.109 | 200
top | 0.467 | 0.383 | 0.125 | 0.086 | 200
pants | 0.519 | 0.45 | 0.107 | 0.066 | 200
skirt | 0.435 | 0.364 | 0.155 | 0.117 | 68
accessoire | 0.399 | 0.27 | 0.112 | 0.066 | 48
one-piece | 0.376 | 0.401 | 0.133 | 0.098 | 100
lingerie | 0.401 | 0.353 | 0.182 | 0.118 | 26
hat | 0.393 | 0.271 | 0.134 | 0.096 | 63
swimwear | 0.322 | 0.235 | 0.084 | 0.064 | 87
footwear | 0.254 | 0.286 | 0.094 | 0.06 | 38
sunglasses | 0.115 | 0.091 | 0.131 | 0.103 | 3
underwear | 0.374 | 0.306 | 0.196 | 0.129 | 11



**v4.5** - 32k cropped product images, train/val sampling (0.03), LoRA, top_p 0.7, temp 0.2
Category| native-BLEU-1 | fashion-BLEU-1 | native-BLEU-1-mean | fashion-BLEU-1-mean | samples for validation
:-------: |:-----------: | :---------:| :---------:| :---------:| :---------:
dress | 0.326 | 0.217 | 0.07 | 0.022 | 98
top | 0.398 | 0.312 | 0.105 | 0.071 | 98
pants | 0.376 | 0.288 | 0.069 | 0.025 | 98
skirt | 0.292 | 0.158 | 0.074 | 0.028 | 18
accessoire | 0.246 | 0.122 | 0.069 | 0.014 | 14
one-piece | 0.334 | 0.215 | 0.078 | 0.015 | 33
lingerie | 0.223 | 0.069 | 0.107 | 0.031 | 5
hat | 0.28 | 0.144 | 0.064 | 0.015 | 19
swimwear | 0.057 | 0.029 | 0.05 | 0.026 | 42
footwear | 0.277 | 0.095 | 0.056 | 0.006 | 11
sunglasses | 0 | 0 | nan | nan | 0
underwear | 0.153 | 0.013 | 0.053 | 0.003 | 3

**v4**  - 125 k non-cropped images, no train/val sampling (low validation diversity!), LoRA, top_p 0.7, temp 0.2
Category| native-BLEU-1 | fashion-BLEU-1 | native-BLEU-1-mean | fashion-BLEU-1-mean | samples for validation
:-------: |:-----------: | :---------:| :---------:| :---------:| :---------:
dress | 0.268 | 0.161 | 0.081 | 0.023 | 98
top | 0.387 | 0.283 | 0.079 | 0.037 | 98
pants | 0.353 | 0.285 | 0.104 | 0.079 | 98
skirt | 0.352 | 0.231 | 0.098 | 0.037 | 98
accessoire | 0.327 | 0.178 | 0.072 | 0.017 | 98
one-piece | 0.336 | 0.22 | 0.087 | 0.026 | 98
lingerie | 0.27 | 0.17 | 0.087 | 0.029 | 56
hat | 0.295 | 0.175 | 0.07 | 0.018 | 98
swimwear | 0.146 | 0.041 | 0.081 | 0.02 | 11
footwear | 0.26 | 0.123 | 0.078 | 0.015 | 98
item | 0.3 | 0.157 | 0.077 | 0.017 | 34
sunglasses | 0.177 | 0.05 | 0.067 | 0.013 | 29
underwear | 0 | 0 | nan | nan | 0


# 🌋 LLaVA: Large Language and Vision Assistant

*Visual instruction tuning towards large language and vision models with GPT-4 level capabilities.*

[[Project Page](https://llava-vl.github.io/)] [[Demo](https://llava.hliu.cc/)]  [[Data](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)] [[Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)]

🤝Community Contributions: [[llama.cpp](https://github.com/ggerganov/llama.cpp/pull/3436)] [[Colab](https://github.com/camenduru/LLaVA-colab)] [[🤗Space](https://huggingface.co/spaces/badayvedat/LLaVA)] [[Replicate](https://replicate.com/yorickvp/llava-13b)] [[AutoGen](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_lmm_llava.ipynb)]  [[BakLLaVA](https://github.com/SkunkworksAI/BakLLaVA)]

**Improved Baselines with Visual Instruction Tuning** [[Paper](https://arxiv.org/abs/2310.03744)] <br>
[Haotian Liu](https://hliu.cc), [Chunyuan Li](https://chunyuan.li/), [Yuheng Li](https://yuheng-li.github.io/), [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/)

**Visual Instruction Tuning** (NeurIPS 2023, **Oral**) [[Paper](https://arxiv.org/abs/2304.08485)]<br>
[Haotian Liu*](https://hliu.cc), [Chunyuan Li*](https://chunyuan.li/), [Qingyang Wu](https://scholar.google.ca/citations?user=HDiw-TsAAAAJ&hl=en/), [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/) (*Equal Contribution)

<!--p align="center">
    <a href="https://llava.hliu.cc/"><img src="images/llava_logo.png" width="50%"></a> <br>
    Generated by <a href="https://gligen.github.io/">GLIGEN</a> via "a cute lava llama with glasses" and box prompt
</p-->


## Release
- [11/10] [LLaVA-Plus](https://llava-vl.github.io/llava-plus/) is released: Learning to Use Tools for Creating Multimodal Agents, with LLaVA-Plus (LLaVA that Plug and Learn to Use Skills). [[Project Page](https://llava-vl.github.io/llava-plus/)] [[Demo](https://llavaplus.ngrok.io/)] [[Code](https://github.com/LLaVA-VL/LLaVA-Plus-Codebase)] [[Paper](https://arxiv.org/abs/2311.05437)]
- [11/2] [LLaVA-Interactive](https://llava-vl.github.io/llava-interactive/) is released: Experience the future of human-AI multimodal interaction with an all-in-one demo for Image Chat, Segmentation, Generation and Editing. [[Project Page](https://llava-vl.github.io/llava-interactive/)] [[Demo](https://llavainteractive.ngrok.io/)] [[Code](https://github.com/LLaVA-VL/LLaVA-Interactive-Demo)] [[Paper](https://arxiv.org/abs/2311.00571)]
- [10/26] 🔥 LLaVA-1.5 with LoRA achieves comparable performance as full-model finetuning, with a reduced GPU RAM requirement ([ckpts](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md#llava-v15), [script](https://github.com/haotian-liu/LLaVA#train)). We also provide a [doc](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md) on how to finetune LLaVA-1.5 on your own dataset with LoRA.
- [10/12] Check out the Korean LLaVA (Ko-LLaVA), created by ETRI, who has generously supported our research! [[🤗 Demo](https://huggingface.co/spaces/etri-vilab/Ko-LLaVA)]
- [10/5] 🔥 LLaVA-1.5 is out! Achieving SoTA on 11 benchmarks, with just simple modifications to the original LLaVA, utilizes all public data, completes training in ~1 day on a single 8-A100 node, and surpasses methods like Qwen-VL-Chat that use billion-scale data. Check out the [technical report](https://arxiv.org/abs/2310.03744), and explore the [demo](https://llava.hliu.cc/)! Models are available in [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md). The training data and scripts of LLaVA-1.5 are released [here](https://github.com/haotian-liu/LLaVA#train), and evaluation scripts are released [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)!
- [9/26] LLaVA is improved with reinforcement learning from human feedback (RLHF) to improve fact grounding and reduce hallucination. Check out the new SFT and RLHF checkpoints at project [[LLavA-RLHF]](https://llava-rlhf.github.io/)
- [9/22] [LLaVA](https://arxiv.org/abs/2304.08485) is accepted by NeurIPS 2023 as **oral presentation**, and [LLaVA-Med](https://arxiv.org/abs/2306.00890) is accepted by NeurIPS 2023 Datasets and Benchmarks Track as **spotlight presentation**.

<details>
<summary>More</summary>

- [11/6] Support **Intel** dGPU and CPU platforms. [More details here.](https://github.com/haotian-liu/LLaVA/tree/intel/docs/intel)
- [10/12] LLaVA is now supported in [llama.cpp](https://github.com/ggerganov/llama.cpp/pull/3436) with 4-bit / 5-bit quantization support!
- [10/11] The training data and scripts of LLaVA-1.5 are released [here](https://github.com/haotian-liu/LLaVA#train), and evaluation scripts are released [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)!
- [10/10] [Roboflow Deep Dive](https://blog.roboflow.com/first-impressions-with-llava-1-5/): First Impressions with LLaVA-1.5.
- [9/20] We summarize our empirical study of training 33B and 65B LLaVA models in a [note](https://arxiv.org/abs/2309.09958). Further, if you are interested in the comprehensive review, evolution and trend of multimodal foundation models, please check out our recent survey paper [``Multimodal Foundation Models: From Specialists to General-Purpose Assistants''.](https://arxiv.org/abs/2309.10020)
<p align="center">
  <img src="https://github.com/Computer-Vision-in-the-Wild/CVinW_Readings/blob/main/images/mfm_evolution.jpeg?raw=true" width=50%/>
</p>

- [7/19] 🔥 We release a major upgrade, including support for LLaMA-2, LoRA training, 4-/8-bit inference, higher resolution (336x336), and a lot more. We release [LLaVA Bench](https://github.com/haotian-liu/LLaVA/blob/main/docs/LLaVA_Bench.md) for benchmarking open-ended visual chat with results from Bard and Bing-Chat. We also support and verify training with RTX 3090 and RTX A6000. Check out [LLaVA-from-LLaMA-2](https://github.com/haotian-liu/LLaVA/blob/main/docs/LLaVA_from_LLaMA2.md), and our [model zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)!
- [6/26] [CVPR 2023 Tutorial](https://vlp-tutorial.github.io/) on **Large Multimodal Models: Towards Building and Surpassing Multimodal GPT-4**!  Please check out [[Slides](https://datarelease.blob.core.windows.net/tutorial/vision_foundation_models_2023/slides/Chunyuan_cvpr2023_tutorial_lmm.pdf)] [[Notes](https://arxiv.org/abs/2306.14895)] [[YouTube](https://youtu.be/mkI7EPD1vp8)] [[Bilibli](https://www.bilibili.com/video/BV1Ng4y1T7v3/)].
- [6/11] We released the preview for the most requested feature: DeepSpeed and LoRA support!  Please see documentations [here](./docs/LoRA.md).
- [6/1] We released **LLaVA-Med: Large Language and Vision Assistant for Biomedicine**, a step towards building biomedical domain large language and vision models with GPT-4 level capabilities.  Checkout the [paper](https://arxiv.org/abs/2306.00890) and [page](https://github.com/microsoft/LLaVA-Med).
- [5/6] We are releasing [LLaVA-Lighting-MPT-7B-preview](https://huggingface.co/liuhaotian/LLaVA-Lightning-MPT-7B-preview), based on MPT-7B-Chat!  See [here](#LLaVA-MPT-7b) for more details.
- [5/2] 🔥 We are releasing LLaVA-Lighting!  Train a lite, multimodal GPT-4 with just $40 in 3 hours!  See [here](#train-llava-lightning) for more details.
- [4/27] Thanks to the community effort, LLaVA-13B with 4-bit quantization allows you to run on a GPU with as few as 12GB VRAM!  Try it out [here](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/llava).
- [4/17] 🔥 We released **LLaVA: Large Language and Vision Assistant**. We propose visual instruction tuning, towards building large language and vision models with GPT-4 level capabilities.  Checkout the [paper](https://arxiv.org/abs/2304.08485) and [demo](https://llava.hliu.cc/).

</details>

<!-- <a href="https://llava.hliu.cc/"><img src="assets/demo.gif" width="70%"></a> -->

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
**Usage and License Notices**: This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses, including but not limited to the [OpenAI Terms of Use](https://openai.com/policies/terms-of-use) for the dataset and the specific licenses for base language models for checkpoints trained using the dataset (e.g. [Llama community license](https://ai.meta.com/llama/license/) for LLaMA-2 and Vicuna-v1.5). This project does not impose any additional constraints beyond those stipulated in the original licenses. Furthermore, users are reminded to ensure that their use of the dataset and checkpoints is in compliance with all applicable laws and regulations.


## Contents
- [Install](#install)
- [LLaVA Weights](#llava-weights)
- [Demo](#Demo)
- [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)
- [Dataset](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

If you are not using Linux, do *NOT* proceed, see instructions for [macOS](https://github.com/haotian-liu/LLaVA/blob/main/docs/macOS.md) and [Windows](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md).

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Upgrade to latest code base

```Shell
git pull
pip install -e .
```

### Quick Start With HuggingFace

<details>
<summary>Example Code</summary>

```Python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
```

Check out the details wth the `load_pretrained_model` function in `llava/model/builder.py`.

You can also use the `eval_model` function in `llava/eval/run_llava.py` to get the output easily. By doing so, you can use this code on Colab directly after downloading this repository.

``` python
model_path = "liuhaotian/llava-v1.5-7b"
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)
```
</details>

## LLaVA Weights
Please check out our [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) for all public LLaVA checkpoints, and the instructions of how to use the weights.

## Demo

To run our demo, you need to prepare LLaVA checkpoints locally.  Please follow the instructions [here](#llava-weights) to download the checkpoints.

### Gradio Web UI

To launch a Gradio demo locally, please run the following commands one by one. If you plan to launch multiple model workers to compare between different checkpoints, you only need to launch the controller and the web server *ONCE*.

```mermaid
flowchart BT
    %% Declare Nodes
    gws("Gradio (UI Server)")
    c("Controller (API Server):<br/>PORT: 10000")
    mw7b("Model Worker:<br/>llava-v1.5-7b<br/>PORT: 40000")
    mw13b("Model Worker:<br/>llava-v1.5-13b<br/>PORT: 40001")

    %% Declare Styles
    classDef data fill:#3af,stroke:#48a,stroke-width:2px,color:#444
    classDef success fill:#8f8,stroke:#0a0,stroke-width:2px,color:#444
    classDef failure fill:#f88,stroke:#f00,stroke-width:2px,color:#444

    %% Assign Styles
    class id,od data;
    class cimg,cs_s,scsim_s success;
    class ncimg,cs_f,scsim_f failure;

    subgraph Demo Connections
        direction BT
        c<-->gws
        
        mw7b<-->c
        mw13b<-->c
    end
```

#### Launch a controller
```Shell
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

#### Launch a gradio web server.
```Shell
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
```
You just launched the Gradio web interface. Now, you can open the web interface with the URL printed on the screen. You may notice that there is no model in the model list. Do not worry, as we have not launched any model worker yet. It will be automatically updated when you launch a model worker.

#### Launch a model worker

This is the actual *worker* that performs the inference on the GPU.  Each worker is responsible for a single model specified in `--model-path`.

```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-13b
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...".  Now, refresh your Gradio web UI, and you will see the model you just launched in the model list.

You can launch as many workers as you want, and compare between different model checkpoints in the same Gradio interface. Please keep the `--controller` the same, and modify the `--port` and `--worker` to a different port number for each worker.
```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port <different from 40000, say 40001> --worker http://localhost:<change accordingly, i.e. 40001> --model-path <ckpt2>
```

If you are using an Apple device with an M1 or M2 chip, you can specify the mps device by using the `--device` flag: `--device mps`.

#### Launch a model worker (Multiple GPUs, when GPU VRAM <= 24GB)

If the VRAM of your GPU is less than 24GB (e.g., RTX 3090, RTX 4090, etc.), you may try running it with multiple GPUs. Our latest code base will automatically try to use multiple GPUs if you have more than one GPU. You can specify which GPUs to use with `CUDA_VISIBLE_DEVICES`. Below is an example of running with the first two GPUs.

```Shell
CUDA_VISIBLE_DEVICES=0,1 python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-13b
```

#### Launch a model worker (4-bit, 8-bit inference, quantized)

You can launch the model worker with quantized bits (4-bit, 8-bit), which allows you to run the inference with reduced GPU memory footprint, potentially allowing you to run on a GPU with as few as 12GB VRAM. Note that inference with quantized bits may not be as accurate as the full-precision model. Simply append `--load-4bit` or `--load-8bit` to the **model worker** command that you are executing. Below is an example of running with 4-bit quantization.

```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-13b --load-4bit
```

#### Launch a model worker (LoRA weights, unmerged)

You can launch the model worker with LoRA weights, without merging them with the base checkpoint, to save disk space. There will be additional loading time, while the inference speed is the same as the merged checkpoints. Unmerged LoRA checkpoints do not have `lora-merge` in the model name, and are usually much smaller (less than 1GB) than the merged checkpoints (13G for 7B, and 25G for 13B).

To load unmerged LoRA weights, you simply need to pass an additional argument `--model-base`, which is the base LLM that is used to train the LoRA weights. You can check the base LLM of each LoRA weights in the [model zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md).

```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1-0719-336px-lora-vicuna-13b-v1.3 --model-base lmsys/vicuna-13b-v1.3
```

### CLI Inference

Chat about images using LLaVA without the need of Gradio interface. It also supports multiple GPUs, 4-bit and 8-bit quantized inference. With 4-bit quantization, for our LLaVA-1.5-7B, it uses less than 8GB VRAM on a single GPU.

```Shell
python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --load-4bit
```

<img src="images/demo_cli.gif" width="70%">

## Train

*Below is the latest training configuration for LLaVA v1.5. For legacy models, please refer to README of [this](https://github.com/haotian-liu/LLaVA/tree/v1.0.1) version for now. We'll add them in a separate doc later.*

LLaVA training consists of two stages: (1) feature alignment stage: use our 558K subset of the LAION-CC-SBU dataset to connect a *frozen pretrained* vision encoder to a *frozen LLM*; (2) visual instruction tuning stage: use 150K GPT-generated multimodal instruction-following data, plus around 515K VQA data from academic-oriented tasks, to teach the model to follow multimodal instructions.

LLaVA is trained on 8 A100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

### Hyperparameters
We use a similar set of hyperparameters as Vicuna in finetuning.  Both hyperparameters used in pretraining and finetuning are provided below.

1. Pretraining

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-v1.5-13B | 256 | 1e-3 | 1 | 2048 | 0 |

2. Finetuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-v1.5-13B | 128 | 2e-5 | 1 | 2048 | 0 |

### Download Vicuna checkpoints (automatically)

Our base model Vicuna v1.5, which is an instruction-tuned chatbot, will be downloaded automatically when you run our provided training scripts. No action is needed.

### Pretrain (feature alignment)

Please download the 558K subset of the LAION-CC-SBU dataset with BLIP captions we use in the paper [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).

Pretrain takes around 5.5 hours for LLaVA-v1.5-13B on 8x A100 (80G), due to the increased resolution to 336px. It takes around 3.5 hours for LLaVA-v1.5-7B.

Training script with DeepSpeed ZeRO-2: [`pretrain.sh`](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/pretrain.sh).

- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.

<details>
<summary>Pretrain takes around 20 hours for LLaVA-7B on 8x V100 (32G)</summary>

 We provide training script with DeepSpeed [here](https://github.com/haotian-liu/LLaVA/blob/main/scripts/pretrain_xformers.sh).
Tips:
- If you are using V100 which is not supported by FlashAttention, you can use the [memory-efficient attention](https://arxiv.org/abs/2112.05682) implemented in [xFormers](https://github.com/facebookresearch/xformers). Install xformers and replace `llava/train/train_mem.py` above with [llava/train/train_xformers.py](llava/train/train_xformers.py).
</details>

### Visual Instruction Tuning

1. Prepare data

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in `./playground/data`,

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

2. Start training!

You may download our pretrained projectors in [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md). It is not recommended to use legacy projectors, as they may be trained with a different version of the codebase, and if any option is off, the model will not function/train as we expected.

Visual instruction tuning takes around 20 hours for LLaVA-v1.5-13B on 8x A100 (80G), due to the increased resolution to 336px. It takes around 10 hours for LLaVA-v1.5-7B on 8x A100 (40G).

Training script with DeepSpeed ZeRO-3: [`finetune.sh`](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune.sh).

If you are do not have enough GPU memory:

- Use LoRA: [`finetune_lora.sh`](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_lora.sh). We are able to fit 13B training in 8-A100-40G/8-A6000, and 7B training in 8-RTX3090. Make sure `per_device_train_batch_size*gradient_accumulation_steps` is the same as the provided script for best reproducibility.
- Replace `zero3.json` with `zero3_offload.json` which offloads some parameters to CPU RAM. This slows down the training speed.

If you are interested in finetuning LLaVA model to your own task/data, please check out [`Finetune_Custom_Data.md`](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md)。

New options to note:

- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.
- `--image_aspect_ratio pad`: this pads the non-square images to square, instead of cropping them; it slightly reduces hallucination.
- `--group_by_modality_length True`: this should only be used when your instruction tuning dataset contains both language (e.g. ShareGPT) and multimodal (e.g. LLaVA-Instruct). It makes the training sampler only sample a single modality (either image or language) during training, which we observe to speed up training by ~25%, and does not affect the final outcome.

## Evaluation

In LLaVA-1.5, we evaluate models on a diverse set of 12 benchmarks. To ensure the reproducibility, we evaluate the models with greedy decoding. We do not evaluate using beam search to make the inference process consistent with the chat demo of real-time outputs.

See [Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

### GPT-assisted Evaluation

Our GPT-assisted evaluation pipeline for multimodal modeling is provided for a comprehensive understanding of the capabilities of vision-language models.  Please see our paper for more details.

1. Generate LLaVA responses

```Shell
python model_vqa.py \
    --model-path ./checkpoints/LLaVA-13B-v0 \
    --question-file \
    playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    /path/to/coco2014_val \
    --answers-file \
    /path/to/answer-file-our.jsonl
```

2. Evaluate the generated responses.  In our case, [`answer-file-ref.jsonl`](./playground/data/coco2014_val_qa_eval/qa90_gpt4_answer.jsonl) is the response generated by text-only GPT-4 (0314), with the context captions/boxes provided.

```Shell
OPENAI_API_KEY="sk-***********************************" python llava/eval/eval_gpt_review_visual.py \
    --question playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --context llava/eval/table/caps_boxes_coco2014_val_80.jsonl \
    --answer-list \
    /path/to/answer-file-ref.jsonl \
    /path/to/answer-file-our.jsonl \
    --rule llava/eval/table/rule.json \
    --output /path/to/review.json
```

3. Summarize the evaluation results

```Shell
python summarize_gpt_review.py
```

## Citation

If you find LLaVA useful for your research and applications, please cite using this BibTeX:
```bibtex

@misc{liu2023improvedllava,
      title={Improved Baselines with Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
      publisher={arXiv:2310.03744},
      year={2023},
}

@misc{liu2023llava,
      title={Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={arXiv:2304.08485},
      year={2023},
}
```

## Acknowledgement

- [Vicuna](https://github.com/lm-sys/FastChat): the codebase we built upon, and our base model Vicuna-13B that has the amazing language capabilities!

## Related Projects

- [Instruction Tuning with GPT-4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://github.com/microsoft/LLaVA-Med)
- [Otter: In-Context Multi-Modal Instruction Tuning](https://github.com/Luodian/Otter)

For future project ideas, please check out:
- [SEEM: Segment Everything Everywhere All at Once](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) to detect, segment, and generate anything by marrying [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and [Segment-Anything](https://github.com/facebookresearch/segment-anything).
