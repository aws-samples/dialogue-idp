# IDEFICS

This demo used a pre-trained IDEFICS-9b-instructVLM model developed by HuggingFaceM4 , a fine-tuned version of IDEFICS-9b followed the training procedure laid out in Flamingo by combining the two open-access pre-trained models (laion/CLIP-ViT-H-14-laion2B-s32B-b79K and huggyllama/llama-7b) with modified Transformer blocks. The IDEFICS-9b was trained on OBELIC, Wikipedia, LAION and PMD  multimodal datasets with total 150B tokens and 1.582B images with 224x224 resolution each. The IDEFICS-9b was based on Llama-7b with 1.31M effective batch size. The IDEFICS-9b-instruct was then fine tuned with ~6.8M multimodality  instruction datasets  created from augmentation using generative AI by unfreezing all the parameters (vision encoder, language model, cross-attentions).  The fine-tuning datasets include the pre-training datawith the following sampling ratios: 5.1% of image-text pairs and 30.7% of OBELICS multimodal web documents. 

The training objective is the standard next token prediction. We use the following hyper and training parameters:
| Parameters | | IDEFICS-80b-instruct | IDEFICS-9b-instruct |
| -- | -- | -- | -- |
| Training | Sequence Length | 2048 | 2048 |
| | Effective Batch Size (# of tokens) | 613K | 205K |
| | Max Training Steps | 22K | 22K |
| | Weight Decay | 0.1 | 0.1 |
| | Optimizer | Adam(0.9, 0.999) | Adam(0.9, 0.999) |
| | Gradient Clipping | 1.0 | 1.0 |
| | [Z-loss](https://huggingface.co/papers/2204.02311) weight | 0. | 0. |
| Learning Rate | Initial Max | 3e-6 | 1e-5 |
| | Initial Final | 3.6e-7 | 1.2e-6 |
| | Decay Schedule | Linear | Linear |
| | Linear warmup Steps | 1K | 1K |
| Large-scale Optimization | Gradient Checkpointing | True | True |
| | Precision | Mixed-pres bf16 | Mixed-pres bf16 |
| | ZeRO Optimization | Stage 3 | Stage 3 |


We note that since IDEFICS was trained on PMD (which contains COCO), the evaluation numbers on COCO are not directly comparable with Flamingo and OpenFlamingo since they did not explicitly have this dataset in the training mixture. Additionally, Flamingo is trained with images of resolution 320 x 320 while IDEFICS and OpenFlamingo were trained with images of 224 x 224 resolution.

| Model | Shots | <nobr>VQAv2<br>OE VQA acc.</nobr> | <nobr>OKVQA<br>OE VQA acc.</nobr> | <nobr>TextVQA<br>OE VQA acc.</nobr> | <nobr>VizWiz<br>OE VQA acc.</nobr> | <nobr>TextCaps<br>CIDEr</nobr> | <nobr>Coco<br>CIDEr</nobr> | <nobr>NoCaps<br>CIDEr</nobr> | <nobr>Flickr<br>CIDEr</nobr> | <nobr>VisDial<br>NDCG</nobr> | <nobr>HatefulMemes<br>ROC AUC</nobr> | <nobr>ScienceQA<br>acc.</nobr> | <nobr>RenderedSST2<br>acc.</nobr> | <nobr>Winoground<br>group/text/image</nobr> |
|:------------|--------:|---------------------:|---------------------:|-----------------------:|----------------------:|-------------------:|---------------:|-----------------:|-----------------:|-----------------:|-------------------------:|-----------------------:|--------------------------:|----------------------------------:|
| IDEFICS 80B |       0 |                 60.0 |                 45.2 |                   30.9 |                  36.0 |               56.8 |           91.8 |             65.0 |             53.7 |             48.8 |                     60.6 |                   68.9 |                      60.5 |                               8.0/18.75/22.5|
|             |       4 |                 63.6 |                 52.4 |                   34.4 |                  40.4 |               72.7 |          110.3 |             99.6 |             73.7 |             48.4 |                     57.8 |                   58.9 |                      66.6 |                              - |
|             |       8 |                 64.8 |                 55.1 |                   35.7 |                  46.1 |               77.6 |          114.3 |            105.7 |             76.6 |             47.9 |                     58.2 |                   - |                      67.8 |                              - |
|             |      16 |                 65.4 |                 56.8 |                   36.3 |                  48.3 |               81.4 |          116.6 |            107.0 |             80.1 |             - |                     55.8 |                   - |                      67.7 |                              - |
|             |      32 |                 65.9 |                 57.8 |                   36.7 |                  50.0 |               82.7 |          116.6 |            107.5 |             81.1 |             - |                     52.5 |                   - |                      67.3 |                              - |
| IDEFICS 9B  |       0 |                 50.9 |                 38.4 |                   25.9 |                  35.5 |               25.4 |           46.0 |             36.8 |             27.3 |             48.7 |                     51.7 |                   44.2 |                      61.8 |                               5.0/16.8/20.8 |
|             |       4 |                 55.4 |                 45.5 |                   27.6 |                  36.9 |               60.0 |           93.0 |             81.3 |             59.7 |             47.9 |                     50.7 |                   37.4 |                      62.3 |                              - |
|             |       8 |                 56.4 |                 47.7 |                   27.5 |                  40.4 |               63.2 |           97.0 |             86.8 |             61.9 |             47.6 |                     51.0 |                   - |                      66.3 |                              - |
|             |      16 |                 57.0 |                 48.4 |                   27.9 |                  42.6 |               67.4 |           99.7 |             89.4 |             64.5 |             - |                     50.9 |                   - |                      67.8 |                              - |
|             |      32 |                 57.9 |                 49.6 |                   28.3 |                  43.7 |               68.1 |           98.0 |             90.5 |             64.4 |             - |                     49.8 |                   - |                      67.0 |                              - |

<br>
For ImageNet-1k, we also report results where the priming samples are selected to be similar (i.e. close in a vector space) to the queried instance. This is the Retrieval-based In-Context Example Selection (RICES in short) approach introduced by [Yang et al. (2021)](https://arxiv.org/abs/2109.05014).

| Model      |   Shots | Support set size | Shots selection | ImageNet-1k<br>Top-1 acc. |
|:-----------|--------:|-----------------:|:----------------|--------------------------:|
| IDEFICS 80B |      16 | 1K               | Random          |                      65.4 |
|            |      16 | 5K               | RICES           |                      72.9 |
| IDEFICS 9B  |      16 | 1K               | Random          |                      53.5 |
|            |      16 | 5K               | RICES           |                      64.5 |

## IDEFICS instruct

Similarly to the base IDEFICS models, we performed checkpoint selection to stop the training. Given that M3IT contains in the training set a handful of the benchmarks we were evaluating on, we used [MMBench](https://huggingface.co/papers/2307.06281) as a held-out validation benchmark to perform checkpoint selection. We select the checkpoint at step 3'000 for IDEFICS-80b-instruct and at step 8'000 for IDEFICS-9b-instruct.

| Model | Shots | <nobr>VQAv2 <br>OE VQA acc.</nobr> | <nobr>OKVQA <br>OE VQA acc.</nobr> | <nobr>TextVQA <br>OE VQA acc.</nobr> | <nobr>VizWiz<br>OE VQA acc.</nobr> | <nobr>TextCaps <br>CIDEr</nobr> | <nobr>Coco <br>CIDEr</nobr> | <nobr>NoCaps<br>CIDEr</nobr> | <nobr>Flickr<br>CIDEr</nobr> | <nobr>VisDial <br>NDCG</nobr> | <nobr>HatefulMemes<br>ROC AUC</nobr> | <nobr>ScienceQA <br>acc.</nobr> | <nobr>RenderedSST2<br>acc.</nobr> | <nobr>Winoground<br>group/text/image</nobr> |
| :--------------------- | --------: | ---------------------: | ---------------------: | -----------------------: | ----------------------: | -------------------: | ---------------: | -----------------: | -----------------: | -----------------: | -------------------------: | -----------------------: | --------------------------: | ----------------------------------: |
| Finetuning data **does not** contain the evaluation dataset | - | &#10006; | &#10006; | &#10006; | &#10004; | &#10006; | &#10006; | &#10006; | &#10004; | &#10006; | &#10004; | &#10006; | &#10004; | &#10006; |
| <nobr>IDEFICS 80B Instruct<br> | 0 | 37.4 (-22.7) | 36.9 (-8.2) | 32.9 (1.9) | 26.2 (-9.8) | 76.5 (19.7) | 117.2 (25.4) | 104.5 (39.5) | 65.3 (11.7) | 49.3 (0.4) | 58.9 (-1.7) | 69.5 (0.5) | 67.3 (6.8) | 9.2/20.0/25.0 (1.2/1.2/2.5) |
|  | 4 | 67.5 (4.0) | 54.0 (1.7) | 37.8 (3.5) | 39.8 (-0.7) | 71.7 (-1.0) | 116.9 (6.6) | 104.0 (4.4) | 67.1 (-6.6) | 48.9 (0.5) | 57.5 (-0.3) | 60.5 (1.6) | 65.5 (-1.1) | - |
|  | 8 | 68.1 (3.4) | 56.9 (1.8) | 38.2 (2.5) | 44.8 (-1.3) | 72.7 (-4.9) | 116.8 (2.5) | 104.8 (-0.9) | 70.7 (-5.9) | 48.2 (0.3) | 58.0 (-0.2) | - | 68.6 (0.8) | - |
|  | 16 | 68.6 (3.2) | 58.2 (1.4) | 39.1 (2.8) | 48.7 (0.4) | 77.0 (-4.5) | 120.5 (4.0) | 107.4 (0.4) | 76.0 (-4.1) | - | 56.4 (0.7) | - | 70.1 (2.4) | - |
|  | 32 | 68.8 (2.9) | 59.5 (1.8) | 39.3 (2.6) | 51.2 (1.2) | 79.7 (-3.0) | 123.2 (6.5) | 108.4 (1.0) | 78.4 (-2.7) | - | 54.9 (2.4) | - | 70.5 (3.2) | - |
| <nobr>IDEFICS 9B Instruct<br> | 0 | 65.8 (15.0) | 46.1 (7.6) | 29.2 (3.3) | 41.2 (5.6) | 67.1 (41.7) | 129.1 (83.0) | 101.1 (64.3) | 71.9 (44.6) | 49.2 (0.5) | 53.5 (1.8) | 60.6 (16.4) | 62.8 (1.0) | 5.8/20.0/18.0 (0.8/2.2/-2.8)|
|  | 4 | 66.2 (10.8) | 48.7 (3.3) | 31.0 (3.4) | 39.0 (2.1) | 68.2 (8.2) | 128.2 (35.1) | 100.9 (19.6) | 74.8 (15.0) | 48.9 (1.0) | 51.8 (1.1) | 53.8 (16.4) | 60.6 (-1.8) | - |
|  | 8 | 66.5 (10.2) | 50.8 (3.1) | 31.0 (3.5) | 41.9 (1.6) | 70.0 (6.7) | 128.8 (31.8) | 101.5 (14.8) | 75.5 (13.6) | 48.2 (0.6) | 51.7 (0.6) | - | 61.3 (-4.9) | - |
|  | 16 | 66.8 (9.8) | 51.7 (3.3) | 31.6 (3.7) | 44.8 (2.3) | 70.2 (2.7) | 128.8 (29.1) | 101.5 (12.2) | 75.8 (11.4) | - | 51.7 (0.7) | - | 63.3 (-4.6) | - |
|  | 32 | 66.9 (9.0) | 52.3 (2.7) | 32.0 (3.7) | 46.0 (2.2) | 71.7 (3.6) | 127.8 (29.8) | 101.0 (10.5) | 76.3 (11.9) | - | 50.8 (1.0) | - | 60.9 (-6.1) | - |


# Llama v2

Llama 2 is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. This is the repository for the 7B pretrained model, converted for the Hugging Face Transformers format. Links to other models can be found in the index at the bottom.

Model Architecture Llama 2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align to human preferences for helpfulness and safety.

Here are the fine tuned Llama v2 models 

|Model|Llama2|Llama2-hf|Llama2-chat|Llama2-chat-hf|
|---|---|---|---|---|
|7B| [Link](https://huggingface.co/llamaste/Llama-2-7b) | [Link](https://huggingface.co/llamaste/Llama-2-7b-hf) | [Link](https://huggingface.co/llamaste/Llama-2-7b-chat) | [Link](https://huggingface.co/llamaste/Llama-2-7b-chat-hf)|
|13B| [Link](https://huggingface.co/llamaste/Llama-2-13b) | [Link](https://huggingface.co/llamaste/Llama-2-13b-hf) | [Link](https://huggingface.co/llamaste/Llama-2-13b-chat) | [Link](https://huggingface.co/llamaste/Llama-2-13b-hf)|
|70B| [Link](https://huggingface.co/llamaste/Llama-2-70b) | [Link](https://huggingface.co/llamaste/Llama-2-70b-hf) | [Link](https://huggingface.co/llamaste/Llama-2-70b-chat) | [Link](https://huggingface.co/llamaste/Llama-2-70b-hf)|

# Whisper

Whisper is a Transformer based encoder-decoder model, also referred to as a _sequence-to-sequence_ model. It was trained on 680k hours of labelled speech data annotated using large-scale weak supervision.  Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. Trained on 680k hours of labelled data, Whisper models demonstrate a strong ability to generalise to many datasets and domains **without** the need
for fine-tuning.

| Size     | Parameters | English-only                                         | Multilingual                                        |
|----------|------------|------------------------------------------------------|-----------------------------------------------------|
| tiny     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny)     |
| base     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)     |
| small    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)    |
| medium   | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium)   |
| large    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)    |
| large-v2 | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large-v2) |


# DistilBERT base uncased finetuned SST-2

This model is a fine-tune checkpoint of [DistilBERT-base-uncased](https://huggingface.co/distilbert-base-uncased), fine-tuned on SST-2.
This model reaches an accuracy of 91.3 on the dev set (for comparison, Bert bert-base-uncased version reaches an accuracy of 92.7).


# Grounding DINO

Grounding DINO is an open-set object detector which combines Transformer-based detector DINO with grounded pre-training, which can detect arbitrary objects with human inputs such as category names or referring expressions. 

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>Data</th>
      <th>box AP on COCO</th>
      <th>Checkpoint</th>
      <th>Config</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>GroundingDINO-T</td>
      <td>Swin-T</td>
      <td>O365,GoldG,Cap4M</td>
      <td>48.4 (zero-shot) / 57.2 (fine-tune)</td>
      <td><a href="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth">GitHub link</a> | <a href="https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth">HF link</a></td>
      <td><a href="https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinT_OGC.py">link</a></td>
    </tr>
    <tr>
      <th>2</th>
      <td>GroundingDINO-B</td>
      <td>Swin-B</td>
      <td>COCO,O365,GoldG,Cap4M,OpenImage,ODinW-35,RefCOCO</td>
      <td>56.7 </td>
      <td><a href="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth">GitHub link</a>  | <a href="https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth">HF link</a> 
      <td><a href="https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinB.cfg.py">link</a></td>
    </tr>
  </tbody>
</table>


# Sagment Anything

The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a dataset of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

Three model versions of the model are available with different backbone sizes. These models can be instantiated by running

```
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
