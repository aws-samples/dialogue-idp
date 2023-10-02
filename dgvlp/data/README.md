# OBELICS

OBELICS is an open, massive, and curated collection of interleaved image-text web documents, containing 141M English documents, 115B text tokens, and 353M images, extracted from Common Crawl dumps between February 2020 and February 2023. The collection and filtering steps are described in our paper.

Interleaved image-text web documents are a succession of text paragraphs interleaved by images, such as web pages that contain images. Models trained on these web documents outperform vision and language models trained solely on image-text pairs on various benchmarks. They can also generate long and coherent text about a set of multiple images. As an example, we trained IDEFICS, a visual language model that accepts arbitrary sequences of image and text inputs and produces text outputs.


~[Image](https://atlas.nomic.ai/map/f2fba2aa-3647-4f49-a0f3-9347daeee499/ee4a84bd-f125-4bcc-a683-1b4e231cb10f)

# Wikipedia

Wikipedia dataset containing cleaned articles of all languages. The datasets are built from the Wikipedia dump (https://dumps.wikimedia.org/) with one split per language. Each example contains the content of one full Wikipedia article with cleaning to strip markdown and unwanted sections (references, etc.).


# The LAION-400M i

LAION-400M a dataset with CLIP-filtered 400 million image-text pairs, their CLIP embeddings and kNN indices that allow efficient similarity search. Multi-modal language-vision models trained on hundreds of millions of image-text pairs (e.g. CLIP, DALL-E) gained a recent surge, showing remarkable capability to perform zero- or few-shot learning and transfer even in absence of per-sample labels on target image data. Despite this trend, to date there has been no publicly available datasets of sufficient scale for training such models from scratch. This dataset is entirely openly, freely accessible.

# PMD

Public Multimodal Dataset (PMD) is a collection of publicly-available image-text pair datasets. PMD contains 70M image-text pairs in total with 68M unique images. The dataset contains pairs from Conceptual Captions, Conceptual Captions 12M, WIT, Localized Narratives, RedCaps, COCO, SBU Captions, Visual Genome and a subset of YFCC100M dataset.



| Data Source | Type of Data                             | Number of Tokens in Source | Number of Images in Source | Epochs | Effective Proportion in Number of Tokens |
|-------------|-----------------------------------------|---------------------------|---------------------------|--------|-----------------------------------------|
| [OBELICS](https://huggingface.co/datasets/HuggingFaceM4/OBELICS)     | Unstructured Multimodal Web Documents    | 114.9B                      | 353M                      | 1      | 73.85%                                  |
| [Wikipedia](https://huggingface.co/datasets/wikipedia)   | Unstructured Multimodal Web Documents    | 3.192B                     | 39M                     | 3      | 6.15%                                  |
| [LAION](https://huggingface.co/datasets/laion/laion2B-en)       | Image-Text Pairs                         | 29.9B                      | 1.120B                      | 1      | 17.18%
| [PMD](https://huggingface.co/datasets/facebook/pmd)         | Image-Text Pairs                         | 1.6B                      | 70M                      | 3      | 2.82%                                   |                                |


# Medical textbook (MTB)

MTB was constructed to create a new multi-modal dataset sourced from a collection of 4,721 textbooks spanning various medical specialties. In the preprocessing phase, each textbook is initially transformed from PDF format to HTML. During this transformation, we eliminate all tags, but image tags are replaced with <image> tokens. Subsequently, the data undergoes a cleaning process, which includes deduplication and content filtering. After cleaning, each textbook, now containing both the refined text and images, is divided into segments for the pre-training phase. Each of these segments contains at least one image, and can have as many as ten images, while adhering to a specified maximum length. In aggregate, the MTB dataset encompasses roughly 0.8 million images and 584 million tokens. For the purposes of pre-training, we utilize 95% of this data for training and reserve the remaining 5% for evaluation.


# PMC

This biomedical dataset consists of 1.6 million image-caption pairs gathered from the Open Access subset of PubMed Central. From this collection, 1.3 million image-caption pairs are designated for training, while 160,000 pairs are set aside for evaluation, in accordance with the provided public split.
