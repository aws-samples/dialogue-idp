from transformers import pipeline
import torch
from typing import Union


def get_device(ignore_mps: bool = False, cuda_as_int: bool = False) -> Union[str, int]:
    """
    Determines and returns a PyTorch device name.

    :param ignore_mps: Manual deactivate MPS
    :param cuda_as_int: Return cuda as device number
    :return: device name as str
    """
    # Define device (either GPU, M1/2, or CPU)
    if torch.cuda.is_available():
        #print('Device: using CUDA')
        return "cuda" if not cuda_as_int else 0
    elif torch.backends.mps.is_available() and not ignore_mps:
        print('Device: using MPS')
        return "mps"
    else:
        print('Device: using CPU :(')
        return "cpu"

def mclass(text_prompt, top_k=3, topics = ['Mask creation', 'Object  detection', 'Inpainting', 'Segmentation', 'Upscaling', 'Creating an image from another one', 'Generating:q an image from text'], model='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'):
    # Get the device (CPU, GPU, Apple M1/2 aka MPS)
    # Ignoring MPS because this particular model contains some int64 ops that are not supported by the MPS backend yet :(
    # see https://github.com/pytorch/pytorch/issues/80784
    device = get_device(ignore_mps=True, cuda_as_int=True)

    # Define a german hypothesis template and the potential candidates for entailment/contradiction
    template_de = 'The topic is {}'

    # Pipeline abstraction from hugging face
    pipe = pipeline(task='zero-shot-classification', model=model, tokenizer=model, device=device)

    # Run pipeline with a test case
    prediction = pipe(text_prompt, topics, hypothesis_template=template_de)

    # Top 3 topics as predicted in zero-shot regime
    #print(f'Zero-shot prediction for: \n {prediction["sequence"]}')
    top_3 = zip(prediction['labels'][0:top_k], prediction['scores'][0:top_k])
    return top_3