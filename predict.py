import json
import os
import random
import subprocess
import time
from io import BytesIO
from pathlib import Path
from pprint import pprint
from threading import Thread
from typing import Union

import requests
import torch
from cog import BasePredictor, ConcatenateIterator, Input, Path
from PIL import Image
from transformers.generation.streamers import TextIteratorStreamer

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

os.environ["HUGGINGFACE_HUB_CACHE"] = os.getcwd() + "/weights"

class Predictor(BasePredictor):
    def setup(self, model_path, model_base, model_name) -> None:
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path=model_path, model_base=model_base, model_name=model_name, load_8bit=False, load_4bit=False)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt to use for text generation"),
        top_p: float = Input(description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens", ge=0.0, le=1.0, default=1.0),
        temperature: float = Input(description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic", default=0.2, ge=0.0),
        max_tokens: int = Input(description="Maximum number of tokens to generate. A word is generally 2-3 tokens", default=1024, ge=0),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
    
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
    
        image_data = load_image(str(image))
        image_tensor = self.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().cuda()
    
        # loop start
    
        # just one turn, always prepend image token
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0)
    
        with torch.inference_mode():
            thread = Thread(target=self.model.generate, kwargs=dict(
                inputs=input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria]))
            thread.start()
            # workaround: second-to-last token is always " "
            # but we want to keep it if it's not the second-to-last token
            prepend_space = False
            for new_text in streamer:
                if new_text == " ":
                    prepend_space = True
                    continue
                if new_text.endswith(stop_str):
                    new_text = new_text[:-len(stop_str)].strip()
                    prepend_space = False
                elif prepend_space:
                    new_text = " " + new_text
                    prepend_space = False
                if len(new_text):
                    yield new_text
            if prepend_space:
                yield " "
            thread.join()
    

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image



class DatasetInference:
    def __init__(self, json_filepath: Union[str, Path], db_path: str) -> None:
        self.json_filepath = json_filepath
        self.ds = self.load_json_file()
        self.db_path = db_path
        print(f"Setup with db_path: {self.db_path}")

    def load_json_file(self):
        try:
            return json.load(open(self.json_filepath, "r"))
        except StopIteration:
            raise FileNotFoundError(
                f"JSON-file not found: {self.json_filepath}. Exiting."
            )

    def load_and_pprint_sample(self, full=False, print=True):
        item = random.randint(0, len(self.ds))
        data = self.ds[item]
        if print:
            if full:
                print("_______________________________________________________")
                print(f'{data["debug_info"]["title"]}, from: {data["debug_info"]["vendor"]}')
                print(f'Inferred type: {data["debug_info"]["type_inference"]["type"]}')
                print(f'Matching key: {data["debug_info"]["type_inference"]["match_key"]}')
                print(
                    f'In attribute value: {data["debug_info"]["type_inference"]["matched_attr"]}'
                )
            print("_______________________________________________________")
            print("Conversation:")
            print(data["conversations"][0]["value"])
            print("")
            print(data["conversations"][1]["value"])
            print("_______________________________________________________")
            if full:
                print("Dumped snippets:")
                pprint(data["debug_info"]["dumped_snippets"])
                print("_______________________________________________________")
                print("Here they might be talking about the backside")
                pprint(data["debug_info"]["back_descr"])
            print(os.path.join(self.db_path, data["image"]))
        imagepath = os.path.join(self.db_path, data["image"])
        item_type = data["debug_info"]["type_inference"]["type"]
        convo = [data["conversations"][0]["value"],data["conversations"][1]["value"]]
        return Path(imagepath), item_type, convo


    def load_sample(self, index):
        data = self.ds[index]
        imagepath = os.path.join(self.db_path, data["image"])
        item_type = data["debug_info"]["type_inference"]["type"]
        convo = [data["conversations"][0]["value"],data["conversations"][1]["value"]]
        return Path(imagepath), item_type, convo