import os
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, ChineseCLIPProcessor, AutoProcessor
from pathlib import Path

class FakenewsDataset(Dataset):
    def __init__(self, data_path, image_root, text_tokenizer, rational_tokenizer, img_processor=None, text_max_len=170, rational_max_len=170, mode='train', indices=None):
        self.data = []
        self.image_root = image_root
        self.text_tokenizer = text_tokenizer
        self.rational_tokenizer = rational_tokenizer
        self.img_processor = img_processor
        self.text_max_len = text_max_len
        self.rational_max_len = rational_max_len
        self.mode = mode
        self.indices = indices

        self._load_data(data_path)

    def _load_data(self, path):
        print(f"Loading data from {path}...")
        raw_data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                raw_data.append(item)
        
        if self.indices is not None:
            raw_data = [raw_data[i] for i in self.indices]

        for item in raw_data:
            # Check image existence if image processor is provided
            if self.img_processor:
                img_name = item.get('image')
                if not img_name:
                    continue
                img_path = os.path.join(self.image_root, img_name)
                if not os.path.exists(img_path):
                    # print(f"Warning: Image not found: {img_path}")
                    continue
                item['img_path'] = img_path

            self.data.append(item)
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Process Text
        content = item['content']
        inputs = self.text_tokenizer(
            content,
            max_length=self.text_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        # 2. Process Label
        label = item['label']

        # 3. Process Auxiliary Features (Rationale)
        # Using .get() with defaults to handle missing keys safely
        # rational_1 = item.get('rational_1', "") or item.get('td_rationale', "")
        # rational_2 = item.get('rational_2', "") or item.get('cs_rationale', "")

        rational_1 = item['rational_1']
        rational_2 = item['rational_2']
        
        rational_1_inputs = self.rational_tokenizer(rational_1, max_length=self.rational_max_len, padding='max_length', truncation=True, return_tensors='pt')
        rational_2_inputs = self.rational_tokenizer(rational_2, max_length=self.rational_max_len, padding='max_length', truncation=True, return_tensors='pt')

        rational_1_ids = rational_1_inputs['input_ids'].squeeze(0)
        rational_1_mask = rational_1_inputs['attention_mask'].squeeze(0)
        rational_2_ids = rational_2_inputs['input_ids'].squeeze(0)
        rational_2_mask = rational_2_inputs['attention_mask'].squeeze(0)

        # 4. Process Auxiliary Labels
        # rational_1_pred = self.label_map_ftr.get(item.get('rational_1_pred'))
        # rational_2_pred = self.label_map_ftr.get(item.get('rational_2_pred'))
        
        # rational_1_acc = int(item.get('rational_1_acc'))
        # rational_2_acc = int(item.get('rational_2_acc'))
        sample = {
            'content': input_ids,
            'content_masks': attention_mask,
            'label': torch.tensor(label, dtype=torch.long),
            'id': str(item.get('id', item.get('source_id', ''))),
            
            'r1': rational_1_ids,
            'r1_masks': rational_1_mask,
            'r2': rational_2_ids,
            'r2_masks': rational_2_mask,
            
        }

        if 'content_rewrite' in item:
            content_rewrite = item['content_rewrite']
            neutral_input = self.text_tokenizer(
                content_rewrite,
                max_length=self.text_max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            sample['content_neutral'] = neutral_input['input_ids'].squeeze(0)
            sample['content_neutral_masks'] = neutral_input['attention_mask'].squeeze(0)
            
        

        # sample = {
        #     'content': input_ids,
        #     'content_masks': attention_mask,
        #     'label': torch.tensor(label, dtype=torch.long),
        #     'id': str(item.get('id', item.get('source_id', ''))),
            
        #     'r1': input_ids,
        #     'r1_masks': attention_mask,
        #     'r2': input_ids,
        #     'r2_masks': attention_mask,
        # }
        # 5. Process Image
        # if self.img_processor:
            
        #     image = Image.open(item['img_path']).resize((224, 224)).convert("RGB")
        #     image_inputs = self.img_processor(images=image, return_tensors="pt", do_resize=False)
        #     sample['img'] = image_inputs['pixel_values'].squeeze(0)

        if self.img_processor:
            with Image.open(item['img_path']) as img:
                # 1. 先处理调色板图片（P 模式）
                if img.mode == "P":
                    # 官方建议：先转 RGBA
                    img = img.convert("RGBA")

                # 2. 如果有 alpha 通道（RGBA），把透明度贴到白底上再变成 RGB
                if img.mode == "RGBA":
                    # 创建白色背景
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    # 用 alpha 通道做 mask 贴过去
                    alpha = img.split()[3]
                    background.paste(img, mask=alpha)
                    img = background  # 现在是纯 RGB 了
                else:
                    # 其它情况统一转成 RGB
                    img = img.convert("RGB")

                # 3. 统一 resize 到 224x224
                img = img.resize((224, 224), Image.BICUBIC)

            # 4. 交给 CLIP 的 image_processor，明确通道在最后
            img = np.array(img)  # (H, W, 3)
            image_inputs = self.img_processor(
                images=img,
                return_tensors="pt",
                input_data_format="channels_last",
            )
            sample['img'] = image_inputs['pixel_values'].squeeze(0)
            
        return sample

def create_dataloader(config, mode='train'):
    text_encoder_path = config['text_encoder_path']
    rational_encoder_path = config['rational_encoder_path']
    # text_encoder = config['text_encoder']
    # text_encoder_path = config['text_encoder_path']
    # img_encoder = config['img_encoder_path']
    img_encoder_path = config.get('img_encoder_path')
    
    text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
    rational_tokenizer = AutoTokenizer.from_pretrained(rational_encoder_path)

    img_processor = None
    if img_encoder_path:
        img_processor = AutoProcessor.from_pretrained(img_encoder_path)

    data_root = config['root_path']
    # Construct file path based on mode and monthly logic if needed
    # Original logic: get_monthly_path(data_type, root_path, 'train.jsonl')
    # We simplify: assume standard names or config provided names
    
    indices = None
    if mode == 'train':
        if 'train_indices' in config:
            indices = config['train_indices']
            file_name = 'train.jsonl'
        else:
            file_name = 'train.jsonl'
        shuffle = True
    elif mode == 'val':
        if 'val_indices' in config:
            indices = config['val_indices']
            file_name = 'train.jsonl'
        else:
            file_name = 'val.jsonl'
        shuffle = False
    else:
        file_name = 'test.jsonl'
        shuffle = False
        
    # Support for the "monthly" folder structure if it exists, otherwise direct path
    # Original code had: os.path.join(root_path, data_type, 'month_'+str(month), file_name)
    # But user args usually just point to a root.
    # Let's try to find the file.
    
    # Simplified path construction:
    # 1. Try root_path/file_name
    # 2. Try root_path/data_name/file_name
    
    file_path = os.path.join(data_root, file_name)
    if not os.path.exists(file_path):
        # Try looking into subfolders if needed, or just fail
        # For now, assume the user provides the directory containing jsonl files
        pass

    # Image root
    # Original code hardcoded: /data01/qy/fakenews/Dataset/weibo
    # We should make this configurable
    image_root = config.get('image_root', '/data01/qy/fakenews/Dataset/weibo')

    dataset = FakenewsDataset(
        data_path=file_path,
        image_root=image_root,
        text_tokenizer=text_tokenizer,
        rational_tokenizer=rational_tokenizer,
        img_processor=img_processor,
        text_max_len=config['text_max_len'],
        rational_max_len=config['rational_max_len'],
        mode=mode,
        indices=indices
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batchsize'],
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader
