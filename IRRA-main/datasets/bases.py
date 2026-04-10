from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from utils.mtp_text import build_full_text, build_rgb_text, build_thermal_text
from prettytable import PrettyTable
import random
import regex as re
import copy


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        # Summarize train/test/val statistics for quick sanity checking.
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    # Add special start/end tokens and pad/truncate to fixed length.
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True,
                 use_modality_text_prompt: bool = False,
                 mtp_color_drop_mode: str = "delete"):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.use_modality_text_prompt = bool(use_modality_text_prompt)
        self.mtp_color_drop_mode = str(mtp_color_drop_mode)
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Each item returns one image-text pair plus identity metadata.
        sample = self.dataset[index]
        # Legacy format: (pid, image_id, img_path, caption)
        # RGBT format:  (pid, image_id, rgb_path, thermal_path, caption)
        if len(sample) == 5:
            pid, image_id, img_path, t_img_path, caption = sample
        else:
            pid, image_id, img_path, caption = sample
            t_img_path = None

        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        if t_img_path is not None:
            # Thermal image uses the same transform pipeline/size as RGB.
            t_img = read_image(t_img_path)
            if self.transform is not None:
                t_img = self.transform(t_img)

        # Tokenize caption into fixed-length CLIP token ids.
        if self.use_modality_text_prompt:
            full_text = build_full_text(caption)
            rgb_text = build_rgb_text(caption)
            thermal_text = build_thermal_text(
                caption, color_drop_mode=self.mtp_color_drop_mode
            )
            tokens = tokenize(
                full_text,
                tokenizer=self.tokenizer,
                text_length=self.text_length,
                truncate=self.truncate,
            )
            tokens_rgb = tokenize(
                rgb_text,
                tokenizer=self.tokenizer,
                text_length=self.text_length,
                truncate=self.truncate,
            )
            tokens_th = tokenize(
                thermal_text,
                tokenizer=self.tokenizer,
                text_length=self.text_length,
                truncate=self.truncate,
            )
        else:
            tokens = tokenize(
                caption,
                tokenizer=self.tokenizer,
                text_length=self.text_length,
                truncate=self.truncate,
            )

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens,
        }
        if self.use_modality_text_prompt:
            ret['caption_ids_rgb'] = tokens_rgb
            ret['caption_ids_th'] = tokens_th
        if t_img_path is not None:
            ret['thermal_images'] = t_img

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        # Used by retrieval evaluator as image gallery.
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class ImageDatasetRGBT(Dataset):
    def __init__(self, image_pids, img_paths_rgb, img_paths_t, transform=None):
        self.image_pids = image_pids
        self.img_paths_rgb = img_paths_rgb
        self.img_paths_t = img_paths_t
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        # Gallery item for text-to-RGBT retrieval evaluation.
        pid = self.image_pids[index]
        rgb_path = self.img_paths_rgb[index]
        t_path = self.img_paths_t[index]

        rgb_img = read_image(rgb_path)
        t_img = read_image(t_path)
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
            t_img = self.transform(t_img)
        return pid, rgb_img, t_img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        # Used by retrieval evaluator as text queries.
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True,
                 use_modality_text_prompt: bool = False,
                 mtp_color_drop_mode: str = "delete"):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.use_modality_text_prompt = bool(use_modality_text_prompt)
        self.mtp_color_drop_mode = str(mtp_color_drop_mode)

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Same as ImageTextDataset, but also builds MLM inputs/labels.
        sample = self.dataset[index]
        # Legacy format: (pid, image_id, img_path, caption)
        # RGBT format:  (pid, image_id, rgb_path, thermal_path, caption)
        if len(sample) == 5:
            pid, image_id, img_path, t_img_path, caption = sample
        else:
            pid, image_id, img_path, caption = sample
            t_img_path = None

        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        if t_img_path is not None:
            # Thermal image uses the same transform pipeline/size as RGB.
            t_img = read_image(t_img_path)
            if self.transform is not None:
                t_img = self.transform(t_img)
        
        if self.use_modality_text_prompt:
            full_text = build_full_text(caption)
            rgb_text = build_rgb_text(caption)
            thermal_text = build_thermal_text(
                caption, color_drop_mode=self.mtp_color_drop_mode
            )
            caption_tokens = tokenize(
                full_text,
                tokenizer=self.tokenizer,
                text_length=self.text_length,
                truncate=self.truncate,
            )
            caption_tokens_rgb = tokenize(
                rgb_text,
                tokenizer=self.tokenizer,
                text_length=self.text_length,
                truncate=self.truncate,
            )
            caption_tokens_th = tokenize(
                thermal_text,
                tokenizer=self.tokenizer,
                text_length=self.text_length,
                truncate=self.truncate,
            )
        else:
            caption_tokens = tokenize(
                caption,
                tokenizer=self.tokenizer,
                text_length=self.text_length,
                truncate=self.truncate,
            )

        # Randomly mask tokens following BERT-style masking policy.
        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels
        }
        if self.use_modality_text_prompt:
            ret['caption_ids_rgb'] = caption_tokens_rgb
            ret['caption_ids_th'] = caption_tokens_th
        if t_img_path is not None:
            ret['thermal_images'] = t_img

        return ret

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # Mask token with 15% probability.
                if prob < 0.15:
                    prob /= 0.15

                    # 80% -> replace by [MASK].
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% -> replace by random token.
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # Remaining 10% -> keep original token.

                    # Store original token as MLM supervision target.
                    labels.append(token)
                else:
                    # 0 means ignore this position in MLM loss.
                    labels.append(0)
            else:
                # Special tokens/padding are not used for MLM supervision.
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # Ensure at least one supervised position to avoid empty MLM target.
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)
