import logging
import os.path as op
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size
from utils.iotools import read_json

from .bases import (
    ImageDataset,
    ImageDatasetRGBT,
    TextDataset,
    ImageTextDataset,
    ImageTextMLMDataset,
)

from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid
from .rgbtpedes import RGBTPEDES

__factory = {
    'CUHK-PEDES': CUHKPEDES,
    'ICFG-PEDES': ICFGPEDES,
    'RSTPReid': RSTPReid,
    'RGBT-PEDES': RGBTPEDES,
}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    # Build image preprocessing pipeline for train/eval.
    height, width = img_size

    # CLIP default normalization statistics.
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        # Evaluation only does deterministic resize + normalize.
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # Training transform, optionally with stronger augmentation.
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform


def collate(batch):
    # Collect all keys from a list of dict samples.
    keys = set([key for b in batch for key in b.keys()])
    # Convert list[dict] into dict[list] to ease tensor stacking.
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        # Convert Python scalars to tensors.
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        # Stack already-tensor fields along batch dimension.
        elif torch.is_tensor(v[0]):
             batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict


def _extract_img_path_from_train_sample(sample):
    # Legacy: (pid, image_id, img_path, caption)
    # RGBT:  (pid, image_id, rgb_path, thermal_path, caption)
    if len(sample) == 5:
        return sample[2]
    if len(sample) == 4:
        return sample[2]
    raise ValueError(
        "Unsupported train sample format for synthetic captions. "
        f"Expected len=4 or len=5, got len={len(sample)}."
    )


def _clone_train_sample_with_caption(sample, caption):
    if len(sample) == 5:
        pid, image_id, img_path, t_img_path, _ = sample
        return (pid, image_id, img_path, t_img_path, caption)
    if len(sample) == 4:
        pid, image_id, img_path, _ = sample
        return (pid, image_id, img_path, caption)
    raise ValueError(
        "Unsupported train sample format for synthetic captions. "
        f"Expected len=4 or len=5, got len={len(sample)}."
    )


def _append_synth_captions_to_train(dataset, args, logger):
    use_synth = bool(getattr(args, "use_synth_caption_bank", False))
    train_mode = str(getattr(args, "synth_caption_train_mode", "append")).lower()
    bank_file = str(getattr(args, "synth_caption_bank_file", ""))
    keep_per_image = int(getattr(args, "synth_caption_keep_per_image", 2))
    use_synth_loss_weighting = bool(
        getattr(args, "use_synth_loss_weighting", False)
    )
    synth_loss_weight = float(getattr(args, "synth_loss_weight", 1.0))
    effective_synth_weight = synth_loss_weight if use_synth_loss_weighting else 1.0

    original_train_count = len(dataset.train)
    added_synth_captions = 0
    enabled = use_synth and train_mode == "append"

    if enabled:
        if keep_per_image < 0:
            raise ValueError(
                "synth_caption_keep_per_image must be >= 0, "
                f"got {keep_per_image}."
            )
        if bank_file == "":
            raise ValueError(
                "use_synth_caption_bank=True requires non-empty synth_caption_bank_file."
            )
        if not op.isfile(bank_file):
            raise RuntimeError(
                f"synth_caption_bank_file='{bank_file}' does not exist."
            )

        synth_bank = read_json(bank_file)
        if not isinstance(synth_bank, dict):
            raise ValueError(
                "synth_caption_bank_file must be a JSON object: "
                "{<stable_image_key>: [caption1, caption2, ...], ...}."
            )

        # One base sample per image_id, preserving first-seen order.
        base_sample_by_image_id = {}
        for sample in dataset.train:
            image_id = sample[1]
            if image_id not in base_sample_by_image_id:
                base_sample_by_image_id[image_id] = sample

        appended_samples = []
        for base_sample in base_sample_by_image_id.values():
            # Stable key aligns to current train sample key: basename(img_path).
            img_path = _extract_img_path_from_train_sample(base_sample)
            stable_image_key = op.basename(str(img_path))
            synth_captions = synth_bank.get(stable_image_key, [])
            if not isinstance(synth_captions, list):
                continue

            kept = 0
            for synth_caption in synth_captions:
                if kept >= keep_per_image:
                    break
                if not isinstance(synth_caption, str):
                    continue
                synth_caption = synth_caption.strip()
                if synth_caption == "":
                    continue
                appended_samples.append(
                    _clone_train_sample_with_caption(base_sample, synth_caption)
                )
                kept += 1

        if appended_samples:
            dataset.train.extend(appended_samples)
            added_synth_captions = len(appended_samples)

    logger.info("[SYNTH] enabled=%s", enabled)
    logger.info("[SYNTH] bank_file=%s", bank_file)
    logger.info("[SYNTH] keep_per_image=%d", keep_per_image)
    logger.info("[SYNTH] added_synth_captions=%d", added_synth_captions)
    logger.info(
        "[SYNTH] total_train_captions_after_append=%d",
        len(dataset.train),
    )
    logger.info("[SYNTH_WEIGHT] enabled=%s", use_synth_loss_weighting)
    logger.info("[SYNTH_WEIGHT] synth_loss_weight=%.6f", synth_loss_weight)
    logger.info("[SYNTH_WEIGHT] original_caption_weight=1.0")

    caption_is_synth = [0] * original_train_count + [1] * added_synth_captions
    caption_weight = (
        [1.0] * original_train_count
        + [effective_synth_weight] * added_synth_captions
    )
    if len(caption_is_synth) != len(dataset.train) or len(caption_weight) != len(dataset.train):
        raise RuntimeError(
            "Synthetic caption metadata length mismatch: "
            f"metadata={len(caption_weight)}, train={len(dataset.train)}."
        )
    return caption_is_synth, caption_weight


def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers
    # Instantiate dataset by name from factory.
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)
    
    if args.training:
        caption_is_synth, caption_weight = _append_synth_captions_to_train(
            dataset, args, logger
        )
        # Build transform pipelines for train and validation.
        train_transforms = build_transforms(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
        val_transforms = build_transforms(img_size=args.img_size,
                                          is_train=False)

        if args.MLM:
            # MLM training needs extra masked tokens and labels.
            train_set = ImageTextMLMDataset(
                dataset.train,
                train_transforms,
                text_length=args.text_length,
                use_modality_text_prompt=bool(
                    getattr(args, "use_modality_text_prompt", False)
                ),
                mtp_color_drop_mode=str(
                    getattr(args, "mtp_color_drop_mode", "delete")
                ),
                caption_is_synth=caption_is_synth,
                caption_weight=caption_weight,
            )
        else:
            train_set = ImageTextDataset(
                dataset.train,
                train_transforms,
                text_length=args.text_length,
                use_modality_text_prompt=bool(
                    getattr(args, "use_modality_text_prompt", False)
                ),
                mtp_color_drop_mode=str(
                    getattr(args, "mtp_color_drop_mode", "delete")
                ),
                caption_is_synth=caption_is_synth,
                caption_weight=caption_weight,
            )

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                # Split global batch across ranks.
                mini_batch_size = args.batch_size // get_world_size()
                # TODO wait to fix bugs
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)

            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                # Identity sampler: each batch has N identities x K instances.
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate)
        elif args.sampler == 'random':
            # TODO add distributed condition
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # Validation split can be val or test, controlled by args.val_dataset.
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        if "img_paths_t" in ds:
            val_img_set = ImageDatasetRGBT(
                ds["image_pids"], ds["img_paths"], ds["img_paths_t"], val_transforms
            )
        else:
            val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                       val_transforms)
        val_txt_set = TextDataset(ds['caption_pids'],
                                  ds['captions'],
                                  text_length=args.text_length)

        # Validation uses independent image/text loaders for retrieval evaluation.
        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)

        return train_loader, val_img_loader, val_txt_loader, num_classes

    else:
        # Build dataloaders for pure testing.
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=args.img_size,
                                               is_train=False)

        eval_split = str(getattr(args, "eval_dataset", "test")).lower()
        if eval_split == "val":
            ds = dataset.val
        elif eval_split == "test":
            ds = dataset.test
        else:
            raise ValueError(
                f"Unsupported eval_dataset={eval_split!r}. Expected 'val' or 'test'."
            )
        logger.info("building eval dataloader on split=%s", eval_split)
        if "img_paths_t" in ds:
            test_img_set = ImageDatasetRGBT(
                ds["image_pids"], ds["img_paths"], ds["img_paths_t"], test_transforms
            )
        else:
            test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                        test_transforms)
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.text_length)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        return test_img_loader, test_txt_loader, num_classes
