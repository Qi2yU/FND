import os
import sys
import argparse
import pathlib
import json
import torch
import shutil
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pdb
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import create_dataloader
from src.trainer import Trainer
from src.utils import setup_seed, get_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Fakenews Detection Training")
    # Config file
    parser.add_argument('--config', type=str, default=None, help='Path to config file (JSON or YAML)')
    
    # Data Params
    parser.add_argument('--root_path', type=str, help='Root path of dataset')
    parser.add_argument('--data_name', type=str, default='weibo', help='Name of dataset (e.g., weibo, gossip)')
    parser.add_argument('--image_root', type=str, default='/data01/qy/fakenews/Dataset/weibo', help='Root path for images')
    # K-Fold Params
    parser.add_argument('--k_folds', type=int, default=0, help='Number of folds for K-Fold Cross Validation. 0 or 1 to disable.')
    parser.add_argument('--kfold_path', type=str, default=None, help='Path to kfold split file (JSON)')

    # Model Params
    parser.add_argument('--model_name', type=str, default='MM', help='Model architecture name')
    parser.add_argument('--text_encoder', type=str, default='bert')
    parser.add_argument('--text_encoder_path', type=str, default='')
    parser.add_argument('--text_max_len', type=int, default=170)
    parser.add_argument('--rational_encoder_path', type=str, default='')
    parser.add_argument('--rational_max_len', type=int, default=170)
    parser.add_argument('--img_encoder', type=str, default='clip')
    parser.add_argument('--img_encoder_path', type=str, default='/data01/qy/models/chinese-clip-vit-base-patch16', help='Path to Image Encoder (CLIP)')
    parser.add_argument('--emb_dim', type=int, default=768)
    parser.add_argument('--co_attention_dim', type=int, default=300)
    parser.add_argument('--num_classes', type=int, default=2)
    
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--w_dim', type=int, default=64)
    # Training Params
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--early_stop', type=int, default=6)
    parser.add_argument('--seed', type=int, default=3759)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_model', type=bool, default=True)
    # Loss Weights
    parser.add_argument('--llm_judgment_predictor_weight', type=float, default=-1)
    parser.add_argument('--rationale_usefulness_evaluator_weight', type=float, default=-1)
    parser.add_argument('--evidential_error_weight', type=float, default=-1)
    parser.add_argument('--kd_loss_weight', type=float, default=1)
    
    parser.add_argument('--align_weight', type=float, default=0.1)
    parser.add_argument('--self_re_weight', type=float, default=1)
    parser.add_argument('--cross_re_weight', type=float, default=1)
    
    # Output Params
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--info', type=str, default=None, help='info')
    return parser.parse_args()


def load_config_from_file(config_path: str) -> dict:
    """Load configuration dict from a JSON or YAML file."""
    p = pathlib.Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    suffix = p.suffix.lower()
    with open(p, 'r', encoding='utf-8') as f:
        if suffix in ['.json']:
            return json.load(f)
        elif suffix in ['.yml', '.yaml']:
            try:
                import yaml  # Optional dependency
            except ImportError:
                raise ImportError("PyYAML is required to load YAML config. Install with: pip install pyyaml")
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file type: {suffix}. Use .json or .yaml/.yml")

def run_kfold(args, logger, base_config):
    logger.info(f"Starting {args.k_folds}-Fold Cross Validation")
    
    # Base output dir for kfold (use the timestamp dir created in main)
    exp_dir = base_config['save_log_dir'].replace('/logs', '') 
    
    splits = []
    if args.kfold_path and os.path.exists(args.kfold_path):
        logger.info(f"Loading k-fold splits from {args.kfold_path}")
        with open(args.kfold_path, 'r') as f:
            splits = json.load(f)
        if len(splits) != args.k_folds:
             logger.warning(f"Loaded splits count ({len(splits)}) does not match k_folds ({args.k_folds}). Using loaded count.")
             args.k_folds = len(splits)
    else:
        # Load original train data to generate splits
        train_path = os.path.join(args.root_path, 'train.jsonl')
        data = []
        labels = []
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                data.append(item)
                # Use raw label for stratification
                label = int(item['label'])
                if label != 0 and label != 1:
                    raise ValueError(f"label is not 0/1 ! {label}")
                labels.append(label)
                
        skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
        
        for train_idx, val_idx in skf.split(data, labels):
            splits.append({
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist()
            })
            
        # Save splits
        split_save_path = os.path.join(exp_dir, 'kfold_splits.json')
        with open(split_save_path, 'w') as f:
            json.dump(splits, f, indent=4)
        logger.info(f"Saved k-fold splits to {split_save_path}")
    
    fold_results = []
    
    for fold, split in enumerate(splits):
        fold_num = fold + 1
        logger.info(f"========== Processing Fold {fold_num}/{args.k_folds} ==========")
        
        fold_dir = os.path.join(exp_dir, f'fold_{fold_num}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Config for this fold
        fold_config = base_config.copy()
        # Do NOT change root_path, keep it pointing to original data
        # fold_config['root_path'] = fold_dir 
        
        # Update save dirs
        fold_config['save_log_dir'] = os.path.join(fold_dir, 'logs')
        fold_config['save_param_dir'] = os.path.join(fold_dir, 'checkpoints')
        fold_config['tensorboard_dir'] = os.path.join(fold_dir, 'tensorboard')
        fold_config['test_save_path'] = os.path.join(fold_dir, 'test.json')
        
        # Add indices
        fold_config['train_indices'] = split['train_indices']
        fold_config['val_indices'] = split['val_indices']
        
        for d in [fold_config['save_log_dir'], fold_config['save_param_dir'], fold_config['tensorboard_dir']]:
            os.makedirs(d, exist_ok=True)
            
        # Dataloaders
        train_loader = create_dataloader(fold_config, mode='train')
        val_loader = create_dataloader(fold_config, mode='val')
        
        test_path = os.path.join(args.root_path, 'test.jsonl')
        if os.path.exists(test_path):
            test_loader = create_dataloader(fold_config, mode='test')
        else:
            test_loader = None
            
        # Trainer
        fold_logger = get_logger(fold_config['save_log_dir'], f'train_fold_{fold_num}')
        
        trainer = Trainer(fold_config, fold_logger)
        metric = trainer.fit(train_loader, val_loader, test_loader)
        fold_results.append(metric)
        
    # Aggregate results
    avg_metrics = {}
    if fold_results:
        keys = fold_results[0].keys()
        for k in keys:
            values = [r[k] for r in fold_results if isinstance(r[k], (int, float))]
            if values:
                avg_metrics[k] = sum(values) / len(values)
    
    logger.info(f"K-Fold Average Results: {avg_metrics}")
    with open(os.path.join(exp_dir, 'kfold_results.json'), 'w') as f:
        json.dump(avg_metrics, f, indent=4)

def main():
    args = parse_args()
    # If a config file is provided, load it and use values to override defaults
    file_cfg = None
    if args.config:
        file_cfg = load_config_from_file(args.config)
        # Merge: file config -> argparse namespace
        # Priority: command-line overrides file content when provided
        for k, v in file_cfg.items():
            if hasattr(args, k):
                # Only set if CLI arg is default (i.e., user didn't provide a different value)
                # We cannot easily distinguish here; prefer file values then let CLI overwrite below
                setattr(args, k, v)
        # Now re-parse CLI to overwrite any file-provided values
        # Note: For simplicity, we keep the current args; users can override key params via CLI
    
    # Setup Environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)
    
    # Create Output Directories
    if args.info is None:
        timestamp = os.path.join(args.output_dir, f"{args.model_name}_{args.data_name}")
    else:
        timestamp = os.path.join(args.output_dir, f"{args.model_name}_{args.data_name}_{args.info}")

    log_dir = os.path.join(timestamp, 'logs')
    param_dir = os.path.join(timestamp, 'checkpoints')
    tensorboard_dir = os.path.join(timestamp, 'tensorboard')
    test_save_path = os.path.join(timestamp, 'test.json')

    for d in [log_dir, param_dir, tensorboard_dir]:
        os.makedirs(d, exist_ok=True)
        
    logger = get_logger(log_dir, 'train')
    logger.info(f"Arguments: {args}")
    
    # Config Dictionary
    config = vars(args)
    # If file had nested dicts like model section, preserve them
    if file_cfg and isinstance(file_cfg, dict):
        # Merge nested dictionaries (shallow merge; file values win unless replaced above)
        for k, v in file_cfg.items():
            if isinstance(v, dict):
                existing = config.get(k, {})
                merged = {**existing, **v}
                config[k] = merged

    config['use_cuda'] = True
    config['save_log_dir'] = log_dir
    config['save_param_dir'] = param_dir
    config['tensorboard_dir'] = tensorboard_dir
    config['test_save_path'] = test_save_path

    # Save Config
    with open(os.path.join(timestamp, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    if args.k_folds > 1 or args.kfold_path:
        run_kfold(args, logger, config)
        return

    # Create Dataloaders
    logger.info("Creating dataloaders...")
    train_loader = create_dataloader(config, mode='train')
    val_loader = create_dataloader(config, mode='val')
    test_path = os.path.join(args.root_path, 'test.jsonl')
    
    if os.path.exists(test_path):
        test_loader = create_dataloader(config, mode='test')
    else:
        test_loader = None
    
    # Initialize Trainer
    trainer = Trainer(config, logger)
    
    # Start Training
    best_metric = trainer.fit(train_loader, val_loader, test_loader)
    
    logger.info(f"Best Metric: {best_metric}")

if __name__ == '__main__':
    main()
