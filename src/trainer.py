import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from src.utils import data2gpu, Averager, Recorder
from src.loss import *
from src.models.multimodal import *
from transformers import get_cosine_schedule_with_warmup
# Import other models as needed



class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if config['use_cuda'] and torch.cuda.is_available() else "cpu")
        
        self._init_model()
        self._init_optimizer()
        self._init_criterion()
        # self._init_scheduler() # Moved to fit() to access train_loader size
        
        self.recorder = Recorder(config['early_stop'])
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=config['tensorboard_dir'])

    def _init_model(self):
        model_name = self.config['model_name']
        self.model_name = model_name
        data_name = self.config['data_name']
        
        self.logger.info(f"Initializing model: {model_name} for data: {data_name}")
        
        # Model selection logic (Decoupled from train loop)
        if model_name == 'uncertainty':
            self.model = MModel_uncertainty(self.config)
        elif model_name == 'evidential':
            self.model = MModel_evidential_learning(self.config)
        elif model_name == 'distangle':
            self.model = MModel_distangle(self.config)
        elif model_name == 'distangle_1':
            self.model = MModel_distangle_1(self.config)
        else:
            # Default model
            self.model = MModel_CE(self.config)
            
        self.model.to(self.device)

    def _init_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['lr'], 
            weight_decay=self.config.get('weight_decay', 5e-5)
        )

    def _init_scheduler(self, num_training_steps):
        # 使用带预热的余弦退火调度器
        warmup_ratio = self.config.get('warmup_ratio', 0.1)
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        
        self.logger.info(f"Initializing Cosine Schedule with Warmup: Total Steps={num_training_steps}, Warmup Steps={num_warmup_steps}")
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def _init_criterion(self):
        # Automatic loss selection based on model type or config
        # Add label smoothing to help mitigate overfitting (configurable)
        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=self.config.get('label_smoothing', 0.0))
        self.criterion_bce = nn.BCELoss()
        if self.model_name == "distangle_1":
            self.criterion_mse = nn.MSELoss()
            self.criterion_cosine = nn.CosineEmbeddingLoss()
            self.criterion_contrast = SupervisedContrastiveLoss(temperature=self.config.get('contrastive_temperature', 0.07))
        # You can add more complex logic here if needed
        self.logger.info("Loss functions initialized.")

    def compute_loss(self, res, batch_data):
        label = batch_data['label']
        # label_r1_pred = batch_data['rational_1_pred']
        # label_r1_acc = batch_data['rational_1_acc']

        # label_r2_pred = batch_data['rational_2_pred']
        # label_r2_acc = batch_data['rational_2_acc']
        # 1. Classification Loss
        loss_classify = self.criterion_cls(res['classify_pred'], label)
        
        loss_info = None
        if self.model_name == 'evidential':
            # loss_evidential_pred = self.criterion_cls(res['evidential_r1']['pred'], label_r1_acc.long()) + self.criterion_cls(res['evidential_r2']['pred'], label_r2_acc.long())

            # label_evidential_onehot = F.one_hot(label_r1_acc, num_classes=2).float()
            # alpha_hat_r1 = label_evidential_onehot + (1.0 - label_evidential_onehot) * res['evidential_r1']['alpha']
            # alpha_hat_r2 = label_evidential_onehot + (1.0 - label_evidential_onehot) * res['evidential_r2']['alpha']

            # beta = torch.ones_like(alpha_hat_r1)
            # kl_per_sample_r1 = dirichlet_kl(alpha_hat_r1, beta)   # [B]
            # error_loss_r1 = kl_per_sample_r1.mean()
            # kl_per_sample_r2 = dirichlet_kl(alpha_hat_r2, beta)   # [B]
            # error_loss_r2 = kl_per_sample_r2.mean()
            
            # loss_evidential = loss_evidential_pred + self.config['model']['evidential_error_weight'] * (error_loss_r1 + error_loss_r2)
            
            # loss_rational_pred = self.criterion_cls(res['r1_pred'], label_r1_pred.long()) + self.criterion_cls(res['r2_pred'], label_r2_pred.long())

            # loss = loss_classify + self.config['model']['llm_judgment_predictor_weight'] * loss_rational_pred + self.config['model']['rationale_usefulness_evaluator_weight'] * loss_evidential
            pass
        elif self.model_name == 'distangle':
            loss_align = F.mse_loss(res['z_m'], res['z_r'])
            loss_recon_self = F.mse_loss(res['m_rec_self'], res['m_target']) + F.mse_loss(res['r_rec_self'], res['r_target'])
            loss_recon_cross = F.mse_loss(res['r_from_m'], res['m_target']) + F.mse_loss(res['m_from_r'], res['r_target'])
            loss = loss_classify + self.config['model']['align_weight'] * loss_align + \
                                   self.config['model']['self_re_weight'] * loss_recon_self + \
                                   self.config['model']['cross_re_weight'] * loss_recon_cross
        elif self.model_name == 'distangle_1':
            # cls loss
            loss_cls_mm = self.criterion_cls(res['logits_m_p'], label.long())
            loss_cls_r1 = self.criterion_cls(res['logits_r1_p'], label.long())
            loss_cls_r2 = self.criterion_cls(res['logits_r2_p'], label.long())
            loss_cls_shared = self.criterion_cls(res['logits_s'], label.long())
            loss_cls_aux = loss_cls_mm + loss_cls_r1 + loss_cls_r2 + loss_cls_shared
            # reconstruct loss
            # recon origin
            loss_recon_mm = self.criterion_mse(res['m_recon'], res['proj_x_m'])
            loss_recon_r1 = self.criterion_mse(res['r1_recon'], res['proj_x_r1'])
            loss_recon_r2 = self.criterion_mse(res['r2_recon'], res['proj_x_r2'])
            loss_recon_origin = loss_recon_mm + loss_recon_r1 + loss_recon_r2
            # recon specific feature
            loss_recon_s_mm = self.criterion_mse(res['m_recon_p'].transpose(0, 1), res['m_p'])
            loss_recon_s_r1 = self.criterion_mse(res['r1_recon_p'].transpose(0, 1), res['r1_p'])
            loss_recon_s_r2 = self.criterion_mse(res['r2_recon_p'].transpose(0, 1), res['r2_p'])
            loss_recon_specific = loss_recon_s_mm + loss_recon_s_r1 + loss_recon_s_r2
            # ort loss
            dim = self.config['model']['embed_dim']
            loss_ort_mm = self.criterion_cosine(res['m_p'].reshape(-1, dim), res['m_s'].reshape(-1, dim), torch.tensor([-1]).to(self.device))
            loss_ort_r1 = self.criterion_cosine(res['r1_p'].reshape(-1, dim), res['r1_s'].reshape(-1, dim), torch.tensor([-1]).to(self.device))
            loss_ort_r2 = self.criterion_cosine(res['r2_p'].reshape(-1, dim), res['r2_s'].reshape(-1, dim), torch.tensor([-1]).to(self.device))
            loss_ort = loss_ort_mm + loss_ort_r1 + loss_ort_r2

            # contrastive loss on modality-specific summaries (bs, dim)
            feats = torch.cat([res['m_s_sim'], res['r1_s_sim'], res['r2_s_sim']], dim=0)
            labels_all = torch.cat([label, label, label], dim=0)
            loss_contrast = self.criterion_contrast(feats, labels_all)

            loss = loss_classify + self.config['model']['hyper_parameters']['cls'] * loss_cls_aux +\
                                   self.config['model']['hyper_parameters']['recon_origin'] * loss_recon_origin +\
                                   self.config['model']['hyper_parameters']['recon_specific'] * loss_recon_specific +\
                                   self.config['model']['hyper_parameters']['ort'] * loss_ort +\
                                   self.config['model']['hyper_parameters']['contrastive'] * loss_contrast
            
            loss_info = {
                'loss': loss.item(),
                'cls': loss_classify.item(),
                'cls_aux_loss': loss_cls_aux.item(),
                'recon_origin': loss_recon_origin.item(),
                'recon_specific': loss_recon_specific.item(),
                'ort': loss_ort.item(),
                'contrastive': loss_contrast.item()
            }
        else:
            # loss_rational_useful = self.criterion_bce(res['r1_useful'], label_r1_acc.float()) + self.criterion_bce(res['r2_useful'], label_r2_acc.float())

            # loss_rational_pred = self.criterion_cls(res['r1_pred'], label_r1_pred.long()) + self.criterion_cls(res['r2_pred'], label_r2_pred.long())

            # loss = loss_classify + self.config['model']['llm_judgment_predictor_weight'] * loss_rational_pred + self.config['model']['rationale_usefulness_evaluator_weight'] * loss_rational_useful
            pass
        return loss, loss_classify, loss_info

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        avg_loss = Averager()
        avg_cls_loss = Averager()
        
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            batch = data2gpu(batch, self.config['use_cuda'])
            
            # Prepare input for model (unpack dict)
            # The model expects kwargs matching config keys + batch keys
            # We can pass the batch dict directly if the model's forward accepts **kwargs
            # But we need to merge config into it or ensure model has access to config
            
            # In original code: batch_input_data = {**self.config, **batch_data}
            # This is a bit messy. Ideally model should use self.config.
            # But let's stick to the pattern to avoid breaking model code.
            model_inputs = {**self.config, **batch}
            
            res, debug_info = self.model(**model_inputs)
            
            loss, cls_loss, loss_info = self.compute_loss(res, batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to stabilize training and reduce overfitting from exploding updates
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
            self.optimizer.step()
            self.scheduler.step()
            
            avg_loss.add(loss.item())
            avg_cls_loss.add(cls_loss.item())
            
            if loss_info is None:
                pbar.set_postfix({'loss': avg_loss.item(), 'cls_loss': avg_cls_loss.item()})
            else:
                pbar.set_postfix(loss_info)
            # Tensorboard logging
            global_step = epoch * len(train_loader) + step
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/Cls_Loss', cls_loss.item(), global_step)

        return avg_loss.item()

    def evaluate(self, val_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        misclassified_indices = []
        base_index = 0  # 作为样本编号回退方案（当数据集中未提供显式ID时）
        
        gate_stats = {
            'r1_useful': [], 'r1_useless': [],
            'r2_useful': [], 'r2_useless': []
        }

        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader, desc="Evaluating"):
                batch = data2gpu(batch, self.config['use_cuda'])
                # label_r1_pred = batch['rational_1_pred']
                # label_r1_acc = batch['rational_1_acc']

                # label_r2_pred = batch['rational_2_pred']
                # label_r2_acc = batch['rational_2_acc']
                model_inputs = {**self.config, **batch}
                
                res, debug_info = self.model(**model_inputs)
                
                # Assuming 'classify_pred' is logits
                preds = torch.argmax(res['classify_pred'], dim=1).cpu().numpy()
                labels = batch['label'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)

                # 记录误分类样本编号：优先使用 batch 提供的显式ID
                if 'idx' in batch:
                    batch_ids = batch['idx']
                elif 'id' in batch:
                    batch_ids = batch['id']
                elif 'sample_id' in batch:
                    batch_ids = batch['sample_id']
                else:
                    # 回退：使用连续编号
                    batch_ids = list(range(base_index, base_index + len(labels)))
                # 找出误分类的位置并记录对应ID
                for i, (p, y, sid) in enumerate(zip(preds, labels, batch_ids)):
                    if p != y:
                        misclassified_indices.append(sid)
                base_index += len(labels)

                # if self.model_name == 'evidential':
                #     gates = debug_info['uncertain_gate_value']

                #     # R1 (Column 0)
                #     # label_r1_acc: 1 (Useful), 0 (Useless)
                #     r1_useful_mask = (label_r1_acc == 1)
                #     r1_useless_mask = (label_r1_acc == 0)
                    
                #     gate_stats['r1_useful'].extend(gates[r1_useful_mask, 0].cpu().tolist())
                #     gate_stats['r1_useless'].extend(gates[r1_useless_mask, 0].cpu().tolist())

                #     # R2 (Column 1)
                #     r2_useful_mask = (label_r2_acc == 1)
                #     r2_useless_mask = (label_r2_acc == 0)
                    
                #     gate_stats['r2_useful'].extend(gates[r2_useful_mask, 1].cpu().tolist())
                #     gate_stats['r2_useless'].extend(gates[r2_useless_mask, 1].cpu().tolist())
        
        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        
        # Calculate per-class metrics (Label 0: Fake, Label 1: Real)
        p_class, r_class, f_class, _ = precision_recall_fscore_support(all_labels, all_preds, labels=[0, 1], average=None, zero_division=0)
        
        metrics = {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            
            # Label 0 (Fake)
            'fake_precision': p_class[0].item(),
            'fake_recall': r_class[0].item(),
            'fake_f1': f_class[0].item(),
            
            # Label 1 (Real)
            'real_precision': p_class[1].item(),
            'real_recall': r_class[1].item(),
            'real_f1': f_class[1].item()
        }

        # if self.model_name == 'evidential':
        #     for key, val_list in gate_stats.items():
        #         if len(val_list) > 0:
        #             metrics[f'{key}_mean'] = np.mean(val_list).item()
        #             metrics[f'{key}_var'] = np.var(val_list).item()
        #         else:
        #             metrics[f'{key}_mean'] = None
        #             metrics[f'{key}_var'] = None
        
        info = {
            'misclassified_indices': misclassified_indices
        }
        return metrics, info

    def fit(self, train_loader, val_loader, test_loader=None):
        self.logger.info("Start training...")
        
        # Initialize scheduler with total training steps
        num_training_steps = len(train_loader) * self.config['epoch']
        self._init_scheduler(num_training_steps)

        for epoch in range(self.config['epoch']):
            train_loss = self.train_epoch(train_loader, epoch)
            self.logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")
            
            val_metrics, val_info = self.evaluate(val_loader)
            self.logger.info(f"Epoch {epoch} | Val Metrics: {val_metrics}")
            self.logger.info(f"Epoch {epoch} | Val Misclassified: {val_info['misclassified_indices']}")
            
            # Tensorboard
            for k, v in val_metrics.items():
                self.writer.add_scalar(f'Val/{k}', v, epoch)
            # 记录误分类样本数量（可选）
            self.writer.add_scalar('Val/Misclassified_Count', len(val_info.get('misclassified_indices', [])), epoch)
            
            # Early Stopping & Saving
            is_best = self.recorder.add(val_metrics['f1']) # Monitor F1
            if is_best:
                self.save_checkpoint('best_model.pth')
                self.logger.info("New best model saved!")
            
            if self.recorder.stop():
                self.logger.info("Early stopping triggered.")
                break
            
            if test_loader is not None:
                epoch_test_metrics, epoch_test_info = self.evaluate(test_loader)
                self.logger.info(f"Epoch {epoch} | test Metrics: {epoch_test_metrics}")
                self.logger.info(f"Epoch {epoch} | Test Misclassified Count: {epoch_test_info['misclassified_indices']}")

        self.logger.info("Training finished.")
        
        # Test if provided
        if test_loader:
            self.load_checkpoint('best_model.pth')
            test_metrics, test_info = self.evaluate(test_loader)
            self.logger.info(f"Test Metrics: {test_metrics}")
            

            with open(self.config['test_save_path'], 'w') as f:
                json.dump({'metrics': test_metrics, 'info': test_info}, f, indent=4)
            return test_metrics
        
        return self.recorder.best_metric

    def save_checkpoint(self, filename):
        if not self.config['save_model']:
            return
        save_path = os.path.join(self.config['save_param_dir'], filename)
        torch.save(self.model.state_dict(), save_path)

    def load_checkpoint(self, filename):
        save_path = os.path.join(self.config['save_param_dir'], filename)
        self.model.load_state_dict(torch.load(save_path))
