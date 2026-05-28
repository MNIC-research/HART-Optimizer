import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset  # pip install datasets 필수!
from lion_pytorch import Lion
import time
import math
import random
import numpy as np
import os
import sys
import datetime
from torch.optim.lr_scheduler import _LRScheduler

class HART(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), beta_window=0.97,
                 range_val=1.2, base_wd=0.05, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, beta_window=beta_window,
                        range_val=range_val, base_wd=base_wd, eps=eps)
        super(HART, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, eps = group['lr'], group['eps']
            beta1, beta2 = group['betas']
            beta_window = group['beta_window']
            range_val = group['range_val']

            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['grad_variance'] = torch.zeros_like(p)
                    state['grad_window_avg'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                grad_var, window_avg = state['grad_variance'], state['grad_window_avg']
                state['step'] += 1
                current_step = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                m_hat = exp_avg / (1.0 - beta1 ** current_step)

                window_avg.mul_(beta_window).add_(grad, alpha=1.0 - beta_window)

                diff = grad - m_hat
                grad_var.mul_(beta2).addcmul_(diff, diff, value=1.0 - beta2)

                raw_signal = torch.norm(window_avg)
                raw_noise = torch.norm(grad_var.sqrt())

                stability_metric = raw_signal / (raw_noise + eps)

                decay_factor_val = torch.clamp(0.5 * stability_metric, min=0.0, max=1.0)
                noise_norm = torch.pow(torch.norm(grad_var), decay_factor_val)

                entropy_ratio = noise_norm / (raw_signal + eps)

                dynamic_scrap = torch.clamp(range_val * (entropy_ratio - 1.0), -range_val, range_val)

                norm_m_hat = torch.norm(m_hat)
                if norm_m_hat > eps:
                    u = m_hat / (norm_m_hat + eps)
                    noise_perp = grad - torch.sum(grad * u) * u

                    cleaned_grad = grad.clone()
                    cleaned_grad.sub_(noise_perp, alpha=dynamic_scrap)
                else:
                    cleaned_grad = grad.clone()

                exp_avg_sq.mul_(beta2).addcmul_(cleaned_grad, cleaned_grad, value=1.0 - beta2)
                v_hat = exp_avg_sq / (1.0 - beta2 ** current_step)

                if group['base_wd'] != 0:
                    p.mul_(1.0 - lr * group['base_wd'])

                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        return loss


class InverseWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, t_max_steps, warm_up_steps,
                 initial_lr=5e-4, target_lr=2.5e-4, eta_min=0.0, last_epoch=-1):

        self.phase1_steps = warm_up_steps
        self.phase2_steps = t_max_steps

        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.eta_min = eta_min

        super(InverseWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self._step_count - 1

        if current_step < self.phase1_steps:
            progress = current_step / max(1, self.phase1_steps) 
            lr = self.initial_lr - (self.initial_lr - self.target_lr) * progress
            return [lr for _ in self.base_lrs]

        else:
            phase2_current_step = current_step - self.phase1_steps
            progress = min(1.0, phase2_current_step / max(1, self.phase2_steps))

            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.eta_min + (self.target_lr - self.eta_min) * cosine_decay
            return [lr for _ in self.base_lrs]

# --- [1. Reproducibility] ---
def set_absolute_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

# --- [2. Dataset] ---
class WikiDataset(Dataset):
    def __init__(self, split="train", seq_len=256):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len = seq_len
        raw_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

        def tokenize_function(examples):
            return self.tokenizer(examples["text"])

        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        tokens = []
        for ids in tokenized_dataset["input_ids"]:
            if len(ids) > 0:
                tokens.extend(ids + [self.tokenizer.eos_token_id])

        self.tokens = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        input_ids = self.tokens[start:end]
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels
        }


# --- [3. Setup] ---

class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"HART_training_log_{current_time}.txt"
sys.stdout = DualLogger(log_filename)
print(f"Real-time log file generated.: {log_filename}")

set_absolute_reproducibility(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
BLOCK_SIZE = 256
GRAD_ACCUM_STEPS = 8
LR = 1e-3
TARGET_LR = 2.5e-4
WD = 0.01

EPOCHS = 10
WARMUP_EPOCHS = 0.0
T_MAX = 9.0
ETA_MIN = 1e-6

train_loader = DataLoader(WikiDataset("train", BLOCK_SIZE), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(WikiDataset("test", BLOCK_SIZE), batch_size=BATCH_SIZE, shuffle=False)

config = GPT2Config()
model = GPT2LMHeadModel(config).to(device)

optimizer = HART(model.parameters(), lr=LR, base_wd=WD, beta_window=0.97, range_val=1.2)
# optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
# optimizer = Lion(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=WD)

total_update_steps = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS) * EPOCHS
warmup_steps = int(math.ceil(len(train_loader) / GRAD_ACCUM_STEPS) * WARMUP_EPOCHS)
t_max_steps = int(math.ceil(len(train_loader) / GRAD_ACCUM_STEPS) * T_MAX)

# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = InverseWarmupCosineAnnealingLR(
    optimizer=optimizer,
    t_max_steps=t_max_steps,
    warm_up_steps=warmup_steps,
    initial_lr=LR,
    target_lr=TARGET_LR,
    eta_min=ETA_MIN
)

best_test_ppl = float('inf')

print("GPT-2 Training Start (WikiText-103)")
print("-" * 70)

# --- [4. Training Loop] ---
CHECKPOINT_LATEST = "checkpoint_latest.pt"
CHECKPOINT_BEST = "checkpoint_best.pt"

start_epoch = 1
global_step = 0
best_test_ppl = float('inf')
best_step_ppl = float('inf')

if os.path.exists(CHECKPOINT_LATEST):
    print(f"Found previous training checkpoint ({CHECKPOINT_LATEST}). Restoring model weights..")
    checkpoint = torch.load(CHECKPOINT_LATEST, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    global_step = checkpoint['global_step']
    best_test_ppl = checkpoint['best_test_ppl']
    best_step_ppl = checkpoint.get('best_step_ppl', float('inf'))

    if 'torch_rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['random_rng_state'])

    print(f"Restoration complete! Advancing training from Epoch {start_epoch}.")
    print(f"   (Saved Best Test PPL: {best_test_ppl:.2f})")
else:
    print("Launching a new GPT-2 training run (WikiText-103)")

print("-" * 70)

log_loss_sum = 0.0
log_step_count = 0

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss / GRAD_ACCUM_STEPS
        loss.backward()

        total_loss += loss.item() * GRAD_ACCUM_STEPS
        log_loss_sum += loss.item() * GRAD_ACCUM_STEPS

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            log_step_count += 1

            if global_step % 10 == 0:
                avg_log_loss = log_loss_sum / (log_step_count * GRAD_ACCUM_STEPS)
                log_ppl = math.exp(avg_log_loss)
                current_lr = optimizer.param_groups[0]["lr"]

                if log_ppl < best_step_ppl:
                    best_step_ppl = log_ppl

                print(
                    f"  [Epoch {epoch} | Step {global_step:5d}] LR: {current_lr:.6e} | Step Loss: {avg_log_loss:.4f} | Step PPL: {log_ppl:.2f} | 🏆 Best Step PPL: {best_step_ppl:.2f}")

                log_loss_sum = 0.0
                log_step_count = 0

    avg_train_loss = total_loss / len(train_loader)
    train_ppl = math.exp(avg_train_loss)

    model.eval()
    total_test_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_test_loss += outputs.loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_ppl = math.exp(avg_test_loss)

    is_best = False
    if test_ppl < best_test_ppl:
        best_test_ppl = test_ppl
        is_best = True

    print("-" * 70)
    print(
        f"HART Epoch {epoch:3d} Completed | Overall Train PPL: {train_ppl:.2f} | Test PPL: {test_ppl:.2f} | Best Test PPL: {best_test_ppl:.2f}")
    # print(
    #     f"AdamW Epoch {epoch:3d} Completed | Overall Train PPL: {train_ppl:.2f} | Test PPL: {test_ppl:.2f} | Best Test PPL: {best_test_ppl:.2f}")
    # print(
    #     f"Lion Epoch {epoch:3d} Completed | Overall Train PPL: {train_ppl:.2f} | Test PPL: {test_ppl:.2f} | Best Test PPL: {best_test_ppl:.2f}")

    # =====================================================================
    checkpoint_state = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_test_ppl': best_test_ppl,
        'best_step_ppl': best_step_ppl,
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'random_rng_state': random.getstate(),

    }

    torch.save(checkpoint_state, CHECKPOINT_LATEST)
    print(f"[Auto-Save] Epoch {epoch} state has been securely saved. ({CHECKPOINT_LATEST})")

    if is_best:
        torch.save(checkpoint_state, CHECKPOINT_BEST)
        print(f"[New Record!] The best-performing model has been securely stored in the vault. ({CHECKPOINT_BEST})")

    print("-" * 70)

print("Finished | Best Test PPL: {:.2f}".format(best_test_ppl))
