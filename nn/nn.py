from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from sklearn.metrics import accuracy_score
from config import cfg
import wandb
import gc


class TextDataset(Dataset):
    def __init__(self, text, labels, tokenizer, max_len):
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        labels = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        return {
            'text': text,
            'labels': labels,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


def create_data_loader(text, labels, tokenizer, batch_size, max_len, shuffle):
    dataset = TextDataset(
        text=text,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


class TgClassifier(nn.Module):
    def __init__(self, n_classes=3, dropout_prob=0.3):
        super(TgClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(
            cfg['model_name'], output_hidden_states=True)
        self.drop = nn.Dropout(p=cfg['dropout_prob'])
        self.out = nn.Linear(256, n_classes)
        self.fc = nn.Linear(4*self.bert.config.hidden_size, 256)
        self.relu = nn.LeakyReLU()
        self.layer_norm = nn.LayerNorm(256)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #pooled_output = outputs[0][:, 0, :]
        hidden_states = outputs[2]
        pooled_output = torch.cat(
            tuple(
                [hidden_states[i]
                 for i in [-4, -3, -2, -1]]),
            dim=-1)
        pooled_output = pooled_output[:, 0, :]
        hidden = self.drop(self.layer_norm(
            self.fc(self.relu(self.drop(pooled_output)))))
        out = self.out(hidden)
        return {'logits': out}


def get_predictions(model, data_loader):
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )['logits'].cpu()
            predictions.extend(outputs)
            true_values.extend(d['labels'])
    predictions = torch.stack(predictions).cpu()
    true_values = torch.stack(true_values).cpu()
    return predictions, true_values


def train_epoch(
        model, loss_fn, optimizer, data_loader, valid_data_loader=None,
        scheduler=None, device='cuda', num_labels=3, window=50, eval_steps=300,
        test_data_loader=None):
    model.train()
    ewm_loss = 0
    ewm_acc = 0.33
    i = 0
    for batch in tqdm(data_loader, desc='TRAIN'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['labels'].to(device)
        targets = targets  # .to(torch.float32)
        # , labels=targets)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = out['logits']
        #loss = out['loss']
        acc = calc_accuracy(outputs.cpu(), targets.cpu())
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        w = 1 / min(i+1, window)
        ewm_loss = ewm_loss * (1-w) + loss * w
        ewm_acc = ewm_acc*(1-w) + acc*w
        i += 1

        wandb.log({"train_acc": ewm_acc})
        wandb.log({"train_loss": ewm_loss})

        if eval_steps and ((i+1) % eval_steps == 0):
            print('=== Validation ===')
            eval_model(model, valid_data_loader, loss_fn, device, how='val')
            eval_model(model, test_data_loader, loss_fn, device, how='test')


def eval_model(model, data_loader, loss_fn, device, how='val'):
    model.eval()
    with torch.no_grad():
        losses = []
        desc = 'VALIDATION' if how == 'val' else 'TESTING'
        accs = []
        maes_fl = []
        maes_int = []
        for batch in tqdm(data_loader, desc=desc):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)
            # ,labels=targets)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            #loss = out['loss']
            outputs = out['logits']
            loss = loss_fn(outputs, targets)
            losses.append(loss.cpu().item())
            accs.append(calc_accuracy(outputs.cpu(), targets.cpu()))
            mae_fl, mae_int = calc_mae(outputs.cpu(), targets.cpu())
            maes_fl.append(mae_fl)
            maes_int.append(mae_int)
        if how == 'val':
            wandb.log({"valid_loss": np.mean(losses)})
            wandb.log({"val_acc": np.mean(accs)})
            wandb.log({"valid_mae_fl": np.mean(maes_fl)})
            wandb.log({"val_mae_i": np.mean(maes_int)})
            print(f'Validation loss: {np.mean(losses)}')
        else:
            wandb.log({"test_loss": np.mean(losses)})
            wandb.log({"test_acc": np.mean(accs)})
            wandb.log({"test_mae_fl": np.mean(maes_fl)})
            wandb.log({"test_mae_i": np.mean(maes_int)})
            print(f'Test loss: {np.mean(losses)}')
    model.train()
