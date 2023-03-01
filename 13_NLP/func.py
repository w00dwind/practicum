import pickle
from tqdm.notebook import tqdm
# from tqdm import tqdm
from pathlib import Path
from torch import cat, tensor
import os
import numpy as np
from sklearn.metrics import f1_score
from torch import cuda
import torch
# from torch.nn.functional import softmax
import wandb
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from datetime import datetime


def make_tokens(tokenizer,
                df,
                tokenized_filepath: str,
                max_length: int,
                force=False,
                fraq=1
                ):
    """Tokenize all of the sentences and map th e tokens to thier word IDs.
    reutrns input_ids, attention_masks, labels
    args:
        tokenizer - tokenizer from huggingface
        df - dataframe with sentence/comment/etc and label (toxic ot not)
        tokenized_filepath - Path to file with tokens, if doesn't exist, will be created and saved.
        max_length - maximum length of token, to make truncation
        data_frac - perc of data to use (between 0 and 1)
        force - force write to file
        fraq - perc of data to make tokens
    """
    if fraq < 1:
        smpl = int(np.floor(len(df) * fraq))
        df = df.sample(smpl)

    # input_ids = []
    # attention_maskss = []
    tokens_dict = {'input_ids': [],
                   'attention_masks': [],
                   'token_type_ids': [],
                   'labels': df.toxic.to_list()
                   }
    sentences = df.text.values

    if not (Path(tokenized_filepath).exists() and os.path.getsize(tokenized_filepath) > 0) or force or (fraq < 1):

        # For every sentence...
        for i, sent in enumerate(tqdm(sentences, desc='making tokens')):
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=max_length,
                padding="max_length",  # Pad & truncate all sentences.
                truncation=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                # Only for BERT finetuning
                return_token_type_ids=True
            )

            # Add the encoded sentence to the list.
            tokens_dict['input_ids'].append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            tokens_dict['attention_masks'].append(encoded_dict['attention_mask'])
            tokens_dict['token_type_ids'].append(encoded_dict['token_type_ids'])

        if fraq == 1:
            with open(tokenized_filepath, 'wb') as tokenized:
                pickle.dump(tokens_dict, tokenized)
                print(f'>>> Tokens saved successful to {tokenized_filepath}')

    elif fraq == 1:

        with open(tokenized_filepath, 'rb') as tokenized:
            print('>>> loading tokens from file, please wait')
            tokens_dict = pickle.load(tokenized)
            print(f'>>> Tokens loaded from {tokenized_filepath}')

    # Convert the lists into tensors.
    tokens_dict['input_ids'] = cat(tokens_dict['input_ids'], dim=0)
    tokens_dict['attention_masks'] = cat(tokens_dict['attention_masks'], dim=0)
    tokens_dict['token_type_ids'] = cat(tokens_dict['token_type_ids'], dim=0)
    tokens_dict['labels'] = tensor(tokens_dict['labels'])

    return tokens_dict


def choose_device(platform: str):
    if platform == 'Darwin':
        return torch.device('mps')
    elif platform == 'Linux' and cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def fit_epoch(train_loader, model, device, optimizer, use_wandb, log_interval):
    train_epoch_loss = []
    train_epoch_true = []
    train_epoch_preds = []

    for step, batch in enumerate(tqdm(train_loader, desc='train', position=0, leave=True)):
        optimizer.zero_grad()

        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        type_ids = batch[2].to(device)
        labels = batch[3].to(device)
        # labels = batch[2].to(device)

        # _, logits = model(input_ids,
        #                      attention_mask=input_mask,
        #                      labels=labels,
        #                      token_type_ids=None,
        #                      return_dict=False
        #                      )

        outputs = model(ids=input_ids,
                       mask=input_mask,
                       token_type_ids=type_ids
                       )
        preds = torch.argmax(outputs, 1).cpu().tolist()
        train_epoch_true.extend(labels.cpu().tolist())
        train_epoch_preds.extend(preds)
        loss = loss_fn(outputs, labels)
        train_epoch_loss.append(loss.item())
        if use_wandb and step % log_interval == 99:
            wandb.log({'runing train loss': np.mean(train_epoch_loss),
                       'running f1': f1_score(train_epoch_true, train_epoch_preds)})

        loss.backward()
        optimizer.step()

    return train_epoch_loss, train_epoch_true, train_epoch_preds


def eval_epoch(valid_loader, model, device):
    model.eval()
    eval_epoch_loss = []
    eval_epoch_true = []
    eval_epoch_preds = []

    for step, batch in enumerate(tqdm(valid_loader, desc='valid', position=0, leave=True)):
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        labels = batch[3].to(device)
        # labels = batch[2].to(device)

        with torch.no_grad():
            # loss, outputs = model(input_ids=input_ids,
            #                      attention_mask=input_mask,
            #                      labels=labels,
            #                      return_dict=False)
            outputs = model(ids=input_ids,
                            mask=input_mask,
                            token_type_ids=token_type_ids)
        loss = loss_fn(outputs, labels)
        # eval_epoch_loss.append(loss.item())
        eval_epoch_loss.append(loss.item())
        preds = torch.argmax(outputs, 1).cpu().tolist()
        eval_epoch_true.extend(labels.cpu().tolist())
        eval_epoch_preds.extend(preds)

    return eval_epoch_loss, eval_epoch_true, eval_epoch_preds


def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)


def train_loop(
        model, train_loader, val_loader,
        epochs,
        optimizer,
        scheduler,
        save_path,
        device,
        # criterion,
        use_wandb,
        log_interval,
        proj_name=None,
        run_name=None,
        model_name=None
):

    # variables for calculate metrics
    train_loss = []
    train_true = []
    train_preds = []
    train_f1 = []

    eval_loss = []
    eval_true = []
    eval_preds = []
    eval_f1 = []
    best_eval_f1 = float('-inf')

    if use_wandb:
        wandb.init(project=proj_name, name=run_name)

    for epoch in tqdm(range(epochs)):
        model.train(True)
        print()
        print(f"epoch: {epoch + 1}")
        run_train_loss, run_train_true, run_train_preds = fit_epoch(train_loader,
                                                                    model,
                                                                    device,
                                                                    optimizer,
                                                                    use_wandb,
                                                                    log_interval)
        run_f1 = f1_score(run_train_true, run_train_preds)
        train_loss.extend(run_train_loss)
        train_true.extend(run_train_true)
        train_preds.extend(run_train_preds)
        train_f1.append(run_f1)

        print(f"train_loss: {np.mean(run_train_loss)} train_f1: {train_f1[-1]}")
        if use_wandb:
            wandb.log({'train epoch loss': np.mean(run_train_loss),
                       'train epoch f1': run_f1})
        model.train(False)
        run_eval_loss, run_eval_true, run_eval_preds = eval_epoch(val_loader, model, device)

        run_eval_f1 = f1_score(run_eval_true, run_eval_preds)
        eval_loss.extend(run_eval_loss)
        eval_true.extend(run_eval_true)
        eval_preds.extend(run_eval_preds)
        eval_f1.append(run_eval_f1)

        print(f"eval_loss: {np.mean(run_eval_loss)} eval_f1: {run_eval_f1}")
        if run_eval_f1 != 0 and run_eval_f1 > best_eval_f1:
            best_eval_f1 = run_eval_f1

            print('>>> Best f1 score achievement, saving model to checkpoint')
            save_model(model, epoch, save_path, model_name, best_eval_f1)
            # torch.save(model.state_dict(), f"{save_path}_{timestamp}_{epoch}_best.pth")
        if use_wandb:
            wandb.log({'eval_loss': np.mean(run_eval_loss),
                       'eval_f1': run_eval_f1})

        scheduler.step()
    print('>>>> Saving last model state dict')
    save_model(model, epoch, save_path, model_name, run_eval_f1, last=True)
    # torch.save(model.state_dict(), f"{save_path}_last.pth")

    if use_wandb:
        wandb.finish()

    res_dct = {"train_loss": train_loss,
               "train_f1": train_f1,
               "valid_loss": eval_loss,
               "valid_f1": eval_f1}
    return res_dct


def save_model(model, epoch, save_path, model_name, eval_f1, last=False, ):
    eval_f1 = f"(f1_{str(eval_f1)[2:6]})"
    # make directory if not exist
    if not Path(save_path).exists():
        Path(save_path).mkdir()

    timestamp = datetime.now().strftime('%Y%m%d_%H_%M')
    filename = f"{save_path}/{model_name}_{timestamp}_{epoch}_{eval_f1}best.pth"
    if last:
        filename = f"{save_path}/{model_name}_{timestamp}_ep{epoch+1}_{eval_f1}_last.pth"
    torch.save(model.state_dict(), filename)
    print(f">>> model saved to {filename}")
