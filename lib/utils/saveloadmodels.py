import shutil
import json
import os
import torch
import torch.nn as nn
from collections import Counter


def print_model_layers(model, DataParallel=False):
    if DataParallel:
        for name in model.module.model_layers:
            print(name)
    else:
        for name in model.model_layers:
            print(name)


def get_model_state(model, DataParallel=False):
    model_state = {}
    if DataParallel:
        for name in model.module.model_layers:
            print("get_model_state model_layers:", name)
            model_state[name] = model.module.model_layers[name].state_dict()
    else:
        for name in model.model_layers:
            model_state[name] = model.model_layers[name].state_dict()
    return model_state


def load_model(path_checkpoint,  model, optimizer, DataParallel=False, Filter_layers = None):

    if os.path.isfile(path_checkpoint):
        print(f"=> loading checkpoint {path_checkpoint}\n")
        checkpoint = torch.load(path_checkpoint)

        data_state = checkpoint['data_state']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model_state = checkpoint['model_state']
        print(f"DataParallel: {DataParallel}\n")
        #print(f"model: {model.module.model_layers}\n")
        if DataParallel:
            for name in model.module.model_layers:

                if name in Filter_layers: continue
                model.module.model_layers[name].load_state_dict(model_state[name])
        else:
            for name in model.model_layers:
                if name in Filter_layers: continue
                model.model_layers[name].load_state_dict(model_state[name])

    else:
        print(f"=> no checkpoint found at {path_checkpoint}")

    return (model, optimizer, data_state)

def save_timepoint(ID, path_checkpoint, model, optimizer,  DataParallel=False , data_state = None):
    filename = f'{path_checkpoint}/{ID}.ckpt.pth.tar'
    model_state = get_model_state(model,DataParallel = DataParallel)
    torch.save({
        'data_state': data_state,
        'model_state': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)



def checkpoint_loss( ID, path_checkpoint, epoch, best_score, current_score,  model, optimizer, DataParallel = False ):

    is_best_score = current_score < best_score
    best_score = min(current_score, best_score)
    data_state = {'epoch': epoch + 1, f'best_score {ID}': best_score}

    print("checkpoint_loss", data_state)

    save_timepoint(ID, path_checkpoint, model, optimizer, data_state=data_state, DataParallel=DataParallel)


    return best_score, is_best_score

def checkpoint_acc( ID, path_checkpoint, epoch, best_score, current_score,  model, optimizer, DataParallel = False ):

    is_best_score = current_score > best_score
    best_score = max(current_score, best_score)
    data_state = {'epoch': epoch + 1, f'best_score {ID}': best_score}

    print(f"checkpoint_acc is_best_score: {is_best_score}, current_score {current_score} , best_score {best_score}")

    if is_best_score:
        print(f"checkpoint save: {ID} , {data_state}")
        save_timepoint(ID, path_checkpoint, model, optimizer, data_state=data_state, DataParallel=DataParallel)



    return best_score, is_best_score


def checkpoint_confusion_matrix( ID, path_checkpoint, epoch, best_score, current_score, accE, conf_matrix  ):

    is_best_score = current_score > best_score
    best_score = max(current_score, best_score)
    data_state = {'epoch': epoch + 1, f'accE': accE, f'conf_matrix': conf_matrix}


    if is_best_score:
        print("checkpoint_acc is_best_score", is_best_score, ID, data_state)
        filename = f'{path_checkpoint}/{ID}.multiclass_confusion'
        torch.save({'data_state': data_state}, filename)


    return best_score, is_best_score



