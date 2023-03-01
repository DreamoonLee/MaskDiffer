import torch
import random
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from data import create_loaders
from logger import Logger
from swin_transformer import swin_tiny_patch4_window7_224 as create_model
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score

from upload_s3 import multiup
from utils import y_true, mse, y_predicted, find_last_checkpoint_file
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from st import SoftTarget
import itertools

post_act = torch.nn.Softmax(dim=1)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def momentum_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def train_batch(inputs, labels, intensity, models, optimizer, criterion, criterion_intensity, criterion_kd):
    # models = {'t_clsmodel': t_clsmodel, 's_clsmodel': s_clsmodel, 't_regmodel': t_regmodel, 's_regmodel': s_regmodel}
    s_regmodel = models["s_regmodel"]
    s_clsmodel = models["s_clsmodel"]
    t_regmodel = models["t_regmodel"]
    t_clsmodel = models["t_clsmodel"]
    s_regmodel.train()
    s_clsmodel.train()
    reg_outputs_s, reg_out_intensity_s = s_regmodel(inputs)
    cls_outputs_s, cls_out_intensity_s = s_clsmodel(inputs)

    reg_outputs_t, reg_out_intensity_t = t_regmodel(inputs)
    cls_outputs_t, cls_out_intensity_t = t_clsmodel(inputs)

    loss_cls = criterion(cls_outputs_s, labels) + 0.2 * criterion(reg_outputs_s, labels)

    loss_intensity = 0.2 * criterion_intensity(cls_out_intensity_s, intensity.unsqueeze(1).float()) \
                     + criterion_intensity(reg_out_intensity_s, intensity.unsqueeze(1).float())

    kd_loss = criterion_kd(cls_outputs_s, reg_outputs_t.detach()) * 0.1 + \
              criterion_kd(cls_out_intensity_s, reg_out_intensity_t.detach()) * 0.1 + \
              criterion_kd(reg_outputs_s, cls_outputs_t.detach()) * 0.1 + \
              criterion_kd(reg_out_intensity_s, cls_out_intensity_t.detach()) * 0.1

    loss = loss_cls + loss_intensity + kd_loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    momentum_update(model=s_regmodel, model_ema=t_regmodel, m=0.999)
    momentum_update(model=s_clsmodel, model_ema=t_clsmodel, m=0.999)

    return loss.item()


@torch.no_grad()
def accuracy(inputs, labels, intensity, models):
    s_regmodel = models["s_regmodel"]
    s_clsmodel = models["s_clsmodel"]
    s_regmodel.eval()
    s_clsmodel.eval()
    _, outputs_intensity = s_regmodel(inputs)
    outputs_cls, _ = s_clsmodel(inputs)

    intensity_acc = r2_score(intensity.cpu().detach().numpy(), outputs_intensity.cpu().detach().numpy())
    mse = mean_squared_error(intensity.cpu().detach().numpy(), outputs_intensity.cpu().detach().numpy())

    # Get the predicted classes
    preds = post_act(outputs_cls)
    # print('preds: ', preds)
    _, pred_classes = torch.max(preds, 1)
    # print('pred_classes: ', pred_classes)
    # print('labels:', labels)
    is_correct = pred_classes == labels
    # print('is_correct: ', is_correct)
    return is_correct.cpu().numpy().tolist(), intensity_acc


@torch.no_grad()
def val_loss(inputs, labels, intensity, models, criterion, criterion_intensity, criterion_kd):
    s_regmodel = models["s_regmodel"]
    s_clsmodel = models["s_clsmodel"]
    t_regmodel = models["t_regmodel"]
    t_clsmodel = models["t_clsmodel"]
    s_regmodel.eval()
    s_clsmodel.eval()
    reg_outputs_s, reg_out_intensity_s = s_regmodel(inputs)
    cls_outputs_s, cls_out_intensity_s = s_clsmodel(inputs)

    reg_outputs_t, reg_out_intensity_t = t_regmodel(inputs)
    cls_outputs_t, cls_out_intensity_t = t_clsmodel(inputs)

    loss_cls = criterion(cls_outputs_s, labels) + 0.2 * criterion(reg_outputs_s, labels)

    loss_intensity = 0.2 * criterion_intensity(cls_out_intensity_s, intensity.unsqueeze(1).float()) \
                     + criterion_intensity(reg_out_intensity_s, intensity.unsqueeze(1).float())

    kd_loss = criterion_kd(cls_outputs_s, reg_outputs_t.detach()) * 0.1 + \
              criterion_kd(cls_out_intensity_s, reg_out_intensity_t.detach()) * 0.1 + \
              criterion_kd(reg_outputs_s, cls_outputs_t.detach()) * 0.1 + \
              criterion_kd(reg_out_intensity_s, cls_out_intensity_t.detach()) * 0.1

    loss = loss_cls + loss_intensity + kd_loss
    return loss.item()


def train_model(device,
                models,
                criterion,
                criterion_intensity,
                criterion_kd,
                optimizer,
                lr_scheduler,
                classification_dataloader_train,
                classification_dataloader_val,
                best_epoch,
                num_epoch,
                best_val_epoch_loss,
                checkpoint_dir,
                saving_dir_experiments,
                logger,
                epoch_start_unfreeze=None,
                block_start_unfreeze=None,
                aws_bucket=None,
                aws_directory=None):
    train_losses = []
    val_losses = []

    print("Start training")
    freezed = True
    for epoch in range(best_epoch, num_epoch):
        logger.log(f'Epoch {epoch}/{num_epoch - 1}')

        if epoch_start_unfreeze is not None and epoch >= epoch_start_unfreeze and freezed:
            print("****************************************")
            print("unfreeze the base model weights")
            if block_start_unfreeze is not None:
                print("unfreeze the layers greater and equal to unfreezing_block: ", block_start_unfreeze)
                # in this case unfreeze only the layers greater and equal the unfreezing_block layer
                for name, param in models["s_regmodel"].named_parameters():
                    param.requires_grad = True

            else:
                # in this case unfreeze all the layers of the model
                print("unfreeze all the layer of the model")
                for name, param in models["s_regmodel"].named_parameters():
                    param.requires_grad = True
            freezed = False

            for name, param in models["s_regmodel"].named_parameters():
                print("Layer name: {} - requires_grad: {}".format(name, param.requires_grad))
            print("****************************************")

        # define empty lists for the values of the loss and the accuracy of train and validation obtained in the batch of the current epoch
        # then at the end I take the average and I get the final values of the whole era
        train_epoch_losses, train_epoch_accuracies, i = [], [], 0.96 ** epoch
        val_epoch_losses, val_epoch_accuracies = [], []

        if epoch_start_unfreeze is not None and epoch >= epoch_start_unfreeze and freezed:
            print("****************************************")
            print("unfreeze the base model weights")

            if block_start_unfreeze is not None:
                print("unfreeze the layers greater and equal to unfreezing_block: ", block_start_unfreeze)
                # in this case unfreeze only the layers greater and equal the unfreezing_block layer
                for name, param in models["s_clsmodel"].named_parameters():
                    param.requires_grad = True

            else:
                # in this case unfreeze all the layers of the model
                print("unfreeze all the layer of the model")
                for name, param in models["s_clsmodel"].named_parameters():
                    param.requires_grad = True
            freezed = False

            for name, param in models["s_clsmodel"].named_parameters():
                print("Layer name: {} - requires_grad: {}".format(name, param.requires_grad))
            print("****************************************")

        # iterate on all train batches of the current epoch by executing the train_batch function
        for inputs, labels, intensity in tqdm(classification_dataloader_train, desc=f"epoch {str(epoch)} | train"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            intensity = intensity[0].to(device)
            batch_loss = train_batch(inputs, labels, intensity, models, optimizer, criterion, criterion_intensity,
                                     criterion_kd)
            train_epoch_losses.append(batch_loss)
        train_epoch_loss = np.array(train_epoch_losses).mean() * i

        for inputs, labels, intensity in tqdm(classification_dataloader_val, desc=f"epoch {str(epoch)} | val"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            intensity = intensity[0].to(device)
            intensity = intensity.to(device)
            validation_loss = val_loss(inputs, labels, intensity, models, criterion, criterion_intensity, criterion_kd)
            val_epoch_losses.append(validation_loss)
        val_epoch_loss = np.mean(val_epoch_losses) * i

        phase = 'train'
        logger.log(
            f'{phase} LR: {lr_scheduler.get_last_lr()} - Loss: {train_epoch_loss:.4f}')
        phase = 'val'
        logger.log(
            f'{phase} LR: {lr_scheduler.get_last_lr()} - Loss: {val_epoch_loss:.4f}')
        print("Epoch: {} - LR:{} - Train Loss: {:.4f} - Val Loss: {:.4f}".format(
            int(epoch), lr_scheduler.get_last_lr(), train_epoch_loss, val_epoch_loss))
        logger.log("-----------")

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

        print("Plot learning curves")

        if best_val_epoch_loss > val_epoch_loss:
            print("We have a new best model! Save the model")

            # update best_val_epoch_loss
            best_val_epoch_loss = val_epoch_loss

            save_obj = {
                's_regmodel': models["s_regmodel"].state_dict(),
                's_clsmodel': models["s_clsmodel"].state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_eva_loss': best_val_epoch_loss
            }
            print("Save best checkpoint at: ", os.path.join(checkpoint_dir, 'best.pth'))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'best.pth'), _use_new_zipfile_serialization=False)
            print("Save checkpoint at: ", os.path.join(checkpoint_dir, 'checkpoint_' + str(epoch) + '.pth'))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'checkpoint_' + str(epoch) + '.pth'),
                       _use_new_zipfile_serialization=False)

        else:
            print("Save the current model")

            save_obj = {
                's_regmodel': models["s_regmodel"].state_dict(),
                's_clsmodel': models["s_clsmodel"].state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_eva_loss': best_val_epoch_loss
            }
            print("Save checkpoint at: ", os.path.join(checkpoint_dir, 'checkpoint_' + str(epoch) + '.pth'))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'checkpoint_' + str(epoch) + '.pth'),
                       _use_new_zipfile_serialization=False)

        if aws_bucket is not None and aws_directory is not None:
            print('Upload on S3')
            multiup(aws_bucket, aws_directory, saving_dir_experiments)

        lr_scheduler.step()
        torch.cuda.empty_cache()
        print("---------------------------------------------------------")

    print("End training")
    return


def test_model(device,
               s_clsmodel,
               s_regmodel,
               classification_dataloader,
               path_save,
               class2label,
               type_dataset):
    y_test_true = []
    y_test_predicted = []
    mse_list = []
    total = 0
    s_clsmodel = s_clsmodel.eval()
    s_regmodel = s_regmodel.eval()

    with torch.no_grad():
        # cycle on all train batches of the current epoch by calculating their accuracy
        for inputs, labels, intensity in classification_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            intensity = intensity[0].to(device)

            outputs_cls, _ = s_clsmodel(inputs)
            _, outputs_reg = s_regmodel(inputs)
            # Get the predicted classes
            preds = post_act(outputs_cls)
            # print('preds: ', preds)
            _, pred_classes = torch.max(preds, 1)
            numero_video = len(labels.cpu().numpy().tolist())
            total += numero_video
            mean_squared_error(intensity.cpu().detach().numpy(), outputs_reg.cpu().detach().numpy())
            y_test_true.extend(labels.cpu().numpy().tolist())
            y_test_predicted.extend(pred_classes.cpu().numpy().tolist())
            mse_list.extend(mse)

        # report predictions and true values to numpy array
        print('Number of tested videos: ', total)
        y_test_true = np.array(y_test_true)
        y_test_predicted = np.array(y_test_predicted)
        print('y_test_true.shape: ', y_test_true.shape)
        print('y_test_predicted.shape: ', y_test_predicted.shape)

        print('mse: ', np.mean(mse_list))
        print('Accuracy: ', accuracy_score(y_true, y_predicted))
        print(metrics.classification_report(y_true, y_predicted))


def run_train_test_model(cfg, do_train, do_test, aws_bucket=None, aws_directory=None):
    seed_everything(42)
    checkpoint = None
    best_epoch = 0
    best_val_epoch_loss = 10

    dataset_path = cfg['dataset_path']
    path_dataset_train_csv = cfg['path_dataset_train_csv']
    path_dataset_val_csv = cfg['path_dataset_val_csv']
    path_dataset_test_csv = cfg.get("path_dataset_test_csv", None)

    saving_dir_experiments = cfg['saving_dir_experiments']
    saving_dir_model = cfg['saving_dir_model']

    num_epoch = cfg['num_epoch']
    learning_rate = cfg['learning_rate']
    scheduler_step_size = cfg['scheduler_step_size']
    scheduler_gamma = cfg['scheduler_gamma']
    epoch_start_unfreeze = cfg.get("epoch_start_unfreeze", None)
    block_start_unfreeze = cfg.get("block_start_unfreeze", None)

    # 1 - load csv dataset
    path_dataset_train_csv = os.path.join(dataset_path, path_dataset_train_csv)
    df_dataset_train = pd.read_csv(path_dataset_train_csv)
    path_dataset_val_csv = os.path.join(dataset_path, path_dataset_val_csv)
    df_dataset_val = pd.read_csv(path_dataset_val_csv)
    df_dataset_test = None
    if path_dataset_test_csv is not None:
        path_dataset_test_csv = os.path.join(dataset_path, path_dataset_test_csv)
        df_dataset_test = pd.read_csv(path_dataset_test_csv)

    # 2 -  create the directory to save the results and checkpoints
    print("Create the project structure")
    print("saving_dir_experiments: ", saving_dir_experiments)
    saving_dir_model = os.path.join(saving_dir_experiments, saving_dir_model)
    print("saving_dir_model: ", saving_dir_model)
    os.makedirs(saving_dir_experiments, exist_ok=True)
    os.makedirs(saving_dir_model, exist_ok=True)

    # 3 - load log configuration
    logger = Logger(exp_path=saving_dir_model)

    # 4 - create the dataloaders
    classification_dataloader_train, classification_dataloader_val, classification_dataloader_test = create_loaders(
        df_dataset_train, df_dataset_val, df_dataset_test, cfg)

    # 5 - set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # 6 - download the model
    # model = VideoClassificationModel(cfg).to(device)
    t_clsmodel = create_model(num_classes=7, mask_patch=True, mask_channel=False).to(device)
    s_clsmodel = create_model(num_classes=7, mask_patch=True, mask_channel=False).to(device)

    t_regmodel = create_model(num_classes=7, mask_patch=False, mask_channel=True).to(device)
    s_regmodel = create_model(num_classes=7, mask_patch=False, mask_channel=True).to(device)

    models = {'t_clsmodel': t_clsmodel, 's_clsmodel': s_clsmodel, 't_regmodel': t_regmodel, 's_regmodel': s_regmodel}

    checkpoint_dir = saving_dir_model

    if do_train:
        # summary of the model
        # summary(model, input_size=(3, 32, 256, 256))

        # 8 - Set the Loss, optimizer and scheduler. ( CrossEntropyLoss wants logits. Perform the softmax internally)
        criterion = nn.CrossEntropyLoss()
        criterion_intensity = torch.nn.MSELoss()
        criterion_kd = SoftTarget(T=4.0)

        parameters = itertools.chain(s_clsmodel.parameters(), s_regmodel.parameters())
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size,
                                                           gamma=scheduler_gamma)
        if checkpoint is not None:
            print('Load the optimizer from the last checkpoint')
            optimizer.load_state_dict(checkpoint['optimizer'])
            exp_lr_scheduler.load_state_dict(checkpoint["scheduler"])

            print('Latest epoch of the checkpoint: ', checkpoint['epoch'])
            print('Setting the new starting epoch: ', checkpoint['epoch'] + 1)
            best_epoch = checkpoint['epoch'] + 1

            print('Setting best_val_epoch_loss from checkpoint: ', checkpoint['best_eva_accuracy'])
            best_val_epoch_loss = checkpoint['best_eva_accuracy']

        # 9 - run train model function
        train_model(device=device,
                    models=models,
                    criterion=criterion,
                    criterion_intensity=criterion_intensity,
                    criterion_kd=criterion_kd,
                    optimizer=optimizer,
                    lr_scheduler=exp_lr_scheduler,
                    classification_dataloader_train=classification_dataloader_train,
                    classification_dataloader_val=classification_dataloader_val,
                    best_epoch=best_epoch,
                    num_epoch=num_epoch,
                    best_val_epoch_loss=best_val_epoch_loss,
                    checkpoint_dir=checkpoint_dir,
                    saving_dir_experiments=saving_dir_experiments,
                    logger=logger,
                    epoch_start_unfreeze=epoch_start_unfreeze,
                    block_start_unfreeze=block_start_unfreeze,
                    aws_bucket=aws_bucket,
                    aws_directory=aws_directory)
        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")

    if do_test:

        print("Execute Inference on Train, Val and Test Dataset with best checkpoint")

        path_last_checkpoint = find_last_checkpoint_file(checkpoint_dir=checkpoint_dir, use_best_checkpoint=True)
        if path_last_checkpoint is not None:
            print("Upload the best checkpoint at the path: ", path_last_checkpoint)
            checkpoint = torch.load(path_last_checkpoint, map_location=torch.device(device))
            s_regmodel.load_state_dict(checkpoint['s_regmodel'])
            s_clsmodel.load_state_dict(checkpoint['s_clsmodel'])
            s_clsmodel = s_clsmodel.to(device)
            s_regmodel = s_regmodel.to(device)

            class2label = {}
            # go through the lines of the dataset
            for index, row in df_dataset_train.iterrows():
                class_name = row["CLASS"]
                label = row["LABEL"]

                if class_name not in class2label:
                    class2label[class_name] = label

            print("class2label: ", class2label)

            print("-------------------------------------------------------------------")
            print("-------------------------------------------------------------------")

            if classification_dataloader_test is not None:
                print("-------------------------------------------------------------------")
                print("-------------------------------------------------------------------")

                print("Inference on test dataset")
                test_model(device=device,
                           s_clsmodel=s_clsmodel,
                           s_regmodel=s_regmodel,
                           classification_dataloader=classification_dataloader_test,
                           path_save=checkpoint_dir,
                           class2label=class2label,
                           type_dataset="test")

            if aws_bucket is not None and aws_directory is not None:
                print("Final upload on S3")
                multiup(aws_bucket, aws_directory, saving_dir_experiments)
