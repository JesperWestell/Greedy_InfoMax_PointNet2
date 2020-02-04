import torch
import numpy as np
import time
import os

## own modules
from GreedyInfoMax.pointnet.data import get_dataloader
from GreedyInfoMax.pointnet.arg_parser import arg_parser
from GreedyInfoMax.pointnet.models import load_pointnet_model
from GreedyInfoMax.utils import utils

from GreedyInfoMax.utils.better_logger import Logger


def train_logistic_regression(opt, context_model, predict_model):
    total_step = len(train_loader)

    starttime = time.time()

    for epoch in range(opt.num_epochs):
        predict_model.train()
        if opt.loss == "supervised":
            context_model.train()

        epoch_acc1 = 0
        epoch_acc5 = 0

        loss_epoch = 0
        for step, (img, target) in enumerate(train_loader):

            context_model.zero_grad()
            predict_model.zero_grad()

            model_input = img.to(opt.device)

            if opt.loss == "supervised":  ## fully supervised training
                _, features = context_model(model_input)
            else:
                with torch.no_grad():
                    _, features = context_model(model_input)
                features = features.detach() #double security that no gradients go to representation learning part of model

            prediction = predict_model(features.squeeze(-1))  # (B, num_classes)

            target = target.to(opt.device).squeeze(-1)  # (B)
            loss = criterion(prediction, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy
            acc1, acc5 = utils.accuracy(prediction.data, target, topk=(1, 5))
            epoch_acc1 += acc1
            epoch_acc5 += acc5

            sample_loss = loss.item()
            loss_epoch += sample_loss

            if step % 10 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}, Acc1: {:.4f}, Acc5: {:.4f}, Loss: {:.4f}".format(
                        epoch + 1,
                        opt.num_epochs,
                        step,
                        total_step,
                        time.time() - starttime,
                        acc1,
                        acc5,
                        sample_loss,
                    )
                )
                starttime = time.time()

        if opt.validate:
            # validate the model - in this case, test_loader loads validation data
            val_acc1, val_acc5, val_loss = test_logistic_regression(
                opt, context_model, predict_model
            )
            logs.log_scalar(val_loss, "val_loss_0")
            logs.log_scalar(val_acc1, "val_acc1")
            logs.log_scalar(val_acc5, "val_acc5")

        print("Overall accuracy for this epoch: ", epoch_acc1 / total_step)

        logs.log_scalar(loss_epoch / total_step, "train_loss_0")
        logs.log_scalar(epoch_acc1 / total_step, "train_acc1")
        logs.log_scalar(epoch_acc5 / total_step, "train_acc5")

        logs.draw_losses(0)
        logs.draw_accs()

        logs.save_encoder_model(context_model, epoch=epoch)
        logs.save_classifier(predict_model, epoch=epoch)
        logs.save_optimizer(optimizer, epoch=0)


def test_logistic_regression(opt, context_model, predict_model):
    total_step = len(test_loader)
    context_model.eval()
    predict_model.eval()

    starttime = time.time()

    loss_epoch = 0
    epoch_acc1 = 0
    epoch_acc5 = 0

    for step, (img, target) in enumerate(test_loader):

        model_input = img.to(opt.device)

        with torch.no_grad():
            _, features = context_model(model_input)
        features = features.detach()
        with torch.no_grad():
            prediction = predict_model(features.squeeze(-1))

        target = target.to(opt.device).squeeze(-1)
        loss = criterion(prediction, target)

        # calculate accuracy
        acc1, acc5 = utils.accuracy(prediction.data, target, topk=(1, 5))
        epoch_acc1 += acc1
        epoch_acc5 += acc5

        sample_loss = loss.item()
        loss_epoch += sample_loss

        if step % 10 == 0:
            print(
                "Step [{}/{}], Time (s): {:.1f}, Acc1: {:.4f}, Acc5: {:.4f}, Loss: {:.4f}".format(
                    step, total_step, time.time() - starttime, acc1, acc5, sample_loss
                )
            )
            starttime = time.time()

    print("Testing Accuracy: ", epoch_acc1 / total_step)
    return epoch_acc1 / total_step, epoch_acc5 / total_step, loss_epoch / total_step

if __name__ == "__main__":

    opt = arg_parser.parse_args()
    add_path_var = "linear_model"
    arg_parser.create_log_path(opt, add_path_var=add_path_var)
    print(opt)

    assert opt.loss in ["supervised", "classifier"], "Invalid --loss argument! One of [supervised, classifier]!"

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # load context model
    reload_model = True if opt.loss == "classifier" else False
    context_model, _ = load_pointnet_model.load_model_and_optimizer(opt, reload_model=reload_model)

    classification_model = load_pointnet_model.load_classification_model(opt)

    _, _, train_loader, _, test_loader, _ = get_dataloader.get_dataloader(opt)

    if opt.loss == "supervised":
        params = list(context_model.parameters()) + list(classification_model.parameters())
    else:
        params = classification_model.parameters()

    optimizer = torch.optim.Adam(params)
    criterion = torch.nn.CrossEntropyLoss()

    logs = Logger(opt)
    logs.save_opt()

    try:
        # Train the model
        train_logistic_regression(opt, context_model, classification_model)

        # Test the model
        acc1, acc5, _ = test_logistic_regression(
            opt, context_model, classification_model
        )
        logs.save_to_txt([acc1, acc5], ["Final acc1", "Final acc5"], "final_test")

    except KeyboardInterrupt:
        print("Training got interrupted")
