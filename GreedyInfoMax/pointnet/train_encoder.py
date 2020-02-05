import torch
import time
import numpy as np

#### own modules
from GreedyInfoMax.utils.better_logger import Logger
from GreedyInfoMax.pointnet.arg_parser import arg_parser
from GreedyInfoMax.pointnet.models import load_pointnet_model
from GreedyInfoMax.pointnet.data import get_dataloader

def validate(opt, model, test_loader):
    total_step = len(test_loader)

    loss_epoch = [0 for _ in range(opt.model_splits)]
    starttime = time.time()

    for step, (img, label) in enumerate(test_loader):

        model_input = img.to(opt.device)

        loss, _ = model(model_input)
        loss = torch.mean(loss, 0)

        loss_epoch += loss.data.cpu().numpy()

    for i in range(opt.model_splits):
        print(
            "Validation Loss Model {}: Time (s): {:.1f} --- {:.4f}".format(
                i, time.time() - starttime, loss_epoch[i] / total_step
            )
        )

    validation_loss = [x/total_step for x in loss_epoch]
    return validation_loss

def train(opt, model):
    total_step = len(train_loader)

    print_idx = 100

    starttime = time.time()
    cur_train_module = opt.train_module

    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch):

        loss_epoch = [0 for _ in range(opt.model_splits)]
        loss_updates = [1 for _ in range(opt.model_splits)]

        for step, (img, label) in enumerate(train_loader):

            if step % print_idx == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Training Block: {}, Time (s): {:.1f}".format(
                        epoch + 1,
                        opt.num_epochs + opt.start_epoch,
                        step,
                        total_step,
                        cur_train_module,
                        time.time() - starttime,
                    )
                )

            starttime = time.time()

            model_input = img.to(opt.device)

            loss, _ = model(model_input)
            loss = torch.mean(loss, 0)  # take mean over outputs of different GPUs

            if cur_train_module != opt.model_splits and opt.model_splits > 1:
                loss = loss[cur_train_module].unsqueeze(0)

            # loop through the losses of the modules and do gradient descent
            for idx, cur_losses in enumerate(loss):
                if len(loss) == 1 and opt.model_splits != 1:
                    idx = cur_train_module

                model.zero_grad()

                if idx == len(loss) - 1:
                    cur_losses.backward()
                else:
                    cur_losses.backward(retain_graph=True)
                optimizer[idx].step()

                print_loss = cur_losses.item()
                if step % print_idx == 0:
                    print("\t \t Loss: \t \t {:.4f}".format(print_loss))

                loss_epoch[idx] += print_loss
                loss_updates[idx] += 1

        for i in range(len(loss_epoch)):
            logs.log_scalar(loss_epoch[i]/total_step, "train_loss_{}".format(i))

        if opt.validate:
            validation_loss = validate(opt, model, test_loader)
            for i in range(len(validation_loss)):
                logs.log_scalar(validation_loss[i], "val_loss_{}".format(i))

        for i in range(len(loss_epoch)):
            logs.draw_losses(module_num=i)

        logs.save_encoder_model(model, epoch=epoch)
        logs.save_optimizer(optimizer, epoch=0)

if __name__ == "__main__":
    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt)
    print(opt)

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    # load model and optimizer
    model, optimizer = load_pointnet_model.load_model_and_optimizer(opt)

    logs = Logger(opt)
    logs.save_opt()

    train_loader, _, supervised_loader, _, test_loader, _ = get_dataloader.get_dataloader(opt)

    if opt.loss == "supervised":
        train_loader = supervised_loader

    try:
        # Train the model
        train(opt, model)
    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")
