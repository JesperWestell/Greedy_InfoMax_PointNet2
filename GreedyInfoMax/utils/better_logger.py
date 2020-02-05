import glob, os
import torch
import matplotlib.pyplot as plt
import numpy as np
import copy


class Logger:
    def __init__(self, opt):
        self.opt = opt
        self.scalars = {}
        self.num_models_to_keep = 1

        if opt.validate:
            self.val_loss = [[] for i in range(opt.model_splits)]
        else:
            self.val_loss = None

        self.train_loss = [[] for i in range(opt.model_splits)]

        if opt.start_epoch > 0:
            for file in glob.glob(os.path.join(opt.model_path, "*.npy")):
                print("Loading logs from file:", file)
                data = np.load(file).tolist()
                self.scalars[os.path.splitext(os.path.basename(file))[0]] = copy.deepcopy(data)

    def save_encoder_model(self, model, epoch=0):
        self.save_model(model, epoch, name="model")

    def save_classifier(self, model, epoch=0):
        self.save_model(model, epoch, name="classification_model")

    def save_optimizer(self, optim, epoch=0):
        if not isinstance(optim, list):
            optim = [optim]
        for i, o in enumerate(optim):
            self.save_model(o, epoch, name="optim_{}".format(i))

    def save_model(self, model, epoch=0, name="model"):
        print("Saving {} to ".format(name) + self.opt.log_path)

        # Save the model checkpoint
        if self.opt.experiment == "vision" and name == "model":
            for idx, layer in enumerate(model.module.encoder):
                torch.save(
                    layer.state_dict(),
                    os.path.join(self.opt.log_path, "{}_{}_{}.ckpt".format(name, idx, epoch)),
                )
        else:
            torch.save(
                model.state_dict(),
                os.path.join(self.opt.log_path, "{}_{}.ckpt".format(name, epoch)),
            )

        ### remove old model files to keep dir uncluttered
        if (epoch - self.num_models_to_keep) % 10 != 0:
            try:
                if self.opt.experiment == "vision" and name == "model":
                    for idx, _ in enumerate(model.module.encoder):
                        os.remove(
                            os.path.join(
                                self.opt.log_path,
                                "{}_{}_{}.ckpt".format(name, idx, epoch - self.num_models_to_keep),
                            )
                        )
                else:
                    os.remove(
                        os.path.join(
                            self.opt.log_path,
                            "{}_{}.ckpt".format(name, epoch - self.num_models_to_keep),
                        )
                    )
            except:
                print("not enough models there yet, nothing to delete")

    def log_scalar(self, s, name, draw_plot=False):
        if name not in self.scalars:
            self.scalars[name] = []
        if not isinstance(s, list):
            s = [s]
        for elem in s:
            self.scalars[name].append(elem)

        # Save numpy arrays
        np.save(os.path.join(self.opt.log_path, name), np.array(self.scalars[name]))

        if draw_plot:
            self.draw_scalars(self.scalars[name], name, name)

    def draw_losses(self, module_num=0):
        loss_names = [k for k in self.scalars.keys() if "loss_{}".format(module_num) in k]
        losses = [self.scalars[k] for k in loss_names]
        self.draw_scalars(losses, loss_names, "loss_{}".format(module_num))

    def draw_accs(self, topks=[1,5]):
        for topk in topks:
            acc_names = [k for k in self.scalars.keys() if "acc{}".format(topk) in k]
            if len(acc_names) == 0:
                continue
            accs = [self.scalars[k] for k in acc_names]
            self.draw_scalars(accs, acc_names, "acc{}".format(topk))

    def draw_scalars(self, scalar_list, scalar_names, file_name):
        if not isinstance(scalar_names, list):
            scalar_list = [scalar_list]
            scalar_names = [scalar_names]
        colors = ["-b", "-g", "-r"]
        for idx, (scalars, name) in enumerate(zip(scalar_list, scalar_names)):
            lst_iter = np.arange(len(scalars))
            plt.plot(lst_iter, np.array(scalars), colors[idx], label=name)

        plt.xlabel("epoch")
        plt.ylabel(file_name)
        if scalars[0] < scalars[-1]:  # Quick and dirty but gets the job done
            plt.legend(loc="lower right")
        else:
            plt.legend(loc="upper right")

        # save image
        plt.savefig(os.path.join(self.opt.log_path, "{}.png".format(file_name)))
        plt.close()

    def save_opt(self):
        with open(os.path.join(self.opt.log_path, "opt.txt"), "w+") as cur_file:
            cur_file.write(str(self.opt))

    def save_to_txt(self, scalars, names, file_name):
        if not isinstance(names, list):
            scalars = [scalars]
            names = [names]
        with open(os.path.join(self.opt.log_path, "{}.txt".format(file_name)), "w+") as cur_file:
            for scalar, name in zip(scalars, names):
                cur_file.write("{}: {}".format(name, str(scalar)))

