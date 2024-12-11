## This implements Moment Matching by Google Deepmind.
# "Multistep Distillation of Diffusion Models via Moment Matching"
# https://arxiv.org/abs/2406.04103
# Using LLama-DiT arch, Sub VP scheduler (Rectified Flow), and CIFAR-10 dataset on single GPU


import copy
import os
import subprocess
from contextlib import nullcontext
from datetime import datetime

import click
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from train_student import RF, setup_trainings

device = "cuda" if torch.cuda.is_available() else "cpu"


class RFMMD:

    # Moment Matching
    # This implements algorithm 2 of the paper, using 2 trainable model student and aux, distilled from one teacher diffusion model.
    # k = 8 in the paper.

    def __init__(self, teacher_model, student_model, aux_model, k=8):

        self.teacher = teacher_model
        self.student = student_model
        self.aux = aux_model
        self.k = k
        self.ln = True
        self.counter = 0

    def loss(self, x, cond):

        train_student = self.counter % 2 == 0
        self.counter += 1

        b = x.size(0)
        if self.ln:
            s_time = torch.sigmoid(torch.randn((b,)).to(x.device))
        else:
            s_time = torch.rand((b,)).to(x.device)

        # we first sample delta time, and sample t. notice that s < t.
        t_time = s_time + torch.rand_like(s_time) * 1 / self.k
        t_time = t_time.clip(0, 1)

        t_expanded = t_time.view([b, *([1] * len(x.shape[1:]))])
        s_expanded = s_time.view([b, *([1] * len(x.shape[1:]))])

        z_1 = torch.randn_like(x)
        # we now get z_t from x.
        z_t = (1 - t_expanded) * x + t_expanded * z_1

        # now we sample x from student and z_s from x. grad flows only when we train student ofc.
        with torch.no_grad() if not train_student else nullcontext():
            x_student = self.student.x_prediction(z_t, t_time, cond)

        # now we use x_student to sample z_s.
        noise_for_z_s = (z_t - x_student * (1 - t_expanded)) / t_expanded
        z_s = (1 - s_expanded) * x_student + noise_for_z_s * s_expanded

        with torch.no_grad() if train_student else nullcontext():
            x_aux = self.aux.x_prediction(z_s, s_time, cond)

        # finally, we need actual x prediction from teacher.
        with torch.no_grad():
            x_teacher = self.teacher.x_prediction(z_s, s_time, cond)

        if not train_student:
            loss = ((x_student - x_aux) ** 2).mean() + ((x_teacher - x_aux) ** 2).mean()
        else:
            loss = (x_student * (x_aux - x_teacher)).mean()

        return loss


def launch_tensorboard(logdir):
    tb_process = subprocess.Popen(["tensorboard", "--logdir", logdir, "--port", "6006"])
    print(f"TensorBoard launched. Open http://localhost:6006 to view")
    return tb_process


@click.command()
@click.option("--cifar", is_flag=True, help="Use CIFAR-10 dataset")
@click.option(
    "--ckpt_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="./checkpoints",
    help="Directory to save checkpoints and TensorBoard logs",
)
@click.option("--epochs", type=int, default=100, help="Number of epochs to train")
@click.option("--batch_size", type=int, default=256, help="Batch size for training")
@click.option("--lr", type=float, default=1e-4, help="Learning rate")
@click.option("--init_ckpt_dir", type=str, default="checkpoints/model_epoch_44.pt")
def main(cifar, ckpt_dir, epochs, batch_size, lr, init_ckpt_dir):
    # Set up datasets and models based on the chosen dataset

    model, fdatasets = setup_trainings("cifar" if cifar else "mnist")
    channels = model.in_channels
    transform = model.transform

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    ckpt_state = torch.load(init_ckpt_dir, map_location="cpu")
    model.load_state_dict(ckpt_state["model_state_dict"])
    rf_teacher = RF(model)
    rf_student = RF(copy.deepcopy(model))
    rf_aux = RF(copy.deepcopy(model))

    rfmmd = RFMMD(rf_teacher, rf_student, rf_aux, k=8)

    rf_teacher.model.requires_grad_(False)
    optimizer = optim.Adam(
        list(rf_student.model.parameters()) + list(rf_aux.model.parameters()), lr=lr
    )

    dataset = fdatasets(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Set up TensorBoard
    log_dir = os.path.join(ckpt_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)

    # Launch TensorBoard
    tb_process = launch_tensorboard(os.path.join(ckpt_dir, "logs"))

    try:
        # Training loop
        for epoch in range(epochs):

            model.eval()
            with torch.no_grad():
                cond = torch.arange(0, 16).to(device) % 10
                uncond = torch.ones_like(cond) * 10

                init_noise = torch.randn(16, channels, 32, 32).to(device)
                images = rf_student.anc_sample(
                    init_noise, cond, uncond, sample_steps=8, cfg=1.0
                )

                # Log the final generated images
                final_images = images[-1]
                final_images = final_images * 0.5 + 0.5  # unnormalize
                final_images = final_images.clamp(0, 1)
                grid = make_grid(final_images, nrow=4)
                writer.add_image("Generated Images", grid, epoch)

                # Save the GIF
                gif = []
                for image in images:
                    image = image * 0.5 + 0.5  # unnormalize
                    image = image.clamp(0, 1)
                    x_as_image = make_grid(image.float(), nrow=4)
                    img = x_as_image.permute(1, 2, 0).cpu().numpy()
                    img = (img * 255).astype(np.uint8)
                    gif.append(Image.fromarray(img))

                gif[0].save(
                    os.path.join(ckpt_dir, f"rfmmd_sample_{epoch}.gif"),
                    save_all=True,
                    append_images=gif[1:],
                    duration=100,
                    loop=0,
                )

            lossbin = {i: 0 for i in range(10)}
            losscnt = {i: 1e-6 for i in range(10)}
            model.train()
            for i, (x, c) in tqdm(enumerate(dataloader), total=len(dataloader)):
                x, c = x.to(device), c.to(device)
                optimizer.zero_grad()
                loss = rfmmd.loss(x, c)
                loss.backward()
                optimizer.step()

                writer.add_scalar(
                    "Loss/train", loss.item(), epoch * len(dataloader) + i
                )

                # # count based on t
                # for t, l in blsct:
                #     lossbin[int(t * 10)] += l
                #     losscnt[int(t * 10)] += 1

            # Log loss bins
            for i in range(10):
                bin_loss = lossbin[i] / losscnt[i]
                writer.add_scalar(f"Loss/bin_{i}", bin_loss, epoch)
                print(f"Epoch: {epoch}, {i} range loss: {bin_loss}")

            # Save model checkpoint
            checkpoint_path = os.path.join(ckpt_dir, f"rfmmd_model_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "student_state_dict": rf_student.model.state_dict(),
                    "aux_state_dict": rf_aRux.model.state_dict(),
                    "loss": loss,
                },
                checkpoint_path,
            )

    finally:
        writer.close()
        # Terminate TensorBoard process
        tb_process.terminate()
        print("TensorBoard process terminated.")


if __name__ == "__main__":
    main()
