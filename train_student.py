import os
import subprocess
from datetime import datetime

import click
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dit import DiT_Llama

device = "cuda" if torch.cuda.is_available() else "cpu"


class RF:
    def __init__(self, model, ln=True):
        self.model = model
        self.ln = ln

    def forward(self, x, cond):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)

        t_expanded = t.view([b, *([1] * len(x.shape[1:]))])

        z_1 = torch.randn_like(x)
        z_t = (1 - t_expanded) * x + t_expanded * z_1

        vtheta = self.model(z_t, t, cond)  # this is supposed to predict z_1 - x.

        # rest is housekeeping actually.
        loss = ((z_1 - x - vtheta) ** 2).mean()

        with torch.no_grad():
            batchwise_mse = ((z_1 - x - vtheta) ** 2).mean(
                dim=list(range(1, len(x.shape)))
            )
            tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
            ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]

        return loss, ttloss

    def x_prediction(self, z_t, t: torch.Tensor, cond):
        # at time t, we get z_t = (1 - t) x + t n
        # because our predictor is simply n - x = v
        # z_t - v t would do.
        # print(z_t.shape, t.shape, cond.shape)
        v_theta = self.model(z_t, t, cond)
        return z_t - v_theta * t.view(-1, *([1] * len(z_t.shape[1:])))

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images

    @torch.no_grad()
    def anc_sample(self, z, cond, null_cond, sample_steps=50, cfg=2.0):
        # sample via ancestral. This is identical to ODE sampler like above, but parameterized around x so MMD can be applicable.
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        init_noise = z.clone()
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)
            t_expanded = t.view([b, *([1] * len(z.shape[1:]))])

            x_t = self.x_prediction(z, t, cond)
            if null_cond is not None:
                x_ut = self.x_prediction(z, t, null_cond)
                x_t = x_ut + cfg * (x_t - x_ut)

            noise_for_z_s = (z - x_t * (1 - t_expanded)) / t_expanded

            t_next = t_expanded - dt

            z = (1 - t_next) * x_t + noise_for_z_s * t_next

            images.append(z)

        return images

    @torch.no_grad()
    def anc_sample_from_paper(self, z, cond, null_cond, sample_steps=50, cfg=2.0):

        # sample via ancstral from paper. for some reason, it's not working.
        # yet to debug...

        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        init_noise = z.clone()
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)
            t_expanded = t.view([b, *([1] * len(z.shape[1:]))])

            x_t = self.x_prediction(z, t, cond)
            if null_cond is not None:
                x_ut = self.x_prediction(z, t, null_cond)
                x_t = x_ut + cfg * (x_t - x_ut)

            noise_for_z_s = torch.randn_like(z)
            t_next = t_expanded - dt

            alpha_t = 1 - t_expanded
            sigma_t = t_expanded

            alpha_s = 1 - t_next
            sigma_s = t_next

            alpha_t_s = alpha_t / alpha_s
            sigma_t_s = sigma_t / sigma_s

            sigma_t_to_s_squared = (sigma_s**-2 + alpha_t_s**2 / sigma_t_s**2) ** -1
            mu_t_s = sigma_t_to_s_squared * (
                (alpha_t_s / (sigma_t_s**2)) * z + alpha_s * x_t / (sigma_s**2)
            )
            z = mu_t_s + (sigma_t_to_s_squared**0.5) * noise_for_z_s

            # alpha, sigma formulation

            images.append(z)

        return images


def setup_trainings(dataset_type: str):
    if dataset_type == "cifar":
        channels = 3
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        model = DiT_Llama(
            channels, 32, dim=512, n_layers=8, n_heads=8, num_classes=10
        ).to(device)
        model.transform = transform
        fdatasets = datasets.CIFAR10

    else:
        channels = 1
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        model = DiT_Llama(
            channels, 32, dim=32, n_layers=3, n_heads=2, num_classes=10
        ).to(device)
        model.transform = transform
        fdatasets = datasets.MNIST

    return model, fdatasets


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
@click.option("--batch_size", type=int, default=64, help="Batch size for training")
@click.option("--lr", type=float, default=5e-4, help="Learning rate")
def main(cifar, ckpt_dir, epochs, batch_size, lr):
    # Set up datasets and models based on the chosen dataset

    model, fdatasets = setup_trainings("cifar" if cifar else "mnist")
    transform = model.transform

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    rf = RF(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema_model_state = model.state_dict()

    dataset = fdatasets(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=32
    )

    # Set up TensorBoard
    log_dir = os.path.join(ckpt_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)

    # Launch TensorBoard
    tb_process = launch_tensorboard(os.path.join(ckpt_dir, "logs"))

    try:
        # Training loop
        for epoch in range(epochs):
            lossbin = {i: 0 for i in range(10)}
            losscnt = {i: 1e-6 for i in range(10)}

            model.train()
            for i, (x, c) in tqdm(enumerate(dataloader), total=len(dataloader)):
                x, c = x.to(device), c.to(device)
                optimizer.zero_grad()
                loss, blsct = rf.forward(x, c)
                loss.backward()
                optimizer.step()

                writer.add_scalar(
                    "Loss/train", loss.item(), epoch * len(dataloader) + i
                )

                if i % 4 == 0:
                    for name, param in model.named_parameters():
                        ema_model_state[name].data.mul_(0.99).add_(
                            param.data, alpha=0.01
                        )

                # count based on t
                for t, l in blsct:
                    lossbin[int(t * 10)] += l
                    losscnt[int(t * 10)] += 1

            # Log loss bins
            for i in range(10):
                bin_loss = lossbin[i] / losscnt[i]
                writer.add_scalar(f"Loss/bin_{i}", bin_loss, epoch)
                print(f"Epoch: {epoch}, {i} range loss: {bin_loss}")

            # Generate and log images
            model.eval()
            rf.model.load_state_dict(ema_model_state)
            with torch.no_grad():
                cond = torch.arange(0, 16).to(device) % 10
                uncond = torch.ones_like(cond) * 10

                init_noise = torch.randn(
                    16, model.in_channels, model.input_size, model.input_size
                ).to(device)
                images = rf.sample(init_noise, cond, uncond)

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
                    os.path.join(ckpt_dir, f"sample_{epoch}.gif"),
                    save_all=True,
                    append_images=gif[1:],
                    duration=100,
                    loop=0,
                )

            # Save model checkpoint
            checkpoint_path = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": ema_model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
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
