from unittest.mock import _ArgsKwargs
from accelerate import Accelerator
from tqdm.auto import tqdm
import os
import torch.nn.functional as F
from PIL import Image
import os
import torch
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

from dataset import load_dataset
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/carotid', help='Directory for the dataset')
    parser.add_argument('--data_type', type=str, default='carotid', help='Type of the data')
    parser.add_argument('--save_dir', type=str, default='checkpoints/carotid_ddpm', help='Directory for output checkpoints')

    parser.add_argument('--image_size', type=int, default=256, help='Generated image resolution')
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=1, help='Number of output channels')

    parser.add_argument('--layers_per_block', type=int, default=2, help='Number of layers per block in UNet')
    parser.add_argument('--block_out_channels', type=int, nargs='+', default=[128, 128, 256, 256, 512, 512], help='Output channels in each block')
    parser.add_argument('--down_block_types', type=str, nargs='+', default=["DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"], help='Types of down-sampling blocks')
    parser.add_argument('--up_block_types', type=str, nargs='+', default=["UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"], help='Types of up-sampling blocks')

    parser.add_argument('--train_batch_size', type=int, default=20, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Evaluation batch size')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_warmup_steps', type=int, default=500, help='Number of warm-up steps for learning rate')
    parser.add_argument('--save_image_epochs', type=int, default=50, help='Frequency of saving images during training (in epochs)')
    parser.add_argument('--save_model_epochs', type=int, default=50, help='Frequency of saving model during training (in epochs)')
    parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16'], help='Use mixed precision training or not')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    return parser.parse_args()


def main():
    args = get_args()
    print(f"train on {args.data_type} data \nsave to {args.save_dir}")

    train_dataloader = load_dataset(
        data_dir=args.data_dir,
        batch_size=args.train_batch_size,
        image_size=args.image_size,
        data_type=args.data_type,
    )

    model = UNet2DModel(
        sample_size=args.image_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        layers_per_block=args.layers_per_block,
        block_out_channels=args.block_out_channels,
        down_block_types=args.down_block_types,
        up_block_types=args.up_block_types
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    train_loop(
        args=args,
        model=model,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        lr_scheduler=lr_scheduler
    )


def train_loop(args, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.save_dir,
    )
    if accelerator.is_main_process:
        if args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.args.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % args.save_image_epochs == 0 or epoch == args.num_epochs - 1:
                evaluate(args, epoch, pipeline)

            if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                pipeline.save_pretrained(args.save_dir)


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("L", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(args, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=args.eval_batch_size,
        generator=torch.manual_seed(args.seed),
        num_inference_steps=50,
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(args.save_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


if __name__ == "__main__":
    main()