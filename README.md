# Moment Matching Distillation for Diffusion Models

This repository implements ["Multistep Distillation of Diffusion Models via Moment Matching"](https://arxiv.org/abs/2406.04103) by Google, using LLaMA-DiT architecture and Flow matching scheduler.

## Requirements

```bash
pip install torch torchvision tqdm tensorboard pillow click
```

## Training

First train the teacher model:

```bash
python train_student.py --cifar --epochs 100 --batch_size 64
```

Then train the student models using moment matching:

```bash 
python train.py --cifar --epochs 100 --batch_size 256 --init_ckpt_dir checkpoints/model_epoch_44.pt
```

## Algorithm 2

The training process follows Algorithm 2 from the paper:

1. Sample time `s < t` and noise `z_t` at time t
2. Get student prediction `x_student` from `z_t`
3. Sample `z_s` using `x_student` 
4. Get auxiliary prediction `x_aux` and teacher prediction `x_teacher` from `z_s`
5. Alternate between:
   - Training student by minimizing `E[x_student * (x_aux - x_teacher)]`
   - Training auxiliary by minimizing `||x_student - x_aux||^2 + ||x_teacher - x_aux||^2`

## Architecture

- Uses LLaMA-DiT (DiT with LLaMA-style blocks) as the base architecture
- Rectified Flow (Sub-VP) as the diffusion scheduler
- Supports both MNIST and CIFAR-10 datasets

## Monitoring

Training progress can be monitored via TensorBoard:
```bash
tensorboard --logdir checkpoints/logs
```

This will show:
- Training loss
- Generated samples
- Sampling process GIFs

## Citation

```bibtex
@article{deepmind2024moment,
  title={Multistep Distillation of Diffusion Models via Moment Matching},
  author={DeepMind},
  journal={arXiv preprint arXiv:2406.04103},
  year={2024}
}
```

## License

MIT