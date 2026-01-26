#!/usr/bin/env python3
"""Test script to verify GPU availability and DDP setup"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def test_gpu_worker(rank, world_size):
    """Test GPU availability for each process"""
    # Setup process group
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Get GPU info
    device = torch.device(f'cuda:{rank}')
    gpu_name = torch.cuda.get_device_name(rank)
    gpu_memory = torch.cuda.get_device_properties(rank).total_memory / 1024**3  # GB

    print(f"[Rank {rank}] Using GPU: {gpu_name}")
    print(f"[Rank {rank}] GPU Memory: {gpu_memory:.2f} GB")

    # Test tensor creation and communication
    tensor = torch.ones(3, 3).to(device) * (rank + 1)
    print(f"[Rank {rank}] Created tensor:\n{tensor}")

    # Cleanup
    dist.destroy_process_group()

def main():
    print("=" * 60)
    print("GPU Detection Test")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return

    num_gpus = torch.cuda.device_count()
    print(f"\nNumber of GPUs detected: {num_gpus}")

    # List all GPUs
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")

    print("\n" + "=" * 60)
    print("Testing Distributed Data Parallel Setup")
    print("=" * 60 + "\n")

    if num_gpus > 1:
        # Test DDP with multiple GPUs
        mp.spawn(test_gpu_worker, args=(num_gpus,), nprocs=num_gpus, join=True)
        print("\n" + "=" * 60)
        print("Multi-GPU DDP test completed successfully!")
        print("=" * 60)
    else:
        print("Only 1 GPU available, skipping DDP test")

    print("\nRecommended settings for your training:")
    print(f"  --num_gpus {num_gpus}")
    print(f"  --batch_size 16  (per GPU)")
    print(f"  Total effective batch size: {16 * num_gpus}")

if __name__ == '__main__':
    main()
