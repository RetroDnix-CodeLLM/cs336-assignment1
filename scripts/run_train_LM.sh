#!/bin/bash

#SBATCH -J train_tinystories
#SBATCH -o logs/train_tinystories.out                      # stdout 重定向到 test.out
#SBATCH -e logs/train_tinystories.err                      # stderr 重定向到 test.err
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 1:30:00                           # 任务运行的最长时间为 1 小时
#SBATCH --mem 32GB
#SBATCH --gres=gpu:a100-pcie-40gb:1
mkdir -p logs

uv run cs336_basics/train.py -c config/train_tinystories.yaml