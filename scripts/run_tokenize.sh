#!/bin/bash

#SBATCH -J tokenize                            # 作业名为 test
#SBATCH -o logs/tokenize.out                      # stdout 重定向到 test.out
#SBATCH -e logs/tokenize.err                      # stderr 重定向到 test.err
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -c 4                           # 任务运行的最长时间为 1 小时
#SBATCH -t 15:00:00                           # 任务运行的最长时间为 1 小时
#SBATCH --mem 128GB

mkdir -p logs

uv run cs336_basics/calc.py