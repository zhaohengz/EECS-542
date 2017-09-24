#!/bin/bash

srun \
  --job-name=debug_job \
  --mail-user=zhaohenz@umich.edu \
  --mail-type=ALL \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem-per-cpu=1024 \
  --gres=gpu:0 \
  --time=00-04:00:00 \
  --pty bash
