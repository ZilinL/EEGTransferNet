#!/usr/bin/env bash
echo Begin EEGTransferNet ERN Experiments!
python EEGTransferNet_ERN/clear_EEGTransferNet_ERN_txt.py
GPU_ID=5
data_dir=./data/ERN

echo Target s0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s1 --tgt_domain s0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s2 --tgt_domain s0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s3 --tgt_domain s0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s4 --tgt_domain s0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s5 --tgt_domain s0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s6 --tgt_domain s0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s7 --tgt_domain s0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s8 --tgt_domain s0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s9 --tgt_domain s0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s10 --tgt_domain s0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s11 --tgt_domain s0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s12 --tgt_domain s0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s13 --tgt_domain s0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s14 --tgt_domain s0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s15 --tgt_domain s0

echo Target s1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s0 --tgt_domain s1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s2 --tgt_domain s1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s3 --tgt_domain s1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s4 --tgt_domain s1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s5 --tgt_domain s1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s6 --tgt_domain s1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s7 --tgt_domain s1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s8 --tgt_domain s1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s9 --tgt_domain s1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s10 --tgt_domain s1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s11 --tgt_domain s1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s12 --tgt_domain s1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s13 --tgt_domain s1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s14 --tgt_domain s1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s15 --tgt_domain s1

echo Target s2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s0 --tgt_domain s2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s1 --tgt_domain s2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s3 --tgt_domain s2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s4 --tgt_domain s2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s5 --tgt_domain s2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s6 --tgt_domain s2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s7 --tgt_domain s2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s8 --tgt_domain s2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s9 --tgt_domain s2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s10 --tgt_domain s2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s11 --tgt_domain s2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s12 --tgt_domain s2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s13 --tgt_domain s2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s14 --tgt_domain s2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s15 --tgt_domain s2

echo Target s3
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s0 --tgt_domain s3
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s1 --tgt_domain s3
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s2 --tgt_domain s3
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s4 --tgt_domain s3
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s5 --tgt_domain s3
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s6 --tgt_domain s3
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s7 --tgt_domain s3
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s8 --tgt_domain s3
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s9 --tgt_domain s3
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s10 --tgt_domain s3
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s11 --tgt_domain s3
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s12 --tgt_domain s3
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s14 --tgt_domain s3
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s15 --tgt_domain s3

echo Target s4
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s0 --tgt_domain s4
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s1 --tgt_domain s4
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s2 --tgt_domain s4
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s3 --tgt_domain s4
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s5 --tgt_domain s4
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s6 --tgt_domain s4
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s7 --tgt_domain s4
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s8 --tgt_domain s4
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s9 --tgt_domain s4
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s10 --tgt_domain s4
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s11 --tgt_domain s4
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s12 --tgt_domain s4
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s14 --tgt_domain s4
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s15 --tgt_domain s4

echo Target s5
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s0 --tgt_domain s5
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s1 --tgt_domain s5
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s2 --tgt_domain s5
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s3 --tgt_domain s5
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s4 --tgt_domain s5
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s6 --tgt_domain s5
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s7 --tgt_domain s5
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s8 --tgt_domain s5
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s9 --tgt_domain s5
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s10 --tgt_domain s5
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s11 --tgt_domain s5
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s12 --tgt_domain s5
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s14 --tgt_domain s5
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s15 --tgt_domain s5

echo Target s6
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s0 --tgt_domain s6
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s1 --tgt_domain s6
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s2 --tgt_domain s6
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s3 --tgt_domain s6
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s4 --tgt_domain s6
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s5 --tgt_domain s6
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s7 --tgt_domain s6
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s8 --tgt_domain s6
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s9 --tgt_domain s6
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s10 --tgt_domain s6
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s11 --tgt_domain s6
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s12 --tgt_domain s6
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s14 --tgt_domain s6
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s15 --tgt_domain s6

echo Target s7
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s0 --tgt_domain s7
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s1 --tgt_domain s7
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s2 --tgt_domain s7
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s3 --tgt_domain s7
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s4 --tgt_domain s7
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s5 --tgt_domain s7
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s6 --tgt_domain s7
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s8 --tgt_domain s7
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s9 --tgt_domain s7
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s10 --tgt_domain s7
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s11 --tgt_domain s7
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s12 --tgt_domain s7
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s14 --tgt_domain s7
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s15 --tgt_domain s7

echo Target s8
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s0 --tgt_domain s8
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s1 --tgt_domain s8
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s2 --tgt_domain s8
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s3 --tgt_domain s8
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s4 --tgt_domain s8
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s5 --tgt_domain s8
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s6 --tgt_domain s8
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s7 --tgt_domain s8
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s9 --tgt_domain s8
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s10 --tgt_domain s8
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s11 --tgt_domain s8
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s12 --tgt_domain s8
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s14 --tgt_domain s8
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s15 --tgt_domain s8

echo Target s9
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s0 --tgt_domain s9
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s1 --tgt_domain s9
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s2 --tgt_domain s9
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s3 --tgt_domain s9
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s4 --tgt_domain s9
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s5 --tgt_domain s9
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s6 --tgt_domain s9
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s7 --tgt_domain s9
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s8 --tgt_domain s9
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s10 --tgt_domain s9
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s11 --tgt_domain s9
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s12 --tgt_domain s9
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s14 --tgt_domain s9
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s15 --tgt_domain s9

echo Target s10
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s0 --tgt_domain s10
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s1 --tgt_domain s10
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s2 --tgt_domain s10
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s3 --tgt_domain s10
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s4 --tgt_domain s10
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s5 --tgt_domain s10
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s6 --tgt_domain s10
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s7 --tgt_domain s10
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s8 --tgt_domain s10
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s9 --tgt_domain s10
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s11 --tgt_domain s10
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s12 --tgt_domain s10
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s13 --tgt_domain s10
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s14 --tgt_domain s10
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s15 --tgt_domain s10

echo Target s11
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s0 --tgt_domain s11
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s1 --tgt_domain s11
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s2 --tgt_domain s11
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s3 --tgt_domain s11
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s4 --tgt_domain s11
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s5 --tgt_domain s11
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s6 --tgt_domain s11
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s7 --tgt_domain s11
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s8 --tgt_domain s11
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s9 --tgt_domain s11
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s10 --tgt_domain s11
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s12 --tgt_domain s11
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s13 --tgt_domain s11
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s14 --tgt_domain s11
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s15 --tgt_domain s11

echo Target s12
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s0 --tgt_domain s12
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s1 --tgt_domain s12
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s2 --tgt_domain s12
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s3 --tgt_domain s12
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s4 --tgt_domain s12
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s5 --tgt_domain s12
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s6 --tgt_domain s12
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s7 --tgt_domain s12
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s8 --tgt_domain s12
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s9 --tgt_domain s12
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s10 --tgt_domain s12
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s11 --tgt_domain s12
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s13 --tgt_domain s12
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s14 --tgt_domain s12
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s15 --tgt_domain s12

echo Target s13
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s0 --tgt_domain s13
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s1 --tgt_domain s13
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s2 --tgt_domain s13
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s3 --tgt_domain s13
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s4 --tgt_domain s13
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s5 --tgt_domain s13
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s6 --tgt_domain s13
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s7 --tgt_domain s13
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s8 --tgt_domain s13
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s9 --tgt_domain s13
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s10 --tgt_domain s13
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s11 --tgt_domain s13
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s12 --tgt_domain s13
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s14 --tgt_domain s13
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s15 --tgt_domain s13

echo Target s14
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s0 --tgt_domain s14
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s1 --tgt_domain s14
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s2 --tgt_domain s14
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s3 --tgt_domain s14
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s4 --tgt_domain s14
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s5 --tgt_domain s14
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s6 --tgt_domain s14
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s7 --tgt_domain s14
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s8 --tgt_domain s14
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s9 --tgt_domain s14
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s10 --tgt_domain s14
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s11 --tgt_domain s14
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s12 --tgt_domain s14
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s13 --tgt_domain s14
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s15 --tgt_domain s14

echo Target s15
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s0 --tgt_domain s15
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s1 --tgt_domain s15
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s2 --tgt_domain s15
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s3 --tgt_domain s15
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s4 --tgt_domain s15
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s5 --tgt_domain s15
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s6 --tgt_domain s15
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s7 --tgt_domain s15
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s8 --tgt_domain s15
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s9 --tgt_domain s15
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s10 --tgt_domain s15
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s11 --tgt_domain s15
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s12 --tgt_domain s15
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s13 --tgt_domain s15
CUDA_VISIBLE_DEVICES=$GPU_ID python main_ERP.py --config EEGTransferNet_ERN/EEGTransferNet_ERN.yaml --data_dir $data_dir --src_domain s14 --tgt_domain s15

python EEGTransferNet_ERN/EEGTransferNet_ERN_AvgTestAcc.py

echo End experiments! 