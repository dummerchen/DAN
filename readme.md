
## requirement
python3.8
torch=1.13.1
...


## Train
Before starting training, please modify the data path in the json file correctly.
```python
# train DAT-DAN-P
torchrun --nnodes=1 --nproc_per_node=2 main_train_psnr --opt options/dat/x4/train_lora_datv2_3_0_3_d180_b64_s4_fdb_t96_wlfreezehb01234_r4.json

# train DAT-DAN-F
torchrun --nnodes=1 --nproc_per_node=2 main_train_psnr --opt options/dat/x4/train_finetune_datv2_3_0_3_d180_b64_s4_fdb_t96.json

```

## Test
```python
# test DAT-DAN-P
python3 main_test_realsr --opt options/dat/x4/train_lora_datv2_3_0_3_d180_b64_s4_fdb_t96_wlfreezehb01234_r4.json --idx -1 -t 0 -d cuda:0
```