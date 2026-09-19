| run | input | frames_in | sf | frames_out | minutes | s_per_out_frame | gpu_peak_gib | holdout_psnr_db | anchor_vae_psnr_db | vae_dtype | adapter_args |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bim-vfi/pyr7_sf10 | test01 | 41 | 10 | 401 | 9.2 | 1.38 | 22.3 |  |  |  | {"pyr_level": 7, "pad_mode": "constant"} |
| ema-vfi/ds025 | test01 | 41 | 10 | 401 | 8.4 | 1.25 | 21.1 |  |  |  | {"down_scale": 0.25} |
| gimm-vfi/R-P_ds0.25 | test01 | 41 | 10 | 401 | 3.2 | 0.47 | 10.6 |  |  |  | {"model_variant": "R-P", "ds_factor": 0.25} |
| ldf-vfi/A_log_sf10 | test01 | 41 | 10 | 401 | 269.8 | 40.37 | 56.9 |  | 45.52 | bf16 | {} |
| vtinker/A_log | test01 | 41 | 10 | 401 | 15.8 | 2.36 | 25.9 |  |  |  | {} |
