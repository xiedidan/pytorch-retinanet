CUDA_VISIBLE_DEVICES=1 python train.py --dataset csv --csv_train ./rsna_no_class/rsna-train-3.csv --csv_class ./rsna_no_class/rsna-class-mapping.csv --csv_val ./rsna_no_class/rsna-val-3.csv --depth 101 --log_prefix resnet_101_baseline_fold3

