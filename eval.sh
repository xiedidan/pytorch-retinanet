CUDA_VISIBLE_DEVICES=0 python eval.py --dataset csv --csv_class ./rsna_no_class/rsna-class-mapping.csv --csv_val ./rsna_no_class/rsna-eval.csv --depth 101 --epochs 100 --checkpoint ./resnet_101_baseline_fold0/retinanet --log_prefix resnet_101_baseline_fold0_eval

