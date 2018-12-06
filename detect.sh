CUDA_VISIBLE_DEVICES=0 python detect.py --dataset csv --csv_classes ./rsna_no_class/rsna-class-mapping.csv --csv_val ./rsna_no_class/rsna-test.csv --depth 101 --checkpoint ./resnet_101_baseline_fold0/retinanet_6.pth --log_prefix resnet_101_baseline_fold0_detect

