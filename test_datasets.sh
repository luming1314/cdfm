export PYTHONPATH=$PYTHONPATH:$(pwd)

# TODO Lytro
echo ">>>>>>>CDFM Far-Near/Lytro>>>>>start!>>>>>>>>>>>>>>>>>>"
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 50 --seed 1234 --model_type "imagenet"  --batch_size 10 --fusion_type Far-Near --dataset_type Lytro --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale
echo "CDFM Far-Near/Lytro is OK!"
echo ">>>>>>CDFM Far-Near/Lytro>>>>>end!>>>>>>>>>>>>>>>>>>"

# TODO MFI-WHU
echo ">>>>>>>CDFM Far-Near/MFI-WHU>>>>>start!>>>>>>>>>>>>>>>>>>"
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 50 --seed 1234 --model_type "imagenet" --batch_size 7 --fusion_type Far-Near --dataset_type MFI-WHU --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 50 --seed 1234 --model_type "imagenet" --batch_size 8 --fusion_type Far-Near --dataset_type MFI-WHU --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p1
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 50 --seed 1234 --model_type "imagenet" --batch_size 7 --fusion_type Far-Near --dataset_type MFI-WHU --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p2
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 50 --seed 1234 --model_type "imagenet" --batch_size 6 --fusion_type Far-Near --dataset_type MFI-WHU --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p3
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 50 --seed 1234 --model_type "imagenet" -batch_size 5 --fusion_type Far-Near --dataset_type MFI-WHU --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p4
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 50 --seed 1234 --model_type "imagenet" --batch_size 4 --fusion_type Far-Near --dataset_type MFI-WHU --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p5
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 50 --seed 1234 --model_type "imagenet" --batch_size 3 --fusion_type Far-Near --dataset_type MFI-WHU --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p6
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 50 --seed 1234 --model_type "imagenet" --batch_size 4 --fusion_type Far-Near --dataset_type MFI-WHU --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p7
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 50 --seed 1234 --model_type "imagenet" --batch_size 1 --fusion_type Far-Near --dataset_type MFI-WHU --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter no_regular
echo "CDFM Far-Near/MFI-WHU is OK!"
echo ">>>>>>>CDFM Far-Near/MFI-WHU>>>>>end!>>>>>>>>>>>>>>>>>>"

# TODO MSRS
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo ">>>>>>>CDFM IR-VI/MSRS>>>>>start!>>>>>>>>>>>>>>>>>>"
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 30 --seed 1234 --model_type "imagenet" --batch_size 10 --fusion_type IR-VI --dataset_type MSRS --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale
echo "CDFM IR-VI/MSRS is OK!"
echo ">>>>>>>CDFM IR-VI/MSRS>>>>>end!>>>>>>>>>>>>>>>>>>"

# TODO M3FD
echo ">>>>>>>CDFM IR-VI/M3FD>>>>start!>>>>>>>>>>>>>>>>>>"
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 50 --seed 1234 --model_type "imagenet" --batch_size 10 --fusion_type IR-VI --dataset_type M3FD --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular
echo "CDFM IR-VI/M3FD regular is OK!"
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 50 --seed 1234 --model_type "imagenet" --batch_size 6 --fusion_type IR-VI --dataset_type M3FD --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p1
echo "CDFM IR-VI/M3FD regular is OK!"
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 50 --seed 1234 --model_type "imagenet" --batch_size 7 --fusion_type IR-VI --dataset_type M3FD --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p2
echo "CDFM IR-VI/M3FD regular is OK!"
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 50 --seed 1234 --model_type "imagenet" --batch_size 1 --fusion_type IR-VI --dataset_type M3FD --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter no_regular
echo "CDFM IR-VI/M3FD no regular is OK!"
echo ">>>>>>>CDFM IR-VI/M3FD>>>>>end!>>>>>>>>>>>>>>>>>>"

# TODO MEFB
echo ">>>>>>>CDFM OE-UE/MEFB>>>>>start!>>>>>>>>>>>>>>>>>>"
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 3 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 5 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p1
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 6 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p2
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 5 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p3
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 4 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p4
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 3 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p5
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 4 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p6
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 5 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p7
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 4 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p8
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 3 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p9
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 1 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter no_regular
echo "CDFM OE-UE/MEFB is OK!"
echo ">>>>>>>CDFM OE-UE/MEFB>>>>>end!>>>>>>>>>>>>>>>>>>"


# TODO MEFB
echo ">>>>>>>CDFM OE-UE/MEFB>>>>>start!>>>>>>>>>>>>>>>>>>"
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 3 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 5 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p1
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 6 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p2
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 5 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p3
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 4 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p4
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 3 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p5
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 4 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p6
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 5 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p7
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 4 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p8
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 3 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p9
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 20 --seed 1234 --model_type "imagenet" --batch_size 1 --fusion_type OE-UE --dataset_type MEFB --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter no_regular
echo "CDFM OE-UE/MEFB is OK!"
echo ">>>>>>>CDFM OE-UE/MEFB>>>>>end!>>>>>>>>>>>>>>>>>>"

## TODO SICE
echo ">>>>>>>CDFM OE-UE/SICE_OU>>>>>start!>>>>>>>>>>>>>>>>>>"
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 2 --seed 1234 --model_type "imagenet" --batch_size 1 --fusion_type OE-UE --dataset_type SICE_OU --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 2 --seed 1234 --model_type "imagenet" --batch_size 1 --fusion_type OE-UE --dataset_type SICE_OU --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p1
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 2 --seed 1234 --model_type "imagenet" --batch_size 1 --fusion_type OE-UE --dataset_type SICE_OU --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p2
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 2 --seed 1234 --model_type "imagenet" --batch_size 1 --fusion_type OE-UE --dataset_type SICE_OU --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p3
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 2 --seed 1234 --model_type "imagenet" --batch_size 1 --fusion_type OE-UE --dataset_type SICE_OU --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter regular_p4
python main.py -i all_imagenet_30 -s all_ddim --doc imagenet --timesteps 2 --seed 1234 --model_type "imagenet" --batch_size 1 --fusion_type OE-UE --dataset_type SICE_OU --root_path /data_mnt/mnt01/lm/datasets/Img_Fusion/fusion_resize_scale --filter no_regular
echo "CDFM OE-UE/SICE_OU is OK!"
echo ">>>>>>>CDFM OE-UE/SICE_OU>>>>>end!>>>>>>>>>>>>>>>>>>"

