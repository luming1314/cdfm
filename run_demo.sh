export PYTHONPATH=$PYTHONPATH:$(pwd)

# TODO demo
echo ">>>>>>>CDFM demo>>>>>start!>>>>>>>>>>>>>>>>>>"
python main.py -i demo -s all_ddim --doc imagenet --timesteps 50 --seed 1234 --model_type "imagenet"  --batch_size 1 --fusion_type Far-Near --dataset_type Lytro --root_path ./images
echo "CDFM demo is OK!"
echo ">>>>>>CDFM demo>>>>>end!>>>>>>>>>>>>>>>>>>"

