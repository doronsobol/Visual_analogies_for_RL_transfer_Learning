# Visual_analogies_for_RL_transfer_Learning

## Acknoweledgments:
The UNIT GAN code is heavily based on the code of the [oficial imlpementation](https://github.com/mingyuliutw/UNIT)
And the RL code is heavily based on the code of [rl_a3c_pytorch](https://github.com/dgriff777/rl_a3c_pytorch)

## Datasets
To Download the data used to create the Mappers:
1. Download the zip files from [here](https://drive.google.com/drive/folders/1B4_n0X0s5ZV3yOhHX2tjJN1G2hX2WpEy?usp=sharing)
2. Unzip the files in datasets/games/

## Training the Mapper
To train the mapper:
```
cd src
python cocogan_train.py --config ../exps/unit/<conf>.yaml --log ../logs/<log_name>
```

## Gaining the Base Network for the Distilation Method
To pre-train the distialation network:
```
cd rl_a3c_pytorch
```
Create the data:
```
 python ./disco_gym_eval.py --model_env <source enviroment> --env <target enviroment> --use_convertor True --convertor 1  --a2b 1 --config <config file of the conversion> --weight <mapper file>  --load-model-dir <directory of the source model>  --transform-action   --keep-images True --num-episodes 100  --labels-file <path to create the labewl file>  --images-dir <directory to save the data>  --blurr (--cuda)
```

