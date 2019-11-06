###

#python train.py --gpu_id 6 --source amazon --target webcam --loss_name JAN --tradeoff 1.0 --using_bottleneck 1

python train.py --gpu_id 1 --source POM_100 --target VR_100 --target_val VR182_100 --loss_name JAN --tradeoff 0.5 --using_bottleneck 1 

#python train.py --gpu_id 6 --source amazon --target dslr --loss_name JAN --tradeoff 1.0 --using_bottleneck 1

#python train.py --gpu_id 6 --source dslr --target amazon --loss_name JAN --tradeoff 1.0 --using_bottleneck 1

#python train.py --gpu_id 6 --source webcam --target dslr --loss_name JAN --tradeoff 1.0 --using_bottleneck 1

#python train.py --gpu_id 6 --source dslr --target webcam --loss_name JAN --tradeoff 1.0 --using_bottleneck 1
