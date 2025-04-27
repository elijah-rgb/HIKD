################### Training URL Model ###################
CUDA_VISIBLE_DEVICES=0 python test_extractor.py --model.dir   --out.dir   --test.c 0.017 --test.distance bolic --model.classifier bolic --model.name fungi-net  --test.distance bolic
################### Training Single Domain Learning Networks ###################
# function train_fn {
#     CUDA_VISIBLE_DEVICES=1 python train_net.py --model.dir /media/yangxilab/DiskB/kdc/URL-master/saved_results/sdl_bolic_c0017_lr_001 --out.dir /media/yangxilab/DiskB/kdc/URL-master/saved_results/test --model.classifier bolic --test.c 0.017 --test.distance hyperbolic --model.name birds-net
# }

# # Train an single domain learning network on every training dataset (the following models could be trained in parallel)

# # #ImageNet
# # NAME="imagenet-net"
# # train_fn $NAME 

# #Omniglot
# NAME="omniglot-net"
# train_fn $NAME 

# # Aircraft
# NAME="aircraft-net"
# train_fn $NAME 

# # Birds
# NAME="birds-net"
# train_fn $NAME 

# # Textures
# NAME="textures-net"
# train_fn $NAME 

# # Quick Draw
# NAME="quickdraw-net"
# train_fn $NAME 

# # Fungi
# NAME="fungi-net"
# train_fn $NAME 

# # VGG Flower
# NAME="vgg_flower-net"
# train_fn $NAME 

# echo "All domain-specific networks are trained!"
