@echo off
setlocal

rem Function to execute a command and log errors
python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split1 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split1_vgg16_SGD --model vgg16
python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split2 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split2_vgg16_SGD --model vgg16
python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split3 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split3_vgg16_SGD --model vgg16
python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split4 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split4_vgg16_SGD --model vgg16
python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split5 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split5_vgg16_SGD --model vgg16

python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split1 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split1_mobilenetv2_SGD --model mobilenetv2
python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split2 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split2_mobilenetv2_SGD --model mobilenetv2
python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split3 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split3_mobilenetv2_SGD --model mobilenetv2
python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split4 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split4_mobilenetv2_SGD --model mobilenetv2
python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split5 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split5_mobilenetv2_SGD --model mobilenetv2

python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split1 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split1_resnet50_SGD --model resnet50
python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split2 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split2_resnet50_SGD --model resnet50
python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split3 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split3_resnet50_SGD --model resnet50
python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split4 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split4_resnet50_SGD --model resnet50
python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split5 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split5_resnet50_SGD --model resnet50

python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split1 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split1_googlenet_SGD --model googlenet
python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split2 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split2_googlenet_SGD --model googlenet
python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split3 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split3_googlenet_SGD --model googlenet
python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split4 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split4_googlenet_SGD --model googlenet
python train.py --data_root C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid\split5 --output_dir C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\result\sigmoid\split5_googlenet_SGD --model googlenet