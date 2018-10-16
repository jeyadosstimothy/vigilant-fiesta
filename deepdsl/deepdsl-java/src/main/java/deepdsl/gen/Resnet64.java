package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.util.CudaRun;


public class Resnet64 extends CudaRun {

public static void main(String[] args){
Resnet64 run = new Resnet64();
run.train(10);
run.test(1);
run.save();
run.free();
}

public Resnet64() {
super("src/main/java/deepdsl/gen/resnet64");
setTrainData(LmdbFactory.getFactory("dataset/imagenet/ilsvrc12_train_lmdb", 1000000, new int[]{64, 3, 224, 224}, 1000, false));
setTestData(LmdbFactory.getFactory("dataset/imagenet/ilsvrc12_val_lmdb", 10000, new int[]{64, 3, 224, 224}, 1000, true));
}

float lrn_rate = -0.01f;
float momentum = 0.9f;
float decay = 5.0E-4f;

JCudnnBatchNorm y136 = addBatchNorm("y136", new int[]{64,64,112,112});
JCudnnBatchNorm y125 = addBatchNorm("y125", new int[]{64,256,55,55});
JCudnnBatchNorm y131 = addBatchNorm("y131", new int[]{64,64,55,55});
JCudnnBatchNorm y129 = addBatchNorm("y129", new int[]{64,64,55,55});
JCudnnBatchNorm y127 = addBatchNorm("y127", new int[]{64,256,55,55});
JCudnnBatchNorm y123 = addBatchNorm("y123", new int[]{64,64,55,55});
JCudnnBatchNorm y121 = addBatchNorm("y121", new int[]{64,64,55,55});
JCudnnBatchNorm y119 = addBatchNorm("y119", new int[]{64,256,55,55});
JCudnnBatchNorm y116 = addBatchNorm("y116", new int[]{64,64,55,55});
JCudnnBatchNorm y113 = addBatchNorm("y113", new int[]{64,64,55,55});
JCudnnBatchNorm y109 = addBatchNorm("y109", new int[]{64,256,55,55});
JCudnnBatchNorm y98 = addBatchNorm("y98", new int[]{64,512,28,28});
JCudnnBatchNorm y105 = addBatchNorm("y105", new int[]{64,128,28,28});
JCudnnBatchNorm y103 = addBatchNorm("y103", new int[]{64,128,28,28});
JCudnnBatchNorm y101 = addBatchNorm("y101", new int[]{64,512,28,28});
JCudnnBatchNorm y96 = addBatchNorm("y96", new int[]{64,128,28,28});
JCudnnBatchNorm y94 = addBatchNorm("y94", new int[]{64,128,28,28});
JCudnnBatchNorm y92 = addBatchNorm("y92", new int[]{64,512,28,28});
JCudnnBatchNorm y90 = addBatchNorm("y90", new int[]{64,128,28,28});
JCudnnBatchNorm y88 = addBatchNorm("y88", new int[]{64,128,28,28});
JCudnnBatchNorm y86 = addBatchNorm("y86", new int[]{64,512,28,28});
JCudnnBatchNorm y83 = addBatchNorm("y83", new int[]{64,128,28,28});
JCudnnBatchNorm y80 = addBatchNorm("y80", new int[]{64,128,28,28});
JCudnnBatchNorm y76 = addBatchNorm("y76", new int[]{64,512,28,28});
JCudnnBatchNorm y65 = addBatchNorm("y65", new int[]{64,1024,14,14});
JCudnnBatchNorm y72 = addBatchNorm("y72", new int[]{64,256,14,14});
JCudnnBatchNorm y70 = addBatchNorm("y70", new int[]{64,256,14,14});
JCudnnBatchNorm y68 = addBatchNorm("y68", new int[]{64,1024,14,14});
JCudnnBatchNorm y63 = addBatchNorm("y63", new int[]{64,256,14,14});
JCudnnBatchNorm y61 = addBatchNorm("y61", new int[]{64,256,14,14});
JCudnnBatchNorm y59 = addBatchNorm("y59", new int[]{64,1024,14,14});
JCudnnBatchNorm y57 = addBatchNorm("y57", new int[]{64,256,14,14});
JCudnnBatchNorm y55 = addBatchNorm("y55", new int[]{64,256,14,14});
JCudnnBatchNorm y53 = addBatchNorm("y53", new int[]{64,1024,14,14});
JCudnnBatchNorm y51 = addBatchNorm("y51", new int[]{64,256,14,14});
JCudnnBatchNorm y49 = addBatchNorm("y49", new int[]{64,256,14,14});
JCudnnBatchNorm y47 = addBatchNorm("y47", new int[]{64,1024,14,14});
JCudnnBatchNorm y45 = addBatchNorm("y45", new int[]{64,256,14,14});
JCudnnBatchNorm y43 = addBatchNorm("y43", new int[]{64,256,14,14});
JCudnnBatchNorm y41 = addBatchNorm("y41", new int[]{64,1024,14,14});
JCudnnBatchNorm y38 = addBatchNorm("y38", new int[]{64,256,14,14});
JCudnnBatchNorm y35 = addBatchNorm("y35", new int[]{64,256,14,14});
JCudnnBatchNorm y31 = addBatchNorm("y31", new int[]{64,1024,14,14});
JCudnnBatchNorm y20 = addBatchNorm("y20", new int[]{64,2048,7,7});
JCudnnBatchNorm y27 = addBatchNorm("y27", new int[]{64,512,7,7});
JCudnnBatchNorm y25 = addBatchNorm("y25", new int[]{64,512,7,7});
JCudnnBatchNorm y23 = addBatchNorm("y23", new int[]{64,2048,7,7});
JCudnnBatchNorm y18 = addBatchNorm("y18", new int[]{64,512,7,7});
JCudnnBatchNorm y16 = addBatchNorm("y16", new int[]{64,512,7,7});
JCudnnBatchNorm y14 = addBatchNorm("y14", new int[]{64,2048,7,7});
JCudnnBatchNorm y11 = addBatchNorm("y11", new int[]{64,512,7,7});
JCudnnBatchNorm y8 = addBatchNorm("y8", new int[]{64,512,7,7});
JCudnnBatchNorm y4 = addBatchNorm("y4", new int[]{64,2048,7,7});
JCudnnConvolution y40 = addConvolution(new int[]{64,1024,14,14},new int[]{256,1024,1,1},new int[]{256}, 1, 0);
JCudnnConvolution y78 = addConvolution(new int[]{64,128,28,28},new int[]{512,128,1,1},new int[]{512}, 1, 0);
JCudnnConvolution y13 = addConvolution(new int[]{64,2048,7,7},new int[]{512,2048,1,1},new int[]{512}, 1, 0);
JCudnnConvolution y33 = addConvolution(new int[]{64,256,14,14},new int[]{1024,256,1,1},new int[]{1024}, 1, 0);
JCudnnConvolution y118 = addConvolution(new int[]{64,256,55,55},new int[]{64,256,1,1},new int[]{64}, 1, 0);
JCudnnConvolution y85 = addConvolution(new int[]{64,512,28,28},new int[]{128,512,1,1},new int[]{128}, 1, 0);
JCudnnConvolution y6 = addConvolution(new int[]{64,512,7,7},new int[]{2048,512,1,1},new int[]{2048}, 1, 0);
JCudnnConvolution y111 = addConvolution(new int[]{64,64,55,55},new int[]{256,64,1,1},new int[]{256}, 1, 0);
JCudnnConvolution y139 = addConvolution(new int[]{64,64,55,55},new int[]{64,64,1,1},new int[]{64}, 1, 0);
JCudnnConvolution y82 = addConvolution(new int[]{64,128,28,28},new int[]{128,128,3,3},new int[]{128}, 1, 1);
JCudnnConvolution y37 = addConvolution(new int[]{64,256,14,14},new int[]{256,256,3,3},new int[]{256}, 1, 1);
JCudnnConvolution y10 = addConvolution(new int[]{64,512,7,7},new int[]{512,512,3,3},new int[]{512}, 1, 1);
JCudnnConvolution y115 = addConvolution(new int[]{64,64,55,55},new int[]{64,64,3,3},new int[]{64}, 1, 1);
JCudnnConvolution y22 = addConvolution(new int[]{64,1024,14,14},new int[]{2048,1024,1,1},new int[]{2048}, 2, 0);
JCudnnConvolution y142 = addConvolution(new int[]{64,1024,14,14},new int[]{512,1024,1,1},new int[]{512}, 2, 0);
JCudnnConvolution y140 = addConvolution(new int[]{64,256,55,55},new int[]{128,256,1,1},new int[]{128}, 2, 0);
JCudnnConvolution y100 = addConvolution(new int[]{64,256,55,55},new int[]{512,256,1,1},new int[]{512}, 2, 0);
JCudnnConvolution y67 = addConvolution(new int[]{64,512,28,28},new int[]{1024,512,1,1},new int[]{1024}, 2, 0);
JCudnnConvolution y141 = addConvolution(new int[]{64,512,28,28},new int[]{256,512,1,1},new int[]{256}, 2, 0);
JCudnnConvolution y138 = addConvolution(new int[]{64,3,224,224},new int[]{64,3,7,7},new int[]{64}, 2, 3);
JCudnnSoftmax y1 = addSoftmax(new int[]{64,1000}, SoftmaxAlgorithm.LOG);
JCudnnPooling y134 = addPooling(new int[]{64,64,112,112}, 3, 2, 0, PoolingType.MAX);
JCudnnPooling y2 = addPooling(new int[]{64,2048,7,7}, 7, 1, 0, PoolingType.AVERAGE_EXCLUDE_PADDING);
JCudnnActivation y30 = addActivation(new int[]{64,1024,14,14}, ActivationMode.RELU);
JCudnnActivation y79 = addActivation(new int[]{64,128,28,28}, ActivationMode.RELU);
JCudnnActivation y3 = addActivation(new int[]{64,2048,7,7}, ActivationMode.RELU);
JCudnnActivation y34 = addActivation(new int[]{64,256,14,14}, ActivationMode.RELU);
JCudnnActivation y108 = addActivation(new int[]{64,256,55,55}, ActivationMode.RELU);
JCudnnActivation y75 = addActivation(new int[]{64,512,28,28}, ActivationMode.RELU);
JCudnnActivation y7 = addActivation(new int[]{64,512,7,7}, ActivationMode.RELU);
JCudnnActivation y135 = addActivation(new int[]{64,64,112,112}, ActivationMode.RELU);
JCudnnActivation y112 = addActivation(new int[]{64,64,55,55}, ActivationMode.RELU);
JCudaTensor _1_bn_bias = addParam("_1_bn_bias", "Constant", 0.0f, 1, 64, 1, 1);
JCudaTensor _1_bn_scale = addParam("_1_bn_scale", "Constant", 1.0f, 1, 64, 1, 1);
JCudaTensor _1_cv_B = addFixedParam("_1_cv_B", "Constant", 0.0f, 64);
JCudaTensor _1_cv_W = addParam("_1_cv_W", "Random", 0.11664237f, 64, 3, 7, 7);
JCudaTensor _2a1_bn_bias = addParam("_2a1_bn_bias", "Constant", 0.0f, 1, 256, 1, 1);
JCudaTensor _2a1_bn_scale = addParam("_2a1_bn_scale", "Constant", 1.0f, 1, 256, 1, 1);
JCudaTensor _2a1_cv_B = addFixedParam("_2a1_cv_B", "Constant", 0.0f, 256);
JCudaTensor _2a1_cv_W = addParam("_2a1_cv_W", "Random", 0.17677669f, 256, 64, 1, 1);
JCudaTensor _2a2_a_bn_bias = addParam("_2a2_a_bn_bias", "Constant", 0.0f, 1, 64, 1, 1);
JCudaTensor _2a2_a_bn_scale = addParam("_2a2_a_bn_scale", "Constant", 1.0f, 1, 64, 1, 1);
JCudaTensor _2a2_a_cv_B = addFixedParam("_2a2_a_cv_B", "Constant", 0.0f, 64);
JCudaTensor _2a2_a_cv_W = addParam("_2a2_a_cv_W", "Random", 0.17677669f, 64, 64, 1, 1);
JCudaTensor _2a2_b_bn_bias = addParam("_2a2_b_bn_bias", "Constant", 0.0f, 1, 64, 1, 1);
JCudaTensor _2a2_b_bn_scale = addParam("_2a2_b_bn_scale", "Constant", 1.0f, 1, 64, 1, 1);
JCudaTensor _2a2_b_cv_B = addFixedParam("_2a2_b_cv_B", "Constant", 0.0f, 64);
JCudaTensor _2a2_b_cv_W = addParam("_2a2_b_cv_W", "Random", 0.058925565f, 64, 64, 3, 3);
JCudaTensor _2a2_c_bn_bias = addParam("_2a2_c_bn_bias", "Constant", 0.0f, 1, 256, 1, 1);
JCudaTensor _2a2_c_bn_scale = addParam("_2a2_c_bn_scale", "Constant", 1.0f, 1, 256, 1, 1);
JCudaTensor _2a2_c_cv_B = addFixedParam("_2a2_c_cv_B", "Constant", 0.0f, 256);
JCudaTensor _2a2_c_cv_W = addParam("_2a2_c_cv_W", "Random", 0.17677669f, 256, 64, 1, 1);
JCudaTensor _2b_a_bn_bias = addParam("_2b_a_bn_bias", "Constant", 0.0f, 1, 64, 1, 1);
JCudaTensor _2b_a_bn_scale = addParam("_2b_a_bn_scale", "Constant", 1.0f, 1, 64, 1, 1);
JCudaTensor _2b_a_cv_B = addFixedParam("_2b_a_cv_B", "Constant", 0.0f, 64);
JCudaTensor _2b_a_cv_W = addParam("_2b_a_cv_W", "Random", 0.088388346f, 64, 256, 1, 1);
JCudaTensor _2b_b_bn_bias = addParam("_2b_b_bn_bias", "Constant", 0.0f, 1, 64, 1, 1);
JCudaTensor _2b_b_bn_scale = addParam("_2b_b_bn_scale", "Constant", 1.0f, 1, 64, 1, 1);
JCudaTensor _2b_b_cv_B = addFixedParam("_2b_b_cv_B", "Constant", 0.0f, 64);
JCudaTensor _2b_b_cv_W = addParam("_2b_b_cv_W", "Random", 0.058925565f, 64, 64, 3, 3);
JCudaTensor _2b_c_bn_bias = addParam("_2b_c_bn_bias", "Constant", 0.0f, 1, 256, 1, 1);
JCudaTensor _2b_c_bn_scale = addParam("_2b_c_bn_scale", "Constant", 1.0f, 1, 256, 1, 1);
JCudaTensor _2b_c_cv_B = addFixedParam("_2b_c_cv_B", "Constant", 0.0f, 256);
JCudaTensor _2b_c_cv_W = addParam("_2b_c_cv_W", "Random", 0.17677669f, 256, 64, 1, 1);
JCudaTensor _2c_a_bn_bias = addParam("_2c_a_bn_bias", "Constant", 0.0f, 1, 64, 1, 1);
JCudaTensor _2c_a_bn_scale = addParam("_2c_a_bn_scale", "Constant", 1.0f, 1, 64, 1, 1);
JCudaTensor _2c_a_cv_B = addFixedParam("_2c_a_cv_B", "Constant", 0.0f, 64);
JCudaTensor _2c_a_cv_W = addParam("_2c_a_cv_W", "Random", 0.088388346f, 64, 256, 1, 1);
JCudaTensor _2c_b_bn_bias = addParam("_2c_b_bn_bias", "Constant", 0.0f, 1, 64, 1, 1);
JCudaTensor _2c_b_bn_scale = addParam("_2c_b_bn_scale", "Constant", 1.0f, 1, 64, 1, 1);
JCudaTensor _2c_b_cv_B = addFixedParam("_2c_b_cv_B", "Constant", 0.0f, 64);
JCudaTensor _2c_b_cv_W = addParam("_2c_b_cv_W", "Random", 0.058925565f, 64, 64, 3, 3);
JCudaTensor _2c_c_bn_bias = addParam("_2c_c_bn_bias", "Constant", 0.0f, 1, 256, 1, 1);
JCudaTensor _2c_c_bn_scale = addParam("_2c_c_bn_scale", "Constant", 1.0f, 1, 256, 1, 1);
JCudaTensor _2c_c_cv_B = addFixedParam("_2c_c_cv_B", "Constant", 0.0f, 256);
JCudaTensor _2c_c_cv_W = addParam("_2c_c_cv_W", "Random", 0.17677669f, 256, 64, 1, 1);
JCudaTensor _3a1_bn_bias = addParam("_3a1_bn_bias", "Constant", 0.0f, 1, 512, 1, 1);
JCudaTensor _3a1_bn_scale = addParam("_3a1_bn_scale", "Constant", 1.0f, 1, 512, 1, 1);
JCudaTensor _3a1_cv_B = addFixedParam("_3a1_cv_B", "Constant", 0.0f, 512);
JCudaTensor _3a1_cv_W = addParam("_3a1_cv_W", "Random", 0.088388346f, 512, 256, 1, 1);
JCudaTensor _3a2_a_bn_bias = addParam("_3a2_a_bn_bias", "Constant", 0.0f, 1, 128, 1, 1);
JCudaTensor _3a2_a_bn_scale = addParam("_3a2_a_bn_scale", "Constant", 1.0f, 1, 128, 1, 1);
JCudaTensor _3a2_a_cv_B = addFixedParam("_3a2_a_cv_B", "Constant", 0.0f, 128);
JCudaTensor _3a2_a_cv_W = addParam("_3a2_a_cv_W", "Random", 0.088388346f, 128, 256, 1, 1);
JCudaTensor _3a2_b_bn_bias = addParam("_3a2_b_bn_bias", "Constant", 0.0f, 1, 128, 1, 1);
JCudaTensor _3a2_b_bn_scale = addParam("_3a2_b_bn_scale", "Constant", 1.0f, 1, 128, 1, 1);
JCudaTensor _3a2_b_cv_B = addFixedParam("_3a2_b_cv_B", "Constant", 0.0f, 128);
JCudaTensor _3a2_b_cv_W = addParam("_3a2_b_cv_W", "Random", 0.041666668f, 128, 128, 3, 3);
JCudaTensor _3a2_c_bn_bias = addParam("_3a2_c_bn_bias", "Constant", 0.0f, 1, 512, 1, 1);
JCudaTensor _3a2_c_bn_scale = addParam("_3a2_c_bn_scale", "Constant", 1.0f, 1, 512, 1, 1);
JCudaTensor _3a2_c_cv_B = addFixedParam("_3a2_c_cv_B", "Constant", 0.0f, 512);
JCudaTensor _3a2_c_cv_W = addParam("_3a2_c_cv_W", "Random", 0.125f, 512, 128, 1, 1);
JCudaTensor _3b_a_bn_bias = addParam("_3b_a_bn_bias", "Constant", 0.0f, 1, 128, 1, 1);
JCudaTensor _3b_a_bn_scale = addParam("_3b_a_bn_scale", "Constant", 1.0f, 1, 128, 1, 1);
JCudaTensor _3b_a_cv_B = addFixedParam("_3b_a_cv_B", "Constant", 0.0f, 128);
JCudaTensor _3b_a_cv_W = addParam("_3b_a_cv_W", "Random", 0.0625f, 128, 512, 1, 1);
JCudaTensor _3b_b_bn_bias = addParam("_3b_b_bn_bias", "Constant", 0.0f, 1, 128, 1, 1);
JCudaTensor _3b_b_bn_scale = addParam("_3b_b_bn_scale", "Constant", 1.0f, 1, 128, 1, 1);
JCudaTensor _3b_b_cv_B = addFixedParam("_3b_b_cv_B", "Constant", 0.0f, 128);
JCudaTensor _3b_b_cv_W = addParam("_3b_b_cv_W", "Random", 0.041666668f, 128, 128, 3, 3);
JCudaTensor _3b_c_bn_bias = addParam("_3b_c_bn_bias", "Constant", 0.0f, 1, 512, 1, 1);
JCudaTensor _3b_c_bn_scale = addParam("_3b_c_bn_scale", "Constant", 1.0f, 1, 512, 1, 1);
JCudaTensor _3b_c_cv_B = addFixedParam("_3b_c_cv_B", "Constant", 0.0f, 512);
JCudaTensor _3b_c_cv_W = addParam("_3b_c_cv_W", "Random", 0.125f, 512, 128, 1, 1);
JCudaTensor _3c_a_bn_bias = addParam("_3c_a_bn_bias", "Constant", 0.0f, 1, 128, 1, 1);
JCudaTensor _3c_a_bn_scale = addParam("_3c_a_bn_scale", "Constant", 1.0f, 1, 128, 1, 1);
JCudaTensor _3c_a_cv_B = addFixedParam("_3c_a_cv_B", "Constant", 0.0f, 128);
JCudaTensor _3c_a_cv_W = addParam("_3c_a_cv_W", "Random", 0.0625f, 128, 512, 1, 1);
JCudaTensor _3c_b_bn_bias = addParam("_3c_b_bn_bias", "Constant", 0.0f, 1, 128, 1, 1);
JCudaTensor _3c_b_bn_scale = addParam("_3c_b_bn_scale", "Constant", 1.0f, 1, 128, 1, 1);
JCudaTensor _3c_b_cv_B = addFixedParam("_3c_b_cv_B", "Constant", 0.0f, 128);
JCudaTensor _3c_b_cv_W = addParam("_3c_b_cv_W", "Random", 0.041666668f, 128, 128, 3, 3);
JCudaTensor _3c_c_bn_bias = addParam("_3c_c_bn_bias", "Constant", 0.0f, 1, 512, 1, 1);
JCudaTensor _3c_c_bn_scale = addParam("_3c_c_bn_scale", "Constant", 1.0f, 1, 512, 1, 1);
JCudaTensor _3c_c_cv_B = addFixedParam("_3c_c_cv_B", "Constant", 0.0f, 512);
JCudaTensor _3c_c_cv_W = addParam("_3c_c_cv_W", "Random", 0.125f, 512, 128, 1, 1);
JCudaTensor _3d_a_bn_bias = addParam("_3d_a_bn_bias", "Constant", 0.0f, 1, 128, 1, 1);
JCudaTensor _3d_a_bn_scale = addParam("_3d_a_bn_scale", "Constant", 1.0f, 1, 128, 1, 1);
JCudaTensor _3d_a_cv_B = addFixedParam("_3d_a_cv_B", "Constant", 0.0f, 128);
JCudaTensor _3d_a_cv_W = addParam("_3d_a_cv_W", "Random", 0.0625f, 128, 512, 1, 1);
JCudaTensor _3d_b_bn_bias = addParam("_3d_b_bn_bias", "Constant", 0.0f, 1, 128, 1, 1);
JCudaTensor _3d_b_bn_scale = addParam("_3d_b_bn_scale", "Constant", 1.0f, 1, 128, 1, 1);
JCudaTensor _3d_b_cv_B = addFixedParam("_3d_b_cv_B", "Constant", 0.0f, 128);
JCudaTensor _3d_b_cv_W = addParam("_3d_b_cv_W", "Random", 0.041666668f, 128, 128, 3, 3);
JCudaTensor _3d_c_bn_bias = addParam("_3d_c_bn_bias", "Constant", 0.0f, 1, 512, 1, 1);
JCudaTensor _3d_c_bn_scale = addParam("_3d_c_bn_scale", "Constant", 1.0f, 1, 512, 1, 1);
JCudaTensor _3d_c_cv_B = addFixedParam("_3d_c_cv_B", "Constant", 0.0f, 512);
JCudaTensor _3d_c_cv_W = addParam("_3d_c_cv_W", "Random", 0.125f, 512, 128, 1, 1);
JCudaTensor _4a1_bn_bias = addParam("_4a1_bn_bias", "Constant", 0.0f, 1, 1024, 1, 1);
JCudaTensor _4a1_bn_scale = addParam("_4a1_bn_scale", "Constant", 1.0f, 1, 1024, 1, 1);
JCudaTensor _4a1_cv_B = addFixedParam("_4a1_cv_B", "Constant", 0.0f, 1024);
JCudaTensor _4a1_cv_W = addParam("_4a1_cv_W", "Random", 0.0625f, 1024, 512, 1, 1);
JCudaTensor _4a2_a_bn_bias = addParam("_4a2_a_bn_bias", "Constant", 0.0f, 1, 256, 1, 1);
JCudaTensor _4a2_a_bn_scale = addParam("_4a2_a_bn_scale", "Constant", 1.0f, 1, 256, 1, 1);
JCudaTensor _4a2_a_cv_B = addFixedParam("_4a2_a_cv_B", "Constant", 0.0f, 256);
JCudaTensor _4a2_a_cv_W = addParam("_4a2_a_cv_W", "Random", 0.0625f, 256, 512, 1, 1);
JCudaTensor _4a2_b_bn_bias = addParam("_4a2_b_bn_bias", "Constant", 0.0f, 1, 256, 1, 1);
JCudaTensor _4a2_b_bn_scale = addParam("_4a2_b_bn_scale", "Constant", 1.0f, 1, 256, 1, 1);
JCudaTensor _4a2_b_cv_B = addFixedParam("_4a2_b_cv_B", "Constant", 0.0f, 256);
JCudaTensor _4a2_b_cv_W = addParam("_4a2_b_cv_W", "Random", 0.029462783f, 256, 256, 3, 3);
JCudaTensor _4a2_c_bn_bias = addParam("_4a2_c_bn_bias", "Constant", 0.0f, 1, 1024, 1, 1);
JCudaTensor _4a2_c_bn_scale = addParam("_4a2_c_bn_scale", "Constant", 1.0f, 1, 1024, 1, 1);
JCudaTensor _4a2_c_cv_B = addFixedParam("_4a2_c_cv_B", "Constant", 0.0f, 1024);
JCudaTensor _4a2_c_cv_W = addParam("_4a2_c_cv_W", "Random", 0.088388346f, 1024, 256, 1, 1);
JCudaTensor _4b_a_bn_bias = addParam("_4b_a_bn_bias", "Constant", 0.0f, 1, 256, 1, 1);
JCudaTensor _4b_a_bn_scale = addParam("_4b_a_bn_scale", "Constant", 1.0f, 1, 256, 1, 1);
JCudaTensor _4b_a_cv_B = addFixedParam("_4b_a_cv_B", "Constant", 0.0f, 256);
JCudaTensor _4b_a_cv_W = addParam("_4b_a_cv_W", "Random", 0.044194173f, 256, 1024, 1, 1);
JCudaTensor _4b_b_bn_bias = addParam("_4b_b_bn_bias", "Constant", 0.0f, 1, 256, 1, 1);
JCudaTensor _4b_b_bn_scale = addParam("_4b_b_bn_scale", "Constant", 1.0f, 1, 256, 1, 1);
JCudaTensor _4b_b_cv_B = addFixedParam("_4b_b_cv_B", "Constant", 0.0f, 256);
JCudaTensor _4b_b_cv_W = addParam("_4b_b_cv_W", "Random", 0.029462783f, 256, 256, 3, 3);
JCudaTensor _4b_c_bn_bias = addParam("_4b_c_bn_bias", "Constant", 0.0f, 1, 1024, 1, 1);
JCudaTensor _4b_c_bn_scale = addParam("_4b_c_bn_scale", "Constant", 1.0f, 1, 1024, 1, 1);
JCudaTensor _4b_c_cv_B = addFixedParam("_4b_c_cv_B", "Constant", 0.0f, 1024);
JCudaTensor _4b_c_cv_W = addParam("_4b_c_cv_W", "Random", 0.088388346f, 1024, 256, 1, 1);
JCudaTensor _4c_a_bn_bias = addParam("_4c_a_bn_bias", "Constant", 0.0f, 1, 256, 1, 1);
JCudaTensor _4c_a_bn_scale = addParam("_4c_a_bn_scale", "Constant", 1.0f, 1, 256, 1, 1);
JCudaTensor _4c_a_cv_B = addFixedParam("_4c_a_cv_B", "Constant", 0.0f, 256);
JCudaTensor _4c_a_cv_W = addParam("_4c_a_cv_W", "Random", 0.044194173f, 256, 1024, 1, 1);
JCudaTensor _4c_b_bn_bias = addParam("_4c_b_bn_bias", "Constant", 0.0f, 1, 256, 1, 1);
JCudaTensor _4c_b_bn_scale = addParam("_4c_b_bn_scale", "Constant", 1.0f, 1, 256, 1, 1);
JCudaTensor _4c_b_cv_B = addFixedParam("_4c_b_cv_B", "Constant", 0.0f, 256);
JCudaTensor _4c_b_cv_W = addParam("_4c_b_cv_W", "Random", 0.029462783f, 256, 256, 3, 3);
JCudaTensor _4c_c_bn_bias = addParam("_4c_c_bn_bias", "Constant", 0.0f, 1, 1024, 1, 1);
JCudaTensor _4c_c_bn_scale = addParam("_4c_c_bn_scale", "Constant", 1.0f, 1, 1024, 1, 1);
JCudaTensor _4c_c_cv_B = addFixedParam("_4c_c_cv_B", "Constant", 0.0f, 1024);
JCudaTensor _4c_c_cv_W = addParam("_4c_c_cv_W", "Random", 0.088388346f, 1024, 256, 1, 1);
JCudaTensor _4d_a_bn_bias = addParam("_4d_a_bn_bias", "Constant", 0.0f, 1, 256, 1, 1);
JCudaTensor _4d_a_bn_scale = addParam("_4d_a_bn_scale", "Constant", 1.0f, 1, 256, 1, 1);
JCudaTensor _4d_a_cv_B = addFixedParam("_4d_a_cv_B", "Constant", 0.0f, 256);
JCudaTensor _4d_a_cv_W = addParam("_4d_a_cv_W", "Random", 0.044194173f, 256, 1024, 1, 1);
JCudaTensor _4d_b_bn_bias = addParam("_4d_b_bn_bias", "Constant", 0.0f, 1, 256, 1, 1);
JCudaTensor _4d_b_bn_scale = addParam("_4d_b_bn_scale", "Constant", 1.0f, 1, 256, 1, 1);
JCudaTensor _4d_b_cv_B = addFixedParam("_4d_b_cv_B", "Constant", 0.0f, 256);
JCudaTensor _4d_b_cv_W = addParam("_4d_b_cv_W", "Random", 0.029462783f, 256, 256, 3, 3);
JCudaTensor _4d_c_bn_bias = addParam("_4d_c_bn_bias", "Constant", 0.0f, 1, 1024, 1, 1);
JCudaTensor _4d_c_bn_scale = addParam("_4d_c_bn_scale", "Constant", 1.0f, 1, 1024, 1, 1);
JCudaTensor _4d_c_cv_B = addFixedParam("_4d_c_cv_B", "Constant", 0.0f, 1024);
JCudaTensor _4d_c_cv_W = addParam("_4d_c_cv_W", "Random", 0.088388346f, 1024, 256, 1, 1);
JCudaTensor _4e_a_bn_bias = addParam("_4e_a_bn_bias", "Constant", 0.0f, 1, 256, 1, 1);
JCudaTensor _4e_a_bn_scale = addParam("_4e_a_bn_scale", "Constant", 1.0f, 1, 256, 1, 1);
JCudaTensor _4e_a_cv_B = addFixedParam("_4e_a_cv_B", "Constant", 0.0f, 256);
JCudaTensor _4e_a_cv_W = addParam("_4e_a_cv_W", "Random", 0.044194173f, 256, 1024, 1, 1);
JCudaTensor _4e_b_bn_bias = addParam("_4e_b_bn_bias", "Constant", 0.0f, 1, 256, 1, 1);
JCudaTensor _4e_b_bn_scale = addParam("_4e_b_bn_scale", "Constant", 1.0f, 1, 256, 1, 1);
JCudaTensor _4e_b_cv_B = addFixedParam("_4e_b_cv_B", "Constant", 0.0f, 256);
JCudaTensor _4e_b_cv_W = addParam("_4e_b_cv_W", "Random", 0.029462783f, 256, 256, 3, 3);
JCudaTensor _4e_c_bn_bias = addParam("_4e_c_bn_bias", "Constant", 0.0f, 1, 1024, 1, 1);
JCudaTensor _4e_c_bn_scale = addParam("_4e_c_bn_scale", "Constant", 1.0f, 1, 1024, 1, 1);
JCudaTensor _4e_c_cv_B = addFixedParam("_4e_c_cv_B", "Constant", 0.0f, 1024);
JCudaTensor _4e_c_cv_W = addParam("_4e_c_cv_W", "Random", 0.088388346f, 1024, 256, 1, 1);
JCudaTensor _4f_a_bn_bias = addParam("_4f_a_bn_bias", "Constant", 0.0f, 1, 256, 1, 1);
JCudaTensor _4f_a_bn_scale = addParam("_4f_a_bn_scale", "Constant", 1.0f, 1, 256, 1, 1);
JCudaTensor _4f_a_cv_B = addFixedParam("_4f_a_cv_B", "Constant", 0.0f, 256);
JCudaTensor _4f_a_cv_W = addParam("_4f_a_cv_W", "Random", 0.044194173f, 256, 1024, 1, 1);
JCudaTensor _4f_b_bn_bias = addParam("_4f_b_bn_bias", "Constant", 0.0f, 1, 256, 1, 1);
JCudaTensor _4f_b_bn_scale = addParam("_4f_b_bn_scale", "Constant", 1.0f, 1, 256, 1, 1);
JCudaTensor _4f_b_cv_B = addFixedParam("_4f_b_cv_B", "Constant", 0.0f, 256);
JCudaTensor _4f_b_cv_W = addParam("_4f_b_cv_W", "Random", 0.029462783f, 256, 256, 3, 3);
JCudaTensor _4f_c_bn_bias = addParam("_4f_c_bn_bias", "Constant", 0.0f, 1, 1024, 1, 1);
JCudaTensor _4f_c_bn_scale = addParam("_4f_c_bn_scale", "Constant", 1.0f, 1, 1024, 1, 1);
JCudaTensor _4f_c_cv_B = addFixedParam("_4f_c_cv_B", "Constant", 0.0f, 1024);
JCudaTensor _4f_c_cv_W = addParam("_4f_c_cv_W", "Random", 0.088388346f, 1024, 256, 1, 1);
JCudaTensor _5a1_bn_bias = addParam("_5a1_bn_bias", "Constant", 0.0f, 1, 2048, 1, 1);
JCudaTensor _5a1_bn_scale = addParam("_5a1_bn_scale", "Constant", 1.0f, 1, 2048, 1, 1);
JCudaTensor _5a1_cv_B = addFixedParam("_5a1_cv_B", "Constant", 0.0f, 2048);
JCudaTensor _5a1_cv_W = addParam("_5a1_cv_W", "Random", 0.044194173f, 2048, 1024, 1, 1);
JCudaTensor _5a2_a_bn_bias = addParam("_5a2_a_bn_bias", "Constant", 0.0f, 1, 512, 1, 1);
JCudaTensor _5a2_a_bn_scale = addParam("_5a2_a_bn_scale", "Constant", 1.0f, 1, 512, 1, 1);
JCudaTensor _5a2_a_cv_B = addFixedParam("_5a2_a_cv_B", "Constant", 0.0f, 512);
JCudaTensor _5a2_a_cv_W = addParam("_5a2_a_cv_W", "Random", 0.044194173f, 512, 1024, 1, 1);
JCudaTensor _5a2_b_bn_bias = addParam("_5a2_b_bn_bias", "Constant", 0.0f, 1, 512, 1, 1);
JCudaTensor _5a2_b_bn_scale = addParam("_5a2_b_bn_scale", "Constant", 1.0f, 1, 512, 1, 1);
JCudaTensor _5a2_b_cv_B = addFixedParam("_5a2_b_cv_B", "Constant", 0.0f, 512);
JCudaTensor _5a2_b_cv_W = addParam("_5a2_b_cv_W", "Random", 0.020833334f, 512, 512, 3, 3);
JCudaTensor _5a2_c_bn_bias = addParam("_5a2_c_bn_bias", "Constant", 0.0f, 1, 2048, 1, 1);
JCudaTensor _5a2_c_bn_scale = addParam("_5a2_c_bn_scale", "Constant", 1.0f, 1, 2048, 1, 1);
JCudaTensor _5a2_c_cv_B = addFixedParam("_5a2_c_cv_B", "Constant", 0.0f, 2048);
JCudaTensor _5a2_c_cv_W = addParam("_5a2_c_cv_W", "Random", 0.0625f, 2048, 512, 1, 1);
JCudaTensor _5b_a_bn_bias = addParam("_5b_a_bn_bias", "Constant", 0.0f, 1, 512, 1, 1);
JCudaTensor _5b_a_bn_scale = addParam("_5b_a_bn_scale", "Constant", 1.0f, 1, 512, 1, 1);
JCudaTensor _5b_a_cv_B = addFixedParam("_5b_a_cv_B", "Constant", 0.0f, 512);
JCudaTensor _5b_a_cv_W = addParam("_5b_a_cv_W", "Random", 0.03125f, 512, 2048, 1, 1);
JCudaTensor _5b_b_bn_bias = addParam("_5b_b_bn_bias", "Constant", 0.0f, 1, 512, 1, 1);
JCudaTensor _5b_b_bn_scale = addParam("_5b_b_bn_scale", "Constant", 1.0f, 1, 512, 1, 1);
JCudaTensor _5b_b_cv_B = addFixedParam("_5b_b_cv_B", "Constant", 0.0f, 512);
JCudaTensor _5b_b_cv_W = addParam("_5b_b_cv_W", "Random", 0.020833334f, 512, 512, 3, 3);
JCudaTensor _5b_c_bn_bias = addParam("_5b_c_bn_bias", "Constant", 0.0f, 1, 2048, 1, 1);
JCudaTensor _5b_c_bn_scale = addParam("_5b_c_bn_scale", "Constant", 1.0f, 1, 2048, 1, 1);
JCudaTensor _5b_c_cv_B = addFixedParam("_5b_c_cv_B", "Constant", 0.0f, 2048);
JCudaTensor _5b_c_cv_W = addParam("_5b_c_cv_W", "Random", 0.0625f, 2048, 512, 1, 1);
JCudaTensor _5c_a_bn_bias = addParam("_5c_a_bn_bias", "Constant", 0.0f, 1, 512, 1, 1);
JCudaTensor _5c_a_bn_scale = addParam("_5c_a_bn_scale", "Constant", 1.0f, 1, 512, 1, 1);
JCudaTensor _5c_a_cv_B = addFixedParam("_5c_a_cv_B", "Constant", 0.0f, 512);
JCudaTensor _5c_a_cv_W = addParam("_5c_a_cv_W", "Random", 0.03125f, 512, 2048, 1, 1);
JCudaTensor _5c_b_bn_bias = addParam("_5c_b_bn_bias", "Constant", 0.0f, 1, 512, 1, 1);
JCudaTensor _5c_b_bn_scale = addParam("_5c_b_bn_scale", "Constant", 1.0f, 1, 512, 1, 1);
JCudaTensor _5c_b_cv_B = addFixedParam("_5c_b_cv_B", "Constant", 0.0f, 512);
JCudaTensor _5c_b_cv_W = addParam("_5c_b_cv_W", "Random", 0.020833334f, 512, 512, 3, 3);
JCudaTensor _5c_c_bn_bias = addParam("_5c_c_bn_bias", "Constant", 0.0f, 1, 2048, 1, 1);
JCudaTensor _5c_c_bn_scale = addParam("_5c_c_bn_scale", "Constant", 1.0f, 1, 2048, 1, 1);
JCudaTensor _5c_c_cv_B = addFixedParam("_5c_c_cv_B", "Constant", 0.0f, 2048);
JCudaTensor _5c_c_cv_W = addParam("_5c_c_cv_W", "Random", 0.0625f, 2048, 512, 1, 1);
JCudaTensor V_1_bn_bias = addParam("V_1_bn_bias", "Constant", 0f, 1, 64, 1, 1);
JCudaTensor V_1_bn_scale = addParam("V_1_bn_scale", "Constant", 0f, 1, 64, 1, 1);
JCudaTensor V_1_cv_W = addParam("V_1_cv_W", "Constant", 0f, 64, 3, 7, 7);
JCudaTensor V_2a1_bn_bias = addParam("V_2a1_bn_bias", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_2a1_bn_scale = addParam("V_2a1_bn_scale", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_2a1_cv_W = addParam("V_2a1_cv_W", "Constant", 0f, 256, 64, 1, 1);
JCudaTensor V_2a2_a_bn_bias = addParam("V_2a2_a_bn_bias", "Constant", 0f, 1, 64, 1, 1);
JCudaTensor V_2a2_a_bn_scale = addParam("V_2a2_a_bn_scale", "Constant", 0f, 1, 64, 1, 1);
JCudaTensor V_2a2_a_cv_W = addParam("V_2a2_a_cv_W", "Constant", 0f, 64, 64, 1, 1);
JCudaTensor V_2a2_b_bn_bias = addParam("V_2a2_b_bn_bias", "Constant", 0f, 1, 64, 1, 1);
JCudaTensor V_2a2_b_bn_scale = addParam("V_2a2_b_bn_scale", "Constant", 0f, 1, 64, 1, 1);
JCudaTensor V_2a2_b_cv_W = addParam("V_2a2_b_cv_W", "Constant", 0f, 64, 64, 3, 3);
JCudaTensor V_2a2_c_bn_bias = addParam("V_2a2_c_bn_bias", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_2a2_c_bn_scale = addParam("V_2a2_c_bn_scale", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_2a2_c_cv_W = addParam("V_2a2_c_cv_W", "Constant", 0f, 256, 64, 1, 1);
JCudaTensor V_2b_a_bn_bias = addParam("V_2b_a_bn_bias", "Constant", 0f, 1, 64, 1, 1);
JCudaTensor V_2b_a_bn_scale = addParam("V_2b_a_bn_scale", "Constant", 0f, 1, 64, 1, 1);
JCudaTensor V_2b_a_cv_W = addParam("V_2b_a_cv_W", "Constant", 0f, 64, 256, 1, 1);
JCudaTensor V_2b_b_bn_bias = addParam("V_2b_b_bn_bias", "Constant", 0f, 1, 64, 1, 1);
JCudaTensor V_2b_b_bn_scale = addParam("V_2b_b_bn_scale", "Constant", 0f, 1, 64, 1, 1);
JCudaTensor V_2b_b_cv_W = addParam("V_2b_b_cv_W", "Constant", 0f, 64, 64, 3, 3);
JCudaTensor V_2b_c_bn_bias = addParam("V_2b_c_bn_bias", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_2b_c_bn_scale = addParam("V_2b_c_bn_scale", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_2b_c_cv_W = addParam("V_2b_c_cv_W", "Constant", 0f, 256, 64, 1, 1);
JCudaTensor V_2c_a_bn_bias = addParam("V_2c_a_bn_bias", "Constant", 0f, 1, 64, 1, 1);
JCudaTensor V_2c_a_bn_scale = addParam("V_2c_a_bn_scale", "Constant", 0f, 1, 64, 1, 1);
JCudaTensor V_2c_a_cv_W = addParam("V_2c_a_cv_W", "Constant", 0f, 64, 256, 1, 1);
JCudaTensor V_2c_b_bn_bias = addParam("V_2c_b_bn_bias", "Constant", 0f, 1, 64, 1, 1);
JCudaTensor V_2c_b_bn_scale = addParam("V_2c_b_bn_scale", "Constant", 0f, 1, 64, 1, 1);
JCudaTensor V_2c_b_cv_W = addParam("V_2c_b_cv_W", "Constant", 0f, 64, 64, 3, 3);
JCudaTensor V_2c_c_bn_bias = addParam("V_2c_c_bn_bias", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_2c_c_bn_scale = addParam("V_2c_c_bn_scale", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_2c_c_cv_W = addParam("V_2c_c_cv_W", "Constant", 0f, 256, 64, 1, 1);
JCudaTensor V_3a1_bn_bias = addParam("V_3a1_bn_bias", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_3a1_bn_scale = addParam("V_3a1_bn_scale", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_3a1_cv_W = addParam("V_3a1_cv_W", "Constant", 0f, 512, 256, 1, 1);
JCudaTensor V_3a2_a_bn_bias = addParam("V_3a2_a_bn_bias", "Constant", 0f, 1, 128, 1, 1);
JCudaTensor V_3a2_a_bn_scale = addParam("V_3a2_a_bn_scale", "Constant", 0f, 1, 128, 1, 1);
JCudaTensor V_3a2_a_cv_W = addParam("V_3a2_a_cv_W", "Constant", 0f, 128, 256, 1, 1);
JCudaTensor V_3a2_b_bn_bias = addParam("V_3a2_b_bn_bias", "Constant", 0f, 1, 128, 1, 1);
JCudaTensor V_3a2_b_bn_scale = addParam("V_3a2_b_bn_scale", "Constant", 0f, 1, 128, 1, 1);
JCudaTensor V_3a2_b_cv_W = addParam("V_3a2_b_cv_W", "Constant", 0f, 128, 128, 3, 3);
JCudaTensor V_3a2_c_bn_bias = addParam("V_3a2_c_bn_bias", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_3a2_c_bn_scale = addParam("V_3a2_c_bn_scale", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_3a2_c_cv_W = addParam("V_3a2_c_cv_W", "Constant", 0f, 512, 128, 1, 1);
JCudaTensor V_3b_a_bn_bias = addParam("V_3b_a_bn_bias", "Constant", 0f, 1, 128, 1, 1);
JCudaTensor V_3b_a_bn_scale = addParam("V_3b_a_bn_scale", "Constant", 0f, 1, 128, 1, 1);
JCudaTensor V_3b_a_cv_W = addParam("V_3b_a_cv_W", "Constant", 0f, 128, 512, 1, 1);
JCudaTensor V_3b_b_bn_bias = addParam("V_3b_b_bn_bias", "Constant", 0f, 1, 128, 1, 1);
JCudaTensor V_3b_b_bn_scale = addParam("V_3b_b_bn_scale", "Constant", 0f, 1, 128, 1, 1);
JCudaTensor V_3b_b_cv_W = addParam("V_3b_b_cv_W", "Constant", 0f, 128, 128, 3, 3);
JCudaTensor V_3b_c_bn_bias = addParam("V_3b_c_bn_bias", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_3b_c_bn_scale = addParam("V_3b_c_bn_scale", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_3b_c_cv_W = addParam("V_3b_c_cv_W", "Constant", 0f, 512, 128, 1, 1);
JCudaTensor V_3c_a_bn_bias = addParam("V_3c_a_bn_bias", "Constant", 0f, 1, 128, 1, 1);
JCudaTensor V_3c_a_bn_scale = addParam("V_3c_a_bn_scale", "Constant", 0f, 1, 128, 1, 1);
JCudaTensor V_3c_a_cv_W = addParam("V_3c_a_cv_W", "Constant", 0f, 128, 512, 1, 1);
JCudaTensor V_3c_b_bn_bias = addParam("V_3c_b_bn_bias", "Constant", 0f, 1, 128, 1, 1);
JCudaTensor V_3c_b_bn_scale = addParam("V_3c_b_bn_scale", "Constant", 0f, 1, 128, 1, 1);
JCudaTensor V_3c_b_cv_W = addParam("V_3c_b_cv_W", "Constant", 0f, 128, 128, 3, 3);
JCudaTensor V_3c_c_bn_bias = addParam("V_3c_c_bn_bias", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_3c_c_bn_scale = addParam("V_3c_c_bn_scale", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_3c_c_cv_W = addParam("V_3c_c_cv_W", "Constant", 0f, 512, 128, 1, 1);
JCudaTensor V_3d_a_bn_bias = addParam("V_3d_a_bn_bias", "Constant", 0f, 1, 128, 1, 1);
JCudaTensor V_3d_a_bn_scale = addParam("V_3d_a_bn_scale", "Constant", 0f, 1, 128, 1, 1);
JCudaTensor V_3d_a_cv_W = addParam("V_3d_a_cv_W", "Constant", 0f, 128, 512, 1, 1);
JCudaTensor V_3d_b_bn_bias = addParam("V_3d_b_bn_bias", "Constant", 0f, 1, 128, 1, 1);
JCudaTensor V_3d_b_bn_scale = addParam("V_3d_b_bn_scale", "Constant", 0f, 1, 128, 1, 1);
JCudaTensor V_3d_b_cv_W = addParam("V_3d_b_cv_W", "Constant", 0f, 128, 128, 3, 3);
JCudaTensor V_3d_c_bn_bias = addParam("V_3d_c_bn_bias", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_3d_c_bn_scale = addParam("V_3d_c_bn_scale", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_3d_c_cv_W = addParam("V_3d_c_cv_W", "Constant", 0f, 512, 128, 1, 1);
JCudaTensor V_4a1_bn_bias = addParam("V_4a1_bn_bias", "Constant", 0f, 1, 1024, 1, 1);
JCudaTensor V_4a1_bn_scale = addParam("V_4a1_bn_scale", "Constant", 0f, 1, 1024, 1, 1);
JCudaTensor V_4a1_cv_W = addParam("V_4a1_cv_W", "Constant", 0f, 1024, 512, 1, 1);
JCudaTensor V_4a2_a_bn_bias = addParam("V_4a2_a_bn_bias", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4a2_a_bn_scale = addParam("V_4a2_a_bn_scale", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4a2_a_cv_W = addParam("V_4a2_a_cv_W", "Constant", 0f, 256, 512, 1, 1);
JCudaTensor V_4a2_b_bn_bias = addParam("V_4a2_b_bn_bias", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4a2_b_bn_scale = addParam("V_4a2_b_bn_scale", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4a2_b_cv_W = addParam("V_4a2_b_cv_W", "Constant", 0f, 256, 256, 3, 3);
JCudaTensor V_4a2_c_bn_bias = addParam("V_4a2_c_bn_bias", "Constant", 0f, 1, 1024, 1, 1);
JCudaTensor V_4a2_c_bn_scale = addParam("V_4a2_c_bn_scale", "Constant", 0f, 1, 1024, 1, 1);
JCudaTensor V_4a2_c_cv_W = addParam("V_4a2_c_cv_W", "Constant", 0f, 1024, 256, 1, 1);
JCudaTensor V_4b_a_bn_bias = addParam("V_4b_a_bn_bias", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4b_a_bn_scale = addParam("V_4b_a_bn_scale", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4b_a_cv_W = addParam("V_4b_a_cv_W", "Constant", 0f, 256, 1024, 1, 1);
JCudaTensor V_4b_b_bn_bias = addParam("V_4b_b_bn_bias", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4b_b_bn_scale = addParam("V_4b_b_bn_scale", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4b_b_cv_W = addParam("V_4b_b_cv_W", "Constant", 0f, 256, 256, 3, 3);
JCudaTensor V_4b_c_bn_bias = addParam("V_4b_c_bn_bias", "Constant", 0f, 1, 1024, 1, 1);
JCudaTensor V_4b_c_bn_scale = addParam("V_4b_c_bn_scale", "Constant", 0f, 1, 1024, 1, 1);
JCudaTensor V_4b_c_cv_W = addParam("V_4b_c_cv_W", "Constant", 0f, 1024, 256, 1, 1);
JCudaTensor V_4c_a_bn_bias = addParam("V_4c_a_bn_bias", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4c_a_bn_scale = addParam("V_4c_a_bn_scale", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4c_a_cv_W = addParam("V_4c_a_cv_W", "Constant", 0f, 256, 1024, 1, 1);
JCudaTensor V_4c_b_bn_bias = addParam("V_4c_b_bn_bias", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4c_b_bn_scale = addParam("V_4c_b_bn_scale", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4c_b_cv_W = addParam("V_4c_b_cv_W", "Constant", 0f, 256, 256, 3, 3);
JCudaTensor V_4c_c_bn_bias = addParam("V_4c_c_bn_bias", "Constant", 0f, 1, 1024, 1, 1);
JCudaTensor V_4c_c_bn_scale = addParam("V_4c_c_bn_scale", "Constant", 0f, 1, 1024, 1, 1);
JCudaTensor V_4c_c_cv_W = addParam("V_4c_c_cv_W", "Constant", 0f, 1024, 256, 1, 1);
JCudaTensor V_4d_a_bn_bias = addParam("V_4d_a_bn_bias", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4d_a_bn_scale = addParam("V_4d_a_bn_scale", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4d_a_cv_W = addParam("V_4d_a_cv_W", "Constant", 0f, 256, 1024, 1, 1);
JCudaTensor V_4d_b_bn_bias = addParam("V_4d_b_bn_bias", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4d_b_bn_scale = addParam("V_4d_b_bn_scale", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4d_b_cv_W = addParam("V_4d_b_cv_W", "Constant", 0f, 256, 256, 3, 3);
JCudaTensor V_4d_c_bn_bias = addParam("V_4d_c_bn_bias", "Constant", 0f, 1, 1024, 1, 1);
JCudaTensor V_4d_c_bn_scale = addParam("V_4d_c_bn_scale", "Constant", 0f, 1, 1024, 1, 1);
JCudaTensor V_4d_c_cv_W = addParam("V_4d_c_cv_W", "Constant", 0f, 1024, 256, 1, 1);
JCudaTensor V_4e_a_bn_bias = addParam("V_4e_a_bn_bias", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4e_a_bn_scale = addParam("V_4e_a_bn_scale", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4e_a_cv_W = addParam("V_4e_a_cv_W", "Constant", 0f, 256, 1024, 1, 1);
JCudaTensor V_4e_b_bn_bias = addParam("V_4e_b_bn_bias", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4e_b_bn_scale = addParam("V_4e_b_bn_scale", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4e_b_cv_W = addParam("V_4e_b_cv_W", "Constant", 0f, 256, 256, 3, 3);
JCudaTensor V_4e_c_bn_bias = addParam("V_4e_c_bn_bias", "Constant", 0f, 1, 1024, 1, 1);
JCudaTensor V_4e_c_bn_scale = addParam("V_4e_c_bn_scale", "Constant", 0f, 1, 1024, 1, 1);
JCudaTensor V_4e_c_cv_W = addParam("V_4e_c_cv_W", "Constant", 0f, 1024, 256, 1, 1);
JCudaTensor V_4f_a_bn_bias = addParam("V_4f_a_bn_bias", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4f_a_bn_scale = addParam("V_4f_a_bn_scale", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4f_a_cv_W = addParam("V_4f_a_cv_W", "Constant", 0f, 256, 1024, 1, 1);
JCudaTensor V_4f_b_bn_bias = addParam("V_4f_b_bn_bias", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4f_b_bn_scale = addParam("V_4f_b_bn_scale", "Constant", 0f, 1, 256, 1, 1);
JCudaTensor V_4f_b_cv_W = addParam("V_4f_b_cv_W", "Constant", 0f, 256, 256, 3, 3);
JCudaTensor V_4f_c_bn_bias = addParam("V_4f_c_bn_bias", "Constant", 0f, 1, 1024, 1, 1);
JCudaTensor V_4f_c_bn_scale = addParam("V_4f_c_bn_scale", "Constant", 0f, 1, 1024, 1, 1);
JCudaTensor V_4f_c_cv_W = addParam("V_4f_c_cv_W", "Constant", 0f, 1024, 256, 1, 1);
JCudaTensor V_5a1_bn_bias = addParam("V_5a1_bn_bias", "Constant", 0f, 1, 2048, 1, 1);
JCudaTensor V_5a1_bn_scale = addParam("V_5a1_bn_scale", "Constant", 0f, 1, 2048, 1, 1);
JCudaTensor V_5a1_cv_W = addParam("V_5a1_cv_W", "Constant", 0f, 2048, 1024, 1, 1);
JCudaTensor V_5a2_a_bn_bias = addParam("V_5a2_a_bn_bias", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_5a2_a_bn_scale = addParam("V_5a2_a_bn_scale", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_5a2_a_cv_W = addParam("V_5a2_a_cv_W", "Constant", 0f, 512, 1024, 1, 1);
JCudaTensor V_5a2_b_bn_bias = addParam("V_5a2_b_bn_bias", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_5a2_b_bn_scale = addParam("V_5a2_b_bn_scale", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_5a2_b_cv_W = addParam("V_5a2_b_cv_W", "Constant", 0f, 512, 512, 3, 3);
JCudaTensor V_5a2_c_bn_bias = addParam("V_5a2_c_bn_bias", "Constant", 0f, 1, 2048, 1, 1);
JCudaTensor V_5a2_c_bn_scale = addParam("V_5a2_c_bn_scale", "Constant", 0f, 1, 2048, 1, 1);
JCudaTensor V_5a2_c_cv_W = addParam("V_5a2_c_cv_W", "Constant", 0f, 2048, 512, 1, 1);
JCudaTensor V_5b_a_bn_bias = addParam("V_5b_a_bn_bias", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_5b_a_bn_scale = addParam("V_5b_a_bn_scale", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_5b_a_cv_W = addParam("V_5b_a_cv_W", "Constant", 0f, 512, 2048, 1, 1);
JCudaTensor V_5b_b_bn_bias = addParam("V_5b_b_bn_bias", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_5b_b_bn_scale = addParam("V_5b_b_bn_scale", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_5b_b_cv_W = addParam("V_5b_b_cv_W", "Constant", 0f, 512, 512, 3, 3);
JCudaTensor V_5b_c_bn_bias = addParam("V_5b_c_bn_bias", "Constant", 0f, 1, 2048, 1, 1);
JCudaTensor V_5b_c_bn_scale = addParam("V_5b_c_bn_scale", "Constant", 0f, 1, 2048, 1, 1);
JCudaTensor V_5b_c_cv_W = addParam("V_5b_c_cv_W", "Constant", 0f, 2048, 512, 1, 1);
JCudaTensor V_5c_a_bn_bias = addParam("V_5c_a_bn_bias", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_5c_a_bn_scale = addParam("V_5c_a_bn_scale", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_5c_a_cv_W = addParam("V_5c_a_cv_W", "Constant", 0f, 512, 2048, 1, 1);
JCudaTensor V_5c_b_bn_bias = addParam("V_5c_b_bn_bias", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_5c_b_bn_scale = addParam("V_5c_b_bn_scale", "Constant", 0f, 1, 512, 1, 1);
JCudaTensor V_5c_b_cv_W = addParam("V_5c_b_cv_W", "Constant", 0f, 512, 512, 3, 3);
JCudaTensor V_5c_c_bn_bias = addParam("V_5c_c_bn_bias", "Constant", 0f, 1, 2048, 1, 1);
JCudaTensor V_5c_c_bn_scale = addParam("V_5c_c_bn_scale", "Constant", 0f, 1, 2048, 1, 1);
JCudaTensor V_5c_c_cv_W = addParam("V_5c_c_cv_W", "Constant", 0f, 2048, 512, 1, 1);
JCudaTensor V_fc_B = addParam("V_fc_B", "Constant", 0f, 1000);
JCudaTensor V_fc_W = addParam("V_fc_W", "Constant", 0f, 1000, 2048);
JCudaTensor fc_B = addParam("fc_B", "Constant", 0.0f, 1000);
JCudaTensor fc_W = addParam("fc_W", "Random", 0.03125f, 1000, 2048);

public float trainFunction(JTensorFloat X, JTensorFloat Y) {
 // val X1325 = Cuda(Indicator(Y, 1000))
JCudaTensor X1325 = Y.asIndicator(1000).asJCudaTensor();
// val X1324 = Cuda(X)
JCudaTensor X1324 = X.asJCudaTensor();
// val X397 = Convolv(2,3)(X1324,1_cv_W,1_cv_B)
JCudaTensor X397 = y138.forward(X1324, _1_cv_W, _1_cv_B);
// val X398 = BatchNorm(1_bn)(X397,1_bn_scale,1_bn_bias)
JCudaTensor X398 = y136.forward(X397, _1_bn_scale, _1_bn_bias);
// val X399 = ReLU()(X398)
JCudaTensor X399 = y135.forward(X398);
// val X400 = Pooling(3,2,0,true)(X399)
JCudaTensor X400 = y134.forward(X399);
// val X401 = Convolv(1,0)(X400,2a1_cv_W,2a1_cv_B)
JCudaTensor X401 = y111.forward(X400, _2a1_cv_W, _2a1_cv_B);
// val X404 = Convolv(1,0)(X400,2a2_a_cv_W,2a2_a_cv_B)
JCudaTensor X404 = y139.forward(X400, _2a2_a_cv_W, _2a2_a_cv_B);
// val X405 = BatchNorm(2a2_a_bn)(X404,2a2_a_bn_scale,2a2_a_bn_bias)
JCudaTensor X405 = y131.forward(X404, _2a2_a_bn_scale, _2a2_a_bn_bias);
// val X402 = BatchNorm(2a1_bn)(X401,2a1_bn_scale,2a1_bn_bias)
JCudaTensor X402 = y125.forward(X401, _2a1_bn_scale, _2a1_bn_bias);
// val X406 = ReLU()(X405)
JCudaTensor X406 = y112.forward(X405);
// val X403 = ReLU()(X402)
JCudaTensor X403 = y108.forward(X402);
// val X407 = Convolv(1,1)(X406,2a2_b_cv_W,2a2_b_cv_B)
JCudaTensor X407 = y115.forward(X406, _2a2_b_cv_W, _2a2_b_cv_B);
// val X408 = BatchNorm(2a2_b_bn)(X407,2a2_b_bn_scale,2a2_b_bn_bias)
JCudaTensor X408 = y129.forward(X407, _2a2_b_bn_scale, _2a2_b_bn_bias);
// val X409 = ReLU()(X408)
JCudaTensor X409 = y112.forward(X408);
// val X410 = Convolv(1,0)(X409,2a2_c_cv_W,2a2_c_cv_B)
JCudaTensor X410 = y111.forward(X409, _2a2_c_cv_W, _2a2_c_cv_B);
// val X411 = BatchNorm(2a2_c_bn)(X410,2a2_c_bn_scale,2a2_c_bn_bias)
JCudaTensor X411 = y127.forward(X410, _2a2_c_bn_scale, _2a2_c_bn_bias);
// val X412 = ReLU()(X411)
JCudaTensor X412 = y108.forward(X411);
// val X413 = (X403.copy + X412)
JCudaTensor X413 = X403.clone().plus_i(X412);;
// val X414 = ReLU()(X413)
JCudaTensor X414 = y108.forward(X413);
// val X415 = Convolv(1,0)(X414,2b_a_cv_W,2b_a_cv_B)
JCudaTensor X415 = y118.forward(X414, _2b_a_cv_W, _2b_a_cv_B);
// val X416 = BatchNorm(2b_a_bn)(X415,2b_a_bn_scale,2b_a_bn_bias)
JCudaTensor X416 = y123.forward(X415, _2b_a_bn_scale, _2b_a_bn_bias);
// val X417 = ReLU()(X416)
JCudaTensor X417 = y112.forward(X416);
// val X418 = Convolv(1,1)(X417,2b_b_cv_W,2b_b_cv_B)
JCudaTensor X418 = y115.forward(X417, _2b_b_cv_W, _2b_b_cv_B);
// val X419 = BatchNorm(2b_b_bn)(X418,2b_b_bn_scale,2b_b_bn_bias)
JCudaTensor X419 = y121.forward(X418, _2b_b_bn_scale, _2b_b_bn_bias);
// val X420 = ReLU()(X419)
JCudaTensor X420 = y112.forward(X419);
// val X421 = Convolv(1,0)(X420,2b_c_cv_W,2b_c_cv_B)
JCudaTensor X421 = y111.forward(X420, _2b_c_cv_W, _2b_c_cv_B);
// val X422 = BatchNorm(2b_c_bn)(X421,2b_c_bn_scale,2b_c_bn_bias)
JCudaTensor X422 = y119.forward(X421, _2b_c_bn_scale, _2b_c_bn_bias);
// val X423 = ReLU()(X422)
JCudaTensor X423 = y108.forward(X422);
// val X424 = (X423.copy + X414)
JCudaTensor X424 = X423.clone().plus_i(X414);;
// val X425 = ReLU()(X424)
JCudaTensor X425 = y108.forward(X424);
// val X426 = Convolv(1,0)(X425,2c_a_cv_W,2c_a_cv_B)
JCudaTensor X426 = y118.forward(X425, _2c_a_cv_W, _2c_a_cv_B);
// val X427 = BatchNorm(2c_a_bn)(X426,2c_a_bn_scale,2c_a_bn_bias)
JCudaTensor X427 = y116.forward(X426, _2c_a_bn_scale, _2c_a_bn_bias);
// val X428 = ReLU()(X427)
JCudaTensor X428 = y112.forward(X427);
// val X429 = Convolv(1,1)(X428,2c_b_cv_W,2c_b_cv_B)
JCudaTensor X429 = y115.forward(X428, _2c_b_cv_W, _2c_b_cv_B);
// val X430 = BatchNorm(2c_b_bn)(X429,2c_b_bn_scale,2c_b_bn_bias)
JCudaTensor X430 = y113.forward(X429, _2c_b_bn_scale, _2c_b_bn_bias);
// val X431 = ReLU()(X430)
JCudaTensor X431 = y112.forward(X430);
// val X432 = Convolv(1,0)(X431,2c_c_cv_W,2c_c_cv_B)
JCudaTensor X432 = y111.forward(X431, _2c_c_cv_W, _2c_c_cv_B);
// val X433 = BatchNorm(2c_c_bn)(X432,2c_c_bn_scale,2c_c_bn_bias)
JCudaTensor X433 = y109.forward(X432, _2c_c_bn_scale, _2c_c_bn_bias);
// val X434 = ReLU()(X433)
JCudaTensor X434 = y108.forward(X433);
// val X435 = (X434.copy + X425)
JCudaTensor X435 = X434.clone().plus_i(X425);;
// val X436 = ReLU()(X435)
JCudaTensor X436 = y108.forward(X435);
// val X440 = Convolv(2,0)(X436,3a2_a_cv_W,3a2_a_cv_B)
JCudaTensor X440 = y140.forward(X436, _3a2_a_cv_W, _3a2_a_cv_B);
// val X437 = Convolv(2,0)(X436,3a1_cv_W,3a1_cv_B)
JCudaTensor X437 = y100.forward(X436, _3a1_cv_W, _3a1_cv_B);
// val X438 = BatchNorm(3a1_bn)(X437,3a1_bn_scale,3a1_bn_bias)
JCudaTensor X438 = y98.forward(X437, _3a1_bn_scale, _3a1_bn_bias);
// val X441 = BatchNorm(3a2_a_bn)(X440,3a2_a_bn_scale,3a2_a_bn_bias)
JCudaTensor X441 = y105.forward(X440, _3a2_a_bn_scale, _3a2_a_bn_bias);
// val X442 = ReLU()(X441)
JCudaTensor X442 = y79.forward(X441);
// val X439 = ReLU()(X438)
JCudaTensor X439 = y75.forward(X438);
// val X443 = Convolv(1,1)(X442,3a2_b_cv_W,3a2_b_cv_B)
JCudaTensor X443 = y82.forward(X442, _3a2_b_cv_W, _3a2_b_cv_B);
// val X444 = BatchNorm(3a2_b_bn)(X443,3a2_b_bn_scale,3a2_b_bn_bias)
JCudaTensor X444 = y103.forward(X443, _3a2_b_bn_scale, _3a2_b_bn_bias);
// val X445 = ReLU()(X444)
JCudaTensor X445 = y79.forward(X444);
// val X446 = Convolv(1,0)(X445,3a2_c_cv_W,3a2_c_cv_B)
JCudaTensor X446 = y78.forward(X445, _3a2_c_cv_W, _3a2_c_cv_B);
// val X447 = BatchNorm(3a2_c_bn)(X446,3a2_c_bn_scale,3a2_c_bn_bias)
JCudaTensor X447 = y101.forward(X446, _3a2_c_bn_scale, _3a2_c_bn_bias);
// val X448 = ReLU()(X447)
JCudaTensor X448 = y75.forward(X447);
// val X449 = (X439.copy + X448)
JCudaTensor X449 = X439.clone().plus_i(X448);;
// val X450 = ReLU()(X449)
JCudaTensor X450 = y75.forward(X449);
// val X451 = Convolv(1,0)(X450,3b_a_cv_W,3b_a_cv_B)
JCudaTensor X451 = y85.forward(X450, _3b_a_cv_W, _3b_a_cv_B);
// val X452 = BatchNorm(3b_a_bn)(X451,3b_a_bn_scale,3b_a_bn_bias)
JCudaTensor X452 = y96.forward(X451, _3b_a_bn_scale, _3b_a_bn_bias);
// val X453 = ReLU()(X452)
JCudaTensor X453 = y79.forward(X452);
// val X454 = Convolv(1,1)(X453,3b_b_cv_W,3b_b_cv_B)
JCudaTensor X454 = y82.forward(X453, _3b_b_cv_W, _3b_b_cv_B);
// val X455 = BatchNorm(3b_b_bn)(X454,3b_b_bn_scale,3b_b_bn_bias)
JCudaTensor X455 = y94.forward(X454, _3b_b_bn_scale, _3b_b_bn_bias);
// val X456 = ReLU()(X455)
JCudaTensor X456 = y79.forward(X455);
// val X457 = Convolv(1,0)(X456,3b_c_cv_W,3b_c_cv_B)
JCudaTensor X457 = y78.forward(X456, _3b_c_cv_W, _3b_c_cv_B);
// val X458 = BatchNorm(3b_c_bn)(X457,3b_c_bn_scale,3b_c_bn_bias)
JCudaTensor X458 = y92.forward(X457, _3b_c_bn_scale, _3b_c_bn_bias);
// val X459 = ReLU()(X458)
JCudaTensor X459 = y75.forward(X458);
// val X460 = (X459.copy + X450)
JCudaTensor X460 = X459.clone().plus_i(X450);;
// val X461 = ReLU()(X460)
JCudaTensor X461 = y75.forward(X460);
// val X462 = Convolv(1,0)(X461,3c_a_cv_W,3c_a_cv_B)
JCudaTensor X462 = y85.forward(X461, _3c_a_cv_W, _3c_a_cv_B);
// val X463 = BatchNorm(3c_a_bn)(X462,3c_a_bn_scale,3c_a_bn_bias)
JCudaTensor X463 = y90.forward(X462, _3c_a_bn_scale, _3c_a_bn_bias);
// val X464 = ReLU()(X463)
JCudaTensor X464 = y79.forward(X463);
// val X465 = Convolv(1,1)(X464,3c_b_cv_W,3c_b_cv_B)
JCudaTensor X465 = y82.forward(X464, _3c_b_cv_W, _3c_b_cv_B);
// val X466 = BatchNorm(3c_b_bn)(X465,3c_b_bn_scale,3c_b_bn_bias)
JCudaTensor X466 = y88.forward(X465, _3c_b_bn_scale, _3c_b_bn_bias);
// val X467 = ReLU()(X466)
JCudaTensor X467 = y79.forward(X466);
// val X468 = Convolv(1,0)(X467,3c_c_cv_W,3c_c_cv_B)
JCudaTensor X468 = y78.forward(X467, _3c_c_cv_W, _3c_c_cv_B);
// val X469 = BatchNorm(3c_c_bn)(X468,3c_c_bn_scale,3c_c_bn_bias)
JCudaTensor X469 = y86.forward(X468, _3c_c_bn_scale, _3c_c_bn_bias);
// val X470 = ReLU()(X469)
JCudaTensor X470 = y75.forward(X469);
// val X471 = (X470.copy + X461)
JCudaTensor X471 = X470.clone().plus_i(X461);;
// val X472 = ReLU()(X471)
JCudaTensor X472 = y75.forward(X471);
// val X473 = Convolv(1,0)(X472,3d_a_cv_W,3d_a_cv_B)
JCudaTensor X473 = y85.forward(X472, _3d_a_cv_W, _3d_a_cv_B);
// val X474 = BatchNorm(3d_a_bn)(X473,3d_a_bn_scale,3d_a_bn_bias)
JCudaTensor X474 = y83.forward(X473, _3d_a_bn_scale, _3d_a_bn_bias);
// val X475 = ReLU()(X474)
JCudaTensor X475 = y79.forward(X474);
// val X476 = Convolv(1,1)(X475,3d_b_cv_W,3d_b_cv_B)
JCudaTensor X476 = y82.forward(X475, _3d_b_cv_W, _3d_b_cv_B);
// val X477 = BatchNorm(3d_b_bn)(X476,3d_b_bn_scale,3d_b_bn_bias)
JCudaTensor X477 = y80.forward(X476, _3d_b_bn_scale, _3d_b_bn_bias);
// val X478 = ReLU()(X477)
JCudaTensor X478 = y79.forward(X477);
// val X479 = Convolv(1,0)(X478,3d_c_cv_W,3d_c_cv_B)
JCudaTensor X479 = y78.forward(X478, _3d_c_cv_W, _3d_c_cv_B);
// val X480 = BatchNorm(3d_c_bn)(X479,3d_c_bn_scale,3d_c_bn_bias)
JCudaTensor X480 = y76.forward(X479, _3d_c_bn_scale, _3d_c_bn_bias);
// val X481 = ReLU()(X480)
JCudaTensor X481 = y75.forward(X480);
// val X482 = (X481.copy + X472)
JCudaTensor X482 = X481.clone().plus_i(X472);;
// val X483 = ReLU()(X482)
JCudaTensor X483 = y75.forward(X482);
// val X487 = Convolv(2,0)(X483,4a2_a_cv_W,4a2_a_cv_B)
JCudaTensor X487 = y141.forward(X483, _4a2_a_cv_W, _4a2_a_cv_B);
// val X484 = Convolv(2,0)(X483,4a1_cv_W,4a1_cv_B)
JCudaTensor X484 = y67.forward(X483, _4a1_cv_W, _4a1_cv_B);
// val X488 = BatchNorm(4a2_a_bn)(X487,4a2_a_bn_scale,4a2_a_bn_bias)
JCudaTensor X488 = y72.forward(X487, _4a2_a_bn_scale, _4a2_a_bn_bias);
// val X485 = BatchNorm(4a1_bn)(X484,4a1_bn_scale,4a1_bn_bias)
JCudaTensor X485 = y65.forward(X484, _4a1_bn_scale, _4a1_bn_bias);
// val X489 = ReLU()(X488)
JCudaTensor X489 = y34.forward(X488);
// val X486 = ReLU()(X485)
JCudaTensor X486 = y30.forward(X485);
// val X490 = Convolv(1,1)(X489,4a2_b_cv_W,4a2_b_cv_B)
JCudaTensor X490 = y37.forward(X489, _4a2_b_cv_W, _4a2_b_cv_B);
// val X491 = BatchNorm(4a2_b_bn)(X490,4a2_b_bn_scale,4a2_b_bn_bias)
JCudaTensor X491 = y70.forward(X490, _4a2_b_bn_scale, _4a2_b_bn_bias);
// val X492 = ReLU()(X491)
JCudaTensor X492 = y34.forward(X491);
// val X493 = Convolv(1,0)(X492,4a2_c_cv_W,4a2_c_cv_B)
JCudaTensor X493 = y33.forward(X492, _4a2_c_cv_W, _4a2_c_cv_B);
// val X494 = BatchNorm(4a2_c_bn)(X493,4a2_c_bn_scale,4a2_c_bn_bias)
JCudaTensor X494 = y68.forward(X493, _4a2_c_bn_scale, _4a2_c_bn_bias);
// val X495 = ReLU()(X494)
JCudaTensor X495 = y30.forward(X494);
// val X496 = (X486.copy + X495)
JCudaTensor X496 = X486.clone().plus_i(X495);;
// val X497 = ReLU()(X496)
JCudaTensor X497 = y30.forward(X496);
// val X498 = Convolv(1,0)(X497,4b_a_cv_W,4b_a_cv_B)
JCudaTensor X498 = y40.forward(X497, _4b_a_cv_W, _4b_a_cv_B);
// val X499 = BatchNorm(4b_a_bn)(X498,4b_a_bn_scale,4b_a_bn_bias)
JCudaTensor X499 = y63.forward(X498, _4b_a_bn_scale, _4b_a_bn_bias);
// val X500 = ReLU()(X499)
JCudaTensor X500 = y34.forward(X499);
// val X501 = Convolv(1,1)(X500,4b_b_cv_W,4b_b_cv_B)
JCudaTensor X501 = y37.forward(X500, _4b_b_cv_W, _4b_b_cv_B);
// val X502 = BatchNorm(4b_b_bn)(X501,4b_b_bn_scale,4b_b_bn_bias)
JCudaTensor X502 = y61.forward(X501, _4b_b_bn_scale, _4b_b_bn_bias);
// val X503 = ReLU()(X502)
JCudaTensor X503 = y34.forward(X502);
// val X504 = Convolv(1,0)(X503,4b_c_cv_W,4b_c_cv_B)
JCudaTensor X504 = y33.forward(X503, _4b_c_cv_W, _4b_c_cv_B);
// val X505 = BatchNorm(4b_c_bn)(X504,4b_c_bn_scale,4b_c_bn_bias)
JCudaTensor X505 = y59.forward(X504, _4b_c_bn_scale, _4b_c_bn_bias);
// val X506 = ReLU()(X505)
JCudaTensor X506 = y30.forward(X505);
// val X507 = (X506.copy + X497)
JCudaTensor X507 = X506.clone().plus_i(X497);;
// val X508 = ReLU()(X507)
JCudaTensor X508 = y30.forward(X507);
// val X509 = Convolv(1,0)(X508,4c_a_cv_W,4c_a_cv_B)
JCudaTensor X509 = y40.forward(X508, _4c_a_cv_W, _4c_a_cv_B);
// val X510 = BatchNorm(4c_a_bn)(X509,4c_a_bn_scale,4c_a_bn_bias)
JCudaTensor X510 = y57.forward(X509, _4c_a_bn_scale, _4c_a_bn_bias);
// val X511 = ReLU()(X510)
JCudaTensor X511 = y34.forward(X510);
// val X512 = Convolv(1,1)(X511,4c_b_cv_W,4c_b_cv_B)
JCudaTensor X512 = y37.forward(X511, _4c_b_cv_W, _4c_b_cv_B);
// val X513 = BatchNorm(4c_b_bn)(X512,4c_b_bn_scale,4c_b_bn_bias)
JCudaTensor X513 = y55.forward(X512, _4c_b_bn_scale, _4c_b_bn_bias);
// val X514 = ReLU()(X513)
JCudaTensor X514 = y34.forward(X513);
// val X515 = Convolv(1,0)(X514,4c_c_cv_W,4c_c_cv_B)
JCudaTensor X515 = y33.forward(X514, _4c_c_cv_W, _4c_c_cv_B);
// val X516 = BatchNorm(4c_c_bn)(X515,4c_c_bn_scale,4c_c_bn_bias)
JCudaTensor X516 = y53.forward(X515, _4c_c_bn_scale, _4c_c_bn_bias);
// val X517 = ReLU()(X516)
JCudaTensor X517 = y30.forward(X516);
// val X518 = (X517.copy + X508)
JCudaTensor X518 = X517.clone().plus_i(X508);;
// val X519 = ReLU()(X518)
JCudaTensor X519 = y30.forward(X518);
// val X520 = Convolv(1,0)(X519,4d_a_cv_W,4d_a_cv_B)
JCudaTensor X520 = y40.forward(X519, _4d_a_cv_W, _4d_a_cv_B);
// val X521 = BatchNorm(4d_a_bn)(X520,4d_a_bn_scale,4d_a_bn_bias)
JCudaTensor X521 = y51.forward(X520, _4d_a_bn_scale, _4d_a_bn_bias);
// val X522 = ReLU()(X521)
JCudaTensor X522 = y34.forward(X521);
// val X523 = Convolv(1,1)(X522,4d_b_cv_W,4d_b_cv_B)
JCudaTensor X523 = y37.forward(X522, _4d_b_cv_W, _4d_b_cv_B);
// val X524 = BatchNorm(4d_b_bn)(X523,4d_b_bn_scale,4d_b_bn_bias)
JCudaTensor X524 = y49.forward(X523, _4d_b_bn_scale, _4d_b_bn_bias);
// val X525 = ReLU()(X524)
JCudaTensor X525 = y34.forward(X524);
// val X526 = Convolv(1,0)(X525,4d_c_cv_W,4d_c_cv_B)
JCudaTensor X526 = y33.forward(X525, _4d_c_cv_W, _4d_c_cv_B);
// val X527 = BatchNorm(4d_c_bn)(X526,4d_c_bn_scale,4d_c_bn_bias)
JCudaTensor X527 = y47.forward(X526, _4d_c_bn_scale, _4d_c_bn_bias);
// val X528 = ReLU()(X527)
JCudaTensor X528 = y30.forward(X527);
// val X529 = (X528.copy + X519)
JCudaTensor X529 = X528.clone().plus_i(X519);;
// val X530 = ReLU()(X529)
JCudaTensor X530 = y30.forward(X529);
// val X531 = Convolv(1,0)(X530,4e_a_cv_W,4e_a_cv_B)
JCudaTensor X531 = y40.forward(X530, _4e_a_cv_W, _4e_a_cv_B);
// val X532 = BatchNorm(4e_a_bn)(X531,4e_a_bn_scale,4e_a_bn_bias)
JCudaTensor X532 = y45.forward(X531, _4e_a_bn_scale, _4e_a_bn_bias);
// val X533 = ReLU()(X532)
JCudaTensor X533 = y34.forward(X532);
// val X534 = Convolv(1,1)(X533,4e_b_cv_W,4e_b_cv_B)
JCudaTensor X534 = y37.forward(X533, _4e_b_cv_W, _4e_b_cv_B);
// val X535 = BatchNorm(4e_b_bn)(X534,4e_b_bn_scale,4e_b_bn_bias)
JCudaTensor X535 = y43.forward(X534, _4e_b_bn_scale, _4e_b_bn_bias);
// val X536 = ReLU()(X535)
JCudaTensor X536 = y34.forward(X535);
// val X537 = Convolv(1,0)(X536,4e_c_cv_W,4e_c_cv_B)
JCudaTensor X537 = y33.forward(X536, _4e_c_cv_W, _4e_c_cv_B);
// val X538 = BatchNorm(4e_c_bn)(X537,4e_c_bn_scale,4e_c_bn_bias)
JCudaTensor X538 = y41.forward(X537, _4e_c_bn_scale, _4e_c_bn_bias);
// val X539 = ReLU()(X538)
JCudaTensor X539 = y30.forward(X538);
// val X540 = (X539.copy + X530)
JCudaTensor X540 = X539.clone().plus_i(X530);;
// val X541 = ReLU()(X540)
JCudaTensor X541 = y30.forward(X540);
// val X542 = Convolv(1,0)(X541,4f_a_cv_W,4f_a_cv_B)
JCudaTensor X542 = y40.forward(X541, _4f_a_cv_W, _4f_a_cv_B);
// val X543 = BatchNorm(4f_a_bn)(X542,4f_a_bn_scale,4f_a_bn_bias)
JCudaTensor X543 = y38.forward(X542, _4f_a_bn_scale, _4f_a_bn_bias);
// val X544 = ReLU()(X543)
JCudaTensor X544 = y34.forward(X543);
// val X545 = Convolv(1,1)(X544,4f_b_cv_W,4f_b_cv_B)
JCudaTensor X545 = y37.forward(X544, _4f_b_cv_W, _4f_b_cv_B);
// val X546 = BatchNorm(4f_b_bn)(X545,4f_b_bn_scale,4f_b_bn_bias)
JCudaTensor X546 = y35.forward(X545, _4f_b_bn_scale, _4f_b_bn_bias);
// val X547 = ReLU()(X546)
JCudaTensor X547 = y34.forward(X546);
// val X548 = Convolv(1,0)(X547,4f_c_cv_W,4f_c_cv_B)
JCudaTensor X548 = y33.forward(X547, _4f_c_cv_W, _4f_c_cv_B);
// val X549 = BatchNorm(4f_c_bn)(X548,4f_c_bn_scale,4f_c_bn_bias)
JCudaTensor X549 = y31.forward(X548, _4f_c_bn_scale, _4f_c_bn_bias);
// val X550 = ReLU()(X549)
JCudaTensor X550 = y30.forward(X549);
// val X551 = (X550.copy + X541)
JCudaTensor X551 = X550.clone().plus_i(X541);;
// val X552 = ReLU()(X551)
JCudaTensor X552 = y30.forward(X551);
// val X556 = Convolv(2,0)(X552,5a2_a_cv_W,5a2_a_cv_B)
JCudaTensor X556 = y142.forward(X552, _5a2_a_cv_W, _5a2_a_cv_B);
// val X553 = Convolv(2,0)(X552,5a1_cv_W,5a1_cv_B)
JCudaTensor X553 = y22.forward(X552, _5a1_cv_W, _5a1_cv_B);
// val X554 = BatchNorm(5a1_bn)(X553,5a1_bn_scale,5a1_bn_bias)
JCudaTensor X554 = y20.forward(X553, _5a1_bn_scale, _5a1_bn_bias);
// val X557 = BatchNorm(5a2_a_bn)(X556,5a2_a_bn_scale,5a2_a_bn_bias)
JCudaTensor X557 = y27.forward(X556, _5a2_a_bn_scale, _5a2_a_bn_bias);
// val X558 = ReLU()(X557)
JCudaTensor X558 = y7.forward(X557);
// val X555 = ReLU()(X554)
JCudaTensor X555 = y3.forward(X554);
// val X559 = Convolv(1,1)(X558,5a2_b_cv_W,5a2_b_cv_B)
JCudaTensor X559 = y10.forward(X558, _5a2_b_cv_W, _5a2_b_cv_B);
// val X560 = BatchNorm(5a2_b_bn)(X559,5a2_b_bn_scale,5a2_b_bn_bias)
JCudaTensor X560 = y25.forward(X559, _5a2_b_bn_scale, _5a2_b_bn_bias);
// val X561 = ReLU()(X560)
JCudaTensor X561 = y7.forward(X560);
// val X562 = Convolv(1,0)(X561,5a2_c_cv_W,5a2_c_cv_B)
JCudaTensor X562 = y6.forward(X561, _5a2_c_cv_W, _5a2_c_cv_B);
// val X563 = BatchNorm(5a2_c_bn)(X562,5a2_c_bn_scale,5a2_c_bn_bias)
JCudaTensor X563 = y23.forward(X562, _5a2_c_bn_scale, _5a2_c_bn_bias);
// val X564 = ReLU()(X563)
JCudaTensor X564 = y3.forward(X563);
// val X565 = (X555.copy + X564)
JCudaTensor X565 = X555.clone().plus_i(X564);;
// val X566 = ReLU()(X565)
JCudaTensor X566 = y3.forward(X565);
// val X567 = Convolv(1,0)(X566,5b_a_cv_W,5b_a_cv_B)
JCudaTensor X567 = y13.forward(X566, _5b_a_cv_W, _5b_a_cv_B);
// val X568 = BatchNorm(5b_a_bn)(X567,5b_a_bn_scale,5b_a_bn_bias)
JCudaTensor X568 = y18.forward(X567, _5b_a_bn_scale, _5b_a_bn_bias);
// val X569 = ReLU()(X568)
JCudaTensor X569 = y7.forward(X568);
// val X570 = Convolv(1,1)(X569,5b_b_cv_W,5b_b_cv_B)
JCudaTensor X570 = y10.forward(X569, _5b_b_cv_W, _5b_b_cv_B);
// val X571 = BatchNorm(5b_b_bn)(X570,5b_b_bn_scale,5b_b_bn_bias)
JCudaTensor X571 = y16.forward(X570, _5b_b_bn_scale, _5b_b_bn_bias);
// val X572 = ReLU()(X571)
JCudaTensor X572 = y7.forward(X571);
// val X573 = Convolv(1,0)(X572,5b_c_cv_W,5b_c_cv_B)
JCudaTensor X573 = y6.forward(X572, _5b_c_cv_W, _5b_c_cv_B);
// val X574 = BatchNorm(5b_c_bn)(X573,5b_c_bn_scale,5b_c_bn_bias)
JCudaTensor X574 = y14.forward(X573, _5b_c_bn_scale, _5b_c_bn_bias);
// val X575 = ReLU()(X574)
JCudaTensor X575 = y3.forward(X574);
// val X576 = (X575.copy + X566)
JCudaTensor X576 = X575.clone().plus_i(X566);;
// val X577 = ReLU()(X576)
JCudaTensor X577 = y3.forward(X576);
// val X578 = Convolv(1,0)(X577,5c_a_cv_W,5c_a_cv_B)
JCudaTensor X578 = y13.forward(X577, _5c_a_cv_W, _5c_a_cv_B);
// val X579 = BatchNorm(5c_a_bn)(X578,5c_a_bn_scale,5c_a_bn_bias)
JCudaTensor X579 = y11.forward(X578, _5c_a_bn_scale, _5c_a_bn_bias);
// val X580 = ReLU()(X579)
JCudaTensor X580 = y7.forward(X579);
// val X581 = Convolv(1,1)(X580,5c_b_cv_W,5c_b_cv_B)
JCudaTensor X581 = y10.forward(X580, _5c_b_cv_W, _5c_b_cv_B);
// val X582 = BatchNorm(5c_b_bn)(X581,5c_b_bn_scale,5c_b_bn_bias)
JCudaTensor X582 = y8.forward(X581, _5c_b_bn_scale, _5c_b_bn_bias);
// val X583 = ReLU()(X582)
JCudaTensor X583 = y7.forward(X582);
// val X584 = Convolv(1,0)(X583,5c_c_cv_W,5c_c_cv_B)
JCudaTensor X584 = y6.forward(X583, _5c_c_cv_W, _5c_c_cv_B);
// val X585 = BatchNorm(5c_c_bn)(X584,5c_c_bn_scale,5c_c_bn_bias)
JCudaTensor X585 = y4.forward(X584, _5c_c_bn_scale, _5c_c_bn_bias);
// val X586 = ReLU()(X585)
JCudaTensor X586 = y3.forward(X585);
// val X587 = (X586.copy + X577)
JCudaTensor X587 = X586.clone().plus_i(X577);;
// val X588 = ReLU()(X587)
JCudaTensor X588 = y3.forward(X587);
// val X589 = Pooling(7,1,0,false)(X588)
JCudaTensor X589 = y2.forward(X588);
// val X1322 = (X589[1><3])(i1 | @) * (fc_W)(i2 | @)
JCudaTensor X1322 = X589.flatten(1, new int[]{2048, 1, 1}).asMatrix(1, true).times(fc_W.asMatrix(1, true));
// val X591 = (X1322 + (i1) => fc_B)
JCudaTensor X591 = fc_B.copy(64, X1322);
// val X592 = LogSoftmax()(X591)
JCudaTensor X592 = y1.forward(X591);
// dealloc X591
X591.free();
// val _loss = ((0 - (X1325 . X592)) / |64|)
float _loss = - X1325.dot(X592) / 64f;
// val X1490 = - X1325.copy
JCudaTensor X1490 = X1325.clone().times_i(-1f);;
// dealloc X1325
X1325.free();
// val X593 = (X1490 / |64|)
JCudaTensor X593 = X1490.times_i(1 / 64f);;
// val X1597 = X593 * d_LogSoftmax()(X592)/d_X591
JCudaTensor X1597 = y1.backward(X593, X592);
// dealloc X592
X592.free();
// dealloc X593
X593.free();
// val m1 = (i413) => X1597[@, i413]
JCudaMatrix m1 = X1597.asMatrix(1, false);
// V_fc_B = ((Sum(m1) * -0.01) + (V_fc_B * 0.9))
m1.sum(V_fc_B, lrn_rate, momentum);
// fc_B = (V_fc_B + (fc_B * (1 + (5.0E-4 * -0.01))))
fc_B.update(V_fc_B, 1f, 1f + decay * lrn_rate);
// val m2 = (i410) => fc_W[@, i410]
JCudaMatrix m2 = fc_W.asMatrix(1, false);
// val m4 = (i16) => X589[1><3][@, i16]
JCudaMatrix m4 = X589.flatten(1, new int[]{2048, 1, 1}).asMatrix(1, false);
// V_fc_W = ((m1 * m4 * -0.01) + (V_fc_W * 0.9))
m1.times(m4, V_fc_W, lrn_rate, momentum);
// fc_W = (V_fc_W + (fc_W * (1 + (5.0E-4 * -0.01))))
fc_W.update(V_fc_W, 1f, 1f + decay * lrn_rate);
// val X1566 = (X1597)(i409 | @) * m2
JCudaTensor X1566 = X1597.asMatrix(1, true).times(m2);
// dealloc X1597
X1597.free();
// val X1483 = X1566[1<>3] * d_Pooling(7,1,0,false)(X589,X588)/d_X588
JCudaTensor X1483 = y2.backward(X1566.unflatten(1, new int[]{2048, 1, 1}), X589, X588);
// dealloc X1566
X1566.free();
// dealloc X589
X589.free();
// val X1468 = X1483 * d_ReLU()(X588)/d_X587
JCudaTensor X1468 = y3.backward(X1483, X588);
// dealloc X588
X588.free();
// val X1735 = X1468.copy * d_ReLU()(X586)/d_X585
JCudaTensor X1735 = y3.backward(X1468.clone(), X586);
// dealloc X586
X586.free();
JCudaTensor[] y5 = y4.backward(X1735,X584,_5c_c_bn_scale);
// val X1547 = X1735 * d_BatchNorm(5c_c_bn)(X584,5c_c_bn_scale)/d_5c_c_bn_bias
JCudaTensor X1547 = y5[2];;
// V_5c_c_bn_bias = ((X1547 * -0.01) + (V_5c_c_bn_bias * 0.9))
V_5c_c_bn_bias.update(X1547, lrn_rate, momentum);
// dealloc X1547
X1547.free();
// 5c_c_bn_bias = (V_5c_c_bn_bias + (5c_c_bn_bias * (1 + (5.0E-4 * -0.01))))
_5c_c_bn_bias.update(V_5c_c_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1408 = X1735 * d_BatchNorm(5c_c_bn)(X584,5c_c_bn_scale)/d_5c_c_bn_scale
JCudaTensor X1408 = y5[1];;
// val X1573 = X1735 * d_BatchNorm(5c_c_bn)(X584,5c_c_bn_scale)/d_X584
JCudaTensor X1573 = y5[0];;
// dealloc X584
X584.free();
// V_5c_c_bn_scale = ((X1408 * -0.01) + (V_5c_c_bn_scale * 0.9))
V_5c_c_bn_scale.update(X1408, lrn_rate, momentum);
// dealloc X1408
X1408.free();
// 5c_c_bn_scale = (V_5c_c_bn_scale + (5c_c_bn_scale * (1 + (5.0E-4 * -0.01))))
_5c_c_bn_scale.update(V_5c_c_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1554 = X1573 * d_Convolv(1,0)(5c_c_cv_W)/d_X583
JCudaTensor X1554 = y6.backward_data(X1573, _5c_c_cv_W);
// V_5c_c_cv_W = ((X1573 * d_Convolv(1,0)(X583)/d_5c_c_cv_W * -0.01) + (V_5c_c_cv_W * 0.9))
y6.backward_filter(X1573, X583, V_5c_c_cv_W, lrn_rate, momentum);
// dealloc X1573
X1573.free();
// 5c_c_cv_W = (V_5c_c_cv_W + (5c_c_cv_W * (1 + (5.0E-4 * -0.01))))
_5c_c_cv_W.update(V_5c_c_cv_W, 1f, 1f + decay * lrn_rate);
// val X1359 = X1554 * d_ReLU()(X583)/d_X582
JCudaTensor X1359 = y7.backward(X1554, X583);
// dealloc X583
X583.free();
JCudaTensor[] y9 = y8.backward(X1359,X581,_5c_b_bn_scale);
// val X1658 = X1359 * d_BatchNorm(5c_b_bn)(X581,5c_b_bn_scale)/d_5c_b_bn_bias
JCudaTensor X1658 = y9[2];;
// V_5c_b_bn_bias = ((X1658 * -0.01) + (V_5c_b_bn_bias * 0.9))
V_5c_b_bn_bias.update(X1658, lrn_rate, momentum);
// dealloc X1658
X1658.free();
// 5c_b_bn_bias = (V_5c_b_bn_bias + (5c_b_bn_bias * (1 + (5.0E-4 * -0.01))))
_5c_b_bn_bias.update(V_5c_b_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1473 = X1359 * d_BatchNorm(5c_b_bn)(X581,5c_b_bn_scale)/d_X581
JCudaTensor X1473 = y9[0];;
// val X1655 = X1359 * d_BatchNorm(5c_b_bn)(X581,5c_b_bn_scale)/d_5c_b_bn_scale
JCudaTensor X1655 = y9[1];;
// dealloc X581
X581.free();
// V_5c_b_bn_scale = ((X1655 * -0.01) + (V_5c_b_bn_scale * 0.9))
V_5c_b_bn_scale.update(X1655, lrn_rate, momentum);
// dealloc X1655
X1655.free();
// 5c_b_bn_scale = (V_5c_b_bn_scale + (5c_b_bn_scale * (1 + (5.0E-4 * -0.01))))
_5c_b_bn_scale.update(V_5c_b_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1545 = X1473 * d_Convolv(1,1)(5c_b_cv_W)/d_X580
JCudaTensor X1545 = y10.backward_data(X1473, _5c_b_cv_W);
// V_5c_b_cv_W = ((X1473 * d_Convolv(1,1)(X580)/d_5c_b_cv_W * -0.01) + (V_5c_b_cv_W * 0.9))
y10.backward_filter(X1473, X580, V_5c_b_cv_W, lrn_rate, momentum);
// dealloc X1473
X1473.free();
// 5c_b_cv_W = (V_5c_b_cv_W + (5c_b_cv_W * (1 + (5.0E-4 * -0.01))))
_5c_b_cv_W.update(V_5c_b_cv_W, 1f, 1f + decay * lrn_rate);
// val X1369 = X1545 * d_ReLU()(X580)/d_X579
JCudaTensor X1369 = y7.backward(X1545, X580);
// dealloc X580
X580.free();
JCudaTensor[] y12 = y11.backward(X1369,X578,_5c_a_bn_scale);
// val X1327 = X1369 * d_BatchNorm(5c_a_bn)(X578,5c_a_bn_scale)/d_5c_a_bn_bias
JCudaTensor X1327 = y12[2];;
// V_5c_a_bn_bias = ((X1327 * -0.01) + (V_5c_a_bn_bias * 0.9))
V_5c_a_bn_bias.update(X1327, lrn_rate, momentum);
// dealloc X1327
X1327.free();
// 5c_a_bn_bias = (V_5c_a_bn_bias + (5c_a_bn_bias * (1 + (5.0E-4 * -0.01))))
_5c_a_bn_bias.update(V_5c_a_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1553 = X1369 * d_BatchNorm(5c_a_bn)(X578,5c_a_bn_scale)/d_X578
JCudaTensor X1553 = y12[0];;
// val X1586 = X1369 * d_BatchNorm(5c_a_bn)(X578,5c_a_bn_scale)/d_5c_a_bn_scale
JCudaTensor X1586 = y12[1];;
// dealloc X578
X578.free();
// V_5c_a_bn_scale = ((X1586 * -0.01) + (V_5c_a_bn_scale * 0.9))
V_5c_a_bn_scale.update(X1586, lrn_rate, momentum);
// dealloc X1586
X1586.free();
// 5c_a_bn_scale = (V_5c_a_bn_scale + (5c_a_bn_scale * (1 + (5.0E-4 * -0.01))))
_5c_a_bn_scale.update(V_5c_a_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1725 = X1553 * d_Convolv(1,0)(5c_a_cv_W)/d_X577
JCudaTensor X1725 = y13.backward_data(X1553, _5c_a_cv_W);
// V_5c_a_cv_W = ((X1553 * d_Convolv(1,0)(X577)/d_5c_a_cv_W * -0.01) + (V_5c_a_cv_W * 0.9))
y13.backward_filter(X1553, X577, V_5c_a_cv_W, lrn_rate, momentum);
// dealloc X1553
X1553.free();
// 5c_a_cv_W = (V_5c_a_cv_W + (5c_a_cv_W * (1 + (5.0E-4 * -0.01))))
_5c_a_cv_W.update(V_5c_a_cv_W, 1f, 1f + decay * lrn_rate);
// val X610 = (X1725 + X1468)
JCudaTensor X610 = X1725.plus_i(X1468);;
// dealloc X1468
X1468.free();
// val X1686 = X610 * d_ReLU()(X577)/d_X576
JCudaTensor X1686 = y3.backward(X610, X577);
// dealloc X577
X577.free();
// val X1626 = X1686.copy * d_ReLU()(X575)/d_X574
JCudaTensor X1626 = y3.backward(X1686.clone(), X575);
// dealloc X575
X575.free();
JCudaTensor[] y15 = y14.backward(X1626,X573,_5b_c_bn_scale);
// val X1560 = X1626 * d_BatchNorm(5b_c_bn)(X573,5b_c_bn_scale)/d_5b_c_bn_bias
JCudaTensor X1560 = y15[2];;
// V_5b_c_bn_bias = ((X1560 * -0.01) + (V_5b_c_bn_bias * 0.9))
V_5b_c_bn_bias.update(X1560, lrn_rate, momentum);
// dealloc X1560
X1560.free();
// 5b_c_bn_bias = (V_5b_c_bn_bias + (5b_c_bn_bias * (1 + (5.0E-4 * -0.01))))
_5b_c_bn_bias.update(V_5b_c_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1455 = X1626 * d_BatchNorm(5b_c_bn)(X573,5b_c_bn_scale)/d_X573
JCudaTensor X1455 = y15[0];;
// val X1616 = X1626 * d_BatchNorm(5b_c_bn)(X573,5b_c_bn_scale)/d_5b_c_bn_scale
JCudaTensor X1616 = y15[1];;
// dealloc X573
X573.free();
// V_5b_c_bn_scale = ((X1616 * -0.01) + (V_5b_c_bn_scale * 0.9))
V_5b_c_bn_scale.update(X1616, lrn_rate, momentum);
// dealloc X1616
X1616.free();
// 5b_c_bn_scale = (V_5b_c_bn_scale + (5b_c_bn_scale * (1 + (5.0E-4 * -0.01))))
_5b_c_bn_scale.update(V_5b_c_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1378 = X1455 * d_Convolv(1,0)(5b_c_cv_W)/d_X572
JCudaTensor X1378 = y6.backward_data(X1455, _5b_c_cv_W);
// V_5b_c_cv_W = ((X1455 * d_Convolv(1,0)(X572)/d_5b_c_cv_W * -0.01) + (V_5b_c_cv_W * 0.9))
y6.backward_filter(X1455, X572, V_5b_c_cv_W, lrn_rate, momentum);
// dealloc X1455
X1455.free();
// 5b_c_cv_W = (V_5b_c_cv_W + (5b_c_cv_W * (1 + (5.0E-4 * -0.01))))
_5b_c_cv_W.update(V_5b_c_cv_W, 1f, 1f + decay * lrn_rate);
// val X1585 = X1378 * d_ReLU()(X572)/d_X571
JCudaTensor X1585 = y7.backward(X1378, X572);
// dealloc X572
X572.free();
JCudaTensor[] y17 = y16.backward(X1585,X570,_5b_b_bn_scale);
// val X1537 = X1585 * d_BatchNorm(5b_b_bn)(X570,5b_b_bn_scale)/d_5b_b_bn_bias
JCudaTensor X1537 = y17[2];;
// V_5b_b_bn_bias = ((X1537 * -0.01) + (V_5b_b_bn_bias * 0.9))
V_5b_b_bn_bias.update(X1537, lrn_rate, momentum);
// dealloc X1537
X1537.free();
// 5b_b_bn_bias = (V_5b_b_bn_bias + (5b_b_bn_bias * (1 + (5.0E-4 * -0.01))))
_5b_b_bn_bias.update(V_5b_b_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1421 = X1585 * d_BatchNorm(5b_b_bn)(X570,5b_b_bn_scale)/d_5b_b_bn_scale
JCudaTensor X1421 = y17[1];;
// val X1434 = X1585 * d_BatchNorm(5b_b_bn)(X570,5b_b_bn_scale)/d_X570
JCudaTensor X1434 = y17[0];;
// dealloc X570
X570.free();
// V_5b_b_bn_scale = ((X1421 * -0.01) + (V_5b_b_bn_scale * 0.9))
V_5b_b_bn_scale.update(X1421, lrn_rate, momentum);
// dealloc X1421
X1421.free();
// 5b_b_bn_scale = (V_5b_b_bn_scale + (5b_b_bn_scale * (1 + (5.0E-4 * -0.01))))
_5b_b_bn_scale.update(V_5b_b_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1389 = X1434 * d_Convolv(1,1)(5b_b_cv_W)/d_X569
JCudaTensor X1389 = y10.backward_data(X1434, _5b_b_cv_W);
// V_5b_b_cv_W = ((X1434 * d_Convolv(1,1)(X569)/d_5b_b_cv_W * -0.01) + (V_5b_b_cv_W * 0.9))
y10.backward_filter(X1434, X569, V_5b_b_cv_W, lrn_rate, momentum);
// dealloc X1434
X1434.free();
// 5b_b_cv_W = (V_5b_b_cv_W + (5b_b_cv_W * (1 + (5.0E-4 * -0.01))))
_5b_b_cv_W.update(V_5b_b_cv_W, 1f, 1f + decay * lrn_rate);
// val X1511 = X1389 * d_ReLU()(X569)/d_X568
JCudaTensor X1511 = y7.backward(X1389, X569);
// dealloc X569
X569.free();
JCudaTensor[] y19 = y18.backward(X1511,X567,_5b_a_bn_scale);
// val X1568 = X1511 * d_BatchNorm(5b_a_bn)(X567,5b_a_bn_scale)/d_5b_a_bn_bias
JCudaTensor X1568 = y19[2];;
// V_5b_a_bn_bias = ((X1568 * -0.01) + (V_5b_a_bn_bias * 0.9))
V_5b_a_bn_bias.update(X1568, lrn_rate, momentum);
// dealloc X1568
X1568.free();
// 5b_a_bn_bias = (V_5b_a_bn_bias + (5b_a_bn_bias * (1 + (5.0E-4 * -0.01))))
_5b_a_bn_bias.update(V_5b_a_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1595 = X1511 * d_BatchNorm(5b_a_bn)(X567,5b_a_bn_scale)/d_5b_a_bn_scale
JCudaTensor X1595 = y19[1];;
// val X1724 = X1511 * d_BatchNorm(5b_a_bn)(X567,5b_a_bn_scale)/d_X567
JCudaTensor X1724 = y19[0];;
// dealloc X567
X567.free();
// V_5b_a_bn_scale = ((X1595 * -0.01) + (V_5b_a_bn_scale * 0.9))
V_5b_a_bn_scale.update(X1595, lrn_rate, momentum);
// dealloc X1595
X1595.free();
// 5b_a_bn_scale = (V_5b_a_bn_scale + (5b_a_bn_scale * (1 + (5.0E-4 * -0.01))))
_5b_a_bn_scale.update(V_5b_a_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1617 = X1724 * d_Convolv(1,0)(5b_a_cv_W)/d_X566
JCudaTensor X1617 = y13.backward_data(X1724, _5b_a_cv_W);
// V_5b_a_cv_W = ((X1724 * d_Convolv(1,0)(X566)/d_5b_a_cv_W * -0.01) + (V_5b_a_cv_W * 0.9))
y13.backward_filter(X1724, X566, V_5b_a_cv_W, lrn_rate, momentum);
// dealloc X1724
X1724.free();
// 5b_a_cv_W = (V_5b_a_cv_W + (5b_a_cv_W * (1 + (5.0E-4 * -0.01))))
_5b_a_cv_W.update(V_5b_a_cv_W, 1f, 1f + decay * lrn_rate);
// val X621 = (X1617 + X1686)
JCudaTensor X621 = X1617.plus_i(X1686);;
// dealloc X1686
X1686.free();
// val X1417 = X621 * d_ReLU()(X566)/d_X565
JCudaTensor X1417 = y3.backward(X621, X566);
// dealloc X566
X566.free();
// val X1740 = X1417.copy * d_ReLU()(X555)/d_X554
JCudaTensor X1740 = y3.backward(X1417.clone(), X555);
// dealloc X555
X555.free();
JCudaTensor[] y21 = y20.backward(X1740,X553,_5a1_bn_scale);
// val X1396 = X1740 * d_BatchNorm(5a1_bn)(X553,5a1_bn_scale)/d_5a1_bn_bias
JCudaTensor X1396 = y21[2];;
// V_5a1_bn_bias = ((X1396 * -0.01) + (V_5a1_bn_bias * 0.9))
V_5a1_bn_bias.update(X1396, lrn_rate, momentum);
// dealloc X1396
X1396.free();
// 5a1_bn_bias = (V_5a1_bn_bias + (5a1_bn_bias * (1 + (5.0E-4 * -0.01))))
_5a1_bn_bias.update(V_5a1_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1446 = X1740 * d_BatchNorm(5a1_bn)(X553,5a1_bn_scale)/d_5a1_bn_scale
JCudaTensor X1446 = y21[1];;
// val X1567 = X1740 * d_BatchNorm(5a1_bn)(X553,5a1_bn_scale)/d_X553
JCudaTensor X1567 = y21[0];;
// dealloc X553
X553.free();
// V_5a1_bn_scale = ((X1446 * -0.01) + (V_5a1_bn_scale * 0.9))
V_5a1_bn_scale.update(X1446, lrn_rate, momentum);
// dealloc X1446
X1446.free();
// 5a1_bn_scale = (V_5a1_bn_scale + (5a1_bn_scale * (1 + (5.0E-4 * -0.01))))
_5a1_bn_scale.update(V_5a1_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1476 = X1567 * d_Convolv(2,0)(5a1_cv_W)/d_X552
JCudaTensor X1476 = y22.backward_data(X1567, _5a1_cv_W);
// V_5a1_cv_W = ((X1567 * d_Convolv(2,0)(X552)/d_5a1_cv_W * -0.01) + (V_5a1_cv_W * 0.9))
y22.backward_filter(X1567, X552, V_5a1_cv_W, lrn_rate, momentum);
// dealloc X1567
X1567.free();
// 5a1_cv_W = (V_5a1_cv_W + (5a1_cv_W * (1 + (5.0E-4 * -0.01))))
_5a1_cv_W.update(V_5a1_cv_W, 1f, 1f + decay * lrn_rate);
// val X1615 = X1417.copy * d_ReLU()(X564)/d_X563
JCudaTensor X1615 = y3.backward(X1417.clone(), X564);
// dealloc X1417
X1417.free();
// dealloc X564
X564.free();
JCudaTensor[] y24 = y23.backward(X1615,X562,_5a2_c_bn_scale);
// val X1732 = X1615 * d_BatchNorm(5a2_c_bn)(X562,5a2_c_bn_scale)/d_5a2_c_bn_bias
JCudaTensor X1732 = y24[2];;
// V_5a2_c_bn_bias = ((X1732 * -0.01) + (V_5a2_c_bn_bias * 0.9))
V_5a2_c_bn_bias.update(X1732, lrn_rate, momentum);
// dealloc X1732
X1732.free();
// 5a2_c_bn_bias = (V_5a2_c_bn_bias + (5a2_c_bn_bias * (1 + (5.0E-4 * -0.01))))
_5a2_c_bn_bias.update(V_5a2_c_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1362 = X1615 * d_BatchNorm(5a2_c_bn)(X562,5a2_c_bn_scale)/d_X562
JCudaTensor X1362 = y24[0];;
// val X1641 = X1615 * d_BatchNorm(5a2_c_bn)(X562,5a2_c_bn_scale)/d_5a2_c_bn_scale
JCudaTensor X1641 = y24[1];;
// dealloc X562
X562.free();
// V_5a2_c_bn_scale = ((X1641 * -0.01) + (V_5a2_c_bn_scale * 0.9))
V_5a2_c_bn_scale.update(X1641, lrn_rate, momentum);
// dealloc X1641
X1641.free();
// 5a2_c_bn_scale = (V_5a2_c_bn_scale + (5a2_c_bn_scale * (1 + (5.0E-4 * -0.01))))
_5a2_c_bn_scale.update(V_5a2_c_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1684 = X1362 * d_Convolv(1,0)(5a2_c_cv_W)/d_X561
JCudaTensor X1684 = y6.backward_data(X1362, _5a2_c_cv_W);
// V_5a2_c_cv_W = ((X1362 * d_Convolv(1,0)(X561)/d_5a2_c_cv_W * -0.01) + (V_5a2_c_cv_W * 0.9))
y6.backward_filter(X1362, X561, V_5a2_c_cv_W, lrn_rate, momentum);
// dealloc X1362
X1362.free();
// 5a2_c_cv_W = (V_5a2_c_cv_W + (5a2_c_cv_W * (1 + (5.0E-4 * -0.01))))
_5a2_c_cv_W.update(V_5a2_c_cv_W, 1f, 1f + decay * lrn_rate);
// val X1639 = X1684 * d_ReLU()(X561)/d_X560
JCudaTensor X1639 = y7.backward(X1684, X561);
// dealloc X561
X561.free();
JCudaTensor[] y26 = y25.backward(X1639,X559,_5a2_b_bn_scale);
// val X1515 = X1639 * d_BatchNorm(5a2_b_bn)(X559,5a2_b_bn_scale)/d_5a2_b_bn_bias
JCudaTensor X1515 = y26[2];;
// V_5a2_b_bn_bias = ((X1515 * -0.01) + (V_5a2_b_bn_bias * 0.9))
V_5a2_b_bn_bias.update(X1515, lrn_rate, momentum);
// dealloc X1515
X1515.free();
// 5a2_b_bn_bias = (V_5a2_b_bn_bias + (5a2_b_bn_bias * (1 + (5.0E-4 * -0.01))))
_5a2_b_bn_bias.update(V_5a2_b_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1402 = X1639 * d_BatchNorm(5a2_b_bn)(X559,5a2_b_bn_scale)/d_5a2_b_bn_scale
JCudaTensor X1402 = y26[1];;
// val X1559 = X1639 * d_BatchNorm(5a2_b_bn)(X559,5a2_b_bn_scale)/d_X559
JCudaTensor X1559 = y26[0];;
// dealloc X559
X559.free();
// V_5a2_b_bn_scale = ((X1402 * -0.01) + (V_5a2_b_bn_scale * 0.9))
V_5a2_b_bn_scale.update(X1402, lrn_rate, momentum);
// dealloc X1402
X1402.free();
// 5a2_b_bn_scale = (V_5a2_b_bn_scale + (5a2_b_bn_scale * (1 + (5.0E-4 * -0.01))))
_5a2_b_bn_scale.update(V_5a2_b_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1713 = X1559 * d_Convolv(1,1)(5a2_b_cv_W)/d_X558
JCudaTensor X1713 = y10.backward_data(X1559, _5a2_b_cv_W);
// V_5a2_b_cv_W = ((X1559 * d_Convolv(1,1)(X558)/d_5a2_b_cv_W * -0.01) + (V_5a2_b_cv_W * 0.9))
y10.backward_filter(X1559, X558, V_5a2_b_cv_W, lrn_rate, momentum);
// dealloc X1559
X1559.free();
// 5a2_b_cv_W = (V_5a2_b_cv_W + (5a2_b_cv_W * (1 + (5.0E-4 * -0.01))))
_5a2_b_cv_W.update(V_5a2_b_cv_W, 1f, 1f + decay * lrn_rate);
// val X1606 = X1713 * d_ReLU()(X558)/d_X557
JCudaTensor X1606 = y7.backward(X1713, X558);
// dealloc X558
X558.free();
JCudaTensor[] y28 = y27.backward(X1606,X556,_5a2_a_bn_scale);
// val X1505 = X1606 * d_BatchNorm(5a2_a_bn)(X556,5a2_a_bn_scale)/d_5a2_a_bn_bias
JCudaTensor X1505 = y28[2];;
// V_5a2_a_bn_bias = ((X1505 * -0.01) + (V_5a2_a_bn_bias * 0.9))
V_5a2_a_bn_bias.update(X1505, lrn_rate, momentum);
// dealloc X1505
X1505.free();
// 5a2_a_bn_bias = (V_5a2_a_bn_bias + (5a2_a_bn_bias * (1 + (5.0E-4 * -0.01))))
_5a2_a_bn_bias.update(V_5a2_a_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1583 = X1606 * d_BatchNorm(5a2_a_bn)(X556,5a2_a_bn_scale)/d_5a2_a_bn_scale
JCudaTensor X1583 = y28[1];;
// val X1719 = X1606 * d_BatchNorm(5a2_a_bn)(X556,5a2_a_bn_scale)/d_X556
JCudaTensor X1719 = y28[0];;
// dealloc X556
X556.free();
// V_5a2_a_bn_scale = ((X1583 * -0.01) + (V_5a2_a_bn_scale * 0.9))
V_5a2_a_bn_scale.update(X1583, lrn_rate, momentum);
// dealloc X1583
X1583.free();
// 5a2_a_bn_scale = (V_5a2_a_bn_scale + (5a2_a_bn_scale * (1 + (5.0E-4 * -0.01))))
_5a2_a_bn_scale.update(V_5a2_a_bn_scale, 1f, 1f + decay * lrn_rate);
// val X626 = (X1476 + X1719 * d_Convolv(2,0)(5a2_a_cv_W)/d_X552)
JCudaTensor X626 = y142.backward_data(X1719,_5a2_a_cv_W, X1476);
// V_5a2_a_cv_W = ((X1719 * d_Convolv(2,0)(X552)/d_5a2_a_cv_W * -0.01) + (V_5a2_a_cv_W * 0.9))
y142.backward_filter(X1719, X552, V_5a2_a_cv_W, lrn_rate, momentum);
// dealloc X1719
X1719.free();
// 5a2_a_cv_W = (V_5a2_a_cv_W + (5a2_a_cv_W * (1 + (5.0E-4 * -0.01))))
_5a2_a_cv_W.update(V_5a2_a_cv_W, 1f, 1f + decay * lrn_rate);
// val X1338 = X626 * d_ReLU()(X552)/d_X551
JCudaTensor X1338 = y30.backward(X626, X552);
// dealloc X552
X552.free();
// val X1593 = X1338.copy * d_ReLU()(X550)/d_X549
JCudaTensor X1593 = y30.backward(X1338.clone(), X550);
// dealloc X550
X550.free();
JCudaTensor[] y32 = y31.backward(X1593,X548,_4f_c_bn_scale);
// val X1418 = X1593 * d_BatchNorm(4f_c_bn)(X548,4f_c_bn_scale)/d_4f_c_bn_bias
JCudaTensor X1418 = y32[2];;
// V_4f_c_bn_bias = ((X1418 * -0.01) + (V_4f_c_bn_bias * 0.9))
V_4f_c_bn_bias.update(X1418, lrn_rate, momentum);
// dealloc X1418
X1418.free();
// 4f_c_bn_bias = (V_4f_c_bn_bias + (4f_c_bn_bias * (1 + (5.0E-4 * -0.01))))
_4f_c_bn_bias.update(V_4f_c_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1411 = X1593 * d_BatchNorm(4f_c_bn)(X548,4f_c_bn_scale)/d_X548
JCudaTensor X1411 = y32[0];;
// val X1697 = X1593 * d_BatchNorm(4f_c_bn)(X548,4f_c_bn_scale)/d_4f_c_bn_scale
JCudaTensor X1697 = y32[1];;
// dealloc X548
X548.free();
// V_4f_c_bn_scale = ((X1697 * -0.01) + (V_4f_c_bn_scale * 0.9))
V_4f_c_bn_scale.update(X1697, lrn_rate, momentum);
// dealloc X1697
X1697.free();
// 4f_c_bn_scale = (V_4f_c_bn_scale + (4f_c_bn_scale * (1 + (5.0E-4 * -0.01))))
_4f_c_bn_scale.update(V_4f_c_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1372 = X1411 * d_Convolv(1,0)(4f_c_cv_W)/d_X547
JCudaTensor X1372 = y33.backward_data(X1411, _4f_c_cv_W);
// V_4f_c_cv_W = ((X1411 * d_Convolv(1,0)(X547)/d_4f_c_cv_W * -0.01) + (V_4f_c_cv_W * 0.9))
y33.backward_filter(X1411, X547, V_4f_c_cv_W, lrn_rate, momentum);
// dealloc X1411
X1411.free();
// 4f_c_cv_W = (V_4f_c_cv_W + (4f_c_cv_W * (1 + (5.0E-4 * -0.01))))
_4f_c_cv_W.update(V_4f_c_cv_W, 1f, 1f + decay * lrn_rate);
// val X1521 = X1372 * d_ReLU()(X547)/d_X546
JCudaTensor X1521 = y34.backward(X1372, X547);
// dealloc X547
X547.free();
JCudaTensor[] y36 = y35.backward(X1521,X545,_4f_b_bn_scale);
// val X1582 = X1521 * d_BatchNorm(4f_b_bn)(X545,4f_b_bn_scale)/d_4f_b_bn_bias
JCudaTensor X1582 = y36[2];;
// V_4f_b_bn_bias = ((X1582 * -0.01) + (V_4f_b_bn_bias * 0.9))
V_4f_b_bn_bias.update(X1582, lrn_rate, momentum);
// dealloc X1582
X1582.free();
// 4f_b_bn_bias = (V_4f_b_bn_bias + (4f_b_bn_bias * (1 + (5.0E-4 * -0.01))))
_4f_b_bn_bias.update(V_4f_b_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1645 = X1521 * d_BatchNorm(4f_b_bn)(X545,4f_b_bn_scale)/d_4f_b_bn_scale
JCudaTensor X1645 = y36[1];;
// val X1663 = X1521 * d_BatchNorm(4f_b_bn)(X545,4f_b_bn_scale)/d_X545
JCudaTensor X1663 = y36[0];;
// dealloc X545
X545.free();
// V_4f_b_bn_scale = ((X1645 * -0.01) + (V_4f_b_bn_scale * 0.9))
V_4f_b_bn_scale.update(X1645, lrn_rate, momentum);
// dealloc X1645
X1645.free();
// 4f_b_bn_scale = (V_4f_b_bn_scale + (4f_b_bn_scale * (1 + (5.0E-4 * -0.01))))
_4f_b_bn_scale.update(V_4f_b_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1439 = X1663 * d_Convolv(1,1)(4f_b_cv_W)/d_X544
JCudaTensor X1439 = y37.backward_data(X1663, _4f_b_cv_W);
// V_4f_b_cv_W = ((X1663 * d_Convolv(1,1)(X544)/d_4f_b_cv_W * -0.01) + (V_4f_b_cv_W * 0.9))
y37.backward_filter(X1663, X544, V_4f_b_cv_W, lrn_rate, momentum);
// dealloc X1663
X1663.free();
// 4f_b_cv_W = (V_4f_b_cv_W + (4f_b_cv_W * (1 + (5.0E-4 * -0.01))))
_4f_b_cv_W.update(V_4f_b_cv_W, 1f, 1f + decay * lrn_rate);
// val X1682 = X1439 * d_ReLU()(X544)/d_X543
JCudaTensor X1682 = y34.backward(X1439, X544);
// dealloc X544
X544.free();
JCudaTensor[] y39 = y38.backward(X1682,X542,_4f_a_bn_scale);
// val X1565 = X1682 * d_BatchNorm(4f_a_bn)(X542,4f_a_bn_scale)/d_4f_a_bn_bias
JCudaTensor X1565 = y39[2];;
// V_4f_a_bn_bias = ((X1565 * -0.01) + (V_4f_a_bn_bias * 0.9))
V_4f_a_bn_bias.update(X1565, lrn_rate, momentum);
// dealloc X1565
X1565.free();
// 4f_a_bn_bias = (V_4f_a_bn_bias + (4f_a_bn_bias * (1 + (5.0E-4 * -0.01))))
_4f_a_bn_bias.update(V_4f_a_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1579 = X1682 * d_BatchNorm(4f_a_bn)(X542,4f_a_bn_scale)/d_4f_a_bn_scale
JCudaTensor X1579 = y39[1];;
// val X1648 = X1682 * d_BatchNorm(4f_a_bn)(X542,4f_a_bn_scale)/d_X542
JCudaTensor X1648 = y39[0];;
// dealloc X542
X542.free();
// V_4f_a_bn_scale = ((X1579 * -0.01) + (V_4f_a_bn_scale * 0.9))
V_4f_a_bn_scale.update(X1579, lrn_rate, momentum);
// dealloc X1579
X1579.free();
// 4f_a_bn_scale = (V_4f_a_bn_scale + (4f_a_bn_scale * (1 + (5.0E-4 * -0.01))))
_4f_a_bn_scale.update(V_4f_a_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1525 = X1648 * d_Convolv(1,0)(4f_a_cv_W)/d_X541
JCudaTensor X1525 = y40.backward_data(X1648, _4f_a_cv_W);
// V_4f_a_cv_W = ((X1648 * d_Convolv(1,0)(X541)/d_4f_a_cv_W * -0.01) + (V_4f_a_cv_W * 0.9))
y40.backward_filter(X1648, X541, V_4f_a_cv_W, lrn_rate, momentum);
// dealloc X1648
X1648.free();
// 4f_a_cv_W = (V_4f_a_cv_W + (4f_a_cv_W * (1 + (5.0E-4 * -0.01))))
_4f_a_cv_W.update(V_4f_a_cv_W, 1f, 1f + decay * lrn_rate);
// val X637 = (X1525 + X1338)
JCudaTensor X637 = X1525.plus_i(X1338);;
// dealloc X1338
X1338.free();
// val X1507 = X637 * d_ReLU()(X541)/d_X540
JCudaTensor X1507 = y30.backward(X637, X541);
// dealloc X541
X541.free();
// val X1581 = X1507.copy * d_ReLU()(X539)/d_X538
JCudaTensor X1581 = y30.backward(X1507.clone(), X539);
// dealloc X539
X539.free();
JCudaTensor[] y42 = y41.backward(X1581,X537,_4e_c_bn_scale);
// val X1518 = X1581 * d_BatchNorm(4e_c_bn)(X537,4e_c_bn_scale)/d_4e_c_bn_bias
JCudaTensor X1518 = y42[2];;
// V_4e_c_bn_bias = ((X1518 * -0.01) + (V_4e_c_bn_bias * 0.9))
V_4e_c_bn_bias.update(X1518, lrn_rate, momentum);
// dealloc X1518
X1518.free();
// 4e_c_bn_bias = (V_4e_c_bn_bias + (4e_c_bn_bias * (1 + (5.0E-4 * -0.01))))
_4e_c_bn_bias.update(V_4e_c_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1551 = X1581 * d_BatchNorm(4e_c_bn)(X537,4e_c_bn_scale)/d_X537
JCudaTensor X1551 = y42[0];;
// val X1651 = X1581 * d_BatchNorm(4e_c_bn)(X537,4e_c_bn_scale)/d_4e_c_bn_scale
JCudaTensor X1651 = y42[1];;
// dealloc X537
X537.free();
// V_4e_c_bn_scale = ((X1651 * -0.01) + (V_4e_c_bn_scale * 0.9))
V_4e_c_bn_scale.update(X1651, lrn_rate, momentum);
// dealloc X1651
X1651.free();
// 4e_c_bn_scale = (V_4e_c_bn_scale + (4e_c_bn_scale * (1 + (5.0E-4 * -0.01))))
_4e_c_bn_scale.update(V_4e_c_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1336 = X1551 * d_Convolv(1,0)(4e_c_cv_W)/d_X536
JCudaTensor X1336 = y33.backward_data(X1551, _4e_c_cv_W);
// V_4e_c_cv_W = ((X1551 * d_Convolv(1,0)(X536)/d_4e_c_cv_W * -0.01) + (V_4e_c_cv_W * 0.9))
y33.backward_filter(X1551, X536, V_4e_c_cv_W, lrn_rate, momentum);
// dealloc X1551
X1551.free();
// 4e_c_cv_W = (V_4e_c_cv_W + (4e_c_cv_W * (1 + (5.0E-4 * -0.01))))
_4e_c_cv_W.update(V_4e_c_cv_W, 1f, 1f + decay * lrn_rate);
// val X1429 = X1336 * d_ReLU()(X536)/d_X535
JCudaTensor X1429 = y34.backward(X1336, X536);
// dealloc X536
X536.free();
JCudaTensor[] y44 = y43.backward(X1429,X534,_4e_b_bn_scale);
// val X1722 = X1429 * d_BatchNorm(4e_b_bn)(X534,4e_b_bn_scale)/d_4e_b_bn_bias
JCudaTensor X1722 = y44[2];;
// V_4e_b_bn_bias = ((X1722 * -0.01) + (V_4e_b_bn_bias * 0.9))
V_4e_b_bn_bias.update(X1722, lrn_rate, momentum);
// dealloc X1722
X1722.free();
// 4e_b_bn_bias = (V_4e_b_bn_bias + (4e_b_bn_bias * (1 + (5.0E-4 * -0.01))))
_4e_b_bn_bias.update(V_4e_b_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1491 = X1429 * d_BatchNorm(4e_b_bn)(X534,4e_b_bn_scale)/d_X534
JCudaTensor X1491 = y44[0];;
// val X1687 = X1429 * d_BatchNorm(4e_b_bn)(X534,4e_b_bn_scale)/d_4e_b_bn_scale
JCudaTensor X1687 = y44[1];;
// dealloc X534
X534.free();
// V_4e_b_bn_scale = ((X1687 * -0.01) + (V_4e_b_bn_scale * 0.9))
V_4e_b_bn_scale.update(X1687, lrn_rate, momentum);
// dealloc X1687
X1687.free();
// 4e_b_bn_scale = (V_4e_b_bn_scale + (4e_b_bn_scale * (1 + (5.0E-4 * -0.01))))
_4e_b_bn_scale.update(V_4e_b_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1347 = X1491 * d_Convolv(1,1)(4e_b_cv_W)/d_X533
JCudaTensor X1347 = y37.backward_data(X1491, _4e_b_cv_W);
// V_4e_b_cv_W = ((X1491 * d_Convolv(1,1)(X533)/d_4e_b_cv_W * -0.01) + (V_4e_b_cv_W * 0.9))
y37.backward_filter(X1491, X533, V_4e_b_cv_W, lrn_rate, momentum);
// dealloc X1491
X1491.free();
// 4e_b_cv_W = (V_4e_b_cv_W + (4e_b_cv_W * (1 + (5.0E-4 * -0.01))))
_4e_b_cv_W.update(V_4e_b_cv_W, 1f, 1f + decay * lrn_rate);
// val X1604 = X1347 * d_ReLU()(X533)/d_X532
JCudaTensor X1604 = y34.backward(X1347, X533);
// dealloc X533
X533.free();
JCudaTensor[] y46 = y45.backward(X1604,X531,_4e_a_bn_scale);
// val X1712 = X1604 * d_BatchNorm(4e_a_bn)(X531,4e_a_bn_scale)/d_4e_a_bn_bias
JCudaTensor X1712 = y46[2];;
// V_4e_a_bn_bias = ((X1712 * -0.01) + (V_4e_a_bn_bias * 0.9))
V_4e_a_bn_bias.update(X1712, lrn_rate, momentum);
// dealloc X1712
X1712.free();
// 4e_a_bn_bias = (V_4e_a_bn_bias + (4e_a_bn_bias * (1 + (5.0E-4 * -0.01))))
_4e_a_bn_bias.update(V_4e_a_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1478 = X1604 * d_BatchNorm(4e_a_bn)(X531,4e_a_bn_scale)/d_4e_a_bn_scale
JCudaTensor X1478 = y46[1];;
// val X1711 = X1604 * d_BatchNorm(4e_a_bn)(X531,4e_a_bn_scale)/d_X531
JCudaTensor X1711 = y46[0];;
// dealloc X531
X531.free();
// V_4e_a_bn_scale = ((X1478 * -0.01) + (V_4e_a_bn_scale * 0.9))
V_4e_a_bn_scale.update(X1478, lrn_rate, momentum);
// dealloc X1478
X1478.free();
// 4e_a_bn_scale = (V_4e_a_bn_scale + (4e_a_bn_scale * (1 + (5.0E-4 * -0.01))))
_4e_a_bn_scale.update(V_4e_a_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1671 = X1711 * d_Convolv(1,0)(4e_a_cv_W)/d_X530
JCudaTensor X1671 = y40.backward_data(X1711, _4e_a_cv_W);
// V_4e_a_cv_W = ((X1711 * d_Convolv(1,0)(X530)/d_4e_a_cv_W * -0.01) + (V_4e_a_cv_W * 0.9))
y40.backward_filter(X1711, X530, V_4e_a_cv_W, lrn_rate, momentum);
// dealloc X1711
X1711.free();
// 4e_a_cv_W = (V_4e_a_cv_W + (4e_a_cv_W * (1 + (5.0E-4 * -0.01))))
_4e_a_cv_W.update(V_4e_a_cv_W, 1f, 1f + decay * lrn_rate);
// val X648 = (X1671 + X1507)
JCudaTensor X648 = X1671.plus_i(X1507);;
// dealloc X1507
X1507.free();
// val X1666 = X648 * d_ReLU()(X530)/d_X529
JCudaTensor X1666 = y30.backward(X648, X530);
// dealloc X530
X530.free();
// val X1504 = X1666.copy * d_ReLU()(X528)/d_X527
JCudaTensor X1504 = y30.backward(X1666.clone(), X528);
// dealloc X528
X528.free();
JCudaTensor[] y48 = y47.backward(X1504,X526,_4d_c_bn_scale);
// val X1680 = X1504 * d_BatchNorm(4d_c_bn)(X526,4d_c_bn_scale)/d_4d_c_bn_bias
JCudaTensor X1680 = y48[2];;
// V_4d_c_bn_bias = ((X1680 * -0.01) + (V_4d_c_bn_bias * 0.9))
V_4d_c_bn_bias.update(X1680, lrn_rate, momentum);
// dealloc X1680
X1680.free();
// 4d_c_bn_bias = (V_4d_c_bn_bias + (4d_c_bn_bias * (1 + (5.0E-4 * -0.01))))
_4d_c_bn_bias.update(V_4d_c_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1353 = X1504 * d_BatchNorm(4d_c_bn)(X526,4d_c_bn_scale)/d_4d_c_bn_scale
JCudaTensor X1353 = y48[1];;
// val X1662 = X1504 * d_BatchNorm(4d_c_bn)(X526,4d_c_bn_scale)/d_X526
JCudaTensor X1662 = y48[0];;
// dealloc X526
X526.free();
// V_4d_c_bn_scale = ((X1353 * -0.01) + (V_4d_c_bn_scale * 0.9))
V_4d_c_bn_scale.update(X1353, lrn_rate, momentum);
// dealloc X1353
X1353.free();
// 4d_c_bn_scale = (V_4d_c_bn_scale + (4d_c_bn_scale * (1 + (5.0E-4 * -0.01))))
_4d_c_bn_scale.update(V_4d_c_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1361 = X1662 * d_Convolv(1,0)(4d_c_cv_W)/d_X525
JCudaTensor X1361 = y33.backward_data(X1662, _4d_c_cv_W);
// V_4d_c_cv_W = ((X1662 * d_Convolv(1,0)(X525)/d_4d_c_cv_W * -0.01) + (V_4d_c_cv_W * 0.9))
y33.backward_filter(X1662, X525, V_4d_c_cv_W, lrn_rate, momentum);
// dealloc X1662
X1662.free();
// 4d_c_cv_W = (V_4d_c_cv_W + (4d_c_cv_W * (1 + (5.0E-4 * -0.01))))
_4d_c_cv_W.update(V_4d_c_cv_W, 1f, 1f + decay * lrn_rate);
// val X1710 = X1361 * d_ReLU()(X525)/d_X524
JCudaTensor X1710 = y34.backward(X1361, X525);
// dealloc X525
X525.free();
JCudaTensor[] y50 = y49.backward(X1710,X523,_4d_b_bn_scale);
// val X1667 = X1710 * d_BatchNorm(4d_b_bn)(X523,4d_b_bn_scale)/d_4d_b_bn_bias
JCudaTensor X1667 = y50[2];;
// V_4d_b_bn_bias = ((X1667 * -0.01) + (V_4d_b_bn_bias * 0.9))
V_4d_b_bn_bias.update(X1667, lrn_rate, momentum);
// dealloc X1667
X1667.free();
// 4d_b_bn_bias = (V_4d_b_bn_bias + (4d_b_bn_bias * (1 + (5.0E-4 * -0.01))))
_4d_b_bn_bias.update(V_4d_b_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1415 = X1710 * d_BatchNorm(4d_b_bn)(X523,4d_b_bn_scale)/d_X523
JCudaTensor X1415 = y50[0];;
// val X1576 = X1710 * d_BatchNorm(4d_b_bn)(X523,4d_b_bn_scale)/d_4d_b_bn_scale
JCudaTensor X1576 = y50[1];;
// dealloc X523
X523.free();
// V_4d_b_bn_scale = ((X1576 * -0.01) + (V_4d_b_bn_scale * 0.9))
V_4d_b_bn_scale.update(X1576, lrn_rate, momentum);
// dealloc X1576
X1576.free();
// 4d_b_bn_scale = (V_4d_b_bn_scale + (4d_b_bn_scale * (1 + (5.0E-4 * -0.01))))
_4d_b_bn_scale.update(V_4d_b_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1631 = X1415 * d_Convolv(1,1)(4d_b_cv_W)/d_X522
JCudaTensor X1631 = y37.backward_data(X1415, _4d_b_cv_W);
// V_4d_b_cv_W = ((X1415 * d_Convolv(1,1)(X522)/d_4d_b_cv_W * -0.01) + (V_4d_b_cv_W * 0.9))
y37.backward_filter(X1415, X522, V_4d_b_cv_W, lrn_rate, momentum);
// dealloc X1415
X1415.free();
// 4d_b_cv_W = (V_4d_b_cv_W + (4d_b_cv_W * (1 + (5.0E-4 * -0.01))))
_4d_b_cv_W.update(V_4d_b_cv_W, 1f, 1f + decay * lrn_rate);
// val X1466 = X1631 * d_ReLU()(X522)/d_X521
JCudaTensor X1466 = y34.backward(X1631, X522);
// dealloc X522
X522.free();
JCudaTensor[] y52 = y51.backward(X1466,X520,_4d_a_bn_scale);
// val X1652 = X1466 * d_BatchNorm(4d_a_bn)(X520,4d_a_bn_scale)/d_4d_a_bn_bias
JCudaTensor X1652 = y52[2];;
// V_4d_a_bn_bias = ((X1652 * -0.01) + (V_4d_a_bn_bias * 0.9))
V_4d_a_bn_bias.update(X1652, lrn_rate, momentum);
// dealloc X1652
X1652.free();
// 4d_a_bn_bias = (V_4d_a_bn_bias + (4d_a_bn_bias * (1 + (5.0E-4 * -0.01))))
_4d_a_bn_bias.update(V_4d_a_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1502 = X1466 * d_BatchNorm(4d_a_bn)(X520,4d_a_bn_scale)/d_4d_a_bn_scale
JCudaTensor X1502 = y52[1];;
// val X1691 = X1466 * d_BatchNorm(4d_a_bn)(X520,4d_a_bn_scale)/d_X520
JCudaTensor X1691 = y52[0];;
// dealloc X520
X520.free();
// V_4d_a_bn_scale = ((X1502 * -0.01) + (V_4d_a_bn_scale * 0.9))
V_4d_a_bn_scale.update(X1502, lrn_rate, momentum);
// dealloc X1502
X1502.free();
// 4d_a_bn_scale = (V_4d_a_bn_scale + (4d_a_bn_scale * (1 + (5.0E-4 * -0.01))))
_4d_a_bn_scale.update(V_4d_a_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1627 = X1691 * d_Convolv(1,0)(4d_a_cv_W)/d_X519
JCudaTensor X1627 = y40.backward_data(X1691, _4d_a_cv_W);
// V_4d_a_cv_W = ((X1691 * d_Convolv(1,0)(X519)/d_4d_a_cv_W * -0.01) + (V_4d_a_cv_W * 0.9))
y40.backward_filter(X1691, X519, V_4d_a_cv_W, lrn_rate, momentum);
// dealloc X1691
X1691.free();
// 4d_a_cv_W = (V_4d_a_cv_W + (4d_a_cv_W * (1 + (5.0E-4 * -0.01))))
_4d_a_cv_W.update(V_4d_a_cv_W, 1f, 1f + decay * lrn_rate);
// val X659 = (X1627 + X1666)
JCudaTensor X659 = X1627.plus_i(X1666);;
// dealloc X1666
X1666.free();
// val X1535 = X659 * d_ReLU()(X519)/d_X518
JCudaTensor X1535 = y30.backward(X659, X519);
// dealloc X519
X519.free();
// val X1715 = X1535.copy * d_ReLU()(X517)/d_X516
JCudaTensor X1715 = y30.backward(X1535.clone(), X517);
// dealloc X517
X517.free();
JCudaTensor[] y54 = y53.backward(X1715,X515,_4c_c_bn_scale);
// val X1383 = X1715 * d_BatchNorm(4c_c_bn)(X515,4c_c_bn_scale)/d_4c_c_bn_bias
JCudaTensor X1383 = y54[2];;
// V_4c_c_bn_bias = ((X1383 * -0.01) + (V_4c_c_bn_bias * 0.9))
V_4c_c_bn_bias.update(X1383, lrn_rate, momentum);
// dealloc X1383
X1383.free();
// 4c_c_bn_bias = (V_4c_c_bn_bias + (4c_c_bn_bias * (1 + (5.0E-4 * -0.01))))
_4c_c_bn_bias.update(V_4c_c_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1546 = X1715 * d_BatchNorm(4c_c_bn)(X515,4c_c_bn_scale)/d_X515
JCudaTensor X1546 = y54[0];;
// val X1693 = X1715 * d_BatchNorm(4c_c_bn)(X515,4c_c_bn_scale)/d_4c_c_bn_scale
JCudaTensor X1693 = y54[1];;
// dealloc X515
X515.free();
// V_4c_c_bn_scale = ((X1693 * -0.01) + (V_4c_c_bn_scale * 0.9))
V_4c_c_bn_scale.update(X1693, lrn_rate, momentum);
// dealloc X1693
X1693.free();
// 4c_c_bn_scale = (V_4c_c_bn_scale + (4c_c_bn_scale * (1 + (5.0E-4 * -0.01))))
_4c_c_bn_scale.update(V_4c_c_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1571 = X1546 * d_Convolv(1,0)(4c_c_cv_W)/d_X514
JCudaTensor X1571 = y33.backward_data(X1546, _4c_c_cv_W);
// V_4c_c_cv_W = ((X1546 * d_Convolv(1,0)(X514)/d_4c_c_cv_W * -0.01) + (V_4c_c_cv_W * 0.9))
y33.backward_filter(X1546, X514, V_4c_c_cv_W, lrn_rate, momentum);
// dealloc X1546
X1546.free();
// 4c_c_cv_W = (V_4c_c_cv_W + (4c_c_cv_W * (1 + (5.0E-4 * -0.01))))
_4c_c_cv_W.update(V_4c_c_cv_W, 1f, 1f + decay * lrn_rate);
// val X1394 = X1571 * d_ReLU()(X514)/d_X513
JCudaTensor X1394 = y34.backward(X1571, X514);
// dealloc X514
X514.free();
JCudaTensor[] y56 = y55.backward(X1394,X512,_4c_b_bn_scale);
// val X1390 = X1394 * d_BatchNorm(4c_b_bn)(X512,4c_b_bn_scale)/d_4c_b_bn_bias
JCudaTensor X1390 = y56[2];;
// V_4c_b_bn_bias = ((X1390 * -0.01) + (V_4c_b_bn_bias * 0.9))
V_4c_b_bn_bias.update(X1390, lrn_rate, momentum);
// dealloc X1390
X1390.free();
// 4c_b_bn_bias = (V_4c_b_bn_bias + (4c_b_bn_bias * (1 + (5.0E-4 * -0.01))))
_4c_b_bn_bias.update(V_4c_b_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1601 = X1394 * d_BatchNorm(4c_b_bn)(X512,4c_b_bn_scale)/d_X512
JCudaTensor X1601 = y56[0];;
// val X1602 = X1394 * d_BatchNorm(4c_b_bn)(X512,4c_b_bn_scale)/d_4c_b_bn_scale
JCudaTensor X1602 = y56[1];;
// dealloc X512
X512.free();
// V_4c_b_bn_scale = ((X1602 * -0.01) + (V_4c_b_bn_scale * 0.9))
V_4c_b_bn_scale.update(X1602, lrn_rate, momentum);
// dealloc X1602
X1602.free();
// 4c_b_bn_scale = (V_4c_b_bn_scale + (4c_b_bn_scale * (1 + (5.0E-4 * -0.01))))
_4c_b_bn_scale.update(V_4c_b_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1558 = X1601 * d_Convolv(1,1)(4c_b_cv_W)/d_X511
JCudaTensor X1558 = y37.backward_data(X1601, _4c_b_cv_W);
// V_4c_b_cv_W = ((X1601 * d_Convolv(1,1)(X511)/d_4c_b_cv_W * -0.01) + (V_4c_b_cv_W * 0.9))
y37.backward_filter(X1601, X511, V_4c_b_cv_W, lrn_rate, momentum);
// dealloc X1601
X1601.free();
// 4c_b_cv_W = (V_4c_b_cv_W + (4c_b_cv_W * (1 + (5.0E-4 * -0.01))))
_4c_b_cv_W.update(V_4c_b_cv_W, 1f, 1f + decay * lrn_rate);
// val X1549 = X1558 * d_ReLU()(X511)/d_X510
JCudaTensor X1549 = y34.backward(X1558, X511);
// dealloc X511
X511.free();
JCudaTensor[] y58 = y57.backward(X1549,X509,_4c_a_bn_scale);
// val X1462 = X1549 * d_BatchNorm(4c_a_bn)(X509,4c_a_bn_scale)/d_4c_a_bn_bias
JCudaTensor X1462 = y58[2];;
// V_4c_a_bn_bias = ((X1462 * -0.01) + (V_4c_a_bn_bias * 0.9))
V_4c_a_bn_bias.update(X1462, lrn_rate, momentum);
// dealloc X1462
X1462.free();
// 4c_a_bn_bias = (V_4c_a_bn_bias + (4c_a_bn_bias * (1 + (5.0E-4 * -0.01))))
_4c_a_bn_bias.update(V_4c_a_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1395 = X1549 * d_BatchNorm(4c_a_bn)(X509,4c_a_bn_scale)/d_X509
JCudaTensor X1395 = y58[0];;
// val X1422 = X1549 * d_BatchNorm(4c_a_bn)(X509,4c_a_bn_scale)/d_4c_a_bn_scale
JCudaTensor X1422 = y58[1];;
// dealloc X509
X509.free();
// V_4c_a_bn_scale = ((X1422 * -0.01) + (V_4c_a_bn_scale * 0.9))
V_4c_a_bn_scale.update(X1422, lrn_rate, momentum);
// dealloc X1422
X1422.free();
// 4c_a_bn_scale = (V_4c_a_bn_scale + (4c_a_bn_scale * (1 + (5.0E-4 * -0.01))))
_4c_a_bn_scale.update(V_4c_a_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1449 = X1395 * d_Convolv(1,0)(4c_a_cv_W)/d_X508
JCudaTensor X1449 = y40.backward_data(X1395, _4c_a_cv_W);
// V_4c_a_cv_W = ((X1395 * d_Convolv(1,0)(X508)/d_4c_a_cv_W * -0.01) + (V_4c_a_cv_W * 0.9))
y40.backward_filter(X1395, X508, V_4c_a_cv_W, lrn_rate, momentum);
// dealloc X1395
X1395.free();
// 4c_a_cv_W = (V_4c_a_cv_W + (4c_a_cv_W * (1 + (5.0E-4 * -0.01))))
_4c_a_cv_W.update(V_4c_a_cv_W, 1f, 1f + decay * lrn_rate);
// val X670 = (X1449 + X1535)
JCudaTensor X670 = X1449.plus_i(X1535);;
// dealloc X1535
X1535.free();
// val X1424 = X670 * d_ReLU()(X508)/d_X507
JCudaTensor X1424 = y30.backward(X670, X508);
// dealloc X508
X508.free();
// val X1660 = X1424.copy * d_ReLU()(X506)/d_X505
JCudaTensor X1660 = y30.backward(X1424.clone(), X506);
// dealloc X506
X506.free();
JCudaTensor[] y60 = y59.backward(X1660,X504,_4b_c_bn_scale);
// val X1385 = X1660 * d_BatchNorm(4b_c_bn)(X504,4b_c_bn_scale)/d_4b_c_bn_bias
JCudaTensor X1385 = y60[2];;
// V_4b_c_bn_bias = ((X1385 * -0.01) + (V_4b_c_bn_bias * 0.9))
V_4b_c_bn_bias.update(X1385, lrn_rate, momentum);
// dealloc X1385
X1385.free();
// 4b_c_bn_bias = (V_4b_c_bn_bias + (4b_c_bn_bias * (1 + (5.0E-4 * -0.01))))
_4b_c_bn_bias.update(V_4b_c_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1494 = X1660 * d_BatchNorm(4b_c_bn)(X504,4b_c_bn_scale)/d_4b_c_bn_scale
JCudaTensor X1494 = y60[1];;
// val X1574 = X1660 * d_BatchNorm(4b_c_bn)(X504,4b_c_bn_scale)/d_X504
JCudaTensor X1574 = y60[0];;
// dealloc X504
X504.free();
// V_4b_c_bn_scale = ((X1494 * -0.01) + (V_4b_c_bn_scale * 0.9))
V_4b_c_bn_scale.update(X1494, lrn_rate, momentum);
// dealloc X1494
X1494.free();
// 4b_c_bn_scale = (V_4b_c_bn_scale + (4b_c_bn_scale * (1 + (5.0E-4 * -0.01))))
_4b_c_bn_scale.update(V_4b_c_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1562 = X1574 * d_Convolv(1,0)(4b_c_cv_W)/d_X503
JCudaTensor X1562 = y33.backward_data(X1574, _4b_c_cv_W);
// V_4b_c_cv_W = ((X1574 * d_Convolv(1,0)(X503)/d_4b_c_cv_W * -0.01) + (V_4b_c_cv_W * 0.9))
y33.backward_filter(X1574, X503, V_4b_c_cv_W, lrn_rate, momentum);
// dealloc X1574
X1574.free();
// 4b_c_cv_W = (V_4b_c_cv_W + (4b_c_cv_W * (1 + (5.0E-4 * -0.01))))
_4b_c_cv_W.update(V_4b_c_cv_W, 1f, 1f + decay * lrn_rate);
// val X1355 = X1562 * d_ReLU()(X503)/d_X502
JCudaTensor X1355 = y34.backward(X1562, X503);
// dealloc X503
X503.free();
JCudaTensor[] y62 = y61.backward(X1355,X501,_4b_b_bn_scale);
// val X1376 = X1355 * d_BatchNorm(4b_b_bn)(X501,4b_b_bn_scale)/d_4b_b_bn_bias
JCudaTensor X1376 = y62[2];;
// V_4b_b_bn_bias = ((X1376 * -0.01) + (V_4b_b_bn_bias * 0.9))
V_4b_b_bn_bias.update(X1376, lrn_rate, momentum);
// dealloc X1376
X1376.free();
// 4b_b_bn_bias = (V_4b_b_bn_bias + (4b_b_bn_bias * (1 + (5.0E-4 * -0.01))))
_4b_b_bn_bias.update(V_4b_b_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1365 = X1355 * d_BatchNorm(4b_b_bn)(X501,4b_b_bn_scale)/d_4b_b_bn_scale
JCudaTensor X1365 = y62[1];;
// val X1723 = X1355 * d_BatchNorm(4b_b_bn)(X501,4b_b_bn_scale)/d_X501
JCudaTensor X1723 = y62[0];;
// dealloc X501
X501.free();
// V_4b_b_bn_scale = ((X1365 * -0.01) + (V_4b_b_bn_scale * 0.9))
V_4b_b_bn_scale.update(X1365, lrn_rate, momentum);
// dealloc X1365
X1365.free();
// 4b_b_bn_scale = (V_4b_b_bn_scale + (4b_b_bn_scale * (1 + (5.0E-4 * -0.01))))
_4b_b_bn_scale.update(V_4b_b_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1414 = X1723 * d_Convolv(1,1)(4b_b_cv_W)/d_X500
JCudaTensor X1414 = y37.backward_data(X1723, _4b_b_cv_W);
// V_4b_b_cv_W = ((X1723 * d_Convolv(1,1)(X500)/d_4b_b_cv_W * -0.01) + (V_4b_b_cv_W * 0.9))
y37.backward_filter(X1723, X500, V_4b_b_cv_W, lrn_rate, momentum);
// dealloc X1723
X1723.free();
// 4b_b_cv_W = (V_4b_b_cv_W + (4b_b_cv_W * (1 + (5.0E-4 * -0.01))))
_4b_b_cv_W.update(V_4b_b_cv_W, 1f, 1f + decay * lrn_rate);
// val X1689 = X1414 * d_ReLU()(X500)/d_X499
JCudaTensor X1689 = y34.backward(X1414, X500);
// dealloc X500
X500.free();
JCudaTensor[] y64 = y63.backward(X1689,X498,_4b_a_bn_scale);
// val X1572 = X1689 * d_BatchNorm(4b_a_bn)(X498,4b_a_bn_scale)/d_4b_a_bn_bias
JCudaTensor X1572 = y64[2];;
// V_4b_a_bn_bias = ((X1572 * -0.01) + (V_4b_a_bn_bias * 0.9))
V_4b_a_bn_bias.update(X1572, lrn_rate, momentum);
// dealloc X1572
X1572.free();
// 4b_a_bn_bias = (V_4b_a_bn_bias + (4b_a_bn_bias * (1 + (5.0E-4 * -0.01))))
_4b_a_bn_bias.update(V_4b_a_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1398 = X1689 * d_BatchNorm(4b_a_bn)(X498,4b_a_bn_scale)/d_4b_a_bn_scale
JCudaTensor X1398 = y64[1];;
// val X1543 = X1689 * d_BatchNorm(4b_a_bn)(X498,4b_a_bn_scale)/d_X498
JCudaTensor X1543 = y64[0];;
// dealloc X498
X498.free();
// V_4b_a_bn_scale = ((X1398 * -0.01) + (V_4b_a_bn_scale * 0.9))
V_4b_a_bn_scale.update(X1398, lrn_rate, momentum);
// dealloc X1398
X1398.free();
// 4b_a_bn_scale = (V_4b_a_bn_scale + (4b_a_bn_scale * (1 + (5.0E-4 * -0.01))))
_4b_a_bn_scale.update(V_4b_a_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1635 = X1543 * d_Convolv(1,0)(4b_a_cv_W)/d_X497
JCudaTensor X1635 = y40.backward_data(X1543, _4b_a_cv_W);
// V_4b_a_cv_W = ((X1543 * d_Convolv(1,0)(X497)/d_4b_a_cv_W * -0.01) + (V_4b_a_cv_W * 0.9))
y40.backward_filter(X1543, X497, V_4b_a_cv_W, lrn_rate, momentum);
// dealloc X1543
X1543.free();
// 4b_a_cv_W = (V_4b_a_cv_W + (4b_a_cv_W * (1 + (5.0E-4 * -0.01))))
_4b_a_cv_W.update(V_4b_a_cv_W, 1f, 1f + decay * lrn_rate);
// val X681 = (X1635 + X1424)
JCudaTensor X681 = X1635.plus_i(X1424);;
// dealloc X1424
X1424.free();
// val X1357 = X681 * d_ReLU()(X497)/d_X496
JCudaTensor X1357 = y30.backward(X681, X497);
// dealloc X497
X497.free();
// val X1364 = X1357.copy * d_ReLU()(X486)/d_X485
JCudaTensor X1364 = y30.backward(X1357.clone(), X486);
// dealloc X486
X486.free();
JCudaTensor[] y66 = y65.backward(X1364,X484,_4a1_bn_scale);
// val X1650 = X1364 * d_BatchNorm(4a1_bn)(X484,4a1_bn_scale)/d_4a1_bn_bias
JCudaTensor X1650 = y66[2];;
// V_4a1_bn_bias = ((X1650 * -0.01) + (V_4a1_bn_bias * 0.9))
V_4a1_bn_bias.update(X1650, lrn_rate, momentum);
// dealloc X1650
X1650.free();
// 4a1_bn_bias = (V_4a1_bn_bias + (4a1_bn_bias * (1 + (5.0E-4 * -0.01))))
_4a1_bn_bias.update(V_4a1_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1461 = X1364 * d_BatchNorm(4a1_bn)(X484,4a1_bn_scale)/d_X484
JCudaTensor X1461 = y66[0];;
// val X1721 = X1364 * d_BatchNorm(4a1_bn)(X484,4a1_bn_scale)/d_4a1_bn_scale
JCudaTensor X1721 = y66[1];;
// dealloc X484
X484.free();
// V_4a1_bn_scale = ((X1721 * -0.01) + (V_4a1_bn_scale * 0.9))
V_4a1_bn_scale.update(X1721, lrn_rate, momentum);
// dealloc X1721
X1721.free();
// 4a1_bn_scale = (V_4a1_bn_scale + (4a1_bn_scale * (1 + (5.0E-4 * -0.01))))
_4a1_bn_scale.update(V_4a1_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1497 = X1461 * d_Convolv(2,0)(4a1_cv_W)/d_X483
JCudaTensor X1497 = y67.backward_data(X1461, _4a1_cv_W);
// V_4a1_cv_W = ((X1461 * d_Convolv(2,0)(X483)/d_4a1_cv_W * -0.01) + (V_4a1_cv_W * 0.9))
y67.backward_filter(X1461, X483, V_4a1_cv_W, lrn_rate, momentum);
// dealloc X1461
X1461.free();
// 4a1_cv_W = (V_4a1_cv_W + (4a1_cv_W * (1 + (5.0E-4 * -0.01))))
_4a1_cv_W.update(V_4a1_cv_W, 1f, 1f + decay * lrn_rate);
// val X1460 = X1357.copy * d_ReLU()(X495)/d_X494
JCudaTensor X1460 = y30.backward(X1357.clone(), X495);
// dealloc X1357
X1357.free();
// dealloc X495
X495.free();
JCudaTensor[] y69 = y68.backward(X1460,X493,_4a2_c_bn_scale);
// val X1524 = X1460 * d_BatchNorm(4a2_c_bn)(X493,4a2_c_bn_scale)/d_4a2_c_bn_bias
JCudaTensor X1524 = y69[2];;
// V_4a2_c_bn_bias = ((X1524 * -0.01) + (V_4a2_c_bn_bias * 0.9))
V_4a2_c_bn_bias.update(X1524, lrn_rate, momentum);
// dealloc X1524
X1524.free();
// 4a2_c_bn_bias = (V_4a2_c_bn_bias + (4a2_c_bn_bias * (1 + (5.0E-4 * -0.01))))
_4a2_c_bn_bias.update(V_4a2_c_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1371 = X1460 * d_BatchNorm(4a2_c_bn)(X493,4a2_c_bn_scale)/d_X493
JCudaTensor X1371 = y69[0];;
// val X1587 = X1460 * d_BatchNorm(4a2_c_bn)(X493,4a2_c_bn_scale)/d_4a2_c_bn_scale
JCudaTensor X1587 = y69[1];;
// dealloc X493
X493.free();
// V_4a2_c_bn_scale = ((X1587 * -0.01) + (V_4a2_c_bn_scale * 0.9))
V_4a2_c_bn_scale.update(X1587, lrn_rate, momentum);
// dealloc X1587
X1587.free();
// 4a2_c_bn_scale = (V_4a2_c_bn_scale + (4a2_c_bn_scale * (1 + (5.0E-4 * -0.01))))
_4a2_c_bn_scale.update(V_4a2_c_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1432 = X1371 * d_Convolv(1,0)(4a2_c_cv_W)/d_X492
JCudaTensor X1432 = y33.backward_data(X1371, _4a2_c_cv_W);
// V_4a2_c_cv_W = ((X1371 * d_Convolv(1,0)(X492)/d_4a2_c_cv_W * -0.01) + (V_4a2_c_cv_W * 0.9))
y33.backward_filter(X1371, X492, V_4a2_c_cv_W, lrn_rate, momentum);
// dealloc X1371
X1371.free();
// 4a2_c_cv_W = (V_4a2_c_cv_W + (4a2_c_cv_W * (1 + (5.0E-4 * -0.01))))
_4a2_c_cv_W.update(V_4a2_c_cv_W, 1f, 1f + decay * lrn_rate);
// val X1717 = X1432 * d_ReLU()(X492)/d_X491
JCudaTensor X1717 = y34.backward(X1432, X492);
// dealloc X492
X492.free();
JCudaTensor[] y71 = y70.backward(X1717,X490,_4a2_b_bn_scale);
// val X1738 = X1717 * d_BatchNorm(4a2_b_bn)(X490,4a2_b_bn_scale)/d_4a2_b_bn_bias
JCudaTensor X1738 = y71[2];;
// V_4a2_b_bn_bias = ((X1738 * -0.01) + (V_4a2_b_bn_bias * 0.9))
V_4a2_b_bn_bias.update(X1738, lrn_rate, momentum);
// dealloc X1738
X1738.free();
// 4a2_b_bn_bias = (V_4a2_b_bn_bias + (4a2_b_bn_bias * (1 + (5.0E-4 * -0.01))))
_4a2_b_bn_bias.update(V_4a2_b_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1496 = X1717 * d_BatchNorm(4a2_b_bn)(X490,4a2_b_bn_scale)/d_4a2_b_bn_scale
JCudaTensor X1496 = y71[1];;
// val X1620 = X1717 * d_BatchNorm(4a2_b_bn)(X490,4a2_b_bn_scale)/d_X490
JCudaTensor X1620 = y71[0];;
// dealloc X490
X490.free();
// V_4a2_b_bn_scale = ((X1496 * -0.01) + (V_4a2_b_bn_scale * 0.9))
V_4a2_b_bn_scale.update(X1496, lrn_rate, momentum);
// dealloc X1496
X1496.free();
// 4a2_b_bn_scale = (V_4a2_b_bn_scale + (4a2_b_bn_scale * (1 + (5.0E-4 * -0.01))))
_4a2_b_bn_scale.update(V_4a2_b_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1464 = X1620 * d_Convolv(1,1)(4a2_b_cv_W)/d_X489
JCudaTensor X1464 = y37.backward_data(X1620, _4a2_b_cv_W);
// V_4a2_b_cv_W = ((X1620 * d_Convolv(1,1)(X489)/d_4a2_b_cv_W * -0.01) + (V_4a2_b_cv_W * 0.9))
y37.backward_filter(X1620, X489, V_4a2_b_cv_W, lrn_rate, momentum);
// dealloc X1620
X1620.free();
// 4a2_b_cv_W = (V_4a2_b_cv_W + (4a2_b_cv_W * (1 + (5.0E-4 * -0.01))))
_4a2_b_cv_W.update(V_4a2_b_cv_W, 1f, 1f + decay * lrn_rate);
// val X1420 = X1464 * d_ReLU()(X489)/d_X488
JCudaTensor X1420 = y34.backward(X1464, X489);
// dealloc X489
X489.free();
JCudaTensor[] y73 = y72.backward(X1420,X487,_4a2_a_bn_scale);
// val X1346 = X1420 * d_BatchNorm(4a2_a_bn)(X487,4a2_a_bn_scale)/d_4a2_a_bn_bias
JCudaTensor X1346 = y73[2];;
// V_4a2_a_bn_bias = ((X1346 * -0.01) + (V_4a2_a_bn_bias * 0.9))
V_4a2_a_bn_bias.update(X1346, lrn_rate, momentum);
// dealloc X1346
X1346.free();
// 4a2_a_bn_bias = (V_4a2_a_bn_bias + (4a2_a_bn_bias * (1 + (5.0E-4 * -0.01))))
_4a2_a_bn_bias.update(V_4a2_a_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1563 = X1420 * d_BatchNorm(4a2_a_bn)(X487,4a2_a_bn_scale)/d_X487
JCudaTensor X1563 = y73[0];;
// val X1698 = X1420 * d_BatchNorm(4a2_a_bn)(X487,4a2_a_bn_scale)/d_4a2_a_bn_scale
JCudaTensor X1698 = y73[1];;
// dealloc X487
X487.free();
// V_4a2_a_bn_scale = ((X1698 * -0.01) + (V_4a2_a_bn_scale * 0.9))
V_4a2_a_bn_scale.update(X1698, lrn_rate, momentum);
// dealloc X1698
X1698.free();
// 4a2_a_bn_scale = (V_4a2_a_bn_scale + (4a2_a_bn_scale * (1 + (5.0E-4 * -0.01))))
_4a2_a_bn_scale.update(V_4a2_a_bn_scale, 1f, 1f + decay * lrn_rate);
// val X686 = (X1497 + X1563 * d_Convolv(2,0)(4a2_a_cv_W)/d_X483)
JCudaTensor X686 = y141.backward_data(X1563,_4a2_a_cv_W, X1497);
// V_4a2_a_cv_W = ((X1563 * d_Convolv(2,0)(X483)/d_4a2_a_cv_W * -0.01) + (V_4a2_a_cv_W * 0.9))
y141.backward_filter(X1563, X483, V_4a2_a_cv_W, lrn_rate, momentum);
// dealloc X1563
X1563.free();
// 4a2_a_cv_W = (V_4a2_a_cv_W + (4a2_a_cv_W * (1 + (5.0E-4 * -0.01))))
_4a2_a_cv_W.update(V_4a2_a_cv_W, 1f, 1f + decay * lrn_rate);
// val X1330 = X686 * d_ReLU()(X483)/d_X482
JCudaTensor X1330 = y75.backward(X686, X483);
// dealloc X483
X483.free();
// val X1590 = X1330.copy * d_ReLU()(X481)/d_X480
JCudaTensor X1590 = y75.backward(X1330.clone(), X481);
// dealloc X481
X481.free();
JCudaTensor[] y77 = y76.backward(X1590,X479,_3d_c_bn_scale);
// val X1370 = X1590 * d_BatchNorm(3d_c_bn)(X479,3d_c_bn_scale)/d_3d_c_bn_bias
JCudaTensor X1370 = y77[2];;
// V_3d_c_bn_bias = ((X1370 * -0.01) + (V_3d_c_bn_bias * 0.9))
V_3d_c_bn_bias.update(X1370, lrn_rate, momentum);
// dealloc X1370
X1370.free();
// 3d_c_bn_bias = (V_3d_c_bn_bias + (3d_c_bn_bias * (1 + (5.0E-4 * -0.01))))
_3d_c_bn_bias.update(V_3d_c_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1485 = X1590 * d_BatchNorm(3d_c_bn)(X479,3d_c_bn_scale)/d_3d_c_bn_scale
JCudaTensor X1485 = y77[1];;
// val X1707 = X1590 * d_BatchNorm(3d_c_bn)(X479,3d_c_bn_scale)/d_X479
JCudaTensor X1707 = y77[0];;
// dealloc X479
X479.free();
// V_3d_c_bn_scale = ((X1485 * -0.01) + (V_3d_c_bn_scale * 0.9))
V_3d_c_bn_scale.update(X1485, lrn_rate, momentum);
// dealloc X1485
X1485.free();
// 3d_c_bn_scale = (V_3d_c_bn_scale + (3d_c_bn_scale * (1 + (5.0E-4 * -0.01))))
_3d_c_bn_scale.update(V_3d_c_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1708 = X1707 * d_Convolv(1,0)(3d_c_cv_W)/d_X478
JCudaTensor X1708 = y78.backward_data(X1707, _3d_c_cv_W);
// V_3d_c_cv_W = ((X1707 * d_Convolv(1,0)(X478)/d_3d_c_cv_W * -0.01) + (V_3d_c_cv_W * 0.9))
y78.backward_filter(X1707, X478, V_3d_c_cv_W, lrn_rate, momentum);
// dealloc X1707
X1707.free();
// 3d_c_cv_W = (V_3d_c_cv_W + (3d_c_cv_W * (1 + (5.0E-4 * -0.01))))
_3d_c_cv_W.update(V_3d_c_cv_W, 1f, 1f + decay * lrn_rate);
// val X1488 = X1708 * d_ReLU()(X478)/d_X477
JCudaTensor X1488 = y79.backward(X1708, X478);
// dealloc X478
X478.free();
JCudaTensor[] y81 = y80.backward(X1488,X476,_3d_b_bn_scale);
// val X1495 = X1488 * d_BatchNorm(3d_b_bn)(X476,3d_b_bn_scale)/d_3d_b_bn_bias
JCudaTensor X1495 = y81[2];;
// V_3d_b_bn_bias = ((X1495 * -0.01) + (V_3d_b_bn_bias * 0.9))
V_3d_b_bn_bias.update(X1495, lrn_rate, momentum);
// dealloc X1495
X1495.free();
// 3d_b_bn_bias = (V_3d_b_bn_bias + (3d_b_bn_bias * (1 + (5.0E-4 * -0.01))))
_3d_b_bn_bias.update(V_3d_b_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1328 = X1488 * d_BatchNorm(3d_b_bn)(X476,3d_b_bn_scale)/d_X476
JCudaTensor X1328 = y81[0];;
// val X1377 = X1488 * d_BatchNorm(3d_b_bn)(X476,3d_b_bn_scale)/d_3d_b_bn_scale
JCudaTensor X1377 = y81[1];;
// dealloc X476
X476.free();
// V_3d_b_bn_scale = ((X1377 * -0.01) + (V_3d_b_bn_scale * 0.9))
V_3d_b_bn_scale.update(X1377, lrn_rate, momentum);
// dealloc X1377
X1377.free();
// 3d_b_bn_scale = (V_3d_b_bn_scale + (3d_b_bn_scale * (1 + (5.0E-4 * -0.01))))
_3d_b_bn_scale.update(V_3d_b_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1512 = X1328 * d_Convolv(1,1)(3d_b_cv_W)/d_X475
JCudaTensor X1512 = y82.backward_data(X1328, _3d_b_cv_W);
// V_3d_b_cv_W = ((X1328 * d_Convolv(1,1)(X475)/d_3d_b_cv_W * -0.01) + (V_3d_b_cv_W * 0.9))
y82.backward_filter(X1328, X475, V_3d_b_cv_W, lrn_rate, momentum);
// dealloc X1328
X1328.free();
// 3d_b_cv_W = (V_3d_b_cv_W + (3d_b_cv_W * (1 + (5.0E-4 * -0.01))))
_3d_b_cv_W.update(V_3d_b_cv_W, 1f, 1f + decay * lrn_rate);
// val X1677 = X1512 * d_ReLU()(X475)/d_X474
JCudaTensor X1677 = y79.backward(X1512, X475);
// dealloc X475
X475.free();
JCudaTensor[] y84 = y83.backward(X1677,X473,_3d_a_bn_scale);
// val X1413 = X1677 * d_BatchNorm(3d_a_bn)(X473,3d_a_bn_scale)/d_3d_a_bn_bias
JCudaTensor X1413 = y84[2];;
// V_3d_a_bn_bias = ((X1413 * -0.01) + (V_3d_a_bn_bias * 0.9))
V_3d_a_bn_bias.update(X1413, lrn_rate, momentum);
// dealloc X1413
X1413.free();
// 3d_a_bn_bias = (V_3d_a_bn_bias + (3d_a_bn_bias * (1 + (5.0E-4 * -0.01))))
_3d_a_bn_bias.update(V_3d_a_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1392 = X1677 * d_BatchNorm(3d_a_bn)(X473,3d_a_bn_scale)/d_X473
JCudaTensor X1392 = y84[0];;
// val X1555 = X1677 * d_BatchNorm(3d_a_bn)(X473,3d_a_bn_scale)/d_3d_a_bn_scale
JCudaTensor X1555 = y84[1];;
// dealloc X473
X473.free();
// V_3d_a_bn_scale = ((X1555 * -0.01) + (V_3d_a_bn_scale * 0.9))
V_3d_a_bn_scale.update(X1555, lrn_rate, momentum);
// dealloc X1555
X1555.free();
// 3d_a_bn_scale = (V_3d_a_bn_scale + (3d_a_bn_scale * (1 + (5.0E-4 * -0.01))))
_3d_a_bn_scale.update(V_3d_a_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1575 = X1392 * d_Convolv(1,0)(3d_a_cv_W)/d_X472
JCudaTensor X1575 = y85.backward_data(X1392, _3d_a_cv_W);
// V_3d_a_cv_W = ((X1392 * d_Convolv(1,0)(X472)/d_3d_a_cv_W * -0.01) + (V_3d_a_cv_W * 0.9))
y85.backward_filter(X1392, X472, V_3d_a_cv_W, lrn_rate, momentum);
// dealloc X1392
X1392.free();
// 3d_a_cv_W = (V_3d_a_cv_W + (3d_a_cv_W * (1 + (5.0E-4 * -0.01))))
_3d_a_cv_W.update(V_3d_a_cv_W, 1f, 1f + decay * lrn_rate);
// val X697 = (X1575 + X1330)
JCudaTensor X697 = X1575.plus_i(X1330);;
// dealloc X1330
X1330.free();
// val X1350 = X697 * d_ReLU()(X472)/d_X471
JCudaTensor X1350 = y75.backward(X697, X472);
// dealloc X472
X472.free();
// val X1448 = X1350.copy * d_ReLU()(X470)/d_X469
JCudaTensor X1448 = y75.backward(X1350.clone(), X470);
// dealloc X470
X470.free();
JCudaTensor[] y87 = y86.backward(X1448,X468,_3c_c_bn_scale);
// val X1618 = X1448 * d_BatchNorm(3c_c_bn)(X468,3c_c_bn_scale)/d_3c_c_bn_bias
JCudaTensor X1618 = y87[2];;
// V_3c_c_bn_bias = ((X1618 * -0.01) + (V_3c_c_bn_bias * 0.9))
V_3c_c_bn_bias.update(X1618, lrn_rate, momentum);
// dealloc X1618
X1618.free();
// 3c_c_bn_bias = (V_3c_c_bn_bias + (3c_c_bn_bias * (1 + (5.0E-4 * -0.01))))
_3c_c_bn_bias.update(V_3c_c_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1412 = X1448 * d_BatchNorm(3c_c_bn)(X468,3c_c_bn_scale)/d_X468
JCudaTensor X1412 = y87[0];;
// val X1624 = X1448 * d_BatchNorm(3c_c_bn)(X468,3c_c_bn_scale)/d_3c_c_bn_scale
JCudaTensor X1624 = y87[1];;
// dealloc X468
X468.free();
// V_3c_c_bn_scale = ((X1624 * -0.01) + (V_3c_c_bn_scale * 0.9))
V_3c_c_bn_scale.update(X1624, lrn_rate, momentum);
// dealloc X1624
X1624.free();
// 3c_c_bn_scale = (V_3c_c_bn_scale + (3c_c_bn_scale * (1 + (5.0E-4 * -0.01))))
_3c_c_bn_scale.update(V_3c_c_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1397 = X1412 * d_Convolv(1,0)(3c_c_cv_W)/d_X467
JCudaTensor X1397 = y78.backward_data(X1412, _3c_c_cv_W);
// V_3c_c_cv_W = ((X1412 * d_Convolv(1,0)(X467)/d_3c_c_cv_W * -0.01) + (V_3c_c_cv_W * 0.9))
y78.backward_filter(X1412, X467, V_3c_c_cv_W, lrn_rate, momentum);
// dealloc X1412
X1412.free();
// 3c_c_cv_W = (V_3c_c_cv_W + (3c_c_cv_W * (1 + (5.0E-4 * -0.01))))
_3c_c_cv_W.update(V_3c_c_cv_W, 1f, 1f + decay * lrn_rate);
// val X1444 = X1397 * d_ReLU()(X467)/d_X466
JCudaTensor X1444 = y79.backward(X1397, X467);
// dealloc X467
X467.free();
JCudaTensor[] y89 = y88.backward(X1444,X465,_3c_b_bn_scale);
// val X1492 = X1444 * d_BatchNorm(3c_b_bn)(X465,3c_b_bn_scale)/d_3c_b_bn_bias
JCudaTensor X1492 = y89[2];;
// V_3c_b_bn_bias = ((X1492 * -0.01) + (V_3c_b_bn_bias * 0.9))
V_3c_b_bn_bias.update(X1492, lrn_rate, momentum);
// dealloc X1492
X1492.free();
// 3c_b_bn_bias = (V_3c_b_bn_bias + (3c_b_bn_bias * (1 + (5.0E-4 * -0.01))))
_3c_b_bn_bias.update(V_3c_b_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1529 = X1444 * d_BatchNorm(3c_b_bn)(X465,3c_b_bn_scale)/d_3c_b_bn_scale
JCudaTensor X1529 = y89[1];;
// val X1720 = X1444 * d_BatchNorm(3c_b_bn)(X465,3c_b_bn_scale)/d_X465
JCudaTensor X1720 = y89[0];;
// dealloc X465
X465.free();
// V_3c_b_bn_scale = ((X1529 * -0.01) + (V_3c_b_bn_scale * 0.9))
V_3c_b_bn_scale.update(X1529, lrn_rate, momentum);
// dealloc X1529
X1529.free();
// 3c_b_bn_scale = (V_3c_b_bn_scale + (3c_b_bn_scale * (1 + (5.0E-4 * -0.01))))
_3c_b_bn_scale.update(V_3c_b_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1598 = X1720 * d_Convolv(1,1)(3c_b_cv_W)/d_X464
JCudaTensor X1598 = y82.backward_data(X1720, _3c_b_cv_W);
// V_3c_b_cv_W = ((X1720 * d_Convolv(1,1)(X464)/d_3c_b_cv_W * -0.01) + (V_3c_b_cv_W * 0.9))
y82.backward_filter(X1720, X464, V_3c_b_cv_W, lrn_rate, momentum);
// dealloc X1720
X1720.free();
// 3c_b_cv_W = (V_3c_b_cv_W + (3c_b_cv_W * (1 + (5.0E-4 * -0.01))))
_3c_b_cv_W.update(V_3c_b_cv_W, 1f, 1f + decay * lrn_rate);
// val X1410 = X1598 * d_ReLU()(X464)/d_X463
JCudaTensor X1410 = y79.backward(X1598, X464);
// dealloc X464
X464.free();
JCudaTensor[] y91 = y90.backward(X1410,X462,_3c_a_bn_scale);
// val X1532 = X1410 * d_BatchNorm(3c_a_bn)(X462,3c_a_bn_scale)/d_3c_a_bn_bias
JCudaTensor X1532 = y91[2];;
// V_3c_a_bn_bias = ((X1532 * -0.01) + (V_3c_a_bn_bias * 0.9))
V_3c_a_bn_bias.update(X1532, lrn_rate, momentum);
// dealloc X1532
X1532.free();
// 3c_a_bn_bias = (V_3c_a_bn_bias + (3c_a_bn_bias * (1 + (5.0E-4 * -0.01))))
_3c_a_bn_bias.update(V_3c_a_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1348 = X1410 * d_BatchNorm(3c_a_bn)(X462,3c_a_bn_scale)/d_X462
JCudaTensor X1348 = y91[0];;
// val X1533 = X1410 * d_BatchNorm(3c_a_bn)(X462,3c_a_bn_scale)/d_3c_a_bn_scale
JCudaTensor X1533 = y91[1];;
// dealloc X462
X462.free();
// V_3c_a_bn_scale = ((X1533 * -0.01) + (V_3c_a_bn_scale * 0.9))
V_3c_a_bn_scale.update(X1533, lrn_rate, momentum);
// dealloc X1533
X1533.free();
// 3c_a_bn_scale = (V_3c_a_bn_scale + (3c_a_bn_scale * (1 + (5.0E-4 * -0.01))))
_3c_a_bn_scale.update(V_3c_a_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1588 = X1348 * d_Convolv(1,0)(3c_a_cv_W)/d_X461
JCudaTensor X1588 = y85.backward_data(X1348, _3c_a_cv_W);
// V_3c_a_cv_W = ((X1348 * d_Convolv(1,0)(X461)/d_3c_a_cv_W * -0.01) + (V_3c_a_cv_W * 0.9))
y85.backward_filter(X1348, X461, V_3c_a_cv_W, lrn_rate, momentum);
// dealloc X1348
X1348.free();
// 3c_a_cv_W = (V_3c_a_cv_W + (3c_a_cv_W * (1 + (5.0E-4 * -0.01))))
_3c_a_cv_W.update(V_3c_a_cv_W, 1f, 1f + decay * lrn_rate);
// val X708 = (X1588 + X1350)
JCudaTensor X708 = X1588.plus_i(X1350);;
// dealloc X1350
X1350.free();
// val X1382 = X708 * d_ReLU()(X461)/d_X460
JCudaTensor X1382 = y75.backward(X708, X461);
// dealloc X461
X461.free();
// val X1400 = X1382.copy * d_ReLU()(X459)/d_X458
JCudaTensor X1400 = y75.backward(X1382.clone(), X459);
// dealloc X459
X459.free();
JCudaTensor[] y93 = y92.backward(X1400,X457,_3b_c_bn_scale);
// val X1656 = X1400 * d_BatchNorm(3b_c_bn)(X457,3b_c_bn_scale)/d_3b_c_bn_bias
JCudaTensor X1656 = y93[2];;
// V_3b_c_bn_bias = ((X1656 * -0.01) + (V_3b_c_bn_bias * 0.9))
V_3b_c_bn_bias.update(X1656, lrn_rate, momentum);
// dealloc X1656
X1656.free();
// 3b_c_bn_bias = (V_3b_c_bn_bias + (3b_c_bn_bias * (1 + (5.0E-4 * -0.01))))
_3b_c_bn_bias.update(V_3b_c_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1352 = X1400 * d_BatchNorm(3b_c_bn)(X457,3b_c_bn_scale)/d_3b_c_bn_scale
JCudaTensor X1352 = y93[1];;
// val X1542 = X1400 * d_BatchNorm(3b_c_bn)(X457,3b_c_bn_scale)/d_X457
JCudaTensor X1542 = y93[0];;
// dealloc X457
X457.free();
// V_3b_c_bn_scale = ((X1352 * -0.01) + (V_3b_c_bn_scale * 0.9))
V_3b_c_bn_scale.update(X1352, lrn_rate, momentum);
// dealloc X1352
X1352.free();
// 3b_c_bn_scale = (V_3b_c_bn_scale + (3b_c_bn_scale * (1 + (5.0E-4 * -0.01))))
_3b_c_bn_scale.update(V_3b_c_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1384 = X1542 * d_Convolv(1,0)(3b_c_cv_W)/d_X456
JCudaTensor X1384 = y78.backward_data(X1542, _3b_c_cv_W);
// V_3b_c_cv_W = ((X1542 * d_Convolv(1,0)(X456)/d_3b_c_cv_W * -0.01) + (V_3b_c_cv_W * 0.9))
y78.backward_filter(X1542, X456, V_3b_c_cv_W, lrn_rate, momentum);
// dealloc X1542
X1542.free();
// 3b_c_cv_W = (V_3b_c_cv_W + (3b_c_cv_W * (1 + (5.0E-4 * -0.01))))
_3b_c_cv_W.update(V_3b_c_cv_W, 1f, 1f + decay * lrn_rate);
// val X1442 = X1384 * d_ReLU()(X456)/d_X455
JCudaTensor X1442 = y79.backward(X1384, X456);
// dealloc X456
X456.free();
JCudaTensor[] y95 = y94.backward(X1442,X454,_3b_b_bn_scale);
// val X1366 = X1442 * d_BatchNorm(3b_b_bn)(X454,3b_b_bn_scale)/d_3b_b_bn_bias
JCudaTensor X1366 = y95[2];;
// V_3b_b_bn_bias = ((X1366 * -0.01) + (V_3b_b_bn_bias * 0.9))
V_3b_b_bn_bias.update(X1366, lrn_rate, momentum);
// dealloc X1366
X1366.free();
// 3b_b_bn_bias = (V_3b_b_bn_bias + (3b_b_bn_bias * (1 + (5.0E-4 * -0.01))))
_3b_b_bn_bias.update(V_3b_b_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1538 = X1442 * d_BatchNorm(3b_b_bn)(X454,3b_b_bn_scale)/d_3b_b_bn_scale
JCudaTensor X1538 = y95[1];;
// val X1594 = X1442 * d_BatchNorm(3b_b_bn)(X454,3b_b_bn_scale)/d_X454
JCudaTensor X1594 = y95[0];;
// dealloc X454
X454.free();
// V_3b_b_bn_scale = ((X1538 * -0.01) + (V_3b_b_bn_scale * 0.9))
V_3b_b_bn_scale.update(X1538, lrn_rate, momentum);
// dealloc X1538
X1538.free();
// 3b_b_bn_scale = (V_3b_b_bn_scale + (3b_b_bn_scale * (1 + (5.0E-4 * -0.01))))
_3b_b_bn_scale.update(V_3b_b_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1704 = X1594 * d_Convolv(1,1)(3b_b_cv_W)/d_X453
JCudaTensor X1704 = y82.backward_data(X1594, _3b_b_cv_W);
// V_3b_b_cv_W = ((X1594 * d_Convolv(1,1)(X453)/d_3b_b_cv_W * -0.01) + (V_3b_b_cv_W * 0.9))
y82.backward_filter(X1594, X453, V_3b_b_cv_W, lrn_rate, momentum);
// dealloc X1594
X1594.free();
// 3b_b_cv_W = (V_3b_b_cv_W + (3b_b_cv_W * (1 + (5.0E-4 * -0.01))))
_3b_b_cv_W.update(V_3b_b_cv_W, 1f, 1f + decay * lrn_rate);
// val X1729 = X1704 * d_ReLU()(X453)/d_X452
JCudaTensor X1729 = y79.backward(X1704, X453);
// dealloc X453
X453.free();
JCudaTensor[] y97 = y96.backward(X1729,X451,_3b_a_bn_scale);
// val X1386 = X1729 * d_BatchNorm(3b_a_bn)(X451,3b_a_bn_scale)/d_3b_a_bn_bias
JCudaTensor X1386 = y97[2];;
// V_3b_a_bn_bias = ((X1386 * -0.01) + (V_3b_a_bn_bias * 0.9))
V_3b_a_bn_bias.update(X1386, lrn_rate, momentum);
// dealloc X1386
X1386.free();
// 3b_a_bn_bias = (V_3b_a_bn_bias + (3b_a_bn_bias * (1 + (5.0E-4 * -0.01))))
_3b_a_bn_bias.update(V_3b_a_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1509 = X1729 * d_BatchNorm(3b_a_bn)(X451,3b_a_bn_scale)/d_3b_a_bn_scale
JCudaTensor X1509 = y97[1];;
// val X1678 = X1729 * d_BatchNorm(3b_a_bn)(X451,3b_a_bn_scale)/d_X451
JCudaTensor X1678 = y97[0];;
// dealloc X451
X451.free();
// V_3b_a_bn_scale = ((X1509 * -0.01) + (V_3b_a_bn_scale * 0.9))
V_3b_a_bn_scale.update(X1509, lrn_rate, momentum);
// dealloc X1509
X1509.free();
// 3b_a_bn_scale = (V_3b_a_bn_scale + (3b_a_bn_scale * (1 + (5.0E-4 * -0.01))))
_3b_a_bn_scale.update(V_3b_a_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1345 = X1678 * d_Convolv(1,0)(3b_a_cv_W)/d_X450
JCudaTensor X1345 = y85.backward_data(X1678, _3b_a_cv_W);
// V_3b_a_cv_W = ((X1678 * d_Convolv(1,0)(X450)/d_3b_a_cv_W * -0.01) + (V_3b_a_cv_W * 0.9))
y85.backward_filter(X1678, X450, V_3b_a_cv_W, lrn_rate, momentum);
// dealloc X1678
X1678.free();
// 3b_a_cv_W = (V_3b_a_cv_W + (3b_a_cv_W * (1 + (5.0E-4 * -0.01))))
_3b_a_cv_W.update(V_3b_a_cv_W, 1f, 1f + decay * lrn_rate);
// val X719 = (X1345 + X1382)
JCudaTensor X719 = X1345.plus_i(X1382);;
// dealloc X1382
X1382.free();
// val X1702 = X719 * d_ReLU()(X450)/d_X449
JCudaTensor X1702 = y75.backward(X719, X450);
// dealloc X450
X450.free();
// val X1731 = X1702.copy * d_ReLU()(X439)/d_X438
JCudaTensor X1731 = y75.backward(X1702.clone(), X439);
// dealloc X439
X439.free();
JCudaTensor[] y99 = y98.backward(X1731,X437,_3a1_bn_scale);
// val X1458 = X1731 * d_BatchNorm(3a1_bn)(X437,3a1_bn_scale)/d_3a1_bn_bias
JCudaTensor X1458 = y99[2];;
// V_3a1_bn_bias = ((X1458 * -0.01) + (V_3a1_bn_bias * 0.9))
V_3a1_bn_bias.update(X1458, lrn_rate, momentum);
// dealloc X1458
X1458.free();
// 3a1_bn_bias = (V_3a1_bn_bias + (3a1_bn_bias * (1 + (5.0E-4 * -0.01))))
_3a1_bn_bias.update(V_3a1_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1331 = X1731 * d_BatchNorm(3a1_bn)(X437,3a1_bn_scale)/d_X437
JCudaTensor X1331 = y99[0];;
// val X1669 = X1731 * d_BatchNorm(3a1_bn)(X437,3a1_bn_scale)/d_3a1_bn_scale
JCudaTensor X1669 = y99[1];;
// dealloc X437
X437.free();
// V_3a1_bn_scale = ((X1669 * -0.01) + (V_3a1_bn_scale * 0.9))
V_3a1_bn_scale.update(X1669, lrn_rate, momentum);
// dealloc X1669
X1669.free();
// 3a1_bn_scale = (V_3a1_bn_scale + (3a1_bn_scale * (1 + (5.0E-4 * -0.01))))
_3a1_bn_scale.update(V_3a1_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1479 = X1331 * d_Convolv(2,0)(3a1_cv_W)/d_X436
JCudaTensor X1479 = y100.backward_data(X1331, _3a1_cv_W);
// V_3a1_cv_W = ((X1331 * d_Convolv(2,0)(X436)/d_3a1_cv_W * -0.01) + (V_3a1_cv_W * 0.9))
y100.backward_filter(X1331, X436, V_3a1_cv_W, lrn_rate, momentum);
// dealloc X1331
X1331.free();
// 3a1_cv_W = (V_3a1_cv_W + (3a1_cv_W * (1 + (5.0E-4 * -0.01))))
_3a1_cv_W.update(V_3a1_cv_W, 1f, 1f + decay * lrn_rate);
// val X1578 = X1702.copy * d_ReLU()(X448)/d_X447
JCudaTensor X1578 = y75.backward(X1702.clone(), X448);
// dealloc X448
X448.free();
// dealloc X1702
X1702.free();
JCudaTensor[] y102 = y101.backward(X1578,X446,_3a2_c_bn_scale);
// val X1642 = X1578 * d_BatchNorm(3a2_c_bn)(X446,3a2_c_bn_scale)/d_3a2_c_bn_bias
JCudaTensor X1642 = y102[2];;
// V_3a2_c_bn_bias = ((X1642 * -0.01) + (V_3a2_c_bn_bias * 0.9))
V_3a2_c_bn_bias.update(X1642, lrn_rate, momentum);
// dealloc X1642
X1642.free();
// 3a2_c_bn_bias = (V_3a2_c_bn_bias + (3a2_c_bn_bias * (1 + (5.0E-4 * -0.01))))
_3a2_c_bn_bias.update(V_3a2_c_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1343 = X1578 * d_BatchNorm(3a2_c_bn)(X446,3a2_c_bn_scale)/d_3a2_c_bn_scale
JCudaTensor X1343 = y102[1];;
// val X1692 = X1578 * d_BatchNorm(3a2_c_bn)(X446,3a2_c_bn_scale)/d_X446
JCudaTensor X1692 = y102[0];;
// dealloc X446
X446.free();
// V_3a2_c_bn_scale = ((X1343 * -0.01) + (V_3a2_c_bn_scale * 0.9))
V_3a2_c_bn_scale.update(X1343, lrn_rate, momentum);
// dealloc X1343
X1343.free();
// 3a2_c_bn_scale = (V_3a2_c_bn_scale + (3a2_c_bn_scale * (1 + (5.0E-4 * -0.01))))
_3a2_c_bn_scale.update(V_3a2_c_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1351 = X1692 * d_Convolv(1,0)(3a2_c_cv_W)/d_X445
JCudaTensor X1351 = y78.backward_data(X1692, _3a2_c_cv_W);
// V_3a2_c_cv_W = ((X1692 * d_Convolv(1,0)(X445)/d_3a2_c_cv_W * -0.01) + (V_3a2_c_cv_W * 0.9))
y78.backward_filter(X1692, X445, V_3a2_c_cv_W, lrn_rate, momentum);
// dealloc X1692
X1692.free();
// 3a2_c_cv_W = (V_3a2_c_cv_W + (3a2_c_cv_W * (1 + (5.0E-4 * -0.01))))
_3a2_c_cv_W.update(V_3a2_c_cv_W, 1f, 1f + decay * lrn_rate);
// val X1706 = X1351 * d_ReLU()(X445)/d_X444
JCudaTensor X1706 = y79.backward(X1351, X445);
// dealloc X445
X445.free();
JCudaTensor[] y104 = y103.backward(X1706,X443,_3a2_b_bn_scale);
// val X1643 = X1706 * d_BatchNorm(3a2_b_bn)(X443,3a2_b_bn_scale)/d_3a2_b_bn_bias
JCudaTensor X1643 = y104[2];;
// V_3a2_b_bn_bias = ((X1643 * -0.01) + (V_3a2_b_bn_bias * 0.9))
V_3a2_b_bn_bias.update(X1643, lrn_rate, momentum);
// dealloc X1643
X1643.free();
// 3a2_b_bn_bias = (V_3a2_b_bn_bias + (3a2_b_bn_bias * (1 + (5.0E-4 * -0.01))))
_3a2_b_bn_bias.update(V_3a2_b_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1435 = X1706 * d_BatchNorm(3a2_b_bn)(X443,3a2_b_bn_scale)/d_X443
JCudaTensor X1435 = y104[0];;
// val X1727 = X1706 * d_BatchNorm(3a2_b_bn)(X443,3a2_b_bn_scale)/d_3a2_b_bn_scale
JCudaTensor X1727 = y104[1];;
// dealloc X443
X443.free();
// V_3a2_b_bn_scale = ((X1727 * -0.01) + (V_3a2_b_bn_scale * 0.9))
V_3a2_b_bn_scale.update(X1727, lrn_rate, momentum);
// dealloc X1727
X1727.free();
// 3a2_b_bn_scale = (V_3a2_b_bn_scale + (3a2_b_bn_scale * (1 + (5.0E-4 * -0.01))))
_3a2_b_bn_scale.update(V_3a2_b_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1636 = X1435 * d_Convolv(1,1)(3a2_b_cv_W)/d_X442
JCudaTensor X1636 = y82.backward_data(X1435, _3a2_b_cv_W);
// V_3a2_b_cv_W = ((X1435 * d_Convolv(1,1)(X442)/d_3a2_b_cv_W * -0.01) + (V_3a2_b_cv_W * 0.9))
y82.backward_filter(X1435, X442, V_3a2_b_cv_W, lrn_rate, momentum);
// dealloc X1435
X1435.free();
// 3a2_b_cv_W = (V_3a2_b_cv_W + (3a2_b_cv_W * (1 + (5.0E-4 * -0.01))))
_3a2_b_cv_W.update(V_3a2_b_cv_W, 1f, 1f + decay * lrn_rate);
// val X1404 = X1636 * d_ReLU()(X442)/d_X441
JCudaTensor X1404 = y79.backward(X1636, X442);
// dealloc X442
X442.free();
JCudaTensor[] y106 = y105.backward(X1404,X440,_3a2_a_bn_scale);
// val X1387 = X1404 * d_BatchNorm(3a2_a_bn)(X440,3a2_a_bn_scale)/d_3a2_a_bn_bias
JCudaTensor X1387 = y106[2];;
// V_3a2_a_bn_bias = ((X1387 * -0.01) + (V_3a2_a_bn_bias * 0.9))
V_3a2_a_bn_bias.update(X1387, lrn_rate, momentum);
// dealloc X1387
X1387.free();
// 3a2_a_bn_bias = (V_3a2_a_bn_bias + (3a2_a_bn_bias * (1 + (5.0E-4 * -0.01))))
_3a2_a_bn_bias.update(V_3a2_a_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1407 = X1404 * d_BatchNorm(3a2_a_bn)(X440,3a2_a_bn_scale)/d_3a2_a_bn_scale
JCudaTensor X1407 = y106[1];;
// val X1517 = X1404 * d_BatchNorm(3a2_a_bn)(X440,3a2_a_bn_scale)/d_X440
JCudaTensor X1517 = y106[0];;
// dealloc X440
X440.free();
// V_3a2_a_bn_scale = ((X1407 * -0.01) + (V_3a2_a_bn_scale * 0.9))
V_3a2_a_bn_scale.update(X1407, lrn_rate, momentum);
// dealloc X1407
X1407.free();
// 3a2_a_bn_scale = (V_3a2_a_bn_scale + (3a2_a_bn_scale * (1 + (5.0E-4 * -0.01))))
_3a2_a_bn_scale.update(V_3a2_a_bn_scale, 1f, 1f + decay * lrn_rate);
// val X724 = (X1479 + X1517 * d_Convolv(2,0)(3a2_a_cv_W)/d_X436)
JCudaTensor X724 = y140.backward_data(X1517,_3a2_a_cv_W, X1479);
// V_3a2_a_cv_W = ((X1517 * d_Convolv(2,0)(X436)/d_3a2_a_cv_W * -0.01) + (V_3a2_a_cv_W * 0.9))
y140.backward_filter(X1517, X436, V_3a2_a_cv_W, lrn_rate, momentum);
// dealloc X1517
X1517.free();
// 3a2_a_cv_W = (V_3a2_a_cv_W + (3a2_a_cv_W * (1 + (5.0E-4 * -0.01))))
_3a2_a_cv_W.update(V_3a2_a_cv_W, 1f, 1f + decay * lrn_rate);
// val X1406 = X724 * d_ReLU()(X436)/d_X435
JCudaTensor X1406 = y108.backward(X724, X436);
// dealloc X436
X436.free();
// val X1629 = X1406.copy * d_ReLU()(X434)/d_X433
JCudaTensor X1629 = y108.backward(X1406.clone(), X434);
// dealloc X434
X434.free();
JCudaTensor[] y110 = y109.backward(X1629,X432,_2c_c_bn_scale);
// val X1426 = X1629 * d_BatchNorm(2c_c_bn)(X432,2c_c_bn_scale)/d_2c_c_bn_bias
JCudaTensor X1426 = y110[2];;
// V_2c_c_bn_bias = ((X1426 * -0.01) + (V_2c_c_bn_bias * 0.9))
V_2c_c_bn_bias.update(X1426, lrn_rate, momentum);
// dealloc X1426
X1426.free();
// 2c_c_bn_bias = (V_2c_c_bn_bias + (2c_c_bn_bias * (1 + (5.0E-4 * -0.01))))
_2c_c_bn_bias.update(V_2c_c_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1552 = X1629 * d_BatchNorm(2c_c_bn)(X432,2c_c_bn_scale)/d_2c_c_bn_scale
JCudaTensor X1552 = y110[1];;
// val X1672 = X1629 * d_BatchNorm(2c_c_bn)(X432,2c_c_bn_scale)/d_X432
JCudaTensor X1672 = y110[0];;
// dealloc X432
X432.free();
// V_2c_c_bn_scale = ((X1552 * -0.01) + (V_2c_c_bn_scale * 0.9))
V_2c_c_bn_scale.update(X1552, lrn_rate, momentum);
// dealloc X1552
X1552.free();
// 2c_c_bn_scale = (V_2c_c_bn_scale + (2c_c_bn_scale * (1 + (5.0E-4 * -0.01))))
_2c_c_bn_scale.update(V_2c_c_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1649 = X1672 * d_Convolv(1,0)(2c_c_cv_W)/d_X431
JCudaTensor X1649 = y111.backward_data(X1672, _2c_c_cv_W);
// V_2c_c_cv_W = ((X1672 * d_Convolv(1,0)(X431)/d_2c_c_cv_W * -0.01) + (V_2c_c_cv_W * 0.9))
y111.backward_filter(X1672, X431, V_2c_c_cv_W, lrn_rate, momentum);
// dealloc X1672
X1672.free();
// 2c_c_cv_W = (V_2c_c_cv_W + (2c_c_cv_W * (1 + (5.0E-4 * -0.01))))
_2c_c_cv_W.update(V_2c_c_cv_W, 1f, 1f + decay * lrn_rate);
// val X1431 = X1649 * d_ReLU()(X431)/d_X430
JCudaTensor X1431 = y112.backward(X1649, X431);
// dealloc X431
X431.free();
JCudaTensor[] y114 = y113.backward(X1431,X429,_2c_b_bn_scale);
// val X1391 = X1431 * d_BatchNorm(2c_b_bn)(X429,2c_b_bn_scale)/d_2c_b_bn_bias
JCudaTensor X1391 = y114[2];;
// V_2c_b_bn_bias = ((X1391 * -0.01) + (V_2c_b_bn_bias * 0.9))
V_2c_b_bn_bias.update(X1391, lrn_rate, momentum);
// dealloc X1391
X1391.free();
// 2c_b_bn_bias = (V_2c_b_bn_bias + (2c_b_bn_bias * (1 + (5.0E-4 * -0.01))))
_2c_b_bn_bias.update(V_2c_b_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1342 = X1431 * d_BatchNorm(2c_b_bn)(X429,2c_b_bn_scale)/d_2c_b_bn_scale
JCudaTensor X1342 = y114[1];;
// val X1646 = X1431 * d_BatchNorm(2c_b_bn)(X429,2c_b_bn_scale)/d_X429
JCudaTensor X1646 = y114[0];;
// dealloc X429
X429.free();
// V_2c_b_bn_scale = ((X1342 * -0.01) + (V_2c_b_bn_scale * 0.9))
V_2c_b_bn_scale.update(X1342, lrn_rate, momentum);
// dealloc X1342
X1342.free();
// 2c_b_bn_scale = (V_2c_b_bn_scale + (2c_b_bn_scale * (1 + (5.0E-4 * -0.01))))
_2c_b_bn_scale.update(V_2c_b_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1599 = X1646 * d_Convolv(1,1)(2c_b_cv_W)/d_X428
JCudaTensor X1599 = y115.backward_data(X1646, _2c_b_cv_W);
// V_2c_b_cv_W = ((X1646 * d_Convolv(1,1)(X428)/d_2c_b_cv_W * -0.01) + (V_2c_b_cv_W * 0.9))
y115.backward_filter(X1646, X428, V_2c_b_cv_W, lrn_rate, momentum);
// dealloc X1646
X1646.free();
// 2c_b_cv_W = (V_2c_b_cv_W + (2c_b_cv_W * (1 + (5.0E-4 * -0.01))))
_2c_b_cv_W.update(V_2c_b_cv_W, 1f, 1f + decay * lrn_rate);
// val X1523 = X1599 * d_ReLU()(X428)/d_X427
JCudaTensor X1523 = y112.backward(X1599, X428);
// dealloc X428
X428.free();
JCudaTensor[] y117 = y116.backward(X1523,X426,_2c_a_bn_scale);
// val X1332 = X1523 * d_BatchNorm(2c_a_bn)(X426,2c_a_bn_scale)/d_2c_a_bn_bias
JCudaTensor X1332 = y117[2];;
// V_2c_a_bn_bias = ((X1332 * -0.01) + (V_2c_a_bn_bias * 0.9))
V_2c_a_bn_bias.update(X1332, lrn_rate, momentum);
// dealloc X1332
X1332.free();
// 2c_a_bn_bias = (V_2c_a_bn_bias + (2c_a_bn_bias * (1 + (5.0E-4 * -0.01))))
_2c_a_bn_bias.update(V_2c_a_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1630 = X1523 * d_BatchNorm(2c_a_bn)(X426,2c_a_bn_scale)/d_2c_a_bn_scale
JCudaTensor X1630 = y117[1];;
// val X1647 = X1523 * d_BatchNorm(2c_a_bn)(X426,2c_a_bn_scale)/d_X426
JCudaTensor X1647 = y117[0];;
// dealloc X426
X426.free();
// V_2c_a_bn_scale = ((X1630 * -0.01) + (V_2c_a_bn_scale * 0.9))
V_2c_a_bn_scale.update(X1630, lrn_rate, momentum);
// dealloc X1630
X1630.free();
// 2c_a_bn_scale = (V_2c_a_bn_scale + (2c_a_bn_scale * (1 + (5.0E-4 * -0.01))))
_2c_a_bn_scale.update(V_2c_a_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1668 = X1647 * d_Convolv(1,0)(2c_a_cv_W)/d_X425
JCudaTensor X1668 = y118.backward_data(X1647, _2c_a_cv_W);
// V_2c_a_cv_W = ((X1647 * d_Convolv(1,0)(X425)/d_2c_a_cv_W * -0.01) + (V_2c_a_cv_W * 0.9))
y118.backward_filter(X1647, X425, V_2c_a_cv_W, lrn_rate, momentum);
// dealloc X1647
X1647.free();
// 2c_a_cv_W = (V_2c_a_cv_W + (2c_a_cv_W * (1 + (5.0E-4 * -0.01))))
_2c_a_cv_W.update(V_2c_a_cv_W, 1f, 1f + decay * lrn_rate);
// val X735 = (X1668 + X1406)
JCudaTensor X735 = X1668.plus_i(X1406);;
// dealloc X1406
X1406.free();
// val X1654 = X735 * d_ReLU()(X425)/d_X424
JCudaTensor X1654 = y108.backward(X735, X425);
// dealloc X425
X425.free();
// val X1375 = X1654.copy * d_ReLU()(X423)/d_X422
JCudaTensor X1375 = y108.backward(X1654.clone(), X423);
// dealloc X423
X423.free();
JCudaTensor[] y120 = y119.backward(X1375,X421,_2b_c_bn_scale);
// val X1425 = X1375 * d_BatchNorm(2b_c_bn)(X421,2b_c_bn_scale)/d_2b_c_bn_bias
JCudaTensor X1425 = y120[2];;
// V_2b_c_bn_bias = ((X1425 * -0.01) + (V_2b_c_bn_bias * 0.9))
V_2b_c_bn_bias.update(X1425, lrn_rate, momentum);
// dealloc X1425
X1425.free();
// 2b_c_bn_bias = (V_2b_c_bn_bias + (2b_c_bn_bias * (1 + (5.0E-4 * -0.01))))
_2b_c_bn_bias.update(V_2b_c_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1456 = X1375 * d_BatchNorm(2b_c_bn)(X421,2b_c_bn_scale)/d_2b_c_bn_scale
JCudaTensor X1456 = y120[1];;
// val X1536 = X1375 * d_BatchNorm(2b_c_bn)(X421,2b_c_bn_scale)/d_X421
JCudaTensor X1536 = y120[0];;
// dealloc X421
X421.free();
// V_2b_c_bn_scale = ((X1456 * -0.01) + (V_2b_c_bn_scale * 0.9))
V_2b_c_bn_scale.update(X1456, lrn_rate, momentum);
// dealloc X1456
X1456.free();
// 2b_c_bn_scale = (V_2b_c_bn_scale + (2b_c_bn_scale * (1 + (5.0E-4 * -0.01))))
_2b_c_bn_scale.update(V_2b_c_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1550 = X1536 * d_Convolv(1,0)(2b_c_cv_W)/d_X420
JCudaTensor X1550 = y111.backward_data(X1536, _2b_c_cv_W);
// V_2b_c_cv_W = ((X1536 * d_Convolv(1,0)(X420)/d_2b_c_cv_W * -0.01) + (V_2b_c_cv_W * 0.9))
y111.backward_filter(X1536, X420, V_2b_c_cv_W, lrn_rate, momentum);
// dealloc X1536
X1536.free();
// 2b_c_cv_W = (V_2b_c_cv_W + (2b_c_cv_W * (1 + (5.0E-4 * -0.01))))
_2b_c_cv_W.update(V_2b_c_cv_W, 1f, 1f + decay * lrn_rate);
// val X1696 = X1550 * d_ReLU()(X420)/d_X419
JCudaTensor X1696 = y112.backward(X1550, X420);
// dealloc X420
X420.free();
JCudaTensor[] y122 = y121.backward(X1696,X418,_2b_b_bn_scale);
// val X1683 = X1696 * d_BatchNorm(2b_b_bn)(X418,2b_b_bn_scale)/d_2b_b_bn_bias
JCudaTensor X1683 = y122[2];;
// V_2b_b_bn_bias = ((X1683 * -0.01) + (V_2b_b_bn_bias * 0.9))
V_2b_b_bn_bias.update(X1683, lrn_rate, momentum);
// dealloc X1683
X1683.free();
// 2b_b_bn_bias = (V_2b_b_bn_bias + (2b_b_bn_bias * (1 + (5.0E-4 * -0.01))))
_2b_b_bn_bias.update(V_2b_b_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1401 = X1696 * d_BatchNorm(2b_b_bn)(X418,2b_b_bn_scale)/d_2b_b_bn_scale
JCudaTensor X1401 = y122[1];;
// val X1540 = X1696 * d_BatchNorm(2b_b_bn)(X418,2b_b_bn_scale)/d_X418
JCudaTensor X1540 = y122[0];;
// dealloc X418
X418.free();
// V_2b_b_bn_scale = ((X1401 * -0.01) + (V_2b_b_bn_scale * 0.9))
V_2b_b_bn_scale.update(X1401, lrn_rate, momentum);
// dealloc X1401
X1401.free();
// 2b_b_bn_scale = (V_2b_b_bn_scale + (2b_b_bn_scale * (1 + (5.0E-4 * -0.01))))
_2b_b_bn_scale.update(V_2b_b_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1450 = X1540 * d_Convolv(1,1)(2b_b_cv_W)/d_X417
JCudaTensor X1450 = y115.backward_data(X1540, _2b_b_cv_W);
// V_2b_b_cv_W = ((X1540 * d_Convolv(1,1)(X417)/d_2b_b_cv_W * -0.01) + (V_2b_b_cv_W * 0.9))
y115.backward_filter(X1540, X417, V_2b_b_cv_W, lrn_rate, momentum);
// dealloc X1540
X1540.free();
// 2b_b_cv_W = (V_2b_b_cv_W + (2b_b_cv_W * (1 + (5.0E-4 * -0.01))))
_2b_b_cv_W.update(V_2b_b_cv_W, 1f, 1f + decay * lrn_rate);
// val X1610 = X1450 * d_ReLU()(X417)/d_X416
JCudaTensor X1610 = y112.backward(X1450, X417);
// dealloc X417
X417.free();
JCudaTensor[] y124 = y123.backward(X1610,X415,_2b_a_bn_scale);
// val X1344 = X1610 * d_BatchNorm(2b_a_bn)(X415,2b_a_bn_scale)/d_2b_a_bn_bias
JCudaTensor X1344 = y124[2];;
// V_2b_a_bn_bias = ((X1344 * -0.01) + (V_2b_a_bn_bias * 0.9))
V_2b_a_bn_bias.update(X1344, lrn_rate, momentum);
// dealloc X1344
X1344.free();
// 2b_a_bn_bias = (V_2b_a_bn_bias + (2b_a_bn_bias * (1 + (5.0E-4 * -0.01))))
_2b_a_bn_bias.update(V_2b_a_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1564 = X1610 * d_BatchNorm(2b_a_bn)(X415,2b_a_bn_scale)/d_X415
JCudaTensor X1564 = y124[0];;
// val X1733 = X1610 * d_BatchNorm(2b_a_bn)(X415,2b_a_bn_scale)/d_2b_a_bn_scale
JCudaTensor X1733 = y124[1];;
// dealloc X415
X415.free();
// V_2b_a_bn_scale = ((X1733 * -0.01) + (V_2b_a_bn_scale * 0.9))
V_2b_a_bn_scale.update(X1733, lrn_rate, momentum);
// dealloc X1733
X1733.free();
// 2b_a_bn_scale = (V_2b_a_bn_scale + (2b_a_bn_scale * (1 + (5.0E-4 * -0.01))))
_2b_a_bn_scale.update(V_2b_a_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1699 = X1564 * d_Convolv(1,0)(2b_a_cv_W)/d_X414
JCudaTensor X1699 = y118.backward_data(X1564, _2b_a_cv_W);
// V_2b_a_cv_W = ((X1564 * d_Convolv(1,0)(X414)/d_2b_a_cv_W * -0.01) + (V_2b_a_cv_W * 0.9))
y118.backward_filter(X1564, X414, V_2b_a_cv_W, lrn_rate, momentum);
// dealloc X1564
X1564.free();
// 2b_a_cv_W = (V_2b_a_cv_W + (2b_a_cv_W * (1 + (5.0E-4 * -0.01))))
_2b_a_cv_W.update(V_2b_a_cv_W, 1f, 1f + decay * lrn_rate);
// val X746 = (X1699 + X1654)
JCudaTensor X746 = X1699.plus_i(X1654);;
// dealloc X1654
X1654.free();
// val X1557 = X746 * d_ReLU()(X414)/d_X413
JCudaTensor X1557 = y108.backward(X746, X414);
// dealloc X414
X414.free();
// val X1501 = X1557.copy * d_ReLU()(X403)/d_X402
JCudaTensor X1501 = y108.backward(X1557.clone(), X403);
// dealloc X403
X403.free();
JCudaTensor[] y126 = y125.backward(X1501,X401,_2a1_bn_scale);
// val X1451 = X1501 * d_BatchNorm(2a1_bn)(X401,2a1_bn_scale)/d_2a1_bn_bias
JCudaTensor X1451 = y126[2];;
// V_2a1_bn_bias = ((X1451 * -0.01) + (V_2a1_bn_bias * 0.9))
V_2a1_bn_bias.update(X1451, lrn_rate, momentum);
// dealloc X1451
X1451.free();
// 2a1_bn_bias = (V_2a1_bn_bias + (2a1_bn_bias * (1 + (5.0E-4 * -0.01))))
_2a1_bn_bias.update(V_2a1_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1335 = X1501 * d_BatchNorm(2a1_bn)(X401,2a1_bn_scale)/d_2a1_bn_scale
JCudaTensor X1335 = y126[1];;
// val X1526 = X1501 * d_BatchNorm(2a1_bn)(X401,2a1_bn_scale)/d_X401
JCudaTensor X1526 = y126[0];;
// dealloc X401
X401.free();
// V_2a1_bn_scale = ((X1335 * -0.01) + (V_2a1_bn_scale * 0.9))
V_2a1_bn_scale.update(X1335, lrn_rate, momentum);
// dealloc X1335
X1335.free();
// 2a1_bn_scale = (V_2a1_bn_scale + (2a1_bn_scale * (1 + (5.0E-4 * -0.01))))
_2a1_bn_scale.update(V_2a1_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1527 = X1526 * d_Convolv(1,0)(2a1_cv_W)/d_X400
JCudaTensor X1527 = y111.backward_data(X1526, _2a1_cv_W);
// V_2a1_cv_W = ((X1526 * d_Convolv(1,0)(X400)/d_2a1_cv_W * -0.01) + (V_2a1_cv_W * 0.9))
y111.backward_filter(X1526, X400, V_2a1_cv_W, lrn_rate, momentum);
// dealloc X1526
X1526.free();
// 2a1_cv_W = (V_2a1_cv_W + (2a1_cv_W * (1 + (5.0E-4 * -0.01))))
_2a1_cv_W.update(V_2a1_cv_W, 1f, 1f + decay * lrn_rate);
// val X1675 = X1557.copy * d_ReLU()(X412)/d_X411
JCudaTensor X1675 = y108.backward(X1557.clone(), X412);
// dealloc X412
X412.free();
// dealloc X1557
X1557.free();
JCudaTensor[] y128 = y127.backward(X1675,X410,_2a2_c_bn_scale);
// val X1561 = X1675 * d_BatchNorm(2a2_c_bn)(X410,2a2_c_bn_scale)/d_2a2_c_bn_bias
JCudaTensor X1561 = y128[2];;
// V_2a2_c_bn_bias = ((X1561 * -0.01) + (V_2a2_c_bn_bias * 0.9))
V_2a2_c_bn_bias.update(X1561, lrn_rate, momentum);
// dealloc X1561
X1561.free();
// 2a2_c_bn_bias = (V_2a2_c_bn_bias + (2a2_c_bn_bias * (1 + (5.0E-4 * -0.01))))
_2a2_c_bn_bias.update(V_2a2_c_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1619 = X1675 * d_BatchNorm(2a2_c_bn)(X410,2a2_c_bn_scale)/d_X410
JCudaTensor X1619 = y128[0];;
// val X1621 = X1675 * d_BatchNorm(2a2_c_bn)(X410,2a2_c_bn_scale)/d_2a2_c_bn_scale
JCudaTensor X1621 = y128[1];;
// dealloc X410
X410.free();
// V_2a2_c_bn_scale = ((X1621 * -0.01) + (V_2a2_c_bn_scale * 0.9))
V_2a2_c_bn_scale.update(X1621, lrn_rate, momentum);
// dealloc X1621
X1621.free();
// 2a2_c_bn_scale = (V_2a2_c_bn_scale + (2a2_c_bn_scale * (1 + (5.0E-4 * -0.01))))
_2a2_c_bn_scale.update(V_2a2_c_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1334 = X1619 * d_Convolv(1,0)(2a2_c_cv_W)/d_X409
JCudaTensor X1334 = y111.backward_data(X1619, _2a2_c_cv_W);
// V_2a2_c_cv_W = ((X1619 * d_Convolv(1,0)(X409)/d_2a2_c_cv_W * -0.01) + (V_2a2_c_cv_W * 0.9))
y111.backward_filter(X1619, X409, V_2a2_c_cv_W, lrn_rate, momentum);
// dealloc X1619
X1619.free();
// 2a2_c_cv_W = (V_2a2_c_cv_W + (2a2_c_cv_W * (1 + (5.0E-4 * -0.01))))
_2a2_c_cv_W.update(V_2a2_c_cv_W, 1f, 1f + decay * lrn_rate);
// val X1633 = X1334 * d_ReLU()(X409)/d_X408
JCudaTensor X1633 = y112.backward(X1334, X409);
// dealloc X409
X409.free();
JCudaTensor[] y130 = y129.backward(X1633,X407,_2a2_b_bn_scale);
// val X1437 = X1633 * d_BatchNorm(2a2_b_bn)(X407,2a2_b_bn_scale)/d_2a2_b_bn_bias
JCudaTensor X1437 = y130[2];;
// V_2a2_b_bn_bias = ((X1437 * -0.01) + (V_2a2_b_bn_bias * 0.9))
V_2a2_b_bn_bias.update(X1437, lrn_rate, momentum);
// dealloc X1437
X1437.free();
// 2a2_b_bn_bias = (V_2a2_b_bn_bias + (2a2_b_bn_bias * (1 + (5.0E-4 * -0.01))))
_2a2_b_bn_bias.update(V_2a2_b_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1541 = X1633 * d_BatchNorm(2a2_b_bn)(X407,2a2_b_bn_scale)/d_X407
JCudaTensor X1541 = y130[0];;
// val X1611 = X1633 * d_BatchNorm(2a2_b_bn)(X407,2a2_b_bn_scale)/d_2a2_b_bn_scale
JCudaTensor X1611 = y130[1];;
// dealloc X407
X407.free();
// V_2a2_b_bn_scale = ((X1611 * -0.01) + (V_2a2_b_bn_scale * 0.9))
V_2a2_b_bn_scale.update(X1611, lrn_rate, momentum);
// dealloc X1611
X1611.free();
// 2a2_b_bn_scale = (V_2a2_b_bn_scale + (2a2_b_bn_scale * (1 + (5.0E-4 * -0.01))))
_2a2_b_bn_scale.update(V_2a2_b_bn_scale, 1f, 1f + decay * lrn_rate);
// val X1673 = X1541 * d_Convolv(1,1)(2a2_b_cv_W)/d_X406
JCudaTensor X1673 = y115.backward_data(X1541, _2a2_b_cv_W);
// V_2a2_b_cv_W = ((X1541 * d_Convolv(1,1)(X406)/d_2a2_b_cv_W * -0.01) + (V_2a2_b_cv_W * 0.9))
y115.backward_filter(X1541, X406, V_2a2_b_cv_W, lrn_rate, momentum);
// dealloc X1541
X1541.free();
// 2a2_b_cv_W = (V_2a2_b_cv_W + (2a2_b_cv_W * (1 + (5.0E-4 * -0.01))))
_2a2_b_cv_W.update(V_2a2_b_cv_W, 1f, 1f + decay * lrn_rate);
// val X1453 = X1673 * d_ReLU()(X406)/d_X405
JCudaTensor X1453 = y112.backward(X1673, X406);
// dealloc X406
X406.free();
JCudaTensor[] y132 = y131.backward(X1453,X404,_2a2_a_bn_scale);
// val X1326 = X1453 * d_BatchNorm(2a2_a_bn)(X404,2a2_a_bn_scale)/d_2a2_a_bn_bias
JCudaTensor X1326 = y132[2];;
// V_2a2_a_bn_bias = ((X1326 * -0.01) + (V_2a2_a_bn_bias * 0.9))
V_2a2_a_bn_bias.update(X1326, lrn_rate, momentum);
// dealloc X1326
X1326.free();
// 2a2_a_bn_bias = (V_2a2_a_bn_bias + (2a2_a_bn_bias * (1 + (5.0E-4 * -0.01))))
_2a2_a_bn_bias.update(V_2a2_a_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1493 = X1453 * d_BatchNorm(2a2_a_bn)(X404,2a2_a_bn_scale)/d_2a2_a_bn_scale
JCudaTensor X1493 = y132[1];;
// val X1607 = X1453 * d_BatchNorm(2a2_a_bn)(X404,2a2_a_bn_scale)/d_X404
JCudaTensor X1607 = y132[0];;
// dealloc X404
X404.free();
// V_2a2_a_bn_scale = ((X1493 * -0.01) + (V_2a2_a_bn_scale * 0.9))
V_2a2_a_bn_scale.update(X1493, lrn_rate, momentum);
// dealloc X1493
X1493.free();
// 2a2_a_bn_scale = (V_2a2_a_bn_scale + (2a2_a_bn_scale * (1 + (5.0E-4 * -0.01))))
_2a2_a_bn_scale.update(V_2a2_a_bn_scale, 1f, 1f + decay * lrn_rate);
// val X751 = (X1527 + X1607 * d_Convolv(1,0)(2a2_a_cv_W)/d_X400)
JCudaTensor X751 = y139.backward_data(X1607,_2a2_a_cv_W, X1527);
// V_2a2_a_cv_W = ((X1607 * d_Convolv(1,0)(X400)/d_2a2_a_cv_W * -0.01) + (V_2a2_a_cv_W * 0.9))
y139.backward_filter(X1607, X400, V_2a2_a_cv_W, lrn_rate, momentum);
// dealloc X1607
X1607.free();
// 2a2_a_cv_W = (V_2a2_a_cv_W + (2a2_a_cv_W * (1 + (5.0E-4 * -0.01))))
_2a2_a_cv_W.update(V_2a2_a_cv_W, 1f, 1f + decay * lrn_rate);
// val X1341 = X751 * d_Pooling(3,2,0,true)(X400,X399)/d_X399
JCudaTensor X1341 = y134.backward(X751, X400, X399);
// dealloc X400
X400.free();
// dealloc X751
X751.free();
// val X1613 = X1341 * d_ReLU()(X399)/d_X398
JCudaTensor X1613 = y135.backward(X1341, X399);
// dealloc X399
X399.free();
JCudaTensor[] y137 = y136.backward(X1613,X397,_1_bn_scale);
// val X1530 = X1613 * d_BatchNorm(1_bn)(X397,1_bn_scale)/d_1_bn_bias
JCudaTensor X1530 = y137[2];;
// V_1_bn_bias = ((X1530 * -0.01) + (V_1_bn_bias * 0.9))
V_1_bn_bias.update(X1530, lrn_rate, momentum);
// dealloc X1530
X1530.free();
// 1_bn_bias = (V_1_bn_bias + (1_bn_bias * (1 + (5.0E-4 * -0.01))))
_1_bn_bias.update(V_1_bn_bias, 1f, 1f + decay * lrn_rate);
// val X1427 = X1613 * d_BatchNorm(1_bn)(X397,1_bn_scale)/d_1_bn_scale
JCudaTensor X1427 = y137[1];;
// val X1644 = X1613 * d_BatchNorm(1_bn)(X397,1_bn_scale)/d_X397
JCudaTensor X1644 = y137[0];;
// dealloc X397
X397.free();
// V_1_bn_scale = ((X1427 * -0.01) + (V_1_bn_scale * 0.9))
V_1_bn_scale.update(X1427, lrn_rate, momentum);
// dealloc X1427
X1427.free();
// 1_bn_scale = (V_1_bn_scale + (1_bn_scale * (1 + (5.0E-4 * -0.01))))
_1_bn_scale.update(V_1_bn_scale, 1f, 1f + decay * lrn_rate);
// V_1_cv_W = ((X1644 * d_Convolv(2,3)(X1324)/d_1_cv_W * -0.01) + (V_1_cv_W * 0.9))
y138.backward_filter(X1644, X1324, V_1_cv_W, lrn_rate, momentum);
// dealloc X1644
X1644.free();
// dealloc X1324
X1324.free();
// 1_cv_W = (V_1_cv_W + (1_cv_W * (1 + (5.0E-4 * -0.01))))
_1_cv_W.update(V_1_cv_W, 1f, 1f + decay * lrn_rate);

return _loss; 
}

public JCudaTensor testFunction(JTensorFloat X) {
 // val X199 = Cuda(X)
JCudaTensor X199 = X.asJCudaTensor();
// val X200 = Convolv(2,3)(X199,1_cv_W,1_cv_B)
JCudaTensor X200 = y138.forward(X199, _1_cv_W, _1_cv_B);
// dealloc X199
X199.free();
// val X201 = BatchNorm(1_bn)(X200,1_bn_scale,1_bn_bias)
JCudaTensor X201 = y136.forward_inference(X200, _1_bn_scale, _1_bn_bias);
// dealloc X200
X200.free();
// val X202 = ReLU()(X201)
JCudaTensor X202 = y135.forward(X201);
// val X203 = Pooling(3,2,0,true)(X202)
JCudaTensor X203 = y134.forward(X202);
// dealloc X202
X202.free();
// val X207 = Convolv(1,0)(X203,2a2_a_cv_W,2a2_a_cv_B)
JCudaTensor X207 = y139.forward(X203, _2a2_a_cv_W, _2a2_a_cv_B);
// val X204 = Convolv(1,0)(X203,2a1_cv_W,2a1_cv_B)
JCudaTensor X204 = y111.forward(X203, _2a1_cv_W, _2a1_cv_B);
// dealloc X203
X203.free();
// val X208 = BatchNorm(2a2_a_bn)(X207,2a2_a_bn_scale,2a2_a_bn_bias)
JCudaTensor X208 = y131.forward_inference(X207, _2a2_a_bn_scale, _2a2_a_bn_bias);
// dealloc X207
X207.free();
// val X205 = BatchNorm(2a1_bn)(X204,2a1_bn_scale,2a1_bn_bias)
JCudaTensor X205 = y125.forward_inference(X204, _2a1_bn_scale, _2a1_bn_bias);
// dealloc X204
X204.free();
// val X209 = ReLU()(X208)
JCudaTensor X209 = y112.forward(X208);
// val X206 = ReLU()(X205)
JCudaTensor X206 = y108.forward(X205);
// val X210 = Convolv(1,1)(X209,2a2_b_cv_W,2a2_b_cv_B)
JCudaTensor X210 = y115.forward(X209, _2a2_b_cv_W, _2a2_b_cv_B);
// dealloc X209
X209.free();
// val X211 = BatchNorm(2a2_b_bn)(X210,2a2_b_bn_scale,2a2_b_bn_bias)
JCudaTensor X211 = y129.forward_inference(X210, _2a2_b_bn_scale, _2a2_b_bn_bias);
// dealloc X210
X210.free();
// val X212 = ReLU()(X211)
JCudaTensor X212 = y112.forward(X211);
// val X213 = Convolv(1,0)(X212,2a2_c_cv_W,2a2_c_cv_B)
JCudaTensor X213 = y111.forward(X212, _2a2_c_cv_W, _2a2_c_cv_B);
// dealloc X212
X212.free();
// val X214 = BatchNorm(2a2_c_bn)(X213,2a2_c_bn_scale,2a2_c_bn_bias)
JCudaTensor X214 = y127.forward_inference(X213, _2a2_c_bn_scale, _2a2_c_bn_bias);
// dealloc X213
X213.free();
// val X215 = ReLU()(X214)
JCudaTensor X215 = y108.forward(X214);
// val X216 = (X206 + X215)
JCudaTensor X216 = X206.plus_i(X215);;
// dealloc X215
X215.free();
// val X217 = ReLU()(X216)
JCudaTensor X217 = y108.forward(X216);
// val X218 = Convolv(1,0)(X217,2b_a_cv_W,2b_a_cv_B)
JCudaTensor X218 = y118.forward(X217, _2b_a_cv_W, _2b_a_cv_B);
// val X219 = BatchNorm(2b_a_bn)(X218,2b_a_bn_scale,2b_a_bn_bias)
JCudaTensor X219 = y123.forward_inference(X218, _2b_a_bn_scale, _2b_a_bn_bias);
// dealloc X218
X218.free();
// val X220 = ReLU()(X219)
JCudaTensor X220 = y112.forward(X219);
// val X221 = Convolv(1,1)(X220,2b_b_cv_W,2b_b_cv_B)
JCudaTensor X221 = y115.forward(X220, _2b_b_cv_W, _2b_b_cv_B);
// dealloc X220
X220.free();
// val X222 = BatchNorm(2b_b_bn)(X221,2b_b_bn_scale,2b_b_bn_bias)
JCudaTensor X222 = y121.forward_inference(X221, _2b_b_bn_scale, _2b_b_bn_bias);
// dealloc X221
X221.free();
// val X223 = ReLU()(X222)
JCudaTensor X223 = y112.forward(X222);
// val X224 = Convolv(1,0)(X223,2b_c_cv_W,2b_c_cv_B)
JCudaTensor X224 = y111.forward(X223, _2b_c_cv_W, _2b_c_cv_B);
// dealloc X223
X223.free();
// val X225 = BatchNorm(2b_c_bn)(X224,2b_c_bn_scale,2b_c_bn_bias)
JCudaTensor X225 = y119.forward_inference(X224, _2b_c_bn_scale, _2b_c_bn_bias);
// dealloc X224
X224.free();
// val X226 = ReLU()(X225)
JCudaTensor X226 = y108.forward(X225);
// val X227 = (X226 + X217)
JCudaTensor X227 = X226.plus_i(X217);;
// dealloc X217
X217.free();
// val X228 = ReLU()(X227)
JCudaTensor X228 = y108.forward(X227);
// val X229 = Convolv(1,0)(X228,2c_a_cv_W,2c_a_cv_B)
JCudaTensor X229 = y118.forward(X228, _2c_a_cv_W, _2c_a_cv_B);
// val X230 = BatchNorm(2c_a_bn)(X229,2c_a_bn_scale,2c_a_bn_bias)
JCudaTensor X230 = y116.forward_inference(X229, _2c_a_bn_scale, _2c_a_bn_bias);
// dealloc X229
X229.free();
// val X231 = ReLU()(X230)
JCudaTensor X231 = y112.forward(X230);
// val X232 = Convolv(1,1)(X231,2c_b_cv_W,2c_b_cv_B)
JCudaTensor X232 = y115.forward(X231, _2c_b_cv_W, _2c_b_cv_B);
// dealloc X231
X231.free();
// val X233 = BatchNorm(2c_b_bn)(X232,2c_b_bn_scale,2c_b_bn_bias)
JCudaTensor X233 = y113.forward_inference(X232, _2c_b_bn_scale, _2c_b_bn_bias);
// dealloc X232
X232.free();
// val X234 = ReLU()(X233)
JCudaTensor X234 = y112.forward(X233);
// val X235 = Convolv(1,0)(X234,2c_c_cv_W,2c_c_cv_B)
JCudaTensor X235 = y111.forward(X234, _2c_c_cv_W, _2c_c_cv_B);
// dealloc X234
X234.free();
// val X236 = BatchNorm(2c_c_bn)(X235,2c_c_bn_scale,2c_c_bn_bias)
JCudaTensor X236 = y109.forward_inference(X235, _2c_c_bn_scale, _2c_c_bn_bias);
// dealloc X235
X235.free();
// val X237 = ReLU()(X236)
JCudaTensor X237 = y108.forward(X236);
// val X238 = (X237 + X228)
JCudaTensor X238 = X237.plus_i(X228);;
// dealloc X228
X228.free();
// val X239 = ReLU()(X238)
JCudaTensor X239 = y108.forward(X238);
// val X243 = Convolv(2,0)(X239,3a2_a_cv_W,3a2_a_cv_B)
JCudaTensor X243 = y140.forward(X239, _3a2_a_cv_W, _3a2_a_cv_B);
// val X240 = Convolv(2,0)(X239,3a1_cv_W,3a1_cv_B)
JCudaTensor X240 = y100.forward(X239, _3a1_cv_W, _3a1_cv_B);
// dealloc X239
X239.free();
// val X241 = BatchNorm(3a1_bn)(X240,3a1_bn_scale,3a1_bn_bias)
JCudaTensor X241 = y98.forward_inference(X240, _3a1_bn_scale, _3a1_bn_bias);
// dealloc X240
X240.free();
// val X244 = BatchNorm(3a2_a_bn)(X243,3a2_a_bn_scale,3a2_a_bn_bias)
JCudaTensor X244 = y105.forward_inference(X243, _3a2_a_bn_scale, _3a2_a_bn_bias);
// dealloc X243
X243.free();
// val X245 = ReLU()(X244)
JCudaTensor X245 = y79.forward(X244);
// val X242 = ReLU()(X241)
JCudaTensor X242 = y75.forward(X241);
// val X246 = Convolv(1,1)(X245,3a2_b_cv_W,3a2_b_cv_B)
JCudaTensor X246 = y82.forward(X245, _3a2_b_cv_W, _3a2_b_cv_B);
// dealloc X245
X245.free();
// val X247 = BatchNorm(3a2_b_bn)(X246,3a2_b_bn_scale,3a2_b_bn_bias)
JCudaTensor X247 = y103.forward_inference(X246, _3a2_b_bn_scale, _3a2_b_bn_bias);
// dealloc X246
X246.free();
// val X248 = ReLU()(X247)
JCudaTensor X248 = y79.forward(X247);
// val X249 = Convolv(1,0)(X248,3a2_c_cv_W,3a2_c_cv_B)
JCudaTensor X249 = y78.forward(X248, _3a2_c_cv_W, _3a2_c_cv_B);
// dealloc X248
X248.free();
// val X250 = BatchNorm(3a2_c_bn)(X249,3a2_c_bn_scale,3a2_c_bn_bias)
JCudaTensor X250 = y101.forward_inference(X249, _3a2_c_bn_scale, _3a2_c_bn_bias);
// dealloc X249
X249.free();
// val X251 = ReLU()(X250)
JCudaTensor X251 = y75.forward(X250);
// val X252 = (X242 + X251)
JCudaTensor X252 = X242.plus_i(X251);;
// dealloc X251
X251.free();
// val X253 = ReLU()(X252)
JCudaTensor X253 = y75.forward(X252);
// val X254 = Convolv(1,0)(X253,3b_a_cv_W,3b_a_cv_B)
JCudaTensor X254 = y85.forward(X253, _3b_a_cv_W, _3b_a_cv_B);
// val X255 = BatchNorm(3b_a_bn)(X254,3b_a_bn_scale,3b_a_bn_bias)
JCudaTensor X255 = y96.forward_inference(X254, _3b_a_bn_scale, _3b_a_bn_bias);
// dealloc X254
X254.free();
// val X256 = ReLU()(X255)
JCudaTensor X256 = y79.forward(X255);
// val X257 = Convolv(1,1)(X256,3b_b_cv_W,3b_b_cv_B)
JCudaTensor X257 = y82.forward(X256, _3b_b_cv_W, _3b_b_cv_B);
// dealloc X256
X256.free();
// val X258 = BatchNorm(3b_b_bn)(X257,3b_b_bn_scale,3b_b_bn_bias)
JCudaTensor X258 = y94.forward_inference(X257, _3b_b_bn_scale, _3b_b_bn_bias);
// dealloc X257
X257.free();
// val X259 = ReLU()(X258)
JCudaTensor X259 = y79.forward(X258);
// val X260 = Convolv(1,0)(X259,3b_c_cv_W,3b_c_cv_B)
JCudaTensor X260 = y78.forward(X259, _3b_c_cv_W, _3b_c_cv_B);
// dealloc X259
X259.free();
// val X261 = BatchNorm(3b_c_bn)(X260,3b_c_bn_scale,3b_c_bn_bias)
JCudaTensor X261 = y92.forward_inference(X260, _3b_c_bn_scale, _3b_c_bn_bias);
// dealloc X260
X260.free();
// val X262 = ReLU()(X261)
JCudaTensor X262 = y75.forward(X261);
// val X263 = (X262 + X253)
JCudaTensor X263 = X262.plus_i(X253);;
// dealloc X253
X253.free();
// val X264 = ReLU()(X263)
JCudaTensor X264 = y75.forward(X263);
// val X265 = Convolv(1,0)(X264,3c_a_cv_W,3c_a_cv_B)
JCudaTensor X265 = y85.forward(X264, _3c_a_cv_W, _3c_a_cv_B);
// val X266 = BatchNorm(3c_a_bn)(X265,3c_a_bn_scale,3c_a_bn_bias)
JCudaTensor X266 = y90.forward_inference(X265, _3c_a_bn_scale, _3c_a_bn_bias);
// dealloc X265
X265.free();
// val X267 = ReLU()(X266)
JCudaTensor X267 = y79.forward(X266);
// val X268 = Convolv(1,1)(X267,3c_b_cv_W,3c_b_cv_B)
JCudaTensor X268 = y82.forward(X267, _3c_b_cv_W, _3c_b_cv_B);
// dealloc X267
X267.free();
// val X269 = BatchNorm(3c_b_bn)(X268,3c_b_bn_scale,3c_b_bn_bias)
JCudaTensor X269 = y88.forward_inference(X268, _3c_b_bn_scale, _3c_b_bn_bias);
// dealloc X268
X268.free();
// val X270 = ReLU()(X269)
JCudaTensor X270 = y79.forward(X269);
// val X271 = Convolv(1,0)(X270,3c_c_cv_W,3c_c_cv_B)
JCudaTensor X271 = y78.forward(X270, _3c_c_cv_W, _3c_c_cv_B);
// dealloc X270
X270.free();
// val X272 = BatchNorm(3c_c_bn)(X271,3c_c_bn_scale,3c_c_bn_bias)
JCudaTensor X272 = y86.forward_inference(X271, _3c_c_bn_scale, _3c_c_bn_bias);
// dealloc X271
X271.free();
// val X273 = ReLU()(X272)
JCudaTensor X273 = y75.forward(X272);
// val X274 = (X273 + X264)
JCudaTensor X274 = X273.plus_i(X264);;
// dealloc X264
X264.free();
// val X275 = ReLU()(X274)
JCudaTensor X275 = y75.forward(X274);
// val X276 = Convolv(1,0)(X275,3d_a_cv_W,3d_a_cv_B)
JCudaTensor X276 = y85.forward(X275, _3d_a_cv_W, _3d_a_cv_B);
// val X277 = BatchNorm(3d_a_bn)(X276,3d_a_bn_scale,3d_a_bn_bias)
JCudaTensor X277 = y83.forward_inference(X276, _3d_a_bn_scale, _3d_a_bn_bias);
// dealloc X276
X276.free();
// val X278 = ReLU()(X277)
JCudaTensor X278 = y79.forward(X277);
// val X279 = Convolv(1,1)(X278,3d_b_cv_W,3d_b_cv_B)
JCudaTensor X279 = y82.forward(X278, _3d_b_cv_W, _3d_b_cv_B);
// dealloc X278
X278.free();
// val X280 = BatchNorm(3d_b_bn)(X279,3d_b_bn_scale,3d_b_bn_bias)
JCudaTensor X280 = y80.forward_inference(X279, _3d_b_bn_scale, _3d_b_bn_bias);
// dealloc X279
X279.free();
// val X281 = ReLU()(X280)
JCudaTensor X281 = y79.forward(X280);
// val X282 = Convolv(1,0)(X281,3d_c_cv_W,3d_c_cv_B)
JCudaTensor X282 = y78.forward(X281, _3d_c_cv_W, _3d_c_cv_B);
// dealloc X281
X281.free();
// val X283 = BatchNorm(3d_c_bn)(X282,3d_c_bn_scale,3d_c_bn_bias)
JCudaTensor X283 = y76.forward_inference(X282, _3d_c_bn_scale, _3d_c_bn_bias);
// dealloc X282
X282.free();
// val X284 = ReLU()(X283)
JCudaTensor X284 = y75.forward(X283);
// val X285 = (X284 + X275)
JCudaTensor X285 = X284.plus_i(X275);;
// dealloc X275
X275.free();
// val X286 = ReLU()(X285)
JCudaTensor X286 = y75.forward(X285);
// val X287 = Convolv(2,0)(X286,4a1_cv_W,4a1_cv_B)
JCudaTensor X287 = y67.forward(X286, _4a1_cv_W, _4a1_cv_B);
// val X290 = Convolv(2,0)(X286,4a2_a_cv_W,4a2_a_cv_B)
JCudaTensor X290 = y141.forward(X286, _4a2_a_cv_W, _4a2_a_cv_B);
// dealloc X286
X286.free();
// val X291 = BatchNorm(4a2_a_bn)(X290,4a2_a_bn_scale,4a2_a_bn_bias)
JCudaTensor X291 = y72.forward_inference(X290, _4a2_a_bn_scale, _4a2_a_bn_bias);
// dealloc X290
X290.free();
// val X288 = BatchNorm(4a1_bn)(X287,4a1_bn_scale,4a1_bn_bias)
JCudaTensor X288 = y65.forward_inference(X287, _4a1_bn_scale, _4a1_bn_bias);
// dealloc X287
X287.free();
// val X292 = ReLU()(X291)
JCudaTensor X292 = y34.forward(X291);
// val X289 = ReLU()(X288)
JCudaTensor X289 = y30.forward(X288);
// val X293 = Convolv(1,1)(X292,4a2_b_cv_W,4a2_b_cv_B)
JCudaTensor X293 = y37.forward(X292, _4a2_b_cv_W, _4a2_b_cv_B);
// dealloc X292
X292.free();
// val X294 = BatchNorm(4a2_b_bn)(X293,4a2_b_bn_scale,4a2_b_bn_bias)
JCudaTensor X294 = y70.forward_inference(X293, _4a2_b_bn_scale, _4a2_b_bn_bias);
// dealloc X293
X293.free();
// val X295 = ReLU()(X294)
JCudaTensor X295 = y34.forward(X294);
// val X296 = Convolv(1,0)(X295,4a2_c_cv_W,4a2_c_cv_B)
JCudaTensor X296 = y33.forward(X295, _4a2_c_cv_W, _4a2_c_cv_B);
// dealloc X295
X295.free();
// val X297 = BatchNorm(4a2_c_bn)(X296,4a2_c_bn_scale,4a2_c_bn_bias)
JCudaTensor X297 = y68.forward_inference(X296, _4a2_c_bn_scale, _4a2_c_bn_bias);
// dealloc X296
X296.free();
// val X298 = ReLU()(X297)
JCudaTensor X298 = y30.forward(X297);
// val X299 = (X289 + X298)
JCudaTensor X299 = X289.plus_i(X298);;
// dealloc X298
X298.free();
// val X300 = ReLU()(X299)
JCudaTensor X300 = y30.forward(X299);
// val X301 = Convolv(1,0)(X300,4b_a_cv_W,4b_a_cv_B)
JCudaTensor X301 = y40.forward(X300, _4b_a_cv_W, _4b_a_cv_B);
// val X302 = BatchNorm(4b_a_bn)(X301,4b_a_bn_scale,4b_a_bn_bias)
JCudaTensor X302 = y63.forward_inference(X301, _4b_a_bn_scale, _4b_a_bn_bias);
// dealloc X301
X301.free();
// val X303 = ReLU()(X302)
JCudaTensor X303 = y34.forward(X302);
// val X304 = Convolv(1,1)(X303,4b_b_cv_W,4b_b_cv_B)
JCudaTensor X304 = y37.forward(X303, _4b_b_cv_W, _4b_b_cv_B);
// dealloc X303
X303.free();
// val X305 = BatchNorm(4b_b_bn)(X304,4b_b_bn_scale,4b_b_bn_bias)
JCudaTensor X305 = y61.forward_inference(X304, _4b_b_bn_scale, _4b_b_bn_bias);
// dealloc X304
X304.free();
// val X306 = ReLU()(X305)
JCudaTensor X306 = y34.forward(X305);
// val X307 = Convolv(1,0)(X306,4b_c_cv_W,4b_c_cv_B)
JCudaTensor X307 = y33.forward(X306, _4b_c_cv_W, _4b_c_cv_B);
// dealloc X306
X306.free();
// val X308 = BatchNorm(4b_c_bn)(X307,4b_c_bn_scale,4b_c_bn_bias)
JCudaTensor X308 = y59.forward_inference(X307, _4b_c_bn_scale, _4b_c_bn_bias);
// dealloc X307
X307.free();
// val X309 = ReLU()(X308)
JCudaTensor X309 = y30.forward(X308);
// val X310 = (X309 + X300)
JCudaTensor X310 = X309.plus_i(X300);;
// dealloc X300
X300.free();
// val X311 = ReLU()(X310)
JCudaTensor X311 = y30.forward(X310);
// val X312 = Convolv(1,0)(X311,4c_a_cv_W,4c_a_cv_B)
JCudaTensor X312 = y40.forward(X311, _4c_a_cv_W, _4c_a_cv_B);
// val X313 = BatchNorm(4c_a_bn)(X312,4c_a_bn_scale,4c_a_bn_bias)
JCudaTensor X313 = y57.forward_inference(X312, _4c_a_bn_scale, _4c_a_bn_bias);
// dealloc X312
X312.free();
// val X314 = ReLU()(X313)
JCudaTensor X314 = y34.forward(X313);
// val X315 = Convolv(1,1)(X314,4c_b_cv_W,4c_b_cv_B)
JCudaTensor X315 = y37.forward(X314, _4c_b_cv_W, _4c_b_cv_B);
// dealloc X314
X314.free();
// val X316 = BatchNorm(4c_b_bn)(X315,4c_b_bn_scale,4c_b_bn_bias)
JCudaTensor X316 = y55.forward_inference(X315, _4c_b_bn_scale, _4c_b_bn_bias);
// dealloc X315
X315.free();
// val X317 = ReLU()(X316)
JCudaTensor X317 = y34.forward(X316);
// val X318 = Convolv(1,0)(X317,4c_c_cv_W,4c_c_cv_B)
JCudaTensor X318 = y33.forward(X317, _4c_c_cv_W, _4c_c_cv_B);
// dealloc X317
X317.free();
// val X319 = BatchNorm(4c_c_bn)(X318,4c_c_bn_scale,4c_c_bn_bias)
JCudaTensor X319 = y53.forward_inference(X318, _4c_c_bn_scale, _4c_c_bn_bias);
// dealloc X318
X318.free();
// val X320 = ReLU()(X319)
JCudaTensor X320 = y30.forward(X319);
// val X321 = (X320 + X311)
JCudaTensor X321 = X320.plus_i(X311);;
// dealloc X311
X311.free();
// val X322 = ReLU()(X321)
JCudaTensor X322 = y30.forward(X321);
// val X323 = Convolv(1,0)(X322,4d_a_cv_W,4d_a_cv_B)
JCudaTensor X323 = y40.forward(X322, _4d_a_cv_W, _4d_a_cv_B);
// val X324 = BatchNorm(4d_a_bn)(X323,4d_a_bn_scale,4d_a_bn_bias)
JCudaTensor X324 = y51.forward_inference(X323, _4d_a_bn_scale, _4d_a_bn_bias);
// dealloc X323
X323.free();
// val X325 = ReLU()(X324)
JCudaTensor X325 = y34.forward(X324);
// val X326 = Convolv(1,1)(X325,4d_b_cv_W,4d_b_cv_B)
JCudaTensor X326 = y37.forward(X325, _4d_b_cv_W, _4d_b_cv_B);
// dealloc X325
X325.free();
// val X327 = BatchNorm(4d_b_bn)(X326,4d_b_bn_scale,4d_b_bn_bias)
JCudaTensor X327 = y49.forward_inference(X326, _4d_b_bn_scale, _4d_b_bn_bias);
// dealloc X326
X326.free();
// val X328 = ReLU()(X327)
JCudaTensor X328 = y34.forward(X327);
// val X329 = Convolv(1,0)(X328,4d_c_cv_W,4d_c_cv_B)
JCudaTensor X329 = y33.forward(X328, _4d_c_cv_W, _4d_c_cv_B);
// dealloc X328
X328.free();
// val X330 = BatchNorm(4d_c_bn)(X329,4d_c_bn_scale,4d_c_bn_bias)
JCudaTensor X330 = y47.forward_inference(X329, _4d_c_bn_scale, _4d_c_bn_bias);
// dealloc X329
X329.free();
// val X331 = ReLU()(X330)
JCudaTensor X331 = y30.forward(X330);
// val X332 = (X331 + X322)
JCudaTensor X332 = X331.plus_i(X322);;
// dealloc X322
X322.free();
// val X333 = ReLU()(X332)
JCudaTensor X333 = y30.forward(X332);
// val X334 = Convolv(1,0)(X333,4e_a_cv_W,4e_a_cv_B)
JCudaTensor X334 = y40.forward(X333, _4e_a_cv_W, _4e_a_cv_B);
// val X335 = BatchNorm(4e_a_bn)(X334,4e_a_bn_scale,4e_a_bn_bias)
JCudaTensor X335 = y45.forward_inference(X334, _4e_a_bn_scale, _4e_a_bn_bias);
// dealloc X334
X334.free();
// val X336 = ReLU()(X335)
JCudaTensor X336 = y34.forward(X335);
// val X337 = Convolv(1,1)(X336,4e_b_cv_W,4e_b_cv_B)
JCudaTensor X337 = y37.forward(X336, _4e_b_cv_W, _4e_b_cv_B);
// dealloc X336
X336.free();
// val X338 = BatchNorm(4e_b_bn)(X337,4e_b_bn_scale,4e_b_bn_bias)
JCudaTensor X338 = y43.forward_inference(X337, _4e_b_bn_scale, _4e_b_bn_bias);
// dealloc X337
X337.free();
// val X339 = ReLU()(X338)
JCudaTensor X339 = y34.forward(X338);
// val X340 = Convolv(1,0)(X339,4e_c_cv_W,4e_c_cv_B)
JCudaTensor X340 = y33.forward(X339, _4e_c_cv_W, _4e_c_cv_B);
// dealloc X339
X339.free();
// val X341 = BatchNorm(4e_c_bn)(X340,4e_c_bn_scale,4e_c_bn_bias)
JCudaTensor X341 = y41.forward_inference(X340, _4e_c_bn_scale, _4e_c_bn_bias);
// dealloc X340
X340.free();
// val X342 = ReLU()(X341)
JCudaTensor X342 = y30.forward(X341);
// val X343 = (X342 + X333)
JCudaTensor X343 = X342.plus_i(X333);;
// dealloc X333
X333.free();
// val X344 = ReLU()(X343)
JCudaTensor X344 = y30.forward(X343);
// val X345 = Convolv(1,0)(X344,4f_a_cv_W,4f_a_cv_B)
JCudaTensor X345 = y40.forward(X344, _4f_a_cv_W, _4f_a_cv_B);
// val X346 = BatchNorm(4f_a_bn)(X345,4f_a_bn_scale,4f_a_bn_bias)
JCudaTensor X346 = y38.forward_inference(X345, _4f_a_bn_scale, _4f_a_bn_bias);
// dealloc X345
X345.free();
// val X347 = ReLU()(X346)
JCudaTensor X347 = y34.forward(X346);
// val X348 = Convolv(1,1)(X347,4f_b_cv_W,4f_b_cv_B)
JCudaTensor X348 = y37.forward(X347, _4f_b_cv_W, _4f_b_cv_B);
// dealloc X347
X347.free();
// val X349 = BatchNorm(4f_b_bn)(X348,4f_b_bn_scale,4f_b_bn_bias)
JCudaTensor X349 = y35.forward_inference(X348, _4f_b_bn_scale, _4f_b_bn_bias);
// dealloc X348
X348.free();
// val X350 = ReLU()(X349)
JCudaTensor X350 = y34.forward(X349);
// val X351 = Convolv(1,0)(X350,4f_c_cv_W,4f_c_cv_B)
JCudaTensor X351 = y33.forward(X350, _4f_c_cv_W, _4f_c_cv_B);
// dealloc X350
X350.free();
// val X352 = BatchNorm(4f_c_bn)(X351,4f_c_bn_scale,4f_c_bn_bias)
JCudaTensor X352 = y31.forward_inference(X351, _4f_c_bn_scale, _4f_c_bn_bias);
// dealloc X351
X351.free();
// val X353 = ReLU()(X352)
JCudaTensor X353 = y30.forward(X352);
// val X354 = (X353 + X344)
JCudaTensor X354 = X353.plus_i(X344);;
// dealloc X344
X344.free();
// val X355 = ReLU()(X354)
JCudaTensor X355 = y30.forward(X354);
// val X356 = Convolv(2,0)(X355,5a1_cv_W,5a1_cv_B)
JCudaTensor X356 = y22.forward(X355, _5a1_cv_W, _5a1_cv_B);
// val X359 = Convolv(2,0)(X355,5a2_a_cv_W,5a2_a_cv_B)
JCudaTensor X359 = y142.forward(X355, _5a2_a_cv_W, _5a2_a_cv_B);
// dealloc X355
X355.free();
// val X357 = BatchNorm(5a1_bn)(X356,5a1_bn_scale,5a1_bn_bias)
JCudaTensor X357 = y20.forward_inference(X356, _5a1_bn_scale, _5a1_bn_bias);
// dealloc X356
X356.free();
// val X360 = BatchNorm(5a2_a_bn)(X359,5a2_a_bn_scale,5a2_a_bn_bias)
JCudaTensor X360 = y27.forward_inference(X359, _5a2_a_bn_scale, _5a2_a_bn_bias);
// dealloc X359
X359.free();
// val X361 = ReLU()(X360)
JCudaTensor X361 = y7.forward(X360);
// val X358 = ReLU()(X357)
JCudaTensor X358 = y3.forward(X357);
// val X362 = Convolv(1,1)(X361,5a2_b_cv_W,5a2_b_cv_B)
JCudaTensor X362 = y10.forward(X361, _5a2_b_cv_W, _5a2_b_cv_B);
// dealloc X361
X361.free();
// val X363 = BatchNorm(5a2_b_bn)(X362,5a2_b_bn_scale,5a2_b_bn_bias)
JCudaTensor X363 = y25.forward_inference(X362, _5a2_b_bn_scale, _5a2_b_bn_bias);
// dealloc X362
X362.free();
// val X364 = ReLU()(X363)
JCudaTensor X364 = y7.forward(X363);
// val X365 = Convolv(1,0)(X364,5a2_c_cv_W,5a2_c_cv_B)
JCudaTensor X365 = y6.forward(X364, _5a2_c_cv_W, _5a2_c_cv_B);
// dealloc X364
X364.free();
// val X366 = BatchNorm(5a2_c_bn)(X365,5a2_c_bn_scale,5a2_c_bn_bias)
JCudaTensor X366 = y23.forward_inference(X365, _5a2_c_bn_scale, _5a2_c_bn_bias);
// dealloc X365
X365.free();
// val X367 = ReLU()(X366)
JCudaTensor X367 = y3.forward(X366);
// val X368 = (X358 + X367)
JCudaTensor X368 = X358.plus_i(X367);;
// dealloc X367
X367.free();
// val X369 = ReLU()(X368)
JCudaTensor X369 = y3.forward(X368);
// val X370 = Convolv(1,0)(X369,5b_a_cv_W,5b_a_cv_B)
JCudaTensor X370 = y13.forward(X369, _5b_a_cv_W, _5b_a_cv_B);
// val X371 = BatchNorm(5b_a_bn)(X370,5b_a_bn_scale,5b_a_bn_bias)
JCudaTensor X371 = y18.forward_inference(X370, _5b_a_bn_scale, _5b_a_bn_bias);
// dealloc X370
X370.free();
// val X372 = ReLU()(X371)
JCudaTensor X372 = y7.forward(X371);
// val X373 = Convolv(1,1)(X372,5b_b_cv_W,5b_b_cv_B)
JCudaTensor X373 = y10.forward(X372, _5b_b_cv_W, _5b_b_cv_B);
// dealloc X372
X372.free();
// val X374 = BatchNorm(5b_b_bn)(X373,5b_b_bn_scale,5b_b_bn_bias)
JCudaTensor X374 = y16.forward_inference(X373, _5b_b_bn_scale, _5b_b_bn_bias);
// dealloc X373
X373.free();
// val X375 = ReLU()(X374)
JCudaTensor X375 = y7.forward(X374);
// val X376 = Convolv(1,0)(X375,5b_c_cv_W,5b_c_cv_B)
JCudaTensor X376 = y6.forward(X375, _5b_c_cv_W, _5b_c_cv_B);
// dealloc X375
X375.free();
// val X377 = BatchNorm(5b_c_bn)(X376,5b_c_bn_scale,5b_c_bn_bias)
JCudaTensor X377 = y14.forward_inference(X376, _5b_c_bn_scale, _5b_c_bn_bias);
// dealloc X376
X376.free();
// val X378 = ReLU()(X377)
JCudaTensor X378 = y3.forward(X377);
// val X379 = (X378 + X369)
JCudaTensor X379 = X378.plus_i(X369);;
// dealloc X369
X369.free();
// val X380 = ReLU()(X379)
JCudaTensor X380 = y3.forward(X379);
// val X381 = Convolv(1,0)(X380,5c_a_cv_W,5c_a_cv_B)
JCudaTensor X381 = y13.forward(X380, _5c_a_cv_W, _5c_a_cv_B);
// val X382 = BatchNorm(5c_a_bn)(X381,5c_a_bn_scale,5c_a_bn_bias)
JCudaTensor X382 = y11.forward_inference(X381, _5c_a_bn_scale, _5c_a_bn_bias);
// dealloc X381
X381.free();
// val X383 = ReLU()(X382)
JCudaTensor X383 = y7.forward(X382);
// val X384 = Convolv(1,1)(X383,5c_b_cv_W,5c_b_cv_B)
JCudaTensor X384 = y10.forward(X383, _5c_b_cv_W, _5c_b_cv_B);
// dealloc X383
X383.free();
// val X385 = BatchNorm(5c_b_bn)(X384,5c_b_bn_scale,5c_b_bn_bias)
JCudaTensor X385 = y8.forward_inference(X384, _5c_b_bn_scale, _5c_b_bn_bias);
// dealloc X384
X384.free();
// val X386 = ReLU()(X385)
JCudaTensor X386 = y7.forward(X385);
// val X387 = Convolv(1,0)(X386,5c_c_cv_W,5c_c_cv_B)
JCudaTensor X387 = y6.forward(X386, _5c_c_cv_W, _5c_c_cv_B);
// dealloc X386
X386.free();
// val X388 = BatchNorm(5c_c_bn)(X387,5c_c_bn_scale,5c_c_bn_bias)
JCudaTensor X388 = y4.forward_inference(X387, _5c_c_bn_scale, _5c_c_bn_bias);
// dealloc X387
X387.free();
// val X389 = ReLU()(X388)
JCudaTensor X389 = y3.forward(X388);
// val X390 = (X389 + X380)
JCudaTensor X390 = X389.plus_i(X380);;
// dealloc X380
X380.free();
// val X391 = ReLU()(X390)
JCudaTensor X391 = y3.forward(X390);
// val X392 = Pooling(7,1,0,false)(X391)
JCudaTensor X392 = y2.forward(X391);
// dealloc X391
X391.free();
// val X395 = (X392[1><3])(i1 | @) * (fc_W)(i2 | @)
JCudaTensor X395 = X392.flatten(1, new int[]{2048, 1, 1}).asMatrix(1, true).times(fc_W.asMatrix(1, true));
// dealloc X392
X392.free();
// val X394 = (X395 + (i1) => fc_B)
JCudaTensor X394 = fc_B.copy(64, X395);

return X394; 
}

}