package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.util.CudaRun;


public class Googlenet256 extends CudaRun {

public static void main(String[] args){
Googlenet256 run = new Googlenet256();
run.train(10);
run.test(1);
run.save();
run.free();
}

public Googlenet256() {
super("src/main/java/deepdsl/gen/googlenet256");
setTrainData(LmdbFactory.getFactory("dataset/imagenet/ilsvrc12_train_lmdb", 1000000, new int[]{256, 3, 224, 224}, 1000, false));
setTestData(LmdbFactory.getFactory("dataset/imagenet/ilsvrc12_val_lmdb", 10000, new int[]{256, 3, 224, 224}, 1000, true));
}

float lrn_rate = -0.01f;
float loss2 = 0.3f;
float loss1 = 0.3f;
float momentum = 0.9f;
float decay = 5.0E-4f;

JCudnnConvolution y98 = addConvolution(new int[]{256,192,28,28},new int[]{16,192,1,1},new int[]{16}, 1, 0);
JCudnnConvolution y83 = addConvolution(new int[]{256,192,28,28},new int[]{32,192,1,1},new int[]{32}, 1, 0);
JCudnnConvolution y82 = addConvolution(new int[]{256,192,28,28},new int[]{64,192,1,1},new int[]{64}, 1, 0);
JCudnnConvolution y97 = addConvolution(new int[]{256,192,28,28},new int[]{96,192,1,1},new int[]{96}, 1, 0);
JCudnnConvolution y106 = addConvolution(new int[]{256,256,14,14},new int[]{16,256,1,1},new int[]{16}, 1, 0);
JCudnnConvolution y37 = addConvolution(new int[]{256,256,14,14},new int[]{32,256,1,1},new int[]{32}, 1, 0);
JCudnnConvolution y32 = addConvolution(new int[]{256,256,14,14},new int[]{64,256,1,1},new int[]{64}, 1, 0);
JCudnnConvolution y104 = addConvolution(new int[]{256,256,14,14},new int[]{96,256,1,1},new int[]{96}, 1, 0);
JCudnnConvolution y102 = addConvolution(new int[]{256,256,28,28},new int[]{16,256,1,1},new int[]{16}, 1, 0);
JCudnnConvolution y74 = addConvolution(new int[]{256,256,28,28},new int[]{32,256,1,1},new int[]{32}, 1, 0);
JCudnnConvolution y69 = addConvolution(new int[]{256,256,28,28},new int[]{64,256,1,1},new int[]{64}, 1, 0);
JCudnnConvolution y101 = addConvolution(new int[]{256,256,28,28},new int[]{96,256,1,1},new int[]{96}, 1, 0);
JCudnnConvolution y5 = addConvolution(new int[]{256,256,4,4},new int[]{128,256,1,1},new int[]{128}, 1, 0);
JCudnnConvolution y110 = addConvolution(new int[]{256,256,7,7},new int[]{16,256,1,1},new int[]{16}, 1, 0);
JCudnnConvolution y17 = addConvolution(new int[]{256,256,7,7},new int[]{32,256,1,1},new int[]{32}, 1, 0);
JCudnnConvolution y12 = addConvolution(new int[]{256,256,7,7},new int[]{64,256,1,1},new int[]{64}, 1, 0);
JCudnnConvolution y111 = addConvolution(new int[]{256,256,7,7},new int[]{96,256,1,1},new int[]{96}, 1, 0);
JCudnnConvolution y92 = addConvolution(new int[]{256,64,56,56},new int[]{64,64,1,1},new int[]{64}, 1, 0);
JCudnnConvolution y90 = addConvolution(new int[]{256,64,56,56},new int[]{192,64,3,3},new int[]{192}, 1, 1);
JCudnnConvolution y34 = addConvolution(new int[]{256,96,14,14},new int[]{128,96,3,3},new int[]{128}, 1, 1);
JCudnnConvolution y71 = addConvolution(new int[]{256,96,28,28},new int[]{128,96,3,3},new int[]{128}, 1, 1);
JCudnnConvolution y14 = addConvolution(new int[]{256,96,7,7},new int[]{128,96,3,3},new int[]{128}, 1, 1);
JCudnnConvolution y36 = addConvolution(new int[]{256,16,14,14},new int[]{32,16,5,5},new int[]{32}, 1, 2);
JCudnnConvolution y73 = addConvolution(new int[]{256,16,28,28},new int[]{32,16,5,5},new int[]{32}, 1, 2);
JCudnnConvolution y16 = addConvolution(new int[]{256,16,7,7},new int[]{32,16,5,5},new int[]{32}, 1, 2);
JCudnnConvolution y96 = addConvolution(new int[]{256,3,224,224},new int[]{64,3,7,7},new int[]{64}, 2, 3);
JCudnnLRN y88 = addLRN(new int[]{256,192,56,56}, 5, 1.0E-4, 0.75);
JCudnnLRN y93 = addLRN(new int[]{256,64,56,56}, 5, 1.0E-4, 0.75);
JCudnnSoftmax y1 = addSoftmax(new int[]{256,1000}, SoftmaxAlgorithm.LOG);
JCudnnPooling y99 = addPooling(new int[]{256,192,28,28}, 3, 1, 1, PoolingType.MAX);
JCudnnPooling y105 = addPooling(new int[]{256,256,14,14}, 3, 1, 1, PoolingType.MAX);
JCudnnPooling y103 = addPooling(new int[]{256,256,28,28}, 3, 1, 1, PoolingType.MAX);
JCudnnPooling y109 = addPooling(new int[]{256,256,7,7}, 3, 1, 1, PoolingType.MAX);
JCudnnPooling y87 = addPooling(new int[]{256,192,56,56}, 3, 2, 1, PoolingType.MAX);
JCudnnPooling y28 = addPooling(new int[]{256,256,14,14}, 3, 2, 1, PoolingType.MAX);
JCudnnPooling y65 = addPooling(new int[]{256,256,28,28}, 3, 2, 1, PoolingType.MAX);
JCudnnPooling y94 = addPooling(new int[]{256,64,112,112}, 3, 2, 1, PoolingType.MAX);
JCudnnPooling y108 = addPooling(new int[]{256,256,14,14}, 5, 3, 0, PoolingType.AVERAGE_EXCLUDE_PADDING);
JCudnnPooling y8 = addPooling(new int[]{256,256,7,7}, 7, 1, 0, PoolingType.AVERAGE_EXCLUDE_PADDING);
JCudnnActivation y3 = addActivation(new int[]{256,1024}, ActivationMode.RELU);
JCudnnActivation y33 = addActivation(new int[]{256,128,14,14}, ActivationMode.RELU);
JCudnnActivation y70 = addActivation(new int[]{256,128,28,28}, ActivationMode.RELU);
JCudnnActivation y4 = addActivation(new int[]{256,128,4,4}, ActivationMode.RELU);
JCudnnActivation y13 = addActivation(new int[]{256,128,7,7}, ActivationMode.RELU);
JCudnnActivation y40 = addActivation(new int[]{256,16,14,14}, ActivationMode.RELU);
JCudnnActivation y77 = addActivation(new int[]{256,16,28,28}, ActivationMode.RELU);
JCudnnActivation y20 = addActivation(new int[]{256,16,7,7}, ActivationMode.RELU);
JCudnnActivation y89 = addActivation(new int[]{256,192,56,56}, ActivationMode.RELU);
JCudnnActivation y35 = addActivation(new int[]{256,32,14,14}, ActivationMode.RELU);
JCudnnActivation y72 = addActivation(new int[]{256,32,28,28}, ActivationMode.RELU);
JCudnnActivation y15 = addActivation(new int[]{256,32,7,7}, ActivationMode.RELU);
JCudnnActivation y95 = addActivation(new int[]{256,64,112,112}, ActivationMode.RELU);
JCudnnActivation y31 = addActivation(new int[]{256,64,14,14}, ActivationMode.RELU);
JCudnnActivation y68 = addActivation(new int[]{256,64,28,28}, ActivationMode.RELU);
JCudnnActivation y91 = addActivation(new int[]{256,64,56,56}, ActivationMode.RELU);
JCudnnActivation y11 = addActivation(new int[]{256,64,7,7}, ActivationMode.RELU);
JCudnnActivation y38 = addActivation(new int[]{256,96,14,14}, ActivationMode.RELU);
JCudnnActivation y75 = addActivation(new int[]{256,96,28,28}, ActivationMode.RELU);
JCudnnActivation y18 = addActivation(new int[]{256,96,7,7}, ActivationMode.RELU);
JCudnnConcat y107 = addConcat(new int[]{256,64,14,14},new int[]{256,128,14,14},new int[]{256,32,14,14},new int[]{256,32,14,14});
JCudnnConcat y100 = addConcat(new int[]{256,64,28,28},new int[]{256,128,28,28},new int[]{256,32,28,28},new int[]{256,32,28,28});
JCudnnConcat y112 = addConcat(new int[]{256,64,7,7},new int[]{256,128,7,7},new int[]{256,32,7,7},new int[]{256,32,7,7});
JCudnnDropout y7 = addDropout("y7", new int[]{256,256,1,1}, 0.4f);
JCudnnDropout y6 = addDropout("y6", new int[]{256,1024}, 0.7f);
JCudnnDropout y2 = addDropout("y2", new int[]{256,1024}, 0.7f);
JCudaTensor V_b1cv_B = addParam("V_b1cv_B", "Constant", 0f, 128);
JCudaTensor V_b1cv_W = addParam("V_b1cv_W", "Constant", 0f, 128, 256, 1, 1);
JCudaTensor V_b1fc1_B = addParam("V_b1fc1_B", "Constant", 0f, 1024);
JCudaTensor V_b1fc1_W = addParam("V_b1fc1_W", "Constant", 0f, 1024, 2048);
JCudaTensor V_b1fc2_B = addParam("V_b1fc2_B", "Constant", 0f, 1000);
JCudaTensor V_b1fc2_W = addParam("V_b1fc2_W", "Constant", 0f, 1000, 1024);
JCudaTensor V_b2cv_B = addParam("V_b2cv_B", "Constant", 0f, 128);
JCudaTensor V_b2cv_W = addParam("V_b2cv_W", "Constant", 0f, 128, 256, 1, 1);
JCudaTensor V_b2fc1_B = addParam("V_b2fc1_B", "Constant", 0f, 1024);
JCudaTensor V_b2fc1_W = addParam("V_b2fc1_W", "Constant", 0f, 1024, 2048);
JCudaTensor V_b2fc2_B = addParam("V_b2fc2_B", "Constant", 0f, 1000);
JCudaTensor V_b2fc2_W = addParam("V_b2fc2_W", "Constant", 0f, 1000, 1024);
JCudaTensor V_cv11_B = addParam("V_cv11_B", "Constant", 0f, 64);
JCudaTensor V_cv11_W = addParam("V_cv11_W", "Constant", 0f, 64, 192, 1, 1);
JCudaTensor V_cv12_B = addParam("V_cv12_B", "Constant", 0f, 96);
JCudaTensor V_cv12_W = addParam("V_cv12_W", "Constant", 0f, 96, 192, 1, 1);
JCudaTensor V_cv13_B = addParam("V_cv13_B", "Constant", 0f, 128);
JCudaTensor V_cv13_W = addParam("V_cv13_W", "Constant", 0f, 128, 96, 3, 3);
JCudaTensor V_cv14_B = addParam("V_cv14_B", "Constant", 0f, 16);
JCudaTensor V_cv14_W = addParam("V_cv14_W", "Constant", 0f, 16, 192, 1, 1);
JCudaTensor V_cv15_B = addParam("V_cv15_B", "Constant", 0f, 32);
JCudaTensor V_cv15_W = addParam("V_cv15_W", "Constant", 0f, 32, 16, 5, 5);
JCudaTensor V_cv16_B = addParam("V_cv16_B", "Constant", 0f, 32);
JCudaTensor V_cv16_W = addParam("V_cv16_W", "Constant", 0f, 32, 192, 1, 1);
JCudaTensor V_cv1_B = addParam("V_cv1_B", "Constant", 0f, 64);
JCudaTensor V_cv1_W = addParam("V_cv1_W", "Constant", 0f, 64, 3, 7, 7);
JCudaTensor V_cv21_B = addParam("V_cv21_B", "Constant", 0f, 64);
JCudaTensor V_cv21_W = addParam("V_cv21_W", "Constant", 0f, 64, 256, 1, 1);
JCudaTensor V_cv22_B = addParam("V_cv22_B", "Constant", 0f, 96);
JCudaTensor V_cv22_W = addParam("V_cv22_W", "Constant", 0f, 96, 256, 1, 1);
JCudaTensor V_cv23_B = addParam("V_cv23_B", "Constant", 0f, 128);
JCudaTensor V_cv23_W = addParam("V_cv23_W", "Constant", 0f, 128, 96, 3, 3);
JCudaTensor V_cv24_B = addParam("V_cv24_B", "Constant", 0f, 16);
JCudaTensor V_cv24_W = addParam("V_cv24_W", "Constant", 0f, 16, 256, 1, 1);
JCudaTensor V_cv25_B = addParam("V_cv25_B", "Constant", 0f, 32);
JCudaTensor V_cv25_W = addParam("V_cv25_W", "Constant", 0f, 32, 16, 5, 5);
JCudaTensor V_cv26_B = addParam("V_cv26_B", "Constant", 0f, 32);
JCudaTensor V_cv26_W = addParam("V_cv26_W", "Constant", 0f, 32, 256, 1, 1);
JCudaTensor V_cv2_B = addParam("V_cv2_B", "Constant", 0f, 64);
JCudaTensor V_cv2_W = addParam("V_cv2_W", "Constant", 0f, 64, 64, 1, 1);
JCudaTensor V_cv31_B = addParam("V_cv31_B", "Constant", 0f, 64);
JCudaTensor V_cv31_W = addParam("V_cv31_W", "Constant", 0f, 64, 256, 1, 1);
JCudaTensor V_cv32_B = addParam("V_cv32_B", "Constant", 0f, 96);
JCudaTensor V_cv32_W = addParam("V_cv32_W", "Constant", 0f, 96, 256, 1, 1);
JCudaTensor V_cv33_B = addParam("V_cv33_B", "Constant", 0f, 128);
JCudaTensor V_cv33_W = addParam("V_cv33_W", "Constant", 0f, 128, 96, 3, 3);
JCudaTensor V_cv34_B = addParam("V_cv34_B", "Constant", 0f, 16);
JCudaTensor V_cv34_W = addParam("V_cv34_W", "Constant", 0f, 16, 256, 1, 1);
JCudaTensor V_cv35_B = addParam("V_cv35_B", "Constant", 0f, 32);
JCudaTensor V_cv35_W = addParam("V_cv35_W", "Constant", 0f, 32, 16, 5, 5);
JCudaTensor V_cv36_B = addParam("V_cv36_B", "Constant", 0f, 32);
JCudaTensor V_cv36_W = addParam("V_cv36_W", "Constant", 0f, 32, 256, 1, 1);
JCudaTensor V_cv3_B = addParam("V_cv3_B", "Constant", 0f, 192);
JCudaTensor V_cv3_W = addParam("V_cv3_W", "Constant", 0f, 192, 64, 3, 3);
JCudaTensor V_cv41_B = addParam("V_cv41_B", "Constant", 0f, 64);
JCudaTensor V_cv41_W = addParam("V_cv41_W", "Constant", 0f, 64, 256, 1, 1);
JCudaTensor V_cv42_B = addParam("V_cv42_B", "Constant", 0f, 96);
JCudaTensor V_cv42_W = addParam("V_cv42_W", "Constant", 0f, 96, 256, 1, 1);
JCudaTensor V_cv43_B = addParam("V_cv43_B", "Constant", 0f, 128);
JCudaTensor V_cv43_W = addParam("V_cv43_W", "Constant", 0f, 128, 96, 3, 3);
JCudaTensor V_cv44_B = addParam("V_cv44_B", "Constant", 0f, 16);
JCudaTensor V_cv44_W = addParam("V_cv44_W", "Constant", 0f, 16, 256, 1, 1);
JCudaTensor V_cv45_B = addParam("V_cv45_B", "Constant", 0f, 32);
JCudaTensor V_cv45_W = addParam("V_cv45_W", "Constant", 0f, 32, 16, 5, 5);
JCudaTensor V_cv46_B = addParam("V_cv46_B", "Constant", 0f, 32);
JCudaTensor V_cv46_W = addParam("V_cv46_W", "Constant", 0f, 32, 256, 1, 1);
JCudaTensor V_cv51_B = addParam("V_cv51_B", "Constant", 0f, 64);
JCudaTensor V_cv51_W = addParam("V_cv51_W", "Constant", 0f, 64, 256, 1, 1);
JCudaTensor V_cv52_B = addParam("V_cv52_B", "Constant", 0f, 96);
JCudaTensor V_cv52_W = addParam("V_cv52_W", "Constant", 0f, 96, 256, 1, 1);
JCudaTensor V_cv53_B = addParam("V_cv53_B", "Constant", 0f, 128);
JCudaTensor V_cv53_W = addParam("V_cv53_W", "Constant", 0f, 128, 96, 3, 3);
JCudaTensor V_cv54_B = addParam("V_cv54_B", "Constant", 0f, 16);
JCudaTensor V_cv54_W = addParam("V_cv54_W", "Constant", 0f, 16, 256, 1, 1);
JCudaTensor V_cv55_B = addParam("V_cv55_B", "Constant", 0f, 32);
JCudaTensor V_cv55_W = addParam("V_cv55_W", "Constant", 0f, 32, 16, 5, 5);
JCudaTensor V_cv56_B = addParam("V_cv56_B", "Constant", 0f, 32);
JCudaTensor V_cv56_W = addParam("V_cv56_W", "Constant", 0f, 32, 256, 1, 1);
JCudaTensor V_cv61_B = addParam("V_cv61_B", "Constant", 0f, 64);
JCudaTensor V_cv61_W = addParam("V_cv61_W", "Constant", 0f, 64, 256, 1, 1);
JCudaTensor V_cv62_B = addParam("V_cv62_B", "Constant", 0f, 96);
JCudaTensor V_cv62_W = addParam("V_cv62_W", "Constant", 0f, 96, 256, 1, 1);
JCudaTensor V_cv63_B = addParam("V_cv63_B", "Constant", 0f, 128);
JCudaTensor V_cv63_W = addParam("V_cv63_W", "Constant", 0f, 128, 96, 3, 3);
JCudaTensor V_cv64_B = addParam("V_cv64_B", "Constant", 0f, 16);
JCudaTensor V_cv64_W = addParam("V_cv64_W", "Constant", 0f, 16, 256, 1, 1);
JCudaTensor V_cv65_B = addParam("V_cv65_B", "Constant", 0f, 32);
JCudaTensor V_cv65_W = addParam("V_cv65_W", "Constant", 0f, 32, 16, 5, 5);
JCudaTensor V_cv66_B = addParam("V_cv66_B", "Constant", 0f, 32);
JCudaTensor V_cv66_W = addParam("V_cv66_W", "Constant", 0f, 32, 256, 1, 1);
JCudaTensor V_cv71_B = addParam("V_cv71_B", "Constant", 0f, 64);
JCudaTensor V_cv71_W = addParam("V_cv71_W", "Constant", 0f, 64, 256, 1, 1);
JCudaTensor V_cv72_B = addParam("V_cv72_B", "Constant", 0f, 96);
JCudaTensor V_cv72_W = addParam("V_cv72_W", "Constant", 0f, 96, 256, 1, 1);
JCudaTensor V_cv73_B = addParam("V_cv73_B", "Constant", 0f, 128);
JCudaTensor V_cv73_W = addParam("V_cv73_W", "Constant", 0f, 128, 96, 3, 3);
JCudaTensor V_cv74_B = addParam("V_cv74_B", "Constant", 0f, 16);
JCudaTensor V_cv74_W = addParam("V_cv74_W", "Constant", 0f, 16, 256, 1, 1);
JCudaTensor V_cv75_B = addParam("V_cv75_B", "Constant", 0f, 32);
JCudaTensor V_cv75_W = addParam("V_cv75_W", "Constant", 0f, 32, 16, 5, 5);
JCudaTensor V_cv76_B = addParam("V_cv76_B", "Constant", 0f, 32);
JCudaTensor V_cv76_W = addParam("V_cv76_W", "Constant", 0f, 32, 256, 1, 1);
JCudaTensor V_cv81_B = addParam("V_cv81_B", "Constant", 0f, 64);
JCudaTensor V_cv81_W = addParam("V_cv81_W", "Constant", 0f, 64, 256, 1, 1);
JCudaTensor V_cv82_B = addParam("V_cv82_B", "Constant", 0f, 96);
JCudaTensor V_cv82_W = addParam("V_cv82_W", "Constant", 0f, 96, 256, 1, 1);
JCudaTensor V_cv83_B = addParam("V_cv83_B", "Constant", 0f, 128);
JCudaTensor V_cv83_W = addParam("V_cv83_W", "Constant", 0f, 128, 96, 3, 3);
JCudaTensor V_cv84_B = addParam("V_cv84_B", "Constant", 0f, 16);
JCudaTensor V_cv84_W = addParam("V_cv84_W", "Constant", 0f, 16, 256, 1, 1);
JCudaTensor V_cv85_B = addParam("V_cv85_B", "Constant", 0f, 32);
JCudaTensor V_cv85_W = addParam("V_cv85_W", "Constant", 0f, 32, 16, 5, 5);
JCudaTensor V_cv86_B = addParam("V_cv86_B", "Constant", 0f, 32);
JCudaTensor V_cv86_W = addParam("V_cv86_W", "Constant", 0f, 32, 256, 1, 1);
JCudaTensor V_cv91_B = addParam("V_cv91_B", "Constant", 0f, 64);
JCudaTensor V_cv91_W = addParam("V_cv91_W", "Constant", 0f, 64, 256, 1, 1);
JCudaTensor V_cv92_B = addParam("V_cv92_B", "Constant", 0f, 96);
JCudaTensor V_cv92_W = addParam("V_cv92_W", "Constant", 0f, 96, 256, 1, 1);
JCudaTensor V_cv93_B = addParam("V_cv93_B", "Constant", 0f, 128);
JCudaTensor V_cv93_W = addParam("V_cv93_W", "Constant", 0f, 128, 96, 3, 3);
JCudaTensor V_cv94_B = addParam("V_cv94_B", "Constant", 0f, 16);
JCudaTensor V_cv94_W = addParam("V_cv94_W", "Constant", 0f, 16, 256, 1, 1);
JCudaTensor V_cv95_B = addParam("V_cv95_B", "Constant", 0f, 32);
JCudaTensor V_cv95_W = addParam("V_cv95_W", "Constant", 0f, 32, 16, 5, 5);
JCudaTensor V_cv96_B = addParam("V_cv96_B", "Constant", 0f, 32);
JCudaTensor V_cv96_W = addParam("V_cv96_W", "Constant", 0f, 32, 256, 1, 1);
JCudaTensor V_fc_B = addParam("V_fc_B", "Constant", 0f, 1000);
JCudaTensor V_fc_W = addParam("V_fc_W", "Constant", 0f, 1000, 256);
JCudaTensor b1cv_B = addParam("b1cv_B", "Constant", 0.2f, 128);
JCudaTensor b1cv_W = addParam("b1cv_W", "Random", 0.088388346f, 128, 256, 1, 1);
JCudaTensor b1fc1_B = addParam("b1fc1_B", "Constant", 0.2f, 1024);
JCudaTensor b1fc1_W = addParam("b1fc1_W", "Random", 0.03125f, 1024, 2048);
JCudaTensor b1fc2_B = addParam("b1fc2_B", "Constant", 0.0f, 1000);
JCudaTensor b1fc2_W = addParam("b1fc2_W", "Random", 0.044194173f, 1000, 1024);
JCudaTensor b2cv_B = addParam("b2cv_B", "Constant", 0.2f, 128);
JCudaTensor b2cv_W = addParam("b2cv_W", "Random", 0.088388346f, 128, 256, 1, 1);
JCudaTensor b2fc1_B = addParam("b2fc1_B", "Constant", 0.2f, 1024);
JCudaTensor b2fc1_W = addParam("b2fc1_W", "Random", 0.03125f, 1024, 2048);
JCudaTensor b2fc2_B = addParam("b2fc2_B", "Constant", 0.0f, 1000);
JCudaTensor b2fc2_W = addParam("b2fc2_W", "Random", 0.044194173f, 1000, 1024);
JCudaTensor cv11_B = addParam("cv11_B", "Constant", 0.2f, 64);
JCudaTensor cv11_W = addParam("cv11_W", "Random", 0.10206208f, 64, 192, 1, 1);
JCudaTensor cv12_B = addParam("cv12_B", "Constant", 0.2f, 96);
JCudaTensor cv12_W = addParam("cv12_W", "Random", 0.10206208f, 96, 192, 1, 1);
JCudaTensor cv13_B = addParam("cv13_B", "Constant", 0.2f, 128);
JCudaTensor cv13_W = addParam("cv13_W", "Random", 0.048112523f, 128, 96, 3, 3);
JCudaTensor cv14_B = addParam("cv14_B", "Constant", 0.2f, 16);
JCudaTensor cv14_W = addParam("cv14_W", "Random", 0.10206208f, 16, 192, 1, 1);
JCudaTensor cv15_B = addParam("cv15_B", "Constant", 0.2f, 32);
JCudaTensor cv15_W = addParam("cv15_W", "Random", 0.07071068f, 32, 16, 5, 5);
JCudaTensor cv16_B = addParam("cv16_B", "Constant", 0.2f, 32);
JCudaTensor cv16_W = addParam("cv16_W", "Random", 0.10206208f, 32, 192, 1, 1);
JCudaTensor cv1_B = addParam("cv1_B", "Constant", 0.2f, 64);
JCudaTensor cv1_W = addParam("cv1_W", "Random", 0.11664237f, 64, 3, 7, 7);
JCudaTensor cv21_B = addParam("cv21_B", "Constant", 0.2f, 64);
JCudaTensor cv21_W = addParam("cv21_W", "Random", 0.088388346f, 64, 256, 1, 1);
JCudaTensor cv22_B = addParam("cv22_B", "Constant", 0.2f, 96);
JCudaTensor cv22_W = addParam("cv22_W", "Random", 0.088388346f, 96, 256, 1, 1);
JCudaTensor cv23_B = addParam("cv23_B", "Constant", 0.2f, 128);
JCudaTensor cv23_W = addParam("cv23_W", "Random", 0.048112523f, 128, 96, 3, 3);
JCudaTensor cv24_B = addParam("cv24_B", "Constant", 0.2f, 16);
JCudaTensor cv24_W = addParam("cv24_W", "Random", 0.088388346f, 16, 256, 1, 1);
JCudaTensor cv25_B = addParam("cv25_B", "Constant", 0.2f, 32);
JCudaTensor cv25_W = addParam("cv25_W", "Random", 0.07071068f, 32, 16, 5, 5);
JCudaTensor cv26_B = addParam("cv26_B", "Constant", 0.2f, 32);
JCudaTensor cv26_W = addParam("cv26_W", "Random", 0.088388346f, 32, 256, 1, 1);
JCudaTensor cv2_B = addParam("cv2_B", "Constant", 0.2f, 64);
JCudaTensor cv2_W = addParam("cv2_W", "Random", 0.17677669f, 64, 64, 1, 1);
JCudaTensor cv31_B = addParam("cv31_B", "Constant", 0.2f, 64);
JCudaTensor cv31_W = addParam("cv31_W", "Random", 0.088388346f, 64, 256, 1, 1);
JCudaTensor cv32_B = addParam("cv32_B", "Constant", 0.2f, 96);
JCudaTensor cv32_W = addParam("cv32_W", "Random", 0.088388346f, 96, 256, 1, 1);
JCudaTensor cv33_B = addParam("cv33_B", "Constant", 0.2f, 128);
JCudaTensor cv33_W = addParam("cv33_W", "Random", 0.048112523f, 128, 96, 3, 3);
JCudaTensor cv34_B = addParam("cv34_B", "Constant", 0.2f, 16);
JCudaTensor cv34_W = addParam("cv34_W", "Random", 0.088388346f, 16, 256, 1, 1);
JCudaTensor cv35_B = addParam("cv35_B", "Constant", 0.2f, 32);
JCudaTensor cv35_W = addParam("cv35_W", "Random", 0.07071068f, 32, 16, 5, 5);
JCudaTensor cv36_B = addParam("cv36_B", "Constant", 0.2f, 32);
JCudaTensor cv36_W = addParam("cv36_W", "Random", 0.088388346f, 32, 256, 1, 1);
JCudaTensor cv3_B = addParam("cv3_B", "Constant", 0.2f, 192);
JCudaTensor cv3_W = addParam("cv3_W", "Random", 0.058925565f, 192, 64, 3, 3);
JCudaTensor cv41_B = addParam("cv41_B", "Constant", 0.2f, 64);
JCudaTensor cv41_W = addParam("cv41_W", "Random", 0.088388346f, 64, 256, 1, 1);
JCudaTensor cv42_B = addParam("cv42_B", "Constant", 0.2f, 96);
JCudaTensor cv42_W = addParam("cv42_W", "Random", 0.088388346f, 96, 256, 1, 1);
JCudaTensor cv43_B = addParam("cv43_B", "Constant", 0.2f, 128);
JCudaTensor cv43_W = addParam("cv43_W", "Random", 0.048112523f, 128, 96, 3, 3);
JCudaTensor cv44_B = addParam("cv44_B", "Constant", 0.2f, 16);
JCudaTensor cv44_W = addParam("cv44_W", "Random", 0.088388346f, 16, 256, 1, 1);
JCudaTensor cv45_B = addParam("cv45_B", "Constant", 0.2f, 32);
JCudaTensor cv45_W = addParam("cv45_W", "Random", 0.07071068f, 32, 16, 5, 5);
JCudaTensor cv46_B = addParam("cv46_B", "Constant", 0.2f, 32);
JCudaTensor cv46_W = addParam("cv46_W", "Random", 0.088388346f, 32, 256, 1, 1);
JCudaTensor cv51_B = addParam("cv51_B", "Constant", 0.2f, 64);
JCudaTensor cv51_W = addParam("cv51_W", "Random", 0.088388346f, 64, 256, 1, 1);
JCudaTensor cv52_B = addParam("cv52_B", "Constant", 0.2f, 96);
JCudaTensor cv52_W = addParam("cv52_W", "Random", 0.088388346f, 96, 256, 1, 1);
JCudaTensor cv53_B = addParam("cv53_B", "Constant", 0.2f, 128);
JCudaTensor cv53_W = addParam("cv53_W", "Random", 0.048112523f, 128, 96, 3, 3);
JCudaTensor cv54_B = addParam("cv54_B", "Constant", 0.2f, 16);
JCudaTensor cv54_W = addParam("cv54_W", "Random", 0.088388346f, 16, 256, 1, 1);
JCudaTensor cv55_B = addParam("cv55_B", "Constant", 0.2f, 32);
JCudaTensor cv55_W = addParam("cv55_W", "Random", 0.07071068f, 32, 16, 5, 5);
JCudaTensor cv56_B = addParam("cv56_B", "Constant", 0.2f, 32);
JCudaTensor cv56_W = addParam("cv56_W", "Random", 0.088388346f, 32, 256, 1, 1);
JCudaTensor cv61_B = addParam("cv61_B", "Constant", 0.2f, 64);
JCudaTensor cv61_W = addParam("cv61_W", "Random", 0.088388346f, 64, 256, 1, 1);
JCudaTensor cv62_B = addParam("cv62_B", "Constant", 0.2f, 96);
JCudaTensor cv62_W = addParam("cv62_W", "Random", 0.088388346f, 96, 256, 1, 1);
JCudaTensor cv63_B = addParam("cv63_B", "Constant", 0.2f, 128);
JCudaTensor cv63_W = addParam("cv63_W", "Random", 0.048112523f, 128, 96, 3, 3);
JCudaTensor cv64_B = addParam("cv64_B", "Constant", 0.2f, 16);
JCudaTensor cv64_W = addParam("cv64_W", "Random", 0.088388346f, 16, 256, 1, 1);
JCudaTensor cv65_B = addParam("cv65_B", "Constant", 0.2f, 32);
JCudaTensor cv65_W = addParam("cv65_W", "Random", 0.07071068f, 32, 16, 5, 5);
JCudaTensor cv66_B = addParam("cv66_B", "Constant", 0.2f, 32);
JCudaTensor cv66_W = addParam("cv66_W", "Random", 0.088388346f, 32, 256, 1, 1);
JCudaTensor cv71_B = addParam("cv71_B", "Constant", 0.2f, 64);
JCudaTensor cv71_W = addParam("cv71_W", "Random", 0.088388346f, 64, 256, 1, 1);
JCudaTensor cv72_B = addParam("cv72_B", "Constant", 0.2f, 96);
JCudaTensor cv72_W = addParam("cv72_W", "Random", 0.088388346f, 96, 256, 1, 1);
JCudaTensor cv73_B = addParam("cv73_B", "Constant", 0.2f, 128);
JCudaTensor cv73_W = addParam("cv73_W", "Random", 0.048112523f, 128, 96, 3, 3);
JCudaTensor cv74_B = addParam("cv74_B", "Constant", 0.2f, 16);
JCudaTensor cv74_W = addParam("cv74_W", "Random", 0.088388346f, 16, 256, 1, 1);
JCudaTensor cv75_B = addParam("cv75_B", "Constant", 0.2f, 32);
JCudaTensor cv75_W = addParam("cv75_W", "Random", 0.07071068f, 32, 16, 5, 5);
JCudaTensor cv76_B = addParam("cv76_B", "Constant", 0.2f, 32);
JCudaTensor cv76_W = addParam("cv76_W", "Random", 0.088388346f, 32, 256, 1, 1);
JCudaTensor cv81_B = addParam("cv81_B", "Constant", 0.2f, 64);
JCudaTensor cv81_W = addParam("cv81_W", "Random", 0.088388346f, 64, 256, 1, 1);
JCudaTensor cv82_B = addParam("cv82_B", "Constant", 0.2f, 96);
JCudaTensor cv82_W = addParam("cv82_W", "Random", 0.088388346f, 96, 256, 1, 1);
JCudaTensor cv83_B = addParam("cv83_B", "Constant", 0.2f, 128);
JCudaTensor cv83_W = addParam("cv83_W", "Random", 0.048112523f, 128, 96, 3, 3);
JCudaTensor cv84_B = addParam("cv84_B", "Constant", 0.2f, 16);
JCudaTensor cv84_W = addParam("cv84_W", "Random", 0.088388346f, 16, 256, 1, 1);
JCudaTensor cv85_B = addParam("cv85_B", "Constant", 0.2f, 32);
JCudaTensor cv85_W = addParam("cv85_W", "Random", 0.07071068f, 32, 16, 5, 5);
JCudaTensor cv86_B = addParam("cv86_B", "Constant", 0.2f, 32);
JCudaTensor cv86_W = addParam("cv86_W", "Random", 0.088388346f, 32, 256, 1, 1);
JCudaTensor cv91_B = addParam("cv91_B", "Constant", 0.2f, 64);
JCudaTensor cv91_W = addParam("cv91_W", "Random", 0.088388346f, 64, 256, 1, 1);
JCudaTensor cv92_B = addParam("cv92_B", "Constant", 0.2f, 96);
JCudaTensor cv92_W = addParam("cv92_W", "Random", 0.088388346f, 96, 256, 1, 1);
JCudaTensor cv93_B = addParam("cv93_B", "Constant", 0.2f, 128);
JCudaTensor cv93_W = addParam("cv93_W", "Random", 0.048112523f, 128, 96, 3, 3);
JCudaTensor cv94_B = addParam("cv94_B", "Constant", 0.2f, 16);
JCudaTensor cv94_W = addParam("cv94_W", "Random", 0.088388346f, 16, 256, 1, 1);
JCudaTensor cv95_B = addParam("cv95_B", "Constant", 0.2f, 32);
JCudaTensor cv95_W = addParam("cv95_W", "Random", 0.07071068f, 32, 16, 5, 5);
JCudaTensor cv96_B = addParam("cv96_B", "Constant", 0.2f, 32);
JCudaTensor cv96_W = addParam("cv96_W", "Random", 0.088388346f, 32, 256, 1, 1);
JCudaTensor fc_B = addParam("fc_B", "Constant", 0.0f, 1000);
JCudaTensor fc_W = addParam("fc_W", "Random", 0.088388346f, 1000, 256);

public float trainFunction(JTensorFloat X, JTensorFloat Y) {
 // val X1041 = Cuda(X)
JCudaTensor X1041 = X.asJCudaTensor();
// val X1044 = Cuda(Indicator(Y, 1000))
JCudaTensor X1044 = Y.asIndicator(1000).asJCudaTensor();
// val X232 = Convolv(2,3)(X1041,cv1_W,cv1_B)
JCudaTensor X232 = y96.forward(X1041, cv1_W, cv1_B);
// val X1130 = - X1044.copy
JCudaTensor X1130 = X1044.clone().times_i(-1f);;
// val X1131 = (X1130 / |256|)
JCudaTensor X1131 = X1130.times_i(1 / 256f);;
// val X233 = ReLU()(X232)
JCudaTensor X233 = y95.forward(X232);
// val X234 = Pooling(3,2,1,true)(X233)
JCudaTensor X234 = y94.forward(X233);
// val X1003 = (X1131.copy * 0.3)
JCudaTensor X1003 = X1131.clone().times_i(loss1);;
// val X235 = LRN(5,1.0E-4,0.75)(X234)
JCudaTensor X235 = y93.forward(X234);
// val X236 = Convolv(1,0)(X235,cv2_W,cv2_B)
JCudaTensor X236 = y92.forward(X235, cv2_W, cv2_B);
// val X237 = ReLU()(X236)
JCudaTensor X237 = y91.forward(X236);
// val X238 = Convolv(1,1)(X237,cv3_W,cv3_B)
JCudaTensor X238 = y90.forward(X237, cv3_W, cv3_B);
// val X239 = ReLU()(X238)
JCudaTensor X239 = y89.forward(X238);
// val X240 = LRN(5,1.0E-4,0.75)(X239)
JCudaTensor X240 = y88.forward(X239);
// val X241 = Pooling(3,2,1,true)(X240)
JCudaTensor X241 = y87.forward(X240);
// val X253 = Convolv(1,0)(X241,cv11_W,cv11_B)
JCudaTensor X253 = y82.forward(X241, cv11_W, cv11_B);
// val X249 = Convolv(1,0)(X241,cv12_W,cv12_B)
JCudaTensor X249 = y97.forward(X241, cv12_W, cv12_B);
// val X245 = Convolv(1,0)(X241,cv14_W,cv14_B)
JCudaTensor X245 = y98.forward(X241, cv14_W, cv14_B);
// val X242 = Pooling(3,1,1,true)(X241)
JCudaTensor X242 = y99.forward(X241);
// val X243 = Convolv(1,0)(X242,cv16_W,cv16_B)
JCudaTensor X243 = y83.forward(X242, cv16_W, cv16_B);
// val X250 = ReLU()(X249)
JCudaTensor X250 = y75.forward(X249);
// val X246 = ReLU()(X245)
JCudaTensor X246 = y77.forward(X245);
// val X254 = ReLU()(X253)
JCudaTensor X254 = y68.forward(X253);
// val X251 = Convolv(1,1)(X250,cv13_W,cv13_B)
JCudaTensor X251 = y71.forward(X250, cv13_W, cv13_B);
// val X247 = Convolv(1,2)(X246,cv15_W,cv15_B)
JCudaTensor X247 = y73.forward(X246, cv15_W, cv15_B);
// val X244 = ReLU()(X243)
JCudaTensor X244 = y72.forward(X243);
// val X248 = ReLU()(X247)
JCudaTensor X248 = y72.forward(X247);
// val X252 = ReLU()(X251)
JCudaTensor X252 = y70.forward(X251);
// val X255 = Concat(X254,X252,X248,X244)
JCudaTensor X255 = y100.forward(X254,X252,X248,X244);
// val X263 = Convolv(1,0)(X255,cv22_W,cv22_B)
JCudaTensor X263 = y101.forward(X255, cv22_W, cv22_B);
// val X267 = Convolv(1,0)(X255,cv21_W,cv21_B)
JCudaTensor X267 = y69.forward(X255, cv21_W, cv21_B);
// val X259 = Convolv(1,0)(X255,cv24_W,cv24_B)
JCudaTensor X259 = y102.forward(X255, cv24_W, cv24_B);
// val X256 = Pooling(3,1,1,true)(X255)
JCudaTensor X256 = y103.forward(X255);
// val X260 = ReLU()(X259)
JCudaTensor X260 = y77.forward(X259);
// val X264 = ReLU()(X263)
JCudaTensor X264 = y75.forward(X263);
// val X268 = ReLU()(X267)
JCudaTensor X268 = y68.forward(X267);
// val X257 = Convolv(1,0)(X256,cv26_W,cv26_B)
JCudaTensor X257 = y74.forward(X256, cv26_W, cv26_B);
// val X261 = Convolv(1,2)(X260,cv25_W,cv25_B)
JCudaTensor X261 = y73.forward(X260, cv25_W, cv25_B);
// val X265 = Convolv(1,1)(X264,cv23_W,cv23_B)
JCudaTensor X265 = y71.forward(X264, cv23_W, cv23_B);
// val X258 = ReLU()(X257)
JCudaTensor X258 = y72.forward(X257);
// val X266 = ReLU()(X265)
JCudaTensor X266 = y70.forward(X265);
// val X262 = ReLU()(X261)
JCudaTensor X262 = y72.forward(X261);
// val X269 = Concat(X268,X266,X262,X258)
JCudaTensor X269 = y100.forward(X268,X266,X262,X258);
// val X270 = Pooling(3,2,1,true)(X269)
JCudaTensor X270 = y65.forward(X269);
// val X282 = Convolv(1,0)(X270,cv31_W,cv31_B)
JCudaTensor X282 = y32.forward(X270, cv31_W, cv31_B);
// val X278 = Convolv(1,0)(X270,cv32_W,cv32_B)
JCudaTensor X278 = y104.forward(X270, cv32_W, cv32_B);
// val X271 = Pooling(3,1,1,true)(X270)
JCudaTensor X271 = y105.forward(X270);
// val X274 = Convolv(1,0)(X270,cv34_W,cv34_B)
JCudaTensor X274 = y106.forward(X270, cv34_W, cv34_B);
// val X272 = Convolv(1,0)(X271,cv36_W,cv36_B)
JCudaTensor X272 = y37.forward(X271, cv36_W, cv36_B);
// val X279 = ReLU()(X278)
JCudaTensor X279 = y38.forward(X278);
// val X275 = ReLU()(X274)
JCudaTensor X275 = y40.forward(X274);
// val X283 = ReLU()(X282)
JCudaTensor X283 = y31.forward(X282);
// val X273 = ReLU()(X272)
JCudaTensor X273 = y35.forward(X272);
// val X276 = Convolv(1,2)(X275,cv35_W,cv35_B)
JCudaTensor X276 = y36.forward(X275, cv35_W, cv35_B);
// val X280 = Convolv(1,1)(X279,cv33_W,cv33_B)
JCudaTensor X280 = y34.forward(X279, cv33_W, cv33_B);
// val X277 = ReLU()(X276)
JCudaTensor X277 = y35.forward(X276);
// val X281 = ReLU()(X280)
JCudaTensor X281 = y33.forward(X280);
// val X284 = Concat(X283,X281,X277,X273)
JCudaTensor X284 = y107.forward(X283,X281,X277,X273);
// val X384 = Pooling(5,3,0,false)(X284)
JCudaTensor X384 = y108.forward(X284);
// val X385 = Convolv(1,0)(X384,b1cv_W,b1cv_B)
JCudaTensor X385 = y5.forward(X384, b1cv_W, b1cv_B);
// val X386 = ReLU()(X385)
JCudaTensor X386 = y4.forward(X385);
// val X1037 = (X386[1><3])(i21 | @) * (b1fc1_W)(i22 | @)
JCudaTensor X1037 = X386.flatten(1, new int[]{128, 4, 4}).asMatrix(1, true).times(b1fc1_W.asMatrix(1, true));
// val X388 = (X1037 + (i21) => b1fc1_B)
JCudaTensor X388 = b1fc1_B.copy(256, X1037);
// val X389 = ReLU()(X388)
JCudaTensor X389 = y3.forward(X388);
// val X390 = Dropout(0.7)(X389)
JCudaTensor X390 = y2.forward(X389);
// val X1033 = (X390)(i24 | @) * (b1fc2_W)(i25 | @)
JCudaTensor X1033 = X390.asMatrix(1, true).times(b1fc2_W.asMatrix(1, true));
// val X391 = (X1033 + (i24) => b1fc2_B)
JCudaTensor X391 = b1fc2_B.copy(256, X1033);
// val X392 = LogSoftmax()(X391)
JCudaTensor X392 = y1.forward(X391);
// dealloc X391
X391.free();
// val X1232 = X1003 * d_LogSoftmax()(X392)/d_X391
JCudaTensor X1232 = y1.backward(X1003, X392);
// dealloc X1003
X1003.free();
// val m12 = (i508) => X1232[@, i508]
JCudaMatrix m12 = X1232.asMatrix(1, false);
// V_b1fc2_B = ((Sum(m12) * (2 * -0.01)) + (V_b1fc2_B * 0.9))
m12.sum(V_b1fc2_B, 2f * lrn_rate, momentum);
// b1fc2_B = (V_b1fc2_B + b1fc2_B)
b1fc2_B.update(V_b1fc2_B, 1f, 1f);
// val m9 = (i530) => b1fc2_W[@, i530]
JCudaMatrix m9 = b1fc2_W.asMatrix(1, false);
// val m13 = (i509) => X390[@, i509]
JCudaMatrix m13 = X390.asMatrix(1, false);
// V_b1fc2_W = ((m12 * m13 * -0.01) + (V_b1fc2_W * 0.9))
m12.times(m13, V_b1fc2_W, lrn_rate, momentum);
// dealloc X390
X390.free();
// b1fc2_W = (V_b1fc2_W + (b1fc2_W * (1 + (5.0E-4 * -0.01))))
b1fc2_W.update(V_b1fc2_W, 1f, 1f + decay * lrn_rate);
// val X1205 = (X1232)(i529 | @) * m9
JCudaTensor X1205 = X1232.asMatrix(1, true).times(m9);
// dealloc X1232
X1232.free();
// val X1102 = X1205 * d_Dropout(0.7)()/d_X389
JCudaTensor X1102 = y2.backward(X1205);
// dealloc X1205
X1205.free();
// val X1253 = X1102 * d_ReLU()(X389)/d_X388
JCudaTensor X1253 = y3.backward(X1102, X389);
// dealloc X389
X389.free();
// val m16 = (i526) => X1253[@, i526]
JCudaMatrix m16 = X1253.asMatrix(1, false);
// V_b1fc1_B = ((Sum(m16) * (2 * -0.01)) + (V_b1fc1_B * 0.9))
m16.sum(V_b1fc1_B, 2f * lrn_rate, momentum);
// b1fc1_B = (V_b1fc1_B + b1fc1_B)
b1fc1_B.update(V_b1fc1_B, 1f, 1f);
// val m3 = (i523) => b1fc1_W[@, i523]
JCudaMatrix m3 = b1fc1_W.asMatrix(1, false);
// val m20 = (i513) => X386[1><3][@, i513]
JCudaMatrix m20 = X386.flatten(1, new int[]{128, 4, 4}).asMatrix(1, false);
// V_b1fc1_W = ((m16 * m20 * -0.01) + (V_b1fc1_W * 0.9))
m16.times(m20, V_b1fc1_W, lrn_rate, momentum);
// b1fc1_W = (V_b1fc1_W + (b1fc1_W * (1 + (5.0E-4 * -0.01))))
b1fc1_W.update(V_b1fc1_W, 1f, 1f + decay * lrn_rate);
// val X1125 = (X1253)(i522 | @) * m3
JCudaTensor X1125 = X1253.asMatrix(1, true).times(m3);
// dealloc X1253
X1253.free();
// val X1248 = X1125[1<>3] * d_ReLU()(X386)/d_X385
JCudaTensor X1248 = y4.backward(X1125.unflatten(1, new int[]{128, 4, 4}), X386);
// dealloc X386
X386.free();
// V_b1cv_B = ((X1248 * d_Convolv(1,0)()/d_b1cv_B * (2 * -0.01)) + (V_b1cv_B * 0.9))
y5.backward_bias(X1248, V_b1cv_B, 2f * lrn_rate, momentum);
// b1cv_B = (V_b1cv_B + b1cv_B)
b1cv_B.update(V_b1cv_B, 1f, 1f);
// val X1293 = X1248 * d_Convolv(1,0)(b1cv_W)/d_X384
JCudaTensor X1293 = y5.backward_data(X1248, b1cv_W);
// V_b1cv_W = ((X1248 * d_Convolv(1,0)(X384)/d_b1cv_W * -0.01) + (V_b1cv_W * 0.9))
y5.backward_filter(X1248, X384, V_b1cv_W, lrn_rate, momentum);
// dealloc X1248
X1248.free();
// b1cv_W = (V_b1cv_W + (b1cv_W * (1 + (5.0E-4 * -0.01))))
b1cv_W.update(V_b1cv_W, 1f, 1f + decay * lrn_rate);
// val X973 = (X1131.copy * 0.3)
JCudaTensor X973 = X1131.clone().times_i(loss2);;
// val X285 = Pooling(3,1,1,true)(X284)
JCudaTensor X285 = y105.forward(X284);
// val X292 = Convolv(1,0)(X284,cv42_W,cv42_B)
JCudaTensor X292 = y104.forward(X284, cv42_W, cv42_B);
// val X288 = Convolv(1,0)(X284,cv44_W,cv44_B)
JCudaTensor X288 = y106.forward(X284, cv44_W, cv44_B);
// val X296 = Convolv(1,0)(X284,cv41_W,cv41_B)
JCudaTensor X296 = y32.forward(X284, cv41_W, cv41_B);
// val X286 = Convolv(1,0)(X285,cv46_W,cv46_B)
JCudaTensor X286 = y37.forward(X285, cv46_W, cv46_B);
// val X293 = ReLU()(X292)
JCudaTensor X293 = y38.forward(X292);
// val X297 = ReLU()(X296)
JCudaTensor X297 = y31.forward(X296);
// val X289 = ReLU()(X288)
JCudaTensor X289 = y40.forward(X288);
// val X287 = ReLU()(X286)
JCudaTensor X287 = y35.forward(X286);
// val X294 = Convolv(1,1)(X293,cv43_W,cv43_B)
JCudaTensor X294 = y34.forward(X293, cv43_W, cv43_B);
// val X290 = Convolv(1,2)(X289,cv45_W,cv45_B)
JCudaTensor X290 = y36.forward(X289, cv45_W, cv45_B);
// val X295 = ReLU()(X294)
JCudaTensor X295 = y33.forward(X294);
// val X291 = ReLU()(X290)
JCudaTensor X291 = y35.forward(X290);
// val X298 = Concat(X297,X295,X291,X287)
JCudaTensor X298 = y107.forward(X297,X295,X291,X287);
// val X299 = Pooling(3,1,1,true)(X298)
JCudaTensor X299 = y105.forward(X298);
// val X310 = Convolv(1,0)(X298,cv51_W,cv51_B)
JCudaTensor X310 = y32.forward(X298, cv51_W, cv51_B);
// val X306 = Convolv(1,0)(X298,cv52_W,cv52_B)
JCudaTensor X306 = y104.forward(X298, cv52_W, cv52_B);
// val X302 = Convolv(1,0)(X298,cv54_W,cv54_B)
JCudaTensor X302 = y106.forward(X298, cv54_W, cv54_B);
// val X307 = ReLU()(X306)
JCudaTensor X307 = y38.forward(X306);
// val X300 = Convolv(1,0)(X299,cv56_W,cv56_B)
JCudaTensor X300 = y37.forward(X299, cv56_W, cv56_B);
// val X303 = ReLU()(X302)
JCudaTensor X303 = y40.forward(X302);
// val X311 = ReLU()(X310)
JCudaTensor X311 = y31.forward(X310);
// val X308 = Convolv(1,1)(X307,cv53_W,cv53_B)
JCudaTensor X308 = y34.forward(X307, cv53_W, cv53_B);
// val X304 = Convolv(1,2)(X303,cv55_W,cv55_B)
JCudaTensor X304 = y36.forward(X303, cv55_W, cv55_B);
// val X301 = ReLU()(X300)
JCudaTensor X301 = y35.forward(X300);
// val X309 = ReLU()(X308)
JCudaTensor X309 = y33.forward(X308);
// val X305 = ReLU()(X304)
JCudaTensor X305 = y35.forward(X304);
// val X312 = Concat(X311,X309,X305,X301)
JCudaTensor X312 = y107.forward(X311,X309,X305,X301);
// val X316 = Convolv(1,0)(X312,cv64_W,cv64_B)
JCudaTensor X316 = y106.forward(X312, cv64_W, cv64_B);
// val X320 = Convolv(1,0)(X312,cv62_W,cv62_B)
JCudaTensor X320 = y104.forward(X312, cv62_W, cv62_B);
// val X324 = Convolv(1,0)(X312,cv61_W,cv61_B)
JCudaTensor X324 = y32.forward(X312, cv61_W, cv61_B);
// val X313 = Pooling(3,1,1,true)(X312)
JCudaTensor X313 = y105.forward(X312);
// val X321 = ReLU()(X320)
JCudaTensor X321 = y38.forward(X320);
// val X314 = Convolv(1,0)(X313,cv66_W,cv66_B)
JCudaTensor X314 = y37.forward(X313, cv66_W, cv66_B);
// val X317 = ReLU()(X316)
JCudaTensor X317 = y40.forward(X316);
// val X325 = ReLU()(X324)
JCudaTensor X325 = y31.forward(X324);
// val X322 = Convolv(1,1)(X321,cv63_W,cv63_B)
JCudaTensor X322 = y34.forward(X321, cv63_W, cv63_B);
// val X315 = ReLU()(X314)
JCudaTensor X315 = y35.forward(X314);
// val X318 = Convolv(1,2)(X317,cv65_W,cv65_B)
JCudaTensor X318 = y36.forward(X317, cv65_W, cv65_B);
// val X323 = ReLU()(X322)
JCudaTensor X323 = y33.forward(X322);
// val X319 = ReLU()(X318)
JCudaTensor X319 = y35.forward(X318);
// val X326 = Concat(X325,X323,X319,X315)
JCudaTensor X326 = y107.forward(X325,X323,X319,X315);
// val X375 = Pooling(5,3,0,false)(X326)
JCudaTensor X375 = y108.forward(X326);
// val X376 = Convolv(1,0)(X375,b2cv_W,b2cv_B)
JCudaTensor X376 = y5.forward(X375, b2cv_W, b2cv_B);
// val X377 = ReLU()(X376)
JCudaTensor X377 = y4.forward(X376);
// val X1042 = (X377[1><3])(i15 | @) * (b2fc1_W)(i16 | @)
JCudaTensor X1042 = X377.flatten(1, new int[]{128, 4, 4}).asMatrix(1, true).times(b2fc1_W.asMatrix(1, true));
// val X379 = (X1042 + (i15) => b2fc1_B)
JCudaTensor X379 = b2fc1_B.copy(256, X1042);
// val X380 = ReLU()(X379)
JCudaTensor X380 = y3.forward(X379);
// val X381 = Dropout(0.7)(X380)
JCudaTensor X381 = y6.forward(X380);
// val X1035 = (X381)(i18 | @) * (b2fc2_W)(i19 | @)
JCudaTensor X1035 = X381.asMatrix(1, true).times(b2fc2_W.asMatrix(1, true));
// val X382 = (X1035 + (i18) => b2fc2_B)
JCudaTensor X382 = b2fc2_B.copy(256, X1035);
// val X383 = LogSoftmax()(X382)
JCudaTensor X383 = y1.forward(X382);
// dealloc X382
X382.free();
// val X1283 = X973 * d_LogSoftmax()(X383)/d_X382
JCudaTensor X1283 = y1.backward(X973, X383);
// dealloc X973
X973.free();
// val m8 = (i503) => X1283[@, i503]
JCudaMatrix m8 = X1283.asMatrix(1, false);
// V_b2fc2_B = ((Sum(m8) * (2 * -0.01)) + (V_b2fc2_B * 0.9))
m8.sum(V_b2fc2_B, 2f * lrn_rate, momentum);
// b2fc2_B = (V_b2fc2_B + b2fc2_B)
b2fc2_B.update(V_b2fc2_B, 1f, 1f);
// val m18 = (i500) => b2fc2_W[@, i500]
JCudaMatrix m18 = b2fc2_W.asMatrix(1, false);
// val m15 = (i479) => X381[@, i479]
JCudaMatrix m15 = X381.asMatrix(1, false);
// V_b2fc2_W = ((m8 * m15 * -0.01) + (V_b2fc2_W * 0.9))
m8.times(m15, V_b2fc2_W, lrn_rate, momentum);
// dealloc X381
X381.free();
// b2fc2_W = (V_b2fc2_W + (b2fc2_W * (1 + (5.0E-4 * -0.01))))
b2fc2_W.update(V_b2fc2_W, 1f, 1f + decay * lrn_rate);
// val X1441 = (X1283)(i499 | @) * m18
JCudaTensor X1441 = X1283.asMatrix(1, true).times(m18);
// dealloc X1283
X1283.free();
// val X1243 = X1441 * d_Dropout(0.7)()/d_X380
JCudaTensor X1243 = y6.backward(X1441);
// dealloc X1441
X1441.free();
// val X1241 = X1243 * d_ReLU()(X380)/d_X379
JCudaTensor X1241 = y3.backward(X1243, X380);
// dealloc X380
X380.free();
// val m4 = (i482) => X1241[@, i482]
JCudaMatrix m4 = X1241.asMatrix(1, false);
// V_b2fc1_B = ((Sum(m4) * (2 * -0.01)) + (V_b2fc1_B * 0.9))
m4.sum(V_b2fc1_B, 2f * lrn_rate, momentum);
// b2fc1_B = (V_b2fc1_B + b2fc1_B)
b2fc1_B.update(V_b2fc1_B, 1f, 1f);
// val m10 = (i493) => b2fc1_W[@, i493]
JCudaMatrix m10 = b2fc1_W.asMatrix(1, false);
// val m5 = (i483) => X377[1><3][@, i483]
JCudaMatrix m5 = X377.flatten(1, new int[]{128, 4, 4}).asMatrix(1, false);
// V_b2fc1_W = ((m4 * m5 * -0.01) + (V_b2fc1_W * 0.9))
m4.times(m5, V_b2fc1_W, lrn_rate, momentum);
// b2fc1_W = (V_b2fc1_W + (b2fc1_W * (1 + (5.0E-4 * -0.01))))
b2fc1_W.update(V_b2fc1_W, 1f, 1f + decay * lrn_rate);
// val X1210 = (X1241)(i492 | @) * m10
JCudaTensor X1210 = X1241.asMatrix(1, true).times(m10);
// dealloc X1241
X1241.free();
// val X1098 = X1210[1<>3] * d_ReLU()(X377)/d_X376
JCudaTensor X1098 = y4.backward(X1210.unflatten(1, new int[]{128, 4, 4}), X377);
// dealloc X377
X377.free();
// V_b2cv_B = ((X1098 * d_Convolv(1,0)()/d_b2cv_B * (2 * -0.01)) + (V_b2cv_B * 0.9))
y5.backward_bias(X1098, V_b2cv_B, 2f * lrn_rate, momentum);
// b2cv_B = (V_b2cv_B + b2cv_B)
b2cv_B.update(V_b2cv_B, 1f, 1f);
// val X1321 = X1098 * d_Convolv(1,0)(b2cv_W)/d_X375
JCudaTensor X1321 = y5.backward_data(X1098, b2cv_W);
// V_b2cv_W = ((X1098 * d_Convolv(1,0)(X375)/d_b2cv_W * -0.01) + (V_b2cv_W * 0.9))
y5.backward_filter(X1098, X375, V_b2cv_W, lrn_rate, momentum);
// dealloc X1098
X1098.free();
// b2cv_W = (V_b2cv_W + (b2cv_W * (1 + (5.0E-4 * -0.01))))
b2cv_W.update(V_b2cv_W, 1f, 1f + decay * lrn_rate);
// val X330 = Convolv(1,0)(X326,cv74_W,cv74_B)
JCudaTensor X330 = y106.forward(X326, cv74_W, cv74_B);
// val X338 = Convolv(1,0)(X326,cv71_W,cv71_B)
JCudaTensor X338 = y32.forward(X326, cv71_W, cv71_B);
// val X327 = Pooling(3,1,1,true)(X326)
JCudaTensor X327 = y105.forward(X326);
// val X334 = Convolv(1,0)(X326,cv72_W,cv72_B)
JCudaTensor X334 = y104.forward(X326, cv72_W, cv72_B);
// val X335 = ReLU()(X334)
JCudaTensor X335 = y38.forward(X334);
// val X339 = ReLU()(X338)
JCudaTensor X339 = y31.forward(X338);
// val X328 = Convolv(1,0)(X327,cv76_W,cv76_B)
JCudaTensor X328 = y37.forward(X327, cv76_W, cv76_B);
// val X331 = ReLU()(X330)
JCudaTensor X331 = y40.forward(X330);
// val X336 = Convolv(1,1)(X335,cv73_W,cv73_B)
JCudaTensor X336 = y34.forward(X335, cv73_W, cv73_B);
// val X332 = Convolv(1,2)(X331,cv75_W,cv75_B)
JCudaTensor X332 = y36.forward(X331, cv75_W, cv75_B);
// val X329 = ReLU()(X328)
JCudaTensor X329 = y35.forward(X328);
// val X337 = ReLU()(X336)
JCudaTensor X337 = y33.forward(X336);
// val X333 = ReLU()(X332)
JCudaTensor X333 = y35.forward(X332);
// val X340 = Concat(X339,X337,X333,X329)
JCudaTensor X340 = y107.forward(X339,X337,X333,X329);
// val X341 = Pooling(3,2,1,true)(X340)
JCudaTensor X341 = y28.forward(X340);
// val X342 = Pooling(3,1,1,true)(X341)
JCudaTensor X342 = y109.forward(X341);
// val X345 = Convolv(1,0)(X341,cv84_W,cv84_B)
JCudaTensor X345 = y110.forward(X341, cv84_W, cv84_B);
// val X353 = Convolv(1,0)(X341,cv81_W,cv81_B)
JCudaTensor X353 = y12.forward(X341, cv81_W, cv81_B);
// val X349 = Convolv(1,0)(X341,cv82_W,cv82_B)
JCudaTensor X349 = y111.forward(X341, cv82_W, cv82_B);
// val X343 = Convolv(1,0)(X342,cv86_W,cv86_B)
JCudaTensor X343 = y17.forward(X342, cv86_W, cv86_B);
// val X350 = ReLU()(X349)
JCudaTensor X350 = y18.forward(X349);
// val X354 = ReLU()(X353)
JCudaTensor X354 = y11.forward(X353);
// val X346 = ReLU()(X345)
JCudaTensor X346 = y20.forward(X345);
// val X347 = Convolv(1,2)(X346,cv85_W,cv85_B)
JCudaTensor X347 = y16.forward(X346, cv85_W, cv85_B);
// val X351 = Convolv(1,1)(X350,cv83_W,cv83_B)
JCudaTensor X351 = y14.forward(X350, cv83_W, cv83_B);
// val X344 = ReLU()(X343)
JCudaTensor X344 = y15.forward(X343);
// val X348 = ReLU()(X347)
JCudaTensor X348 = y15.forward(X347);
// val X352 = ReLU()(X351)
JCudaTensor X352 = y13.forward(X351);
// val X355 = Concat(X354,X352,X348,X344)
JCudaTensor X355 = y112.forward(X354,X352,X348,X344);
// val X363 = Convolv(1,0)(X355,cv92_W,cv92_B)
JCudaTensor X363 = y111.forward(X355, cv92_W, cv92_B);
// val X359 = Convolv(1,0)(X355,cv94_W,cv94_B)
JCudaTensor X359 = y110.forward(X355, cv94_W, cv94_B);
// val X356 = Pooling(3,1,1,true)(X355)
JCudaTensor X356 = y109.forward(X355);
// val X367 = Convolv(1,0)(X355,cv91_W,cv91_B)
JCudaTensor X367 = y12.forward(X355, cv91_W, cv91_B);
// val X364 = ReLU()(X363)
JCudaTensor X364 = y18.forward(X363);
// val X357 = Convolv(1,0)(X356,cv96_W,cv96_B)
JCudaTensor X357 = y17.forward(X356, cv96_W, cv96_B);
// val X360 = ReLU()(X359)
JCudaTensor X360 = y20.forward(X359);
// val X368 = ReLU()(X367)
JCudaTensor X368 = y11.forward(X367);
// val X358 = ReLU()(X357)
JCudaTensor X358 = y15.forward(X357);
// val X361 = Convolv(1,2)(X360,cv95_W,cv95_B)
JCudaTensor X361 = y16.forward(X360, cv95_W, cv95_B);
// val X365 = Convolv(1,1)(X364,cv93_W,cv93_B)
JCudaTensor X365 = y14.forward(X364, cv93_W, cv93_B);
// val X366 = ReLU()(X365)
JCudaTensor X366 = y13.forward(X365);
// val X362 = ReLU()(X361)
JCudaTensor X362 = y15.forward(X361);
// val X369 = Concat(X368,X366,X362,X358)
JCudaTensor X369 = y112.forward(X368,X366,X362,X358);
// val X370 = Pooling(7,1,0,false)(X369)
JCudaTensor X370 = y8.forward(X369);
// val X371 = Dropout(0.4)(X370)
JCudaTensor X371 = y7.forward(X370);
// val X1039 = (X371[1><3])(i12 | @) * (fc_W)(i13 | @)
JCudaTensor X1039 = X371.flatten(1, new int[]{256, 1, 1}).asMatrix(1, true).times(fc_W.asMatrix(1, true));
// val X373 = (X1039 + (i12) => fc_B)
JCudaTensor X373 = fc_B.copy(256, X1039);
// val X374 = LogSoftmax()(X373)
JCudaTensor X374 = y1.forward(X373);
// dealloc X373
X373.free();
// val _loss = ((((0 - (X1044 . X374)) / |256|) + (((0 - (X1044 . X383)) / |256|) * 0.3)) + (((0 - (X1044 . X392)) / |256|) * 0.3))
float _loss = - X1044.dot(X374) / 256f + - X1044.dot(X383) / 256f * loss2 + - X1044.dot(X392) / 256f * loss1;
// dealloc X1044
X1044.free();
// dealloc X392
X392.free();
// dealloc X383
X383.free();
// val X1246 = X1131 * d_LogSoftmax()(X374)/d_X373
JCudaTensor X1246 = y1.backward(X1131, X374);
// dealloc X374
X374.free();
// dealloc X1131
X1131.free();
// val m1 = (i27) => X1246[@, i27]
JCudaMatrix m1 = X1246.asMatrix(1, false);
// V_fc_B = ((Sum(m1) * (2 * -0.01)) + (V_fc_B * 0.9))
m1.sum(V_fc_B, 2f * lrn_rate, momentum);
// fc_B = (V_fc_B + fc_B)
fc_B.update(V_fc_B, 1f, 1f);
// val m6 = (i470) => fc_W[@, i470]
JCudaMatrix m6 = fc_W.asMatrix(1, false);
// val m2 = (i28) => X371[1><3][@, i28]
JCudaMatrix m2 = X371.flatten(1, new int[]{256, 1, 1}).asMatrix(1, false);
// V_fc_W = ((m1 * m2 * -0.01) + (V_fc_W * 0.9))
m1.times(m2, V_fc_W, lrn_rate, momentum);
// dealloc X371
X371.free();
// fc_W = (V_fc_W + (fc_W * (1 + (5.0E-4 * -0.01))))
fc_W.update(V_fc_W, 1f, 1f + decay * lrn_rate);
// val X1193 = (X1246)(i469 | @) * m6
JCudaTensor X1193 = X1246.asMatrix(1, true).times(m6);
// dealloc X1246
X1246.free();
// val X1208 = X1193[1<>3] * d_Dropout(0.4)()/d_X370
JCudaTensor X1208 = y7.backward(X1193.unflatten(1, new int[]{256, 1, 1}));
// dealloc X1193
X1193.free();
// val X1403 = X1208 * d_Pooling(7,1,0,false)(X370,X369)/d_X369
JCudaTensor X1403 = y8.backward(X1208, X370, X369);
// dealloc X370
X370.free();
// dealloc X1208
X1208.free();
// dealloc X369
X369.free();
JCudaTensor[] y9 = y112.backward(X1403);
// val X1254 = Proj(X1403, X368,X366,X362,X358, 0)
JCudaTensor X1254 = y9[0];
// val X1207 = X1254 * d_ReLU()(X368)/d_X367
JCudaTensor X1207 = y11.backward(X1254, X368);
// V_cv91_B = ((X1207 * d_Convolv(1,0)()/d_cv91_B * (2 * -0.01)) + (V_cv91_B * 0.9))
y12.backward_bias(X1207, V_cv91_B, 2f * lrn_rate, momentum);
// cv91_B = (V_cv91_B + cv91_B)
cv91_B.update(V_cv91_B, 1f, 1f);
// val X1271 = X1207 * d_Convolv(1,0)(cv91_W)/d_X355
JCudaTensor X1271 = y12.backward_data(X1207, cv91_W);
// V_cv91_W = ((X1207 * d_Convolv(1,0)(X355)/d_cv91_W * -0.01) + (V_cv91_W * 0.9))
y12.backward_filter(X1207, X355, V_cv91_W, lrn_rate, momentum);
// dealloc X1207
X1207.free();
// cv91_W = (V_cv91_W + (cv91_W * (1 + (5.0E-4 * -0.01))))
cv91_W.update(V_cv91_W, 1f, 1f + decay * lrn_rate);
// val X1322 = Proj(X1403, X368,X366,X362,X358, 1)
JCudaTensor X1322 = y9[1];
// val X1201 = X1322 * d_ReLU()(X366)/d_X365
JCudaTensor X1201 = y13.backward(X1322, X366);
// V_cv93_B = ((X1201 * d_Convolv(1,1)()/d_cv93_B * (2 * -0.01)) + (V_cv93_B * 0.9))
y14.backward_bias(X1201, V_cv93_B, 2f * lrn_rate, momentum);
// cv93_B = (V_cv93_B + cv93_B)
cv93_B.update(V_cv93_B, 1f, 1f);
// val X1235 = X1201 * d_Convolv(1,1)(cv93_W)/d_X364
JCudaTensor X1235 = y14.backward_data(X1201, cv93_W);
// V_cv93_W = ((X1201 * d_Convolv(1,1)(X364)/d_cv93_W * -0.01) + (V_cv93_W * 0.9))
y14.backward_filter(X1201, X364, V_cv93_W, lrn_rate, momentum);
// dealloc X1201
X1201.free();
// cv93_W = (V_cv93_W + (cv93_W * (1 + (5.0E-4 * -0.01))))
cv93_W.update(V_cv93_W, 1f, 1f + decay * lrn_rate);
// val X1113 = Proj(X1403, X368,X366,X362,X358, 2)
JCudaTensor X1113 = y9[2];
// val X1109 = X1113 * d_ReLU()(X362)/d_X361
JCudaTensor X1109 = y15.backward(X1113, X362);
// V_cv95_B = ((X1109 * d_Convolv(1,2)()/d_cv95_B * (2 * -0.01)) + (V_cv95_B * 0.9))
y16.backward_bias(X1109, V_cv95_B, 2f * lrn_rate, momentum);
// cv95_B = (V_cv95_B + cv95_B)
cv95_B.update(V_cv95_B, 1f, 1f);
// val X1343 = X1109 * d_Convolv(1,2)(cv95_W)/d_X360
JCudaTensor X1343 = y16.backward_data(X1109, cv95_W);
// V_cv95_W = ((X1109 * d_Convolv(1,2)(X360)/d_cv95_W * -0.01) + (V_cv95_W * 0.9))
y16.backward_filter(X1109, X360, V_cv95_W, lrn_rate, momentum);
// dealloc X1109
X1109.free();
// cv95_W = (V_cv95_W + (cv95_W * (1 + (5.0E-4 * -0.01))))
cv95_W.update(V_cv95_W, 1f, 1f + decay * lrn_rate);
// val X1216 = Proj(X1403, X368,X366,X362,X358, 3)
JCudaTensor X1216 = y9[3];
// dealloc X366
X366.free();
// dealloc X1403
X1403.free();
// dealloc X368
X368.free();
// dealloc X362
X362.free();
// val X1351 = X1216 * d_ReLU()(X358)/d_X357
JCudaTensor X1351 = y15.backward(X1216, X358);
// dealloc X358
X358.free();
// V_cv96_B = ((X1351 * d_Convolv(1,0)()/d_cv96_B * (2 * -0.01)) + (V_cv96_B * 0.9))
y17.backward_bias(X1351, V_cv96_B, 2f * lrn_rate, momentum);
// cv96_B = (V_cv96_B + cv96_B)
cv96_B.update(V_cv96_B, 1f, 1f);
// val X1415 = X1351 * d_Convolv(1,0)(cv96_W)/d_X356
JCudaTensor X1415 = y17.backward_data(X1351, cv96_W);
// V_cv96_W = ((X1351 * d_Convolv(1,0)(X356)/d_cv96_W * -0.01) + (V_cv96_W * 0.9))
y17.backward_filter(X1351, X356, V_cv96_W, lrn_rate, momentum);
// dealloc X1351
X1351.free();
// cv96_W = (V_cv96_W + (cv96_W * (1 + (5.0E-4 * -0.01))))
cv96_W.update(V_cv96_W, 1f, 1f + decay * lrn_rate);
// val X1215 = X1235 * d_ReLU()(X364)/d_X363
JCudaTensor X1215 = y18.backward(X1235, X364);
// dealloc X364
X364.free();
// V_cv92_B = ((X1215 * d_Convolv(1,0)()/d_cv92_B * (2 * -0.01)) + (V_cv92_B * 0.9))
y111.backward_bias(X1215, V_cv92_B, 2f * lrn_rate, momentum);
// cv92_B = (V_cv92_B + cv92_B)
cv92_B.update(V_cv92_B, 1f, 1f);
// val X1273 = (X1271 + X1215 * d_Convolv(1,0)(cv92_W)/d_X355)
JCudaTensor X1273 = y111.backward_data(X1215,cv92_W, X1271);
// V_cv92_W = ((X1215 * d_Convolv(1,0)(X355)/d_cv92_W * -0.01) + (V_cv92_W * 0.9))
y111.backward_filter(X1215, X355, V_cv92_W, lrn_rate, momentum);
// dealloc X1215
X1215.free();
// cv92_W = (V_cv92_W + (cv92_W * (1 + (5.0E-4 * -0.01))))
cv92_W.update(V_cv92_W, 1f, 1f + decay * lrn_rate);
// val X1227 = X1343 * d_ReLU()(X360)/d_X359
JCudaTensor X1227 = y20.backward(X1343, X360);
// dealloc X360
X360.free();
// V_cv94_B = ((X1227 * d_Convolv(1,0)()/d_cv94_B * (2 * -0.01)) + (V_cv94_B * 0.9))
y110.backward_bias(X1227, V_cv94_B, 2f * lrn_rate, momentum);
// cv94_B = (V_cv94_B + cv94_B)
cv94_B.update(V_cv94_B, 1f, 1f);
// V_cv94_W = ((X1227 * d_Convolv(1,0)(X355)/d_cv94_W * -0.01) + (V_cv94_W * 0.9))
y110.backward_filter(X1227, X355, V_cv94_W, lrn_rate, momentum);
// val X1275 = (X1273 + X1227 * d_Convolv(1,0)(cv94_W)/d_X355)
JCudaTensor X1275 = y110.backward_data(X1227,cv94_W, X1273);
// dealloc X1227
X1227.free();
// cv94_W = (V_cv94_W + (cv94_W * (1 + (5.0E-4 * -0.01))))
cv94_W.update(V_cv94_W, 1f, 1f + decay * lrn_rate);
// val X403 = (X1275 + X1415 * d_Pooling(3,1,1,true)(X356,X355)/d_X355)
JCudaTensor X403 = y109.backward(X1415,X356,X355, X1275);
// dealloc X355
X355.free();
// dealloc X1415
X1415.free();
// dealloc X356
X356.free();
JCudaTensor[] y23 = y112.backward(X403);
// val X1055 = Proj(X403, X354,X352,X348,X344, 0)
JCudaTensor X1055 = y23[0];
// val X1176 = X1055 * d_ReLU()(X354)/d_X353
JCudaTensor X1176 = y11.backward(X1055, X354);
// V_cv81_B = ((X1176 * d_Convolv(1,0)()/d_cv81_B * (2 * -0.01)) + (V_cv81_B * 0.9))
y12.backward_bias(X1176, V_cv81_B, 2f * lrn_rate, momentum);
// cv81_B = (V_cv81_B + cv81_B)
cv81_B.update(V_cv81_B, 1f, 1f);
// val X1090 = X1176 * d_Convolv(1,0)(cv81_W)/d_X341
JCudaTensor X1090 = y12.backward_data(X1176, cv81_W);
// V_cv81_W = ((X1176 * d_Convolv(1,0)(X341)/d_cv81_W * -0.01) + (V_cv81_W * 0.9))
y12.backward_filter(X1176, X341, V_cv81_W, lrn_rate, momentum);
// dealloc X1176
X1176.free();
// cv81_W = (V_cv81_W + (cv81_W * (1 + (5.0E-4 * -0.01))))
cv81_W.update(V_cv81_W, 1f, 1f + decay * lrn_rate);
// val X1455 = Proj(X403, X354,X352,X348,X344, 1)
JCudaTensor X1455 = y23[1];
// val X1281 = X1455 * d_ReLU()(X352)/d_X351
JCudaTensor X1281 = y13.backward(X1455, X352);
// V_cv83_B = ((X1281 * d_Convolv(1,1)()/d_cv83_B * (2 * -0.01)) + (V_cv83_B * 0.9))
y14.backward_bias(X1281, V_cv83_B, 2f * lrn_rate, momentum);
// cv83_B = (V_cv83_B + cv83_B)
cv83_B.update(V_cv83_B, 1f, 1f);
// val X1405 = X1281 * d_Convolv(1,1)(cv83_W)/d_X350
JCudaTensor X1405 = y14.backward_data(X1281, cv83_W);
// V_cv83_W = ((X1281 * d_Convolv(1,1)(X350)/d_cv83_W * -0.01) + (V_cv83_W * 0.9))
y14.backward_filter(X1281, X350, V_cv83_W, lrn_rate, momentum);
// dealloc X1281
X1281.free();
// cv83_W = (V_cv83_W + (cv83_W * (1 + (5.0E-4 * -0.01))))
cv83_W.update(V_cv83_W, 1f, 1f + decay * lrn_rate);
// val X1439 = Proj(X403, X354,X352,X348,X344, 2)
JCudaTensor X1439 = y23[2];
// val X1290 = X1439 * d_ReLU()(X348)/d_X347
JCudaTensor X1290 = y15.backward(X1439, X348);
// V_cv85_B = ((X1290 * d_Convolv(1,2)()/d_cv85_B * (2 * -0.01)) + (V_cv85_B * 0.9))
y16.backward_bias(X1290, V_cv85_B, 2f * lrn_rate, momentum);
// cv85_B = (V_cv85_B + cv85_B)
cv85_B.update(V_cv85_B, 1f, 1f);
// val X1319 = X1290 * d_Convolv(1,2)(cv85_W)/d_X346
JCudaTensor X1319 = y16.backward_data(X1290, cv85_W);
// V_cv85_W = ((X1290 * d_Convolv(1,2)(X346)/d_cv85_W * -0.01) + (V_cv85_W * 0.9))
y16.backward_filter(X1290, X346, V_cv85_W, lrn_rate, momentum);
// dealloc X1290
X1290.free();
// cv85_W = (V_cv85_W + (cv85_W * (1 + (5.0E-4 * -0.01))))
cv85_W.update(V_cv85_W, 1f, 1f + decay * lrn_rate);
// val X1260 = Proj(X403, X354,X352,X348,X344, 3)
JCudaTensor X1260 = y23[3];
// dealloc X352
X352.free();
// dealloc X348
X348.free();
// dealloc X403
X403.free();
// dealloc X354
X354.free();
// val X1328 = X1260 * d_ReLU()(X344)/d_X343
JCudaTensor X1328 = y15.backward(X1260, X344);
// dealloc X344
X344.free();
// V_cv86_B = ((X1328 * d_Convolv(1,0)()/d_cv86_B * (2 * -0.01)) + (V_cv86_B * 0.9))
y17.backward_bias(X1328, V_cv86_B, 2f * lrn_rate, momentum);
// cv86_B = (V_cv86_B + cv86_B)
cv86_B.update(V_cv86_B, 1f, 1f);
// val X1228 = X1328 * d_Convolv(1,0)(cv86_W)/d_X342
JCudaTensor X1228 = y17.backward_data(X1328, cv86_W);
// V_cv86_W = ((X1328 * d_Convolv(1,0)(X342)/d_cv86_W * -0.01) + (V_cv86_W * 0.9))
y17.backward_filter(X1328, X342, V_cv86_W, lrn_rate, momentum);
// dealloc X1328
X1328.free();
// cv86_W = (V_cv86_W + (cv86_W * (1 + (5.0E-4 * -0.01))))
cv86_W.update(V_cv86_W, 1f, 1f + decay * lrn_rate);
// val X1401 = X1405 * d_ReLU()(X350)/d_X349
JCudaTensor X1401 = y18.backward(X1405, X350);
// dealloc X350
X350.free();
// V_cv82_B = ((X1401 * d_Convolv(1,0)()/d_cv82_B * (2 * -0.01)) + (V_cv82_B * 0.9))
y111.backward_bias(X1401, V_cv82_B, 2f * lrn_rate, momentum);
// cv82_B = (V_cv82_B + cv82_B)
cv82_B.update(V_cv82_B, 1f, 1f);
// val X1092 = (X1090 + X1401 * d_Convolv(1,0)(cv82_W)/d_X341)
JCudaTensor X1092 = y111.backward_data(X1401,cv82_W, X1090);
// V_cv82_W = ((X1401 * d_Convolv(1,0)(X341)/d_cv82_W * -0.01) + (V_cv82_W * 0.9))
y111.backward_filter(X1401, X341, V_cv82_W, lrn_rate, momentum);
// dealloc X1401
X1401.free();
// cv82_W = (V_cv82_W + (cv82_W * (1 + (5.0E-4 * -0.01))))
cv82_W.update(V_cv82_W, 1f, 1f + decay * lrn_rate);
// val X1073 = X1319 * d_ReLU()(X346)/d_X345
JCudaTensor X1073 = y20.backward(X1319, X346);
// dealloc X346
X346.free();
// V_cv84_B = ((X1073 * d_Convolv(1,0)()/d_cv84_B * (2 * -0.01)) + (V_cv84_B * 0.9))
y110.backward_bias(X1073, V_cv84_B, 2f * lrn_rate, momentum);
// cv84_B = (V_cv84_B + cv84_B)
cv84_B.update(V_cv84_B, 1f, 1f);
// V_cv84_W = ((X1073 * d_Convolv(1,0)(X341)/d_cv84_W * -0.01) + (V_cv84_W * 0.9))
y110.backward_filter(X1073, X341, V_cv84_W, lrn_rate, momentum);
// val X1094 = (X1092 + X1073 * d_Convolv(1,0)(cv84_W)/d_X341)
JCudaTensor X1094 = y110.backward_data(X1073,cv84_W, X1092);
// dealloc X1073
X1073.free();
// cv84_W = (V_cv84_W + (cv84_W * (1 + (5.0E-4 * -0.01))))
cv84_W.update(V_cv84_W, 1f, 1f + decay * lrn_rate);
// val X406 = (X1094 + X1228 * d_Pooling(3,1,1,true)(X342,X341)/d_X341)
JCudaTensor X406 = y109.backward(X1228,X342,X341, X1094);
// dealloc X1228
X1228.free();
// dealloc X342
X342.free();
// val X1065 = X406 * d_Pooling(3,2,1,true)(X341,X340)/d_X340
JCudaTensor X1065 = y28.backward(X406, X341, X340);
// dealloc X340
X340.free();
// dealloc X406
X406.free();
// dealloc X341
X341.free();
JCudaTensor[] y29 = y107.backward(X1065);
// val X1369 = Proj(X1065, X339,X337,X333,X329, 0)
JCudaTensor X1369 = y29[0];
// val X1340 = X1369 * d_ReLU()(X339)/d_X338
JCudaTensor X1340 = y31.backward(X1369, X339);
// V_cv71_B = ((X1340 * d_Convolv(1,0)()/d_cv71_B * (2 * -0.01)) + (V_cv71_B * 0.9))
y32.backward_bias(X1340, V_cv71_B, 2f * lrn_rate, momentum);
// cv71_B = (V_cv71_B + cv71_B)
cv71_B.update(V_cv71_B, 1f, 1f);
// val X1386 = X1340 * d_Convolv(1,0)(cv71_W)/d_X326
JCudaTensor X1386 = y32.backward_data(X1340, cv71_W);
// V_cv71_W = ((X1340 * d_Convolv(1,0)(X326)/d_cv71_W * -0.01) + (V_cv71_W * 0.9))
y32.backward_filter(X1340, X326, V_cv71_W, lrn_rate, momentum);
// dealloc X1340
X1340.free();
// cv71_W = (V_cv71_W + (cv71_W * (1 + (5.0E-4 * -0.01))))
cv71_W.update(V_cv71_W, 1f, 1f + decay * lrn_rate);
// val X1398 = Proj(X1065, X339,X337,X333,X329, 1)
JCudaTensor X1398 = y29[1];
// val X1450 = X1398 * d_ReLU()(X337)/d_X336
JCudaTensor X1450 = y33.backward(X1398, X337);
// V_cv73_B = ((X1450 * d_Convolv(1,1)()/d_cv73_B * (2 * -0.01)) + (V_cv73_B * 0.9))
y34.backward_bias(X1450, V_cv73_B, 2f * lrn_rate, momentum);
// cv73_B = (V_cv73_B + cv73_B)
cv73_B.update(V_cv73_B, 1f, 1f);
// val X1320 = X1450 * d_Convolv(1,1)(cv73_W)/d_X335
JCudaTensor X1320 = y34.backward_data(X1450, cv73_W);
// V_cv73_W = ((X1450 * d_Convolv(1,1)(X335)/d_cv73_W * -0.01) + (V_cv73_W * 0.9))
y34.backward_filter(X1450, X335, V_cv73_W, lrn_rate, momentum);
// dealloc X1450
X1450.free();
// cv73_W = (V_cv73_W + (cv73_W * (1 + (5.0E-4 * -0.01))))
cv73_W.update(V_cv73_W, 1f, 1f + decay * lrn_rate);
// val X1363 = Proj(X1065, X339,X337,X333,X329, 2)
JCudaTensor X1363 = y29[2];
// val X1057 = X1363 * d_ReLU()(X333)/d_X332
JCudaTensor X1057 = y35.backward(X1363, X333);
// V_cv75_B = ((X1057 * d_Convolv(1,2)()/d_cv75_B * (2 * -0.01)) + (V_cv75_B * 0.9))
y36.backward_bias(X1057, V_cv75_B, 2f * lrn_rate, momentum);
// cv75_B = (V_cv75_B + cv75_B)
cv75_B.update(V_cv75_B, 1f, 1f);
// val X1230 = X1057 * d_Convolv(1,2)(cv75_W)/d_X331
JCudaTensor X1230 = y36.backward_data(X1057, cv75_W);
// V_cv75_W = ((X1057 * d_Convolv(1,2)(X331)/d_cv75_W * -0.01) + (V_cv75_W * 0.9))
y36.backward_filter(X1057, X331, V_cv75_W, lrn_rate, momentum);
// dealloc X1057
X1057.free();
// cv75_W = (V_cv75_W + (cv75_W * (1 + (5.0E-4 * -0.01))))
cv75_W.update(V_cv75_W, 1f, 1f + decay * lrn_rate);
// val X1314 = Proj(X1065, X339,X337,X333,X329, 3)
JCudaTensor X1314 = y29[3];
// dealloc X1065
X1065.free();
// dealloc X339
X339.free();
// dealloc X333
X333.free();
// dealloc X337
X337.free();
// val X1338 = X1314 * d_ReLU()(X329)/d_X328
JCudaTensor X1338 = y35.backward(X1314, X329);
// dealloc X329
X329.free();
// V_cv76_B = ((X1338 * d_Convolv(1,0)()/d_cv76_B * (2 * -0.01)) + (V_cv76_B * 0.9))
y37.backward_bias(X1338, V_cv76_B, 2f * lrn_rate, momentum);
// cv76_B = (V_cv76_B + cv76_B)
cv76_B.update(V_cv76_B, 1f, 1f);
// val X1212 = X1338 * d_Convolv(1,0)(cv76_W)/d_X327
JCudaTensor X1212 = y37.backward_data(X1338, cv76_W);
// V_cv76_W = ((X1338 * d_Convolv(1,0)(X327)/d_cv76_W * -0.01) + (V_cv76_W * 0.9))
y37.backward_filter(X1338, X327, V_cv76_W, lrn_rate, momentum);
// dealloc X1338
X1338.free();
// cv76_W = (V_cv76_W + (cv76_W * (1 + (5.0E-4 * -0.01))))
cv76_W.update(V_cv76_W, 1f, 1f + decay * lrn_rate);
// val X1309 = X1320 * d_ReLU()(X335)/d_X334
JCudaTensor X1309 = y38.backward(X1320, X335);
// dealloc X335
X335.free();
// V_cv72_B = ((X1309 * d_Convolv(1,0)()/d_cv72_B * (2 * -0.01)) + (V_cv72_B * 0.9))
y104.backward_bias(X1309, V_cv72_B, 2f * lrn_rate, momentum);
// cv72_B = (V_cv72_B + cv72_B)
cv72_B.update(V_cv72_B, 1f, 1f);
// val X1388 = (X1386 + X1309 * d_Convolv(1,0)(cv72_W)/d_X326)
JCudaTensor X1388 = y104.backward_data(X1309,cv72_W, X1386);
// V_cv72_W = ((X1309 * d_Convolv(1,0)(X326)/d_cv72_W * -0.01) + (V_cv72_W * 0.9))
y104.backward_filter(X1309, X326, V_cv72_W, lrn_rate, momentum);
// dealloc X1309
X1309.free();
// cv72_W = (V_cv72_W + (cv72_W * (1 + (5.0E-4 * -0.01))))
cv72_W.update(V_cv72_W, 1f, 1f + decay * lrn_rate);
// val X1414 = X1230 * d_ReLU()(X331)/d_X330
JCudaTensor X1414 = y40.backward(X1230, X331);
// dealloc X331
X331.free();
// V_cv74_B = ((X1414 * d_Convolv(1,0)()/d_cv74_B * (2 * -0.01)) + (V_cv74_B * 0.9))
y106.backward_bias(X1414, V_cv74_B, 2f * lrn_rate, momentum);
// cv74_B = (V_cv74_B + cv74_B)
cv74_B.update(V_cv74_B, 1f, 1f);
// V_cv74_W = ((X1414 * d_Convolv(1,0)(X326)/d_cv74_W * -0.01) + (V_cv74_W * 0.9))
y106.backward_filter(X1414, X326, V_cv74_W, lrn_rate, momentum);
// val X1390 = (X1388 + X1414 * d_Convolv(1,0)(cv74_W)/d_X326)
JCudaTensor X1390 = y106.backward_data(X1414,cv74_W, X1388);
// dealloc X1414
X1414.free();
// cv74_W = (V_cv74_W + (cv74_W * (1 + (5.0E-4 * -0.01))))
cv74_W.update(V_cv74_W, 1f, 1f + decay * lrn_rate);
// val X1393 = (X1390 + X1212 * d_Pooling(3,1,1,true)(X327,X326)/d_X326)
JCudaTensor X1393 = y105.backward(X1212,X327,X326, X1390);
// dealloc X1212
X1212.free();
// dealloc X327
X327.free();
// val X410 = (X1393 + X1321 * d_Pooling(5,3,0,false)(X375,X326)/d_X326)
JCudaTensor X410 = y108.backward(X1321,X375,X326, X1393);
// dealloc X1321
X1321.free();
// dealloc X326
X326.free();
// dealloc X375
X375.free();
JCudaTensor[] y44 = y107.backward(X410);
// val X1303 = Proj(X410, X325,X323,X319,X315, 0)
JCudaTensor X1303 = y44[0];
// val X1239 = X1303 * d_ReLU()(X325)/d_X324
JCudaTensor X1239 = y31.backward(X1303, X325);
// V_cv61_B = ((X1239 * d_Convolv(1,0)()/d_cv61_B * (2 * -0.01)) + (V_cv61_B * 0.9))
y32.backward_bias(X1239, V_cv61_B, 2f * lrn_rate, momentum);
// cv61_B = (V_cv61_B + cv61_B)
cv61_B.update(V_cv61_B, 1f, 1f);
// val X1115 = X1239 * d_Convolv(1,0)(cv61_W)/d_X312
JCudaTensor X1115 = y32.backward_data(X1239, cv61_W);
// V_cv61_W = ((X1239 * d_Convolv(1,0)(X312)/d_cv61_W * -0.01) + (V_cv61_W * 0.9))
y32.backward_filter(X1239, X312, V_cv61_W, lrn_rate, momentum);
// dealloc X1239
X1239.free();
// cv61_W = (V_cv61_W + (cv61_W * (1 + (5.0E-4 * -0.01))))
cv61_W.update(V_cv61_W, 1f, 1f + decay * lrn_rate);
// val X1442 = Proj(X410, X325,X323,X319,X315, 1)
JCudaTensor X1442 = y44[1];
// val X1140 = X1442 * d_ReLU()(X323)/d_X322
JCudaTensor X1140 = y33.backward(X1442, X323);
// V_cv63_B = ((X1140 * d_Convolv(1,1)()/d_cv63_B * (2 * -0.01)) + (V_cv63_B * 0.9))
y34.backward_bias(X1140, V_cv63_B, 2f * lrn_rate, momentum);
// cv63_B = (V_cv63_B + cv63_B)
cv63_B.update(V_cv63_B, 1f, 1f);
// val X1311 = X1140 * d_Convolv(1,1)(cv63_W)/d_X321
JCudaTensor X1311 = y34.backward_data(X1140, cv63_W);
// V_cv63_W = ((X1140 * d_Convolv(1,1)(X321)/d_cv63_W * -0.01) + (V_cv63_W * 0.9))
y34.backward_filter(X1140, X321, V_cv63_W, lrn_rate, momentum);
// dealloc X1140
X1140.free();
// cv63_W = (V_cv63_W + (cv63_W * (1 + (5.0E-4 * -0.01))))
cv63_W.update(V_cv63_W, 1f, 1f + decay * lrn_rate);
// val X1048 = Proj(X410, X325,X323,X319,X315, 2)
JCudaTensor X1048 = y44[2];
// val X1292 = X1048 * d_ReLU()(X319)/d_X318
JCudaTensor X1292 = y35.backward(X1048, X319);
// V_cv65_B = ((X1292 * d_Convolv(1,2)()/d_cv65_B * (2 * -0.01)) + (V_cv65_B * 0.9))
y36.backward_bias(X1292, V_cv65_B, 2f * lrn_rate, momentum);
// cv65_B = (V_cv65_B + cv65_B)
cv65_B.update(V_cv65_B, 1f, 1f);
// val X1304 = X1292 * d_Convolv(1,2)(cv65_W)/d_X317
JCudaTensor X1304 = y36.backward_data(X1292, cv65_W);
// V_cv65_W = ((X1292 * d_Convolv(1,2)(X317)/d_cv65_W * -0.01) + (V_cv65_W * 0.9))
y36.backward_filter(X1292, X317, V_cv65_W, lrn_rate, momentum);
// dealloc X1292
X1292.free();
// cv65_W = (V_cv65_W + (cv65_W * (1 + (5.0E-4 * -0.01))))
cv65_W.update(V_cv65_W, 1f, 1f + decay * lrn_rate);
// val X1138 = Proj(X410, X325,X323,X319,X315, 3)
JCudaTensor X1138 = y44[3];
// dealloc X325
X325.free();
// dealloc X323
X323.free();
// dealloc X319
X319.free();
// dealloc X410
X410.free();
// val X1437 = X1138 * d_ReLU()(X315)/d_X314
JCudaTensor X1437 = y35.backward(X1138, X315);
// dealloc X315
X315.free();
// V_cv66_B = ((X1437 * d_Convolv(1,0)()/d_cv66_B * (2 * -0.01)) + (V_cv66_B * 0.9))
y37.backward_bias(X1437, V_cv66_B, 2f * lrn_rate, momentum);
// cv66_B = (V_cv66_B + cv66_B)
cv66_B.update(V_cv66_B, 1f, 1f);
// val X1177 = X1437 * d_Convolv(1,0)(cv66_W)/d_X313
JCudaTensor X1177 = y37.backward_data(X1437, cv66_W);
// V_cv66_W = ((X1437 * d_Convolv(1,0)(X313)/d_cv66_W * -0.01) + (V_cv66_W * 0.9))
y37.backward_filter(X1437, X313, V_cv66_W, lrn_rate, momentum);
// dealloc X1437
X1437.free();
// cv66_W = (V_cv66_W + (cv66_W * (1 + (5.0E-4 * -0.01))))
cv66_W.update(V_cv66_W, 1f, 1f + decay * lrn_rate);
// val X1167 = X1311 * d_ReLU()(X321)/d_X320
JCudaTensor X1167 = y38.backward(X1311, X321);
// dealloc X321
X321.free();
// V_cv62_B = ((X1167 * d_Convolv(1,0)()/d_cv62_B * (2 * -0.01)) + (V_cv62_B * 0.9))
y104.backward_bias(X1167, V_cv62_B, 2f * lrn_rate, momentum);
// cv62_B = (V_cv62_B + cv62_B)
cv62_B.update(V_cv62_B, 1f, 1f);
// val X1117 = (X1115 + X1167 * d_Convolv(1,0)(cv62_W)/d_X312)
JCudaTensor X1117 = y104.backward_data(X1167,cv62_W, X1115);
// V_cv62_W = ((X1167 * d_Convolv(1,0)(X312)/d_cv62_W * -0.01) + (V_cv62_W * 0.9))
y104.backward_filter(X1167, X312, V_cv62_W, lrn_rate, momentum);
// dealloc X1167
X1167.free();
// cv62_W = (V_cv62_W + (cv62_W * (1 + (5.0E-4 * -0.01))))
cv62_W.update(V_cv62_W, 1f, 1f + decay * lrn_rate);
// val X1331 = X1304 * d_ReLU()(X317)/d_X316
JCudaTensor X1331 = y40.backward(X1304, X317);
// dealloc X317
X317.free();
// V_cv64_B = ((X1331 * d_Convolv(1,0)()/d_cv64_B * (2 * -0.01)) + (V_cv64_B * 0.9))
y106.backward_bias(X1331, V_cv64_B, 2f * lrn_rate, momentum);
// cv64_B = (V_cv64_B + cv64_B)
cv64_B.update(V_cv64_B, 1f, 1f);
// V_cv64_W = ((X1331 * d_Convolv(1,0)(X312)/d_cv64_W * -0.01) + (V_cv64_W * 0.9))
y106.backward_filter(X1331, X312, V_cv64_W, lrn_rate, momentum);
// val X1119 = (X1117 + X1331 * d_Convolv(1,0)(cv64_W)/d_X312)
JCudaTensor X1119 = y106.backward_data(X1331,cv64_W, X1117);
// dealloc X1331
X1331.free();
// cv64_W = (V_cv64_W + (cv64_W * (1 + (5.0E-4 * -0.01))))
cv64_W.update(V_cv64_W, 1f, 1f + decay * lrn_rate);
// val X413 = (X1119 + X1177 * d_Pooling(3,1,1,true)(X313,X312)/d_X312)
JCudaTensor X413 = y105.backward(X1177,X313,X312, X1119);
// dealloc X312
X312.free();
// dealloc X313
X313.free();
// dealloc X1177
X1177.free();
JCudaTensor[] y49 = y107.backward(X413);
// val X1133 = Proj(X413, X311,X309,X305,X301, 0)
JCudaTensor X1133 = y49[0];
// val X1267 = X1133 * d_ReLU()(X311)/d_X310
JCudaTensor X1267 = y31.backward(X1133, X311);
// V_cv51_B = ((X1267 * d_Convolv(1,0)()/d_cv51_B * (2 * -0.01)) + (V_cv51_B * 0.9))
y32.backward_bias(X1267, V_cv51_B, 2f * lrn_rate, momentum);
// cv51_B = (V_cv51_B + cv51_B)
cv51_B.update(V_cv51_B, 1f, 1f);
// val X1373 = X1267 * d_Convolv(1,0)(cv51_W)/d_X298
JCudaTensor X1373 = y32.backward_data(X1267, cv51_W);
// V_cv51_W = ((X1267 * d_Convolv(1,0)(X298)/d_cv51_W * -0.01) + (V_cv51_W * 0.9))
y32.backward_filter(X1267, X298, V_cv51_W, lrn_rate, momentum);
// dealloc X1267
X1267.free();
// cv51_W = (V_cv51_W + (cv51_W * (1 + (5.0E-4 * -0.01))))
cv51_W.update(V_cv51_W, 1f, 1f + decay * lrn_rate);
// val X1211 = Proj(X413, X311,X309,X305,X301, 1)
JCudaTensor X1211 = y49[1];
// val X1078 = X1211 * d_ReLU()(X309)/d_X308
JCudaTensor X1078 = y33.backward(X1211, X309);
// V_cv53_B = ((X1078 * d_Convolv(1,1)()/d_cv53_B * (2 * -0.01)) + (V_cv53_B * 0.9))
y34.backward_bias(X1078, V_cv53_B, 2f * lrn_rate, momentum);
// cv53_B = (V_cv53_B + cv53_B)
cv53_B.update(V_cv53_B, 1f, 1f);
// val X1059 = X1078 * d_Convolv(1,1)(cv53_W)/d_X307
JCudaTensor X1059 = y34.backward_data(X1078, cv53_W);
// V_cv53_W = ((X1078 * d_Convolv(1,1)(X307)/d_cv53_W * -0.01) + (V_cv53_W * 0.9))
y34.backward_filter(X1078, X307, V_cv53_W, lrn_rate, momentum);
// dealloc X1078
X1078.free();
// cv53_W = (V_cv53_W + (cv53_W * (1 + (5.0E-4 * -0.01))))
cv53_W.update(V_cv53_W, 1f, 1f + decay * lrn_rate);
// val X1465 = Proj(X413, X311,X309,X305,X301, 2)
JCudaTensor X1465 = y49[2];
// val X1190 = X1465 * d_ReLU()(X305)/d_X304
JCudaTensor X1190 = y35.backward(X1465, X305);
// V_cv55_B = ((X1190 * d_Convolv(1,2)()/d_cv55_B * (2 * -0.01)) + (V_cv55_B * 0.9))
y36.backward_bias(X1190, V_cv55_B, 2f * lrn_rate, momentum);
// cv55_B = (V_cv55_B + cv55_B)
cv55_B.update(V_cv55_B, 1f, 1f);
// val X1068 = X1190 * d_Convolv(1,2)(cv55_W)/d_X303
JCudaTensor X1068 = y36.backward_data(X1190, cv55_W);
// V_cv55_W = ((X1190 * d_Convolv(1,2)(X303)/d_cv55_W * -0.01) + (V_cv55_W * 0.9))
y36.backward_filter(X1190, X303, V_cv55_W, lrn_rate, momentum);
// dealloc X1190
X1190.free();
// cv55_W = (V_cv55_W + (cv55_W * (1 + (5.0E-4 * -0.01))))
cv55_W.update(V_cv55_W, 1f, 1f + decay * lrn_rate);
// val X1349 = Proj(X413, X311,X309,X305,X301, 3)
JCudaTensor X1349 = y49[3];
// dealloc X311
X311.free();
// dealloc X413
X413.free();
// dealloc X305
X305.free();
// dealloc X309
X309.free();
// val X1083 = X1349 * d_ReLU()(X301)/d_X300
JCudaTensor X1083 = y35.backward(X1349, X301);
// dealloc X301
X301.free();
// V_cv56_B = ((X1083 * d_Convolv(1,0)()/d_cv56_B * (2 * -0.01)) + (V_cv56_B * 0.9))
y37.backward_bias(X1083, V_cv56_B, 2f * lrn_rate, momentum);
// cv56_B = (V_cv56_B + cv56_B)
cv56_B.update(V_cv56_B, 1f, 1f);
// val X1408 = X1083 * d_Convolv(1,0)(cv56_W)/d_X299
JCudaTensor X1408 = y37.backward_data(X1083, cv56_W);
// V_cv56_W = ((X1083 * d_Convolv(1,0)(X299)/d_cv56_W * -0.01) + (V_cv56_W * 0.9))
y37.backward_filter(X1083, X299, V_cv56_W, lrn_rate, momentum);
// dealloc X1083
X1083.free();
// cv56_W = (V_cv56_W + (cv56_W * (1 + (5.0E-4 * -0.01))))
cv56_W.update(V_cv56_W, 1f, 1f + decay * lrn_rate);
// val X1432 = X1059 * d_ReLU()(X307)/d_X306
JCudaTensor X1432 = y38.backward(X1059, X307);
// dealloc X307
X307.free();
// V_cv52_B = ((X1432 * d_Convolv(1,0)()/d_cv52_B * (2 * -0.01)) + (V_cv52_B * 0.9))
y104.backward_bias(X1432, V_cv52_B, 2f * lrn_rate, momentum);
// cv52_B = (V_cv52_B + cv52_B)
cv52_B.update(V_cv52_B, 1f, 1f);
// val X1375 = (X1373 + X1432 * d_Convolv(1,0)(cv52_W)/d_X298)
JCudaTensor X1375 = y104.backward_data(X1432,cv52_W, X1373);
// V_cv52_W = ((X1432 * d_Convolv(1,0)(X298)/d_cv52_W * -0.01) + (V_cv52_W * 0.9))
y104.backward_filter(X1432, X298, V_cv52_W, lrn_rate, momentum);
// dealloc X1432
X1432.free();
// cv52_W = (V_cv52_W + (cv52_W * (1 + (5.0E-4 * -0.01))))
cv52_W.update(V_cv52_W, 1f, 1f + decay * lrn_rate);
// val X1250 = X1068 * d_ReLU()(X303)/d_X302
JCudaTensor X1250 = y40.backward(X1068, X303);
// dealloc X303
X303.free();
// V_cv54_B = ((X1250 * d_Convolv(1,0)()/d_cv54_B * (2 * -0.01)) + (V_cv54_B * 0.9))
y106.backward_bias(X1250, V_cv54_B, 2f * lrn_rate, momentum);
// cv54_B = (V_cv54_B + cv54_B)
cv54_B.update(V_cv54_B, 1f, 1f);
// V_cv54_W = ((X1250 * d_Convolv(1,0)(X298)/d_cv54_W * -0.01) + (V_cv54_W * 0.9))
y106.backward_filter(X1250, X298, V_cv54_W, lrn_rate, momentum);
// val X1377 = (X1375 + X1250 * d_Convolv(1,0)(cv54_W)/d_X298)
JCudaTensor X1377 = y106.backward_data(X1250,cv54_W, X1375);
// dealloc X1250
X1250.free();
// cv54_W = (V_cv54_W + (cv54_W * (1 + (5.0E-4 * -0.01))))
cv54_W.update(V_cv54_W, 1f, 1f + decay * lrn_rate);
// val X416 = (X1377 + X1408 * d_Pooling(3,1,1,true)(X299,X298)/d_X298)
JCudaTensor X416 = y105.backward(X1408,X299,X298, X1377);
// dealloc X298
X298.free();
// dealloc X299
X299.free();
// dealloc X1408
X1408.free();
JCudaTensor[] y54 = y107.backward(X416);
// val X1217 = Proj(X416, X297,X295,X291,X287, 0)
JCudaTensor X1217 = y54[0];
// val X1412 = X1217 * d_ReLU()(X297)/d_X296
JCudaTensor X1412 = y31.backward(X1217, X297);
// V_cv41_B = ((X1412 * d_Convolv(1,0)()/d_cv41_B * (2 * -0.01)) + (V_cv41_B * 0.9))
y32.backward_bias(X1412, V_cv41_B, 2f * lrn_rate, momentum);
// cv41_B = (V_cv41_B + cv41_B)
cv41_B.update(V_cv41_B, 1f, 1f);
// val X1353 = X1412 * d_Convolv(1,0)(cv41_W)/d_X284
JCudaTensor X1353 = y32.backward_data(X1412, cv41_W);
// V_cv41_W = ((X1412 * d_Convolv(1,0)(X284)/d_cv41_W * -0.01) + (V_cv41_W * 0.9))
y32.backward_filter(X1412, X284, V_cv41_W, lrn_rate, momentum);
// dealloc X1412
X1412.free();
// cv41_W = (V_cv41_W + (cv41_W * (1 + (5.0E-4 * -0.01))))
cv41_W.update(V_cv41_W, 1f, 1f + decay * lrn_rate);
// val X1221 = Proj(X416, X297,X295,X291,X287, 1)
JCudaTensor X1221 = y54[1];
// val X1075 = X1221 * d_ReLU()(X295)/d_X294
JCudaTensor X1075 = y33.backward(X1221, X295);
// V_cv43_B = ((X1075 * d_Convolv(1,1)()/d_cv43_B * (2 * -0.01)) + (V_cv43_B * 0.9))
y34.backward_bias(X1075, V_cv43_B, 2f * lrn_rate, momentum);
// cv43_B = (V_cv43_B + cv43_B)
cv43_B.update(V_cv43_B, 1f, 1f);
// val X1049 = X1075 * d_Convolv(1,1)(cv43_W)/d_X293
JCudaTensor X1049 = y34.backward_data(X1075, cv43_W);
// V_cv43_W = ((X1075 * d_Convolv(1,1)(X293)/d_cv43_W * -0.01) + (V_cv43_W * 0.9))
y34.backward_filter(X1075, X293, V_cv43_W, lrn_rate, momentum);
// dealloc X1075
X1075.free();
// cv43_W = (V_cv43_W + (cv43_W * (1 + (5.0E-4 * -0.01))))
cv43_W.update(V_cv43_W, 1f, 1f + decay * lrn_rate);
// val X1397 = Proj(X416, X297,X295,X291,X287, 2)
JCudaTensor X1397 = y54[2];
// val X1198 = X1397 * d_ReLU()(X291)/d_X290
JCudaTensor X1198 = y35.backward(X1397, X291);
// V_cv45_B = ((X1198 * d_Convolv(1,2)()/d_cv45_B * (2 * -0.01)) + (V_cv45_B * 0.9))
y36.backward_bias(X1198, V_cv45_B, 2f * lrn_rate, momentum);
// cv45_B = (V_cv45_B + cv45_B)
cv45_B.update(V_cv45_B, 1f, 1f);
// val X1204 = X1198 * d_Convolv(1,2)(cv45_W)/d_X289
JCudaTensor X1204 = y36.backward_data(X1198, cv45_W);
// V_cv45_W = ((X1198 * d_Convolv(1,2)(X289)/d_cv45_W * -0.01) + (V_cv45_W * 0.9))
y36.backward_filter(X1198, X289, V_cv45_W, lrn_rate, momentum);
// dealloc X1198
X1198.free();
// cv45_W = (V_cv45_W + (cv45_W * (1 + (5.0E-4 * -0.01))))
cv45_W.update(V_cv45_W, 1f, 1f + decay * lrn_rate);
// val X1084 = Proj(X416, X297,X295,X291,X287, 3)
JCudaTensor X1084 = y54[3];
// dealloc X291
X291.free();
// dealloc X297
X297.free();
// dealloc X295
X295.free();
// dealloc X416
X416.free();
// val X1052 = X1084 * d_ReLU()(X287)/d_X286
JCudaTensor X1052 = y35.backward(X1084, X287);
// dealloc X287
X287.free();
// V_cv46_B = ((X1052 * d_Convolv(1,0)()/d_cv46_B * (2 * -0.01)) + (V_cv46_B * 0.9))
y37.backward_bias(X1052, V_cv46_B, 2f * lrn_rate, momentum);
// cv46_B = (V_cv46_B + cv46_B)
cv46_B.update(V_cv46_B, 1f, 1f);
// val X1126 = X1052 * d_Convolv(1,0)(cv46_W)/d_X285
JCudaTensor X1126 = y37.backward_data(X1052, cv46_W);
// V_cv46_W = ((X1052 * d_Convolv(1,0)(X285)/d_cv46_W * -0.01) + (V_cv46_W * 0.9))
y37.backward_filter(X1052, X285, V_cv46_W, lrn_rate, momentum);
// dealloc X1052
X1052.free();
// cv46_W = (V_cv46_W + (cv46_W * (1 + (5.0E-4 * -0.01))))
cv46_W.update(V_cv46_W, 1f, 1f + decay * lrn_rate);
// val X1367 = X1049 * d_ReLU()(X293)/d_X292
JCudaTensor X1367 = y38.backward(X1049, X293);
// dealloc X293
X293.free();
// V_cv42_B = ((X1367 * d_Convolv(1,0)()/d_cv42_B * (2 * -0.01)) + (V_cv42_B * 0.9))
y104.backward_bias(X1367, V_cv42_B, 2f * lrn_rate, momentum);
// cv42_B = (V_cv42_B + cv42_B)
cv42_B.update(V_cv42_B, 1f, 1f);
// val X1355 = (X1353 + X1367 * d_Convolv(1,0)(cv42_W)/d_X284)
JCudaTensor X1355 = y104.backward_data(X1367,cv42_W, X1353);
// V_cv42_W = ((X1367 * d_Convolv(1,0)(X284)/d_cv42_W * -0.01) + (V_cv42_W * 0.9))
y104.backward_filter(X1367, X284, V_cv42_W, lrn_rate, momentum);
// dealloc X1367
X1367.free();
// cv42_W = (V_cv42_W + (cv42_W * (1 + (5.0E-4 * -0.01))))
cv42_W.update(V_cv42_W, 1f, 1f + decay * lrn_rate);
// val X1107 = X1204 * d_ReLU()(X289)/d_X288
JCudaTensor X1107 = y40.backward(X1204, X289);
// dealloc X289
X289.free();
// V_cv44_B = ((X1107 * d_Convolv(1,0)()/d_cv44_B * (2 * -0.01)) + (V_cv44_B * 0.9))
y106.backward_bias(X1107, V_cv44_B, 2f * lrn_rate, momentum);
// cv44_B = (V_cv44_B + cv44_B)
cv44_B.update(V_cv44_B, 1f, 1f);
// V_cv44_W = ((X1107 * d_Convolv(1,0)(X284)/d_cv44_W * -0.01) + (V_cv44_W * 0.9))
y106.backward_filter(X1107, X284, V_cv44_W, lrn_rate, momentum);
// val X1357 = (X1355 + X1107 * d_Convolv(1,0)(cv44_W)/d_X284)
JCudaTensor X1357 = y106.backward_data(X1107,cv44_W, X1355);
// dealloc X1107
X1107.free();
// cv44_W = (V_cv44_W + (cv44_W * (1 + (5.0E-4 * -0.01))))
cv44_W.update(V_cv44_W, 1f, 1f + decay * lrn_rate);
// val X1360 = (X1357 + X1126 * d_Pooling(3,1,1,true)(X285,X284)/d_X284)
JCudaTensor X1360 = y105.backward(X1126,X285,X284, X1357);
// dealloc X1126
X1126.free();
// dealloc X285
X285.free();
// val X419 = (X1360 + X1293 * d_Pooling(5,3,0,false)(X384,X284)/d_X284)
JCudaTensor X419 = y108.backward(X1293,X384,X284, X1360);
// dealloc X1293
X1293.free();
// dealloc X284
X284.free();
// dealloc X384
X384.free();
JCudaTensor[] y60 = y107.backward(X419);
// val X1336 = Proj(X419, X283,X281,X277,X273, 0)
JCudaTensor X1336 = y60[0];
// val X1173 = X1336 * d_ReLU()(X283)/d_X282
JCudaTensor X1173 = y31.backward(X1336, X283);
// V_cv31_B = ((X1173 * d_Convolv(1,0)()/d_cv31_B * (2 * -0.01)) + (V_cv31_B * 0.9))
y32.backward_bias(X1173, V_cv31_B, 2f * lrn_rate, momentum);
// cv31_B = (V_cv31_B + cv31_B)
cv31_B.update(V_cv31_B, 1f, 1f);
// val X1153 = X1173 * d_Convolv(1,0)(cv31_W)/d_X270
JCudaTensor X1153 = y32.backward_data(X1173, cv31_W);
// V_cv31_W = ((X1173 * d_Convolv(1,0)(X270)/d_cv31_W * -0.01) + (V_cv31_W * 0.9))
y32.backward_filter(X1173, X270, V_cv31_W, lrn_rate, momentum);
// dealloc X1173
X1173.free();
// cv31_W = (V_cv31_W + (cv31_W * (1 + (5.0E-4 * -0.01))))
cv31_W.update(V_cv31_W, 1f, 1f + decay * lrn_rate);
// val X1464 = Proj(X419, X283,X281,X277,X273, 1)
JCudaTensor X1464 = y60[1];
// val X1162 = X1464 * d_ReLU()(X281)/d_X280
JCudaTensor X1162 = y33.backward(X1464, X281);
// V_cv33_B = ((X1162 * d_Convolv(1,1)()/d_cv33_B * (2 * -0.01)) + (V_cv33_B * 0.9))
y34.backward_bias(X1162, V_cv33_B, 2f * lrn_rate, momentum);
// cv33_B = (V_cv33_B + cv33_B)
cv33_B.update(V_cv33_B, 1f, 1f);
// val X1180 = X1162 * d_Convolv(1,1)(cv33_W)/d_X279
JCudaTensor X1180 = y34.backward_data(X1162, cv33_W);
// V_cv33_W = ((X1162 * d_Convolv(1,1)(X279)/d_cv33_W * -0.01) + (V_cv33_W * 0.9))
y34.backward_filter(X1162, X279, V_cv33_W, lrn_rate, momentum);
// dealloc X1162
X1162.free();
// cv33_W = (V_cv33_W + (cv33_W * (1 + (5.0E-4 * -0.01))))
cv33_W.update(V_cv33_W, 1f, 1f + decay * lrn_rate);
// val X1259 = Proj(X419, X283,X281,X277,X273, 2)
JCudaTensor X1259 = y60[2];
// val X1326 = X1259 * d_ReLU()(X277)/d_X276
JCudaTensor X1326 = y35.backward(X1259, X277);
// V_cv35_B = ((X1326 * d_Convolv(1,2)()/d_cv35_B * (2 * -0.01)) + (V_cv35_B * 0.9))
y36.backward_bias(X1326, V_cv35_B, 2f * lrn_rate, momentum);
// cv35_B = (V_cv35_B + cv35_B)
cv35_B.update(V_cv35_B, 1f, 1f);
// val X1448 = X1326 * d_Convolv(1,2)(cv35_W)/d_X275
JCudaTensor X1448 = y36.backward_data(X1326, cv35_W);
// V_cv35_W = ((X1326 * d_Convolv(1,2)(X275)/d_cv35_W * -0.01) + (V_cv35_W * 0.9))
y36.backward_filter(X1326, X275, V_cv35_W, lrn_rate, momentum);
// dealloc X1326
X1326.free();
// cv35_W = (V_cv35_W + (cv35_W * (1 + (5.0E-4 * -0.01))))
cv35_W.update(V_cv35_W, 1f, 1f + decay * lrn_rate);
// val X1186 = Proj(X419, X283,X281,X277,X273, 3)
JCudaTensor X1186 = y60[3];
// dealloc X281
X281.free();
// dealloc X283
X283.free();
// dealloc X419
X419.free();
// dealloc X277
X277.free();
// val X1262 = X1186 * d_ReLU()(X273)/d_X272
JCudaTensor X1262 = y35.backward(X1186, X273);
// dealloc X273
X273.free();
// V_cv36_B = ((X1262 * d_Convolv(1,0)()/d_cv36_B * (2 * -0.01)) + (V_cv36_B * 0.9))
y37.backward_bias(X1262, V_cv36_B, 2f * lrn_rate, momentum);
// cv36_B = (V_cv36_B + cv36_B)
cv36_B.update(V_cv36_B, 1f, 1f);
// val X1184 = X1262 * d_Convolv(1,0)(cv36_W)/d_X271
JCudaTensor X1184 = y37.backward_data(X1262, cv36_W);
// V_cv36_W = ((X1262 * d_Convolv(1,0)(X271)/d_cv36_W * -0.01) + (V_cv36_W * 0.9))
y37.backward_filter(X1262, X271, V_cv36_W, lrn_rate, momentum);
// dealloc X1262
X1262.free();
// cv36_W = (V_cv36_W + (cv36_W * (1 + (5.0E-4 * -0.01))))
cv36_W.update(V_cv36_W, 1f, 1f + decay * lrn_rate);
// val X1444 = X1180 * d_ReLU()(X279)/d_X278
JCudaTensor X1444 = y38.backward(X1180, X279);
// dealloc X279
X279.free();
// V_cv32_B = ((X1444 * d_Convolv(1,0)()/d_cv32_B * (2 * -0.01)) + (V_cv32_B * 0.9))
y104.backward_bias(X1444, V_cv32_B, 2f * lrn_rate, momentum);
// cv32_B = (V_cv32_B + cv32_B)
cv32_B.update(V_cv32_B, 1f, 1f);
// val X1155 = (X1153 + X1444 * d_Convolv(1,0)(cv32_W)/d_X270)
JCudaTensor X1155 = y104.backward_data(X1444,cv32_W, X1153);
// V_cv32_W = ((X1444 * d_Convolv(1,0)(X270)/d_cv32_W * -0.01) + (V_cv32_W * 0.9))
y104.backward_filter(X1444, X270, V_cv32_W, lrn_rate, momentum);
// dealloc X1444
X1444.free();
// cv32_W = (V_cv32_W + (cv32_W * (1 + (5.0E-4 * -0.01))))
cv32_W.update(V_cv32_W, 1f, 1f + decay * lrn_rate);
// val X1101 = X1448 * d_ReLU()(X275)/d_X274
JCudaTensor X1101 = y40.backward(X1448, X275);
// dealloc X275
X275.free();
// V_cv34_B = ((X1101 * d_Convolv(1,0)()/d_cv34_B * (2 * -0.01)) + (V_cv34_B * 0.9))
y106.backward_bias(X1101, V_cv34_B, 2f * lrn_rate, momentum);
// cv34_B = (V_cv34_B + cv34_B)
cv34_B.update(V_cv34_B, 1f, 1f);
// V_cv34_W = ((X1101 * d_Convolv(1,0)(X270)/d_cv34_W * -0.01) + (V_cv34_W * 0.9))
y106.backward_filter(X1101, X270, V_cv34_W, lrn_rate, momentum);
// val X1157 = (X1155 + X1101 * d_Convolv(1,0)(cv34_W)/d_X270)
JCudaTensor X1157 = y106.backward_data(X1101,cv34_W, X1155);
// dealloc X1101
X1101.free();
// cv34_W = (V_cv34_W + (cv34_W * (1 + (5.0E-4 * -0.01))))
cv34_W.update(V_cv34_W, 1f, 1f + decay * lrn_rate);
// val X422 = (X1157 + X1184 * d_Pooling(3,1,1,true)(X271,X270)/d_X270)
JCudaTensor X422 = y105.backward(X1184,X271,X270, X1157);
// dealloc X271
X271.free();
// dealloc X1184
X1184.free();
// val X1410 = X422 * d_Pooling(3,2,1,true)(X270,X269)/d_X269
JCudaTensor X1410 = y65.backward(X422, X270, X269);
// dealloc X270
X270.free();
// dealloc X422
X422.free();
// dealloc X269
X269.free();
JCudaTensor[] y66 = y100.backward(X1410);
// val X1086 = Proj(X1410, X268,X266,X262,X258, 0)
JCudaTensor X1086 = y66[0];
// val X1382 = X1086 * d_ReLU()(X268)/d_X267
JCudaTensor X1382 = y68.backward(X1086, X268);
// V_cv21_B = ((X1382 * d_Convolv(1,0)()/d_cv21_B * (2 * -0.01)) + (V_cv21_B * 0.9))
y69.backward_bias(X1382, V_cv21_B, 2f * lrn_rate, momentum);
// cv21_B = (V_cv21_B + cv21_B)
cv21_B.update(V_cv21_B, 1f, 1f);
// val X1416 = X1382 * d_Convolv(1,0)(cv21_W)/d_X255
JCudaTensor X1416 = y69.backward_data(X1382, cv21_W);
// V_cv21_W = ((X1382 * d_Convolv(1,0)(X255)/d_cv21_W * -0.01) + (V_cv21_W * 0.9))
y69.backward_filter(X1382, X255, V_cv21_W, lrn_rate, momentum);
// dealloc X1382
X1382.free();
// cv21_W = (V_cv21_W + (cv21_W * (1 + (5.0E-4 * -0.01))))
cv21_W.update(V_cv21_W, 1f, 1f + decay * lrn_rate);
// val X1062 = Proj(X1410, X268,X266,X262,X258, 1)
JCudaTensor X1062 = y66[1];
// val X1080 = X1062 * d_ReLU()(X266)/d_X265
JCudaTensor X1080 = y70.backward(X1062, X266);
// V_cv23_B = ((X1080 * d_Convolv(1,1)()/d_cv23_B * (2 * -0.01)) + (V_cv23_B * 0.9))
y71.backward_bias(X1080, V_cv23_B, 2f * lrn_rate, momentum);
// cv23_B = (V_cv23_B + cv23_B)
cv23_B.update(V_cv23_B, 1f, 1f);
// val X1452 = X1080 * d_Convolv(1,1)(cv23_W)/d_X264
JCudaTensor X1452 = y71.backward_data(X1080, cv23_W);
// V_cv23_W = ((X1080 * d_Convolv(1,1)(X264)/d_cv23_W * -0.01) + (V_cv23_W * 0.9))
y71.backward_filter(X1080, X264, V_cv23_W, lrn_rate, momentum);
// dealloc X1080
X1080.free();
// cv23_W = (V_cv23_W + (cv23_W * (1 + (5.0E-4 * -0.01))))
cv23_W.update(V_cv23_W, 1f, 1f + decay * lrn_rate);
// val X1332 = Proj(X1410, X268,X266,X262,X258, 2)
JCudaTensor X1332 = y66[2];
// val X1424 = X1332 * d_ReLU()(X262)/d_X261
JCudaTensor X1424 = y72.backward(X1332, X262);
// V_cv25_B = ((X1424 * d_Convolv(1,2)()/d_cv25_B * (2 * -0.01)) + (V_cv25_B * 0.9))
y73.backward_bias(X1424, V_cv25_B, 2f * lrn_rate, momentum);
// cv25_B = (V_cv25_B + cv25_B)
cv25_B.update(V_cv25_B, 1f, 1f);
// val X1174 = X1424 * d_Convolv(1,2)(cv25_W)/d_X260
JCudaTensor X1174 = y73.backward_data(X1424, cv25_W);
// V_cv25_W = ((X1424 * d_Convolv(1,2)(X260)/d_cv25_W * -0.01) + (V_cv25_W * 0.9))
y73.backward_filter(X1424, X260, V_cv25_W, lrn_rate, momentum);
// dealloc X1424
X1424.free();
// cv25_W = (V_cv25_W + (cv25_W * (1 + (5.0E-4 * -0.01))))
cv25_W.update(V_cv25_W, 1f, 1f + decay * lrn_rate);
// val X1112 = Proj(X1410, X268,X266,X262,X258, 3)
JCudaTensor X1112 = y66[3];
// dealloc X268
X268.free();
// dealloc X262
X262.free();
// dealloc X1410
X1410.free();
// dealloc X266
X266.free();
// val X1288 = X1112 * d_ReLU()(X258)/d_X257
JCudaTensor X1288 = y72.backward(X1112, X258);
// dealloc X258
X258.free();
// V_cv26_B = ((X1288 * d_Convolv(1,0)()/d_cv26_B * (2 * -0.01)) + (V_cv26_B * 0.9))
y74.backward_bias(X1288, V_cv26_B, 2f * lrn_rate, momentum);
// cv26_B = (V_cv26_B + cv26_B)
cv26_B.update(V_cv26_B, 1f, 1f);
// val X1067 = X1288 * d_Convolv(1,0)(cv26_W)/d_X256
JCudaTensor X1067 = y74.backward_data(X1288, cv26_W);
// V_cv26_W = ((X1288 * d_Convolv(1,0)(X256)/d_cv26_W * -0.01) + (V_cv26_W * 0.9))
y74.backward_filter(X1288, X256, V_cv26_W, lrn_rate, momentum);
// dealloc X1288
X1288.free();
// cv26_W = (V_cv26_W + (cv26_W * (1 + (5.0E-4 * -0.01))))
cv26_W.update(V_cv26_W, 1f, 1f + decay * lrn_rate);
// val X1428 = X1452 * d_ReLU()(X264)/d_X263
JCudaTensor X1428 = y75.backward(X1452, X264);
// dealloc X264
X264.free();
// V_cv22_B = ((X1428 * d_Convolv(1,0)()/d_cv22_B * (2 * -0.01)) + (V_cv22_B * 0.9))
y101.backward_bias(X1428, V_cv22_B, 2f * lrn_rate, momentum);
// cv22_B = (V_cv22_B + cv22_B)
cv22_B.update(V_cv22_B, 1f, 1f);
// val X1418 = (X1416 + X1428 * d_Convolv(1,0)(cv22_W)/d_X255)
JCudaTensor X1418 = y101.backward_data(X1428,cv22_W, X1416);
// V_cv22_W = ((X1428 * d_Convolv(1,0)(X255)/d_cv22_W * -0.01) + (V_cv22_W * 0.9))
y101.backward_filter(X1428, X255, V_cv22_W, lrn_rate, momentum);
// dealloc X1428
X1428.free();
// cv22_W = (V_cv22_W + (cv22_W * (1 + (5.0E-4 * -0.01))))
cv22_W.update(V_cv22_W, 1f, 1f + decay * lrn_rate);
// val X1164 = X1174 * d_ReLU()(X260)/d_X259
JCudaTensor X1164 = y77.backward(X1174, X260);
// dealloc X260
X260.free();
// V_cv24_B = ((X1164 * d_Convolv(1,0)()/d_cv24_B * (2 * -0.01)) + (V_cv24_B * 0.9))
y102.backward_bias(X1164, V_cv24_B, 2f * lrn_rate, momentum);
// cv24_B = (V_cv24_B + cv24_B)
cv24_B.update(V_cv24_B, 1f, 1f);
// V_cv24_W = ((X1164 * d_Convolv(1,0)(X255)/d_cv24_W * -0.01) + (V_cv24_W * 0.9))
y102.backward_filter(X1164, X255, V_cv24_W, lrn_rate, momentum);
// val X1420 = (X1418 + X1164 * d_Convolv(1,0)(cv24_W)/d_X255)
JCudaTensor X1420 = y102.backward_data(X1164,cv24_W, X1418);
// dealloc X1164
X1164.free();
// cv24_W = (V_cv24_W + (cv24_W * (1 + (5.0E-4 * -0.01))))
cv24_W.update(V_cv24_W, 1f, 1f + decay * lrn_rate);
// val X426 = (X1420 + X1067 * d_Pooling(3,1,1,true)(X256,X255)/d_X255)
JCudaTensor X426 = y103.backward(X1067,X256,X255, X1420);
// dealloc X256
X256.free();
// dealloc X1067
X1067.free();
// dealloc X255
X255.free();
JCudaTensor[] y80 = y100.backward(X426);
// val X1104 = Proj(X426, X254,X252,X248,X244, 0)
JCudaTensor X1104 = y80[0];
// val X1461 = X1104 * d_ReLU()(X254)/d_X253
JCudaTensor X1461 = y68.backward(X1104, X254);
// V_cv11_B = ((X1461 * d_Convolv(1,0)()/d_cv11_B * (2 * -0.01)) + (V_cv11_B * 0.9))
y82.backward_bias(X1461, V_cv11_B, 2f * lrn_rate, momentum);
// cv11_B = (V_cv11_B + cv11_B)
cv11_B.update(V_cv11_B, 1f, 1f);
// val X1144 = X1461 * d_Convolv(1,0)(cv11_W)/d_X241
JCudaTensor X1144 = y82.backward_data(X1461, cv11_W);
// V_cv11_W = ((X1461 * d_Convolv(1,0)(X241)/d_cv11_W * -0.01) + (V_cv11_W * 0.9))
y82.backward_filter(X1461, X241, V_cv11_W, lrn_rate, momentum);
// dealloc X1461
X1461.free();
// cv11_W = (V_cv11_W + (cv11_W * (1 + (5.0E-4 * -0.01))))
cv11_W.update(V_cv11_W, 1f, 1f + decay * lrn_rate);
// val X1105 = Proj(X426, X254,X252,X248,X244, 1)
JCudaTensor X1105 = y80[1];
// val X1470 = X1105 * d_ReLU()(X252)/d_X251
JCudaTensor X1470 = y70.backward(X1105, X252);
// V_cv13_B = ((X1470 * d_Convolv(1,1)()/d_cv13_B * (2 * -0.01)) + (V_cv13_B * 0.9))
y71.backward_bias(X1470, V_cv13_B, 2f * lrn_rate, momentum);
// cv13_B = (V_cv13_B + cv13_B)
cv13_B.update(V_cv13_B, 1f, 1f);
// val X1071 = X1470 * d_Convolv(1,1)(cv13_W)/d_X250
JCudaTensor X1071 = y71.backward_data(X1470, cv13_W);
// V_cv13_W = ((X1470 * d_Convolv(1,1)(X250)/d_cv13_W * -0.01) + (V_cv13_W * 0.9))
y71.backward_filter(X1470, X250, V_cv13_W, lrn_rate, momentum);
// dealloc X1470
X1470.free();
// cv13_W = (V_cv13_W + (cv13_W * (1 + (5.0E-4 * -0.01))))
cv13_W.update(V_cv13_W, 1f, 1f + decay * lrn_rate);
// val X1329 = Proj(X426, X254,X252,X248,X244, 2)
JCudaTensor X1329 = y80[2];
// val X1258 = X1329 * d_ReLU()(X248)/d_X247
JCudaTensor X1258 = y72.backward(X1329, X248);
// V_cv15_B = ((X1258 * d_Convolv(1,2)()/d_cv15_B * (2 * -0.01)) + (V_cv15_B * 0.9))
y73.backward_bias(X1258, V_cv15_B, 2f * lrn_rate, momentum);
// cv15_B = (V_cv15_B + cv15_B)
cv15_B.update(V_cv15_B, 1f, 1f);
// val X1047 = X1258 * d_Convolv(1,2)(cv15_W)/d_X246
JCudaTensor X1047 = y73.backward_data(X1258, cv15_W);
// V_cv15_W = ((X1258 * d_Convolv(1,2)(X246)/d_cv15_W * -0.01) + (V_cv15_W * 0.9))
y73.backward_filter(X1258, X246, V_cv15_W, lrn_rate, momentum);
// dealloc X1258
X1258.free();
// cv15_W = (V_cv15_W + (cv15_W * (1 + (5.0E-4 * -0.01))))
cv15_W.update(V_cv15_W, 1f, 1f + decay * lrn_rate);
// val X1318 = Proj(X426, X254,X252,X248,X244, 3)
JCudaTensor X1318 = y80[3];
// dealloc X248
X248.free();
// dealloc X252
X252.free();
// dealloc X426
X426.free();
// dealloc X254
X254.free();
// val X1297 = X1318 * d_ReLU()(X244)/d_X243
JCudaTensor X1297 = y72.backward(X1318, X244);
// dealloc X244
X244.free();
// V_cv16_B = ((X1297 * d_Convolv(1,0)()/d_cv16_B * (2 * -0.01)) + (V_cv16_B * 0.9))
y83.backward_bias(X1297, V_cv16_B, 2f * lrn_rate, momentum);
// cv16_B = (V_cv16_B + cv16_B)
cv16_B.update(V_cv16_B, 1f, 1f);
// val X1209 = X1297 * d_Convolv(1,0)(cv16_W)/d_X242
JCudaTensor X1209 = y83.backward_data(X1297, cv16_W);
// V_cv16_W = ((X1297 * d_Convolv(1,0)(X242)/d_cv16_W * -0.01) + (V_cv16_W * 0.9))
y83.backward_filter(X1297, X242, V_cv16_W, lrn_rate, momentum);
// dealloc X1297
X1297.free();
// cv16_W = (V_cv16_W + (cv16_W * (1 + (5.0E-4 * -0.01))))
cv16_W.update(V_cv16_W, 1f, 1f + decay * lrn_rate);
// val X1435 = X1071 * d_ReLU()(X250)/d_X249
JCudaTensor X1435 = y75.backward(X1071, X250);
// dealloc X250
X250.free();
// V_cv12_B = ((X1435 * d_Convolv(1,0)()/d_cv12_B * (2 * -0.01)) + (V_cv12_B * 0.9))
y97.backward_bias(X1435, V_cv12_B, 2f * lrn_rate, momentum);
// cv12_B = (V_cv12_B + cv12_B)
cv12_B.update(V_cv12_B, 1f, 1f);
// val X1146 = (X1144 + X1435 * d_Convolv(1,0)(cv12_W)/d_X241)
JCudaTensor X1146 = y97.backward_data(X1435,cv12_W, X1144);
// V_cv12_W = ((X1435 * d_Convolv(1,0)(X241)/d_cv12_W * -0.01) + (V_cv12_W * 0.9))
y97.backward_filter(X1435, X241, V_cv12_W, lrn_rate, momentum);
// dealloc X1435
X1435.free();
// cv12_W = (V_cv12_W + (cv12_W * (1 + (5.0E-4 * -0.01))))
cv12_W.update(V_cv12_W, 1f, 1f + decay * lrn_rate);
// val X1372 = X1047 * d_ReLU()(X246)/d_X245
JCudaTensor X1372 = y77.backward(X1047, X246);
// dealloc X246
X246.free();
// V_cv14_B = ((X1372 * d_Convolv(1,0)()/d_cv14_B * (2 * -0.01)) + (V_cv14_B * 0.9))
y98.backward_bias(X1372, V_cv14_B, 2f * lrn_rate, momentum);
// cv14_B = (V_cv14_B + cv14_B)
cv14_B.update(V_cv14_B, 1f, 1f);
// V_cv14_W = ((X1372 * d_Convolv(1,0)(X241)/d_cv14_W * -0.01) + (V_cv14_W * 0.9))
y98.backward_filter(X1372, X241, V_cv14_W, lrn_rate, momentum);
// val X1148 = (X1146 + X1372 * d_Convolv(1,0)(cv14_W)/d_X241)
JCudaTensor X1148 = y98.backward_data(X1372,cv14_W, X1146);
// dealloc X1372
X1372.free();
// cv14_W = (V_cv14_W + (cv14_W * (1 + (5.0E-4 * -0.01))))
cv14_W.update(V_cv14_W, 1f, 1f + decay * lrn_rate);
// val X429 = (X1148 + X1209 * d_Pooling(3,1,1,true)(X242,X241)/d_X241)
JCudaTensor X429 = y99.backward(X1209,X242,X241, X1148);
// dealloc X242
X242.free();
// dealloc X1209
X1209.free();
// val X1307 = X429 * d_Pooling(3,2,1,true)(X241,X240)/d_X240
JCudaTensor X1307 = y87.backward(X429, X241, X240);
// dealloc X429
X429.free();
// dealloc X241
X241.free();
// val X1446 = X1307 * d_LRN(5,1.0E-4,0.75)(X240,X239)/d_X239
JCudaTensor X1446 = y88.backward(X1307, X240, X239);
// dealloc X240
X240.free();
// val X1384 = X1446 * d_ReLU()(X239)/d_X238
JCudaTensor X1384 = y89.backward(X1446, X239);
// dealloc X239
X239.free();
// V_cv3_B = ((X1384 * d_Convolv(1,1)()/d_cv3_B * (2 * -0.01)) + (V_cv3_B * 0.9))
y90.backward_bias(X1384, V_cv3_B, 2f * lrn_rate, momentum);
// cv3_B = (V_cv3_B + cv3_B)
cv3_B.update(V_cv3_B, 1f, 1f);
// val X1194 = X1384 * d_Convolv(1,1)(cv3_W)/d_X237
JCudaTensor X1194 = y90.backward_data(X1384, cv3_W);
// V_cv3_W = ((X1384 * d_Convolv(1,1)(X237)/d_cv3_W * -0.01) + (V_cv3_W * 0.9))
y90.backward_filter(X1384, X237, V_cv3_W, lrn_rate, momentum);
// dealloc X1384
X1384.free();
// cv3_W = (V_cv3_W + (cv3_W * (1 + (5.0E-4 * -0.01))))
cv3_W.update(V_cv3_W, 1f, 1f + decay * lrn_rate);
// val X1171 = X1194 * d_ReLU()(X237)/d_X236
JCudaTensor X1171 = y91.backward(X1194, X237);
// dealloc X237
X237.free();
// V_cv2_B = ((X1171 * d_Convolv(1,0)()/d_cv2_B * (2 * -0.01)) + (V_cv2_B * 0.9))
y92.backward_bias(X1171, V_cv2_B, 2f * lrn_rate, momentum);
// cv2_B = (V_cv2_B + cv2_B)
cv2_B.update(V_cv2_B, 1f, 1f);
// val X1447 = X1171 * d_Convolv(1,0)(cv2_W)/d_X235
JCudaTensor X1447 = y92.backward_data(X1171, cv2_W);
// V_cv2_W = ((X1171 * d_Convolv(1,0)(X235)/d_cv2_W * -0.01) + (V_cv2_W * 0.9))
y92.backward_filter(X1171, X235, V_cv2_W, lrn_rate, momentum);
// dealloc X1171
X1171.free();
// cv2_W = (V_cv2_W + (cv2_W * (1 + (5.0E-4 * -0.01))))
cv2_W.update(V_cv2_W, 1f, 1f + decay * lrn_rate);
// val X1179 = X1447 * d_LRN(5,1.0E-4,0.75)(X235,X234)/d_X234
JCudaTensor X1179 = y93.backward(X1447, X235, X234);
// dealloc X235
X235.free();
// val X1070 = X1179 * d_Pooling(3,2,1,true)(X234,X233)/d_X233
JCudaTensor X1070 = y94.backward(X1179, X234, X233);
// dealloc X234
X234.free();
// dealloc X1179
X1179.free();
// val X1286 = X1070 * d_ReLU()(X233)/d_X232
JCudaTensor X1286 = y95.backward(X1070, X233);
// dealloc X233
X233.free();
// V_cv1_B = ((X1286 * d_Convolv(2,3)()/d_cv1_B * (2 * -0.01)) + (V_cv1_B * 0.9))
y96.backward_bias(X1286, V_cv1_B, 2f * lrn_rate, momentum);
// cv1_B = (V_cv1_B + cv1_B)
cv1_B.update(V_cv1_B, 1f, 1f);
// V_cv1_W = ((X1286 * d_Convolv(2,3)(X1041)/d_cv1_W * -0.01) + (V_cv1_W * 0.9))
y96.backward_filter(X1286, X1041, V_cv1_W, lrn_rate, momentum);
// dealloc X1286
X1286.free();
// dealloc X1041
X1041.free();
// cv1_W = (V_cv1_W + (cv1_W * (1 + (5.0E-4 * -0.01))))
cv1_W.update(V_cv1_W, 1f, 1f + decay * lrn_rate);

return _loss; 
}

public JCudaTensor testFunction(JTensorFloat X) {
 // val X88 = Cuda(X)
JCudaTensor X88 = X.asJCudaTensor();
// val X89 = Convolv(2,3)(X88,cv1_W,cv1_B)
JCudaTensor X89 = y96.forward(X88, cv1_W, cv1_B);
// dealloc X88
X88.free();
// val X90 = ReLU()(X89)
JCudaTensor X90 = y95.forward(X89);
// val X91 = Pooling(3,2,1,true)(X90)
JCudaTensor X91 = y94.forward(X90);
// dealloc X90
X90.free();
// val X92 = LRN(5,1.0E-4,0.75)(X91)
JCudaTensor X92 = y93.forward(X91);
// dealloc X91
X91.free();
// val X93 = Convolv(1,0)(X92,cv2_W,cv2_B)
JCudaTensor X93 = y92.forward(X92, cv2_W, cv2_B);
// dealloc X92
X92.free();
// val X94 = ReLU()(X93)
JCudaTensor X94 = y91.forward(X93);
// val X95 = Convolv(1,1)(X94,cv3_W,cv3_B)
JCudaTensor X95 = y90.forward(X94, cv3_W, cv3_B);
// dealloc X94
X94.free();
// val X96 = ReLU()(X95)
JCudaTensor X96 = y89.forward(X95);
// val X97 = LRN(5,1.0E-4,0.75)(X96)
JCudaTensor X97 = y88.forward(X96);
// dealloc X96
X96.free();
// val X98 = Pooling(3,2,1,true)(X97)
JCudaTensor X98 = y87.forward(X97);
// dealloc X97
X97.free();
// val X99 = Pooling(3,1,1,true)(X98)
JCudaTensor X99 = y99.forward(X98);
// val X110 = Convolv(1,0)(X98,cv11_W,cv11_B)
JCudaTensor X110 = y82.forward(X98, cv11_W, cv11_B);
// val X102 = Convolv(1,0)(X98,cv14_W,cv14_B)
JCudaTensor X102 = y98.forward(X98, cv14_W, cv14_B);
// val X106 = Convolv(1,0)(X98,cv12_W,cv12_B)
JCudaTensor X106 = y97.forward(X98, cv12_W, cv12_B);
// dealloc X98
X98.free();
// val X107 = ReLU()(X106)
JCudaTensor X107 = y75.forward(X106);
// val X103 = ReLU()(X102)
JCudaTensor X103 = y77.forward(X102);
// val X111 = ReLU()(X110)
JCudaTensor X111 = y68.forward(X110);
// val X100 = Convolv(1,0)(X99,cv16_W,cv16_B)
JCudaTensor X100 = y83.forward(X99, cv16_W, cv16_B);
// dealloc X99
X99.free();
// val X101 = ReLU()(X100)
JCudaTensor X101 = y72.forward(X100);
// val X108 = Convolv(1,1)(X107,cv13_W,cv13_B)
JCudaTensor X108 = y71.forward(X107, cv13_W, cv13_B);
// dealloc X107
X107.free();
// val X104 = Convolv(1,2)(X103,cv15_W,cv15_B)
JCudaTensor X104 = y73.forward(X103, cv15_W, cv15_B);
// dealloc X103
X103.free();
// val X105 = ReLU()(X104)
JCudaTensor X105 = y72.forward(X104);
// val X109 = ReLU()(X108)
JCudaTensor X109 = y70.forward(X108);
// val X112 = Concat(X111,X109,X105,X101)
JCudaTensor X112 = y100.forward(X111,X109,X105,X101);
// dealloc X109
X109.free();
// dealloc X111
X111.free();
// dealloc X101
X101.free();
// dealloc X105
X105.free();
// val X124 = Convolv(1,0)(X112,cv21_W,cv21_B)
JCudaTensor X124 = y69.forward(X112, cv21_W, cv21_B);
// val X113 = Pooling(3,1,1,true)(X112)
JCudaTensor X113 = y103.forward(X112);
// val X116 = Convolv(1,0)(X112,cv24_W,cv24_B)
JCudaTensor X116 = y102.forward(X112, cv24_W, cv24_B);
// val X120 = Convolv(1,0)(X112,cv22_W,cv22_B)
JCudaTensor X120 = y101.forward(X112, cv22_W, cv22_B);
// dealloc X112
X112.free();
// val X121 = ReLU()(X120)
JCudaTensor X121 = y75.forward(X120);
// val X114 = Convolv(1,0)(X113,cv26_W,cv26_B)
JCudaTensor X114 = y74.forward(X113, cv26_W, cv26_B);
// dealloc X113
X113.free();
// val X117 = ReLU()(X116)
JCudaTensor X117 = y77.forward(X116);
// val X125 = ReLU()(X124)
JCudaTensor X125 = y68.forward(X124);
// val X118 = Convolv(1,2)(X117,cv25_W,cv25_B)
JCudaTensor X118 = y73.forward(X117, cv25_W, cv25_B);
// dealloc X117
X117.free();
// val X122 = Convolv(1,1)(X121,cv23_W,cv23_B)
JCudaTensor X122 = y71.forward(X121, cv23_W, cv23_B);
// dealloc X121
X121.free();
// val X115 = ReLU()(X114)
JCudaTensor X115 = y72.forward(X114);
// val X119 = ReLU()(X118)
JCudaTensor X119 = y72.forward(X118);
// val X123 = ReLU()(X122)
JCudaTensor X123 = y70.forward(X122);
// val X126 = Concat(X125,X123,X119,X115)
JCudaTensor X126 = y100.forward(X125,X123,X119,X115);
// dealloc X123
X123.free();
// dealloc X125
X125.free();
// dealloc X115
X115.free();
// dealloc X119
X119.free();
// val X127 = Pooling(3,2,1,true)(X126)
JCudaTensor X127 = y65.forward(X126);
// dealloc X126
X126.free();
// val X131 = Convolv(1,0)(X127,cv34_W,cv34_B)
JCudaTensor X131 = y106.forward(X127, cv34_W, cv34_B);
// val X139 = Convolv(1,0)(X127,cv31_W,cv31_B)
JCudaTensor X139 = y32.forward(X127, cv31_W, cv31_B);
// val X135 = Convolv(1,0)(X127,cv32_W,cv32_B)
JCudaTensor X135 = y104.forward(X127, cv32_W, cv32_B);
// val X128 = Pooling(3,1,1,true)(X127)
JCudaTensor X128 = y105.forward(X127);
// dealloc X127
X127.free();
// val X129 = Convolv(1,0)(X128,cv36_W,cv36_B)
JCudaTensor X129 = y37.forward(X128, cv36_W, cv36_B);
// dealloc X128
X128.free();
// val X140 = ReLU()(X139)
JCudaTensor X140 = y31.forward(X139);
// val X136 = ReLU()(X135)
JCudaTensor X136 = y38.forward(X135);
// val X132 = ReLU()(X131)
JCudaTensor X132 = y40.forward(X131);
// val X130 = ReLU()(X129)
JCudaTensor X130 = y35.forward(X129);
// val X133 = Convolv(1,2)(X132,cv35_W,cv35_B)
JCudaTensor X133 = y36.forward(X132, cv35_W, cv35_B);
// dealloc X132
X132.free();
// val X137 = Convolv(1,1)(X136,cv33_W,cv33_B)
JCudaTensor X137 = y34.forward(X136, cv33_W, cv33_B);
// dealloc X136
X136.free();
// val X138 = ReLU()(X137)
JCudaTensor X138 = y33.forward(X137);
// val X134 = ReLU()(X133)
JCudaTensor X134 = y35.forward(X133);
// val X141 = Concat(X140,X138,X134,X130)
JCudaTensor X141 = y107.forward(X140,X138,X134,X130);
// dealloc X130
X130.free();
// dealloc X138
X138.free();
// dealloc X140
X140.free();
// dealloc X134
X134.free();
// val X142 = Pooling(3,1,1,true)(X141)
JCudaTensor X142 = y105.forward(X141);
// val X145 = Convolv(1,0)(X141,cv44_W,cv44_B)
JCudaTensor X145 = y106.forward(X141, cv44_W, cv44_B);
// val X153 = Convolv(1,0)(X141,cv41_W,cv41_B)
JCudaTensor X153 = y32.forward(X141, cv41_W, cv41_B);
// val X149 = Convolv(1,0)(X141,cv42_W,cv42_B)
JCudaTensor X149 = y104.forward(X141, cv42_W, cv42_B);
// dealloc X141
X141.free();
// val X150 = ReLU()(X149)
JCudaTensor X150 = y38.forward(X149);
// val X143 = Convolv(1,0)(X142,cv46_W,cv46_B)
JCudaTensor X143 = y37.forward(X142, cv46_W, cv46_B);
// dealloc X142
X142.free();
// val X146 = ReLU()(X145)
JCudaTensor X146 = y40.forward(X145);
// val X154 = ReLU()(X153)
JCudaTensor X154 = y31.forward(X153);
// val X147 = Convolv(1,2)(X146,cv45_W,cv45_B)
JCudaTensor X147 = y36.forward(X146, cv45_W, cv45_B);
// dealloc X146
X146.free();
// val X144 = ReLU()(X143)
JCudaTensor X144 = y35.forward(X143);
// val X151 = Convolv(1,1)(X150,cv43_W,cv43_B)
JCudaTensor X151 = y34.forward(X150, cv43_W, cv43_B);
// dealloc X150
X150.free();
// val X148 = ReLU()(X147)
JCudaTensor X148 = y35.forward(X147);
// val X152 = ReLU()(X151)
JCudaTensor X152 = y33.forward(X151);
// val X155 = Concat(X154,X152,X148,X144)
JCudaTensor X155 = y107.forward(X154,X152,X148,X144);
// dealloc X152
X152.free();
// dealloc X154
X154.free();
// dealloc X148
X148.free();
// dealloc X144
X144.free();
// val X156 = Pooling(3,1,1,true)(X155)
JCudaTensor X156 = y105.forward(X155);
// val X163 = Convolv(1,0)(X155,cv52_W,cv52_B)
JCudaTensor X163 = y104.forward(X155, cv52_W, cv52_B);
// val X159 = Convolv(1,0)(X155,cv54_W,cv54_B)
JCudaTensor X159 = y106.forward(X155, cv54_W, cv54_B);
// val X167 = Convolv(1,0)(X155,cv51_W,cv51_B)
JCudaTensor X167 = y32.forward(X155, cv51_W, cv51_B);
// dealloc X155
X155.free();
// val X157 = Convolv(1,0)(X156,cv56_W,cv56_B)
JCudaTensor X157 = y37.forward(X156, cv56_W, cv56_B);
// dealloc X156
X156.free();
// val X160 = ReLU()(X159)
JCudaTensor X160 = y40.forward(X159);
// val X168 = ReLU()(X167)
JCudaTensor X168 = y31.forward(X167);
// val X164 = ReLU()(X163)
JCudaTensor X164 = y38.forward(X163);
// val X165 = Convolv(1,1)(X164,cv53_W,cv53_B)
JCudaTensor X165 = y34.forward(X164, cv53_W, cv53_B);
// dealloc X164
X164.free();
// val X158 = ReLU()(X157)
JCudaTensor X158 = y35.forward(X157);
// val X161 = Convolv(1,2)(X160,cv55_W,cv55_B)
JCudaTensor X161 = y36.forward(X160, cv55_W, cv55_B);
// dealloc X160
X160.free();
// val X162 = ReLU()(X161)
JCudaTensor X162 = y35.forward(X161);
// val X166 = ReLU()(X165)
JCudaTensor X166 = y33.forward(X165);
// val X169 = Concat(X168,X166,X162,X158)
JCudaTensor X169 = y107.forward(X168,X166,X162,X158);
// dealloc X158
X158.free();
// dealloc X168
X168.free();
// dealloc X166
X166.free();
// dealloc X162
X162.free();
// val X173 = Convolv(1,0)(X169,cv64_W,cv64_B)
JCudaTensor X173 = y106.forward(X169, cv64_W, cv64_B);
// val X170 = Pooling(3,1,1,true)(X169)
JCudaTensor X170 = y105.forward(X169);
// val X181 = Convolv(1,0)(X169,cv61_W,cv61_B)
JCudaTensor X181 = y32.forward(X169, cv61_W, cv61_B);
// val X177 = Convolv(1,0)(X169,cv62_W,cv62_B)
JCudaTensor X177 = y104.forward(X169, cv62_W, cv62_B);
// dealloc X169
X169.free();
// val X174 = ReLU()(X173)
JCudaTensor X174 = y40.forward(X173);
// val X182 = ReLU()(X181)
JCudaTensor X182 = y31.forward(X181);
// val X178 = ReLU()(X177)
JCudaTensor X178 = y38.forward(X177);
// val X171 = Convolv(1,0)(X170,cv66_W,cv66_B)
JCudaTensor X171 = y37.forward(X170, cv66_W, cv66_B);
// dealloc X170
X170.free();
// val X175 = Convolv(1,2)(X174,cv65_W,cv65_B)
JCudaTensor X175 = y36.forward(X174, cv65_W, cv65_B);
// dealloc X174
X174.free();
// val X179 = Convolv(1,1)(X178,cv63_W,cv63_B)
JCudaTensor X179 = y34.forward(X178, cv63_W, cv63_B);
// dealloc X178
X178.free();
// val X172 = ReLU()(X171)
JCudaTensor X172 = y35.forward(X171);
// val X176 = ReLU()(X175)
JCudaTensor X176 = y35.forward(X175);
// val X180 = ReLU()(X179)
JCudaTensor X180 = y33.forward(X179);
// val X183 = Concat(X182,X180,X176,X172)
JCudaTensor X183 = y107.forward(X182,X180,X176,X172);
// dealloc X172
X172.free();
// dealloc X176
X176.free();
// dealloc X180
X180.free();
// dealloc X182
X182.free();
// val X191 = Convolv(1,0)(X183,cv72_W,cv72_B)
JCudaTensor X191 = y104.forward(X183, cv72_W, cv72_B);
// val X187 = Convolv(1,0)(X183,cv74_W,cv74_B)
JCudaTensor X187 = y106.forward(X183, cv74_W, cv74_B);
// val X184 = Pooling(3,1,1,true)(X183)
JCudaTensor X184 = y105.forward(X183);
// val X195 = Convolv(1,0)(X183,cv71_W,cv71_B)
JCudaTensor X195 = y32.forward(X183, cv71_W, cv71_B);
// dealloc X183
X183.free();
// val X188 = ReLU()(X187)
JCudaTensor X188 = y40.forward(X187);
// val X185 = Convolv(1,0)(X184,cv76_W,cv76_B)
JCudaTensor X185 = y37.forward(X184, cv76_W, cv76_B);
// dealloc X184
X184.free();
// val X196 = ReLU()(X195)
JCudaTensor X196 = y31.forward(X195);
// val X192 = ReLU()(X191)
JCudaTensor X192 = y38.forward(X191);
// val X189 = Convolv(1,2)(X188,cv75_W,cv75_B)
JCudaTensor X189 = y36.forward(X188, cv75_W, cv75_B);
// dealloc X188
X188.free();
// val X193 = Convolv(1,1)(X192,cv73_W,cv73_B)
JCudaTensor X193 = y34.forward(X192, cv73_W, cv73_B);
// dealloc X192
X192.free();
// val X186 = ReLU()(X185)
JCudaTensor X186 = y35.forward(X185);
// val X194 = ReLU()(X193)
JCudaTensor X194 = y33.forward(X193);
// val X190 = ReLU()(X189)
JCudaTensor X190 = y35.forward(X189);
// val X197 = Concat(X196,X194,X190,X186)
JCudaTensor X197 = y107.forward(X196,X194,X190,X186);
// dealloc X186
X186.free();
// dealloc X194
X194.free();
// dealloc X196
X196.free();
// dealloc X190
X190.free();
// val X198 = Pooling(3,2,1,true)(X197)
JCudaTensor X198 = y28.forward(X197);
// dealloc X197
X197.free();
// val X206 = Convolv(1,0)(X198,cv82_W,cv82_B)
JCudaTensor X206 = y111.forward(X198, cv82_W, cv82_B);
// val X210 = Convolv(1,0)(X198,cv81_W,cv81_B)
JCudaTensor X210 = y12.forward(X198, cv81_W, cv81_B);
// val X199 = Pooling(3,1,1,true)(X198)
JCudaTensor X199 = y109.forward(X198);
// val X202 = Convolv(1,0)(X198,cv84_W,cv84_B)
JCudaTensor X202 = y110.forward(X198, cv84_W, cv84_B);
// dealloc X198
X198.free();
// val X211 = ReLU()(X210)
JCudaTensor X211 = y11.forward(X210);
// val X207 = ReLU()(X206)
JCudaTensor X207 = y18.forward(X206);
// val X200 = Convolv(1,0)(X199,cv86_W,cv86_B)
JCudaTensor X200 = y17.forward(X199, cv86_W, cv86_B);
// dealloc X199
X199.free();
// val X203 = ReLU()(X202)
JCudaTensor X203 = y20.forward(X202);
// val X208 = Convolv(1,1)(X207,cv83_W,cv83_B)
JCudaTensor X208 = y14.forward(X207, cv83_W, cv83_B);
// dealloc X207
X207.free();
// val X204 = Convolv(1,2)(X203,cv85_W,cv85_B)
JCudaTensor X204 = y16.forward(X203, cv85_W, cv85_B);
// dealloc X203
X203.free();
// val X201 = ReLU()(X200)
JCudaTensor X201 = y15.forward(X200);
// val X205 = ReLU()(X204)
JCudaTensor X205 = y15.forward(X204);
// val X209 = ReLU()(X208)
JCudaTensor X209 = y13.forward(X208);
// val X212 = Concat(X211,X209,X205,X201)
JCudaTensor X212 = y112.forward(X211,X209,X205,X201);
// dealloc X209
X209.free();
// dealloc X211
X211.free();
// dealloc X201
X201.free();
// dealloc X205
X205.free();
// val X216 = Convolv(1,0)(X212,cv94_W,cv94_B)
JCudaTensor X216 = y110.forward(X212, cv94_W, cv94_B);
// val X220 = Convolv(1,0)(X212,cv92_W,cv92_B)
JCudaTensor X220 = y111.forward(X212, cv92_W, cv92_B);
// val X224 = Convolv(1,0)(X212,cv91_W,cv91_B)
JCudaTensor X224 = y12.forward(X212, cv91_W, cv91_B);
// val X213 = Pooling(3,1,1,true)(X212)
JCudaTensor X213 = y109.forward(X212);
// dealloc X212
X212.free();
// val X217 = ReLU()(X216)
JCudaTensor X217 = y20.forward(X216);
// val X225 = ReLU()(X224)
JCudaTensor X225 = y11.forward(X224);
// val X221 = ReLU()(X220)
JCudaTensor X221 = y18.forward(X220);
// val X214 = Convolv(1,0)(X213,cv96_W,cv96_B)
JCudaTensor X214 = y17.forward(X213, cv96_W, cv96_B);
// dealloc X213
X213.free();
// val X215 = ReLU()(X214)
JCudaTensor X215 = y15.forward(X214);
// val X222 = Convolv(1,1)(X221,cv93_W,cv93_B)
JCudaTensor X222 = y14.forward(X221, cv93_W, cv93_B);
// dealloc X221
X221.free();
// val X218 = Convolv(1,2)(X217,cv95_W,cv95_B)
JCudaTensor X218 = y16.forward(X217, cv95_W, cv95_B);
// dealloc X217
X217.free();
// val X223 = ReLU()(X222)
JCudaTensor X223 = y13.forward(X222);
// val X219 = ReLU()(X218)
JCudaTensor X219 = y15.forward(X218);
// val X226 = Concat(X225,X223,X219,X215)
JCudaTensor X226 = y112.forward(X225,X223,X219,X215);
// dealloc X225
X225.free();
// dealloc X223
X223.free();
// dealloc X215
X215.free();
// dealloc X219
X219.free();
// val X227 = Pooling(7,1,0,false)(X226)
JCudaTensor X227 = y8.forward(X226);
// dealloc X226
X226.free();
// val X230 = (X227[1><3])(i12 | @) * (fc_W)(i13 | @)
JCudaTensor X230 = X227.flatten(1, new int[]{256, 1, 1}).asMatrix(1, true).times(fc_W.asMatrix(1, true));
// dealloc X227
X227.free();
// val X229 = (X230 + (i12) => fc_B)
JCudaTensor X229 = fc_B.copy(256, X230);

return X229; 
}

}