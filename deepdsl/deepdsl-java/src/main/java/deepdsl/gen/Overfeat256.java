package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.util.CudaRun;


public class Overfeat256 extends CudaRun {

public static void main(String[] args){
Overfeat256 run = new Overfeat256();
run.train(10);
run.test(10);
run.save();
run.free();
}

public Overfeat256() {
super("src/main/java/deepdsl/gen/overfeat256");
setTrainData(LmdbFactory.getFactory("dataset/imagenet/ilsvrc12_train_lmdb", 1000000, new int[]{256, 3, 224, 224}, 1000, false));
setTestData(LmdbFactory.getFactory("dataset/imagenet/ilsvrc12_val_lmdb", 10000, new int[]{256, 3, 224, 224}, 1000, true));
}

float lrn_rate = -0.01f;
float momentum = 0.9f;
float decay = 5.0E-4f;

JCudnnConvolution y4 = addConvolution(new int[]{256,1024,13,13},new int[]{1024,1024,3,3},new int[]{1024}, 1, 1);
JCudnnConvolution y7 = addConvolution(new int[]{256,256,13,13},new int[]{512,256,3,3},new int[]{512}, 1, 1);
JCudnnConvolution y5 = addConvolution(new int[]{256,512,13,13},new int[]{1024,512,3,3},new int[]{1024}, 1, 1);
JCudnnConvolution y10 = addConvolution(new int[]{256,96,27,27},new int[]{256,96,5,5},new int[]{256}, 1, 2);
JCudnnConvolution y13 = addConvolution(new int[]{256,3,224,224},new int[]{96,3,11,11},new int[]{96}, 4, 0);
JCudnnSoftmax y1 = addSoftmax(new int[]{256,1000}, SoftmaxAlgorithm.LOG);
JCudnnPooling y2 = addPooling(new int[]{256,1024,13,13}, 2, 2, 0, PoolingType.MAX);
JCudnnPooling y8 = addPooling(new int[]{256,256,27,27}, 2, 2, 0, PoolingType.MAX);
JCudnnPooling y11 = addPooling(new int[]{256,96,54,54}, 2, 2, 0, PoolingType.MAX);
JCudnnActivation y3 = addActivation(new int[]{256,1024,13,13}, ActivationMode.RELU);
JCudnnActivation y9 = addActivation(new int[]{256,256,27,27}, ActivationMode.RELU);
JCudnnActivation y6 = addActivation(new int[]{256,512,13,13}, ActivationMode.RELU);
JCudnnActivation y12 = addActivation(new int[]{256,96,54,54}, ActivationMode.RELU);
JCudaTensor V_cv1_B = addParam("V_cv1_B", "Constant", 0f, 96);
JCudaTensor V_cv1_W = addParam("V_cv1_W", "Constant", 0f, 96, 3, 11, 11);
JCudaTensor V_cv2_B = addParam("V_cv2_B", "Constant", 0f, 256);
JCudaTensor V_cv2_W = addParam("V_cv2_W", "Constant", 0f, 256, 96, 5, 5);
JCudaTensor V_cv3_B = addParam("V_cv3_B", "Constant", 0f, 512);
JCudaTensor V_cv3_W = addParam("V_cv3_W", "Constant", 0f, 512, 256, 3, 3);
JCudaTensor V_cv4_B = addParam("V_cv4_B", "Constant", 0f, 1024);
JCudaTensor V_cv4_W = addParam("V_cv4_W", "Constant", 0f, 1024, 512, 3, 3);
JCudaTensor V_cv5_B = addParam("V_cv5_B", "Constant", 0f, 1024);
JCudaTensor V_cv5_W = addParam("V_cv5_W", "Constant", 0f, 1024, 1024, 3, 3);
JCudaTensor V_fc6_B = addParam("V_fc6_B", "Constant", 0f, 3072);
JCudaTensor V_fc6_W = addParam("V_fc6_W", "Constant", 0f, 3072, 36864);
JCudaTensor V_fc7_B = addParam("V_fc7_B", "Constant", 0f, 4096);
JCudaTensor V_fc7_W = addParam("V_fc7_W", "Constant", 0f, 4096, 3072);
JCudaTensor V_fc8_B = addParam("V_fc8_B", "Constant", 0f, 1000);
JCudaTensor V_fc8_W = addParam("V_fc8_W", "Constant", 0f, 1000, 4096);
JCudaTensor cv1_B = addParam("cv1_B", "Constant", 0.2f, 96);
JCudaTensor cv1_W = addParam("cv1_W", "Random", 0.07422696f, 96, 3, 11, 11);
JCudaTensor cv2_B = addParam("cv2_B", "Constant", 0.2f, 256);
JCudaTensor cv2_W = addParam("cv2_W", "Random", 0.028867513f, 256, 96, 5, 5);
JCudaTensor cv3_B = addParam("cv3_B", "Constant", 0.2f, 512);
JCudaTensor cv3_W = addParam("cv3_W", "Random", 0.029462783f, 512, 256, 3, 3);
JCudaTensor cv4_B = addParam("cv4_B", "Constant", 0.2f, 1024);
JCudaTensor cv4_W = addParam("cv4_W", "Random", 0.020833334f, 1024, 512, 3, 3);
JCudaTensor cv5_B = addParam("cv5_B", "Constant", 0.2f, 1024);
JCudaTensor cv5_W = addParam("cv5_W", "Random", 0.014731391f, 1024, 1024, 3, 3);
JCudaTensor fc6_B = addParam("fc6_B", "Constant", 0.0f, 3072);
JCudaTensor fc6_W = addParam("fc6_W", "Random", 0.0073656957f, 3072, 36864);
JCudaTensor fc7_B = addParam("fc7_B", "Constant", 0.0f, 4096);
JCudaTensor fc7_W = addParam("fc7_W", "Random", 0.02551552f, 4096, 3072);
JCudaTensor fc8_B = addParam("fc8_B", "Constant", 0.0f, 1000);
JCudaTensor fc8_W = addParam("fc8_W", "Random", 0.022097087f, 1000, 4096);

public float trainFunction(JTensorFloat X, JTensorFloat Y) {
 // val X130 = Cuda(Indicator(Y, 1000))
JCudaTensor X130 = Y.asIndicator(1000).asJCudaTensor();
// val X123 = Cuda(X)
JCudaTensor X123 = X.asJCudaTensor();
// val X38 = Convolv(4,0)(X123,cv1_W,cv1_B)
JCudaTensor X38 = y13.forward(X123, cv1_W, cv1_B);
// val X39 = ReLU()(X38)
JCudaTensor X39 = y12.forward(X38);
// val X40 = Pooling(2,2,0,true)(X39)
JCudaTensor X40 = y11.forward(X39);
// val X41 = Convolv(1,2)(X40,cv2_W,cv2_B)
JCudaTensor X41 = y10.forward(X40, cv2_W, cv2_B);
// val X42 = ReLU()(X41)
JCudaTensor X42 = y9.forward(X41);
// val X43 = Pooling(2,2,0,true)(X42)
JCudaTensor X43 = y8.forward(X42);
// val X44 = Convolv(1,1)(X43,cv3_W,cv3_B)
JCudaTensor X44 = y7.forward(X43, cv3_W, cv3_B);
// val X45 = ReLU()(X44)
JCudaTensor X45 = y6.forward(X44);
// val X46 = Convolv(1,1)(X45,cv4_W,cv4_B)
JCudaTensor X46 = y5.forward(X45, cv4_W, cv4_B);
// val X47 = ReLU()(X46)
JCudaTensor X47 = y3.forward(X46);
// val X48 = Convolv(1,1)(X47,cv5_W,cv5_B)
JCudaTensor X48 = y4.forward(X47, cv5_W, cv5_B);
// val X49 = ReLU()(X48)
JCudaTensor X49 = y3.forward(X48);
// val X50 = Pooling(2,2,0,true)(X49)
JCudaTensor X50 = y2.forward(X49);
// val X124 = (X50[1><3])(i1 | @) * (fc6_W)(i2 | @)
JCudaTensor X124 = X50.flatten(1, new int[]{1024, 6, 6}).asMatrix(1, true).times(fc6_W.asMatrix(1, true));
// val X52 = (X124 + (i1) => fc6_B)
JCudaTensor X52 = fc6_B.copy(256, X124);
// val X126 = (X52)(i4 | @) * (fc7_W)(i5 | @)
JCudaTensor X126 = X52.asMatrix(1, true).times(fc7_W.asMatrix(1, true));
// val X53 = (X126 + (i4) => fc7_B)
JCudaTensor X53 = fc7_B.copy(256, X126);
// val X128 = (X53)(i7 | @) * (fc8_W)(i8 | @)
JCudaTensor X128 = X53.asMatrix(1, true).times(fc8_W.asMatrix(1, true));
// val X54 = (X128 + (i7) => fc8_B)
JCudaTensor X54 = fc8_B.copy(256, X128);
// val X55 = LogSoftmax()(X54)
JCudaTensor X55 = y1.forward(X54);
// dealloc X54
X54.free();
// val _loss = ((0 - (X130 . X55)) / |256|)
float _loss = - X130.dot(X55) / 256f;
// val X176 = - X130.copy
JCudaTensor X176 = X130.clone().times_i(-1f);;
// dealloc X130
X130.free();
// val X56 = (X176 / |256|)
JCudaTensor X56 = X176.times_i(1 / 256f);;
// val X156 = X56 * d_LogSoftmax()(X55)/d_X54
JCudaTensor X156 = y1.backward(X56, X55);
// dealloc X55
X55.free();
// dealloc X56
X56.free();
// val m1 = (i21) => X156[@, i21]
JCudaMatrix m1 = X156.asMatrix(1, false);
// V_fc8_B = ((Sum(m1) * -0.01) + (V_fc8_B * 0.9))
m1.sum(V_fc8_B, lrn_rate, momentum);
// fc8_B = (V_fc8_B + (fc8_B * (1 + (5.0E-4 * -0.01))))
fc8_B.update(V_fc8_B, 1f, 1f + decay * lrn_rate);
// val m7 = (i54) => fc8_W[@, i54]
JCudaMatrix m7 = fc8_W.asMatrix(1, false);
// val m2 = (i22) => X53[@, i22]
JCudaMatrix m2 = X53.asMatrix(1, false);
// V_fc8_W = ((m1 * m2 * -0.01) + (V_fc8_W * 0.9))
m1.times(m2, V_fc8_W, lrn_rate, momentum);
// dealloc X53
X53.free();
// fc8_W = (V_fc8_W + (fc8_W * (1 + (5.0E-4 * -0.01))))
fc8_W.update(V_fc8_W, 1f, 1f + decay * lrn_rate);
// val X158 = (X156)(i53 | @) * m7
JCudaTensor X158 = X156.asMatrix(1, true).times(m7);
// dealloc X156
X156.free();
// val m3 = (i25) => X158[@, i25]
JCudaMatrix m3 = X158.asMatrix(1, false);
// V_fc7_B = ((Sum(m3) * -0.01) + (V_fc7_B * 0.9))
m3.sum(V_fc7_B, lrn_rate, momentum);
// fc7_B = (V_fc7_B + (fc7_B * (1 + (5.0E-4 * -0.01))))
fc7_B.update(V_fc7_B, 1f, 1f + decay * lrn_rate);
// val m5 = (i47) => fc7_W[@, i47]
JCudaMatrix m5 = fc7_W.asMatrix(1, false);
// val m4 = (i26) => X52[@, i26]
JCudaMatrix m4 = X52.asMatrix(1, false);
// V_fc7_W = ((m3 * m4 * -0.01) + (V_fc7_W * 0.9))
m3.times(m4, V_fc7_W, lrn_rate, momentum);
// dealloc X52
X52.free();
// fc7_W = (V_fc7_W + (fc7_W * (1 + (5.0E-4 * -0.01))))
fc7_W.update(V_fc7_W, 1f, 1f + decay * lrn_rate);
// val X149 = (X158)(i46 | @) * m5
JCudaTensor X149 = X158.asMatrix(1, true).times(m5);
// dealloc X158
X158.free();
// val m10 = (i29) => X149[@, i29]
JCudaMatrix m10 = X149.asMatrix(1, false);
// V_fc6_B = ((Sum(m10) * -0.01) + (V_fc6_B * 0.9))
m10.sum(V_fc6_B, lrn_rate, momentum);
// fc6_B = (V_fc6_B + (fc6_B * (1 + (5.0E-4 * -0.01))))
fc6_B.update(V_fc6_B, 1f, 1f + decay * lrn_rate);
// val m9 = (i40) => fc6_W[@, i40]
JCudaMatrix m9 = fc6_W.asMatrix(1, false);
// val m11 = (i30) => X50[1><3][@, i30]
JCudaMatrix m11 = X50.flatten(1, new int[]{1024, 6, 6}).asMatrix(1, false);
// V_fc6_W = ((m10 * m11 * -0.01) + (V_fc6_W * 0.9))
m10.times(m11, V_fc6_W, lrn_rate, momentum);
// fc6_W = (V_fc6_W + (fc6_W * (1 + (5.0E-4 * -0.01))))
fc6_W.update(V_fc6_W, 1f, 1f + decay * lrn_rate);
// val X161 = (X149)(i39 | @) * m9
JCudaTensor X161 = X149.asMatrix(1, true).times(m9);
// dealloc X149
X149.free();
// val X147 = X161[1<>3] * d_Pooling(2,2,0,true)(X50,X49)/d_X49
JCudaTensor X147 = y2.backward(X161.unflatten(1, new int[]{1024, 6, 6}), X50, X49);
// dealloc X50
X50.free();
// dealloc X161
X161.free();
// val X168 = X147 * d_ReLU()(X49)/d_X48
JCudaTensor X168 = y3.backward(X147, X49);
// dealloc X49
X49.free();
// V_cv5_B = ((X168 * d_Convolv(1,1)()/d_cv5_B * (2 * -0.01)) + (V_cv5_B * 0.9))
y4.backward_bias(X168, V_cv5_B, 2f * lrn_rate, momentum);
// cv5_B = (V_cv5_B + cv5_B)
cv5_B.update(V_cv5_B, 1f, 1f);
// val X139 = X168 * d_Convolv(1,1)(cv5_W)/d_X47
JCudaTensor X139 = y4.backward_data(X168, cv5_W);
// V_cv5_W = ((X168 * d_Convolv(1,1)(X47)/d_cv5_W * -0.01) + (V_cv5_W * 0.9))
y4.backward_filter(X168, X47, V_cv5_W, lrn_rate, momentum);
// dealloc X168
X168.free();
// cv5_W = (V_cv5_W + (cv5_W * (1 + (5.0E-4 * -0.01))))
cv5_W.update(V_cv5_W, 1f, 1f + decay * lrn_rate);
// val X141 = X139 * d_ReLU()(X47)/d_X46
JCudaTensor X141 = y3.backward(X139, X47);
// dealloc X47
X47.free();
// V_cv4_B = ((X141 * d_Convolv(1,1)()/d_cv4_B * (2 * -0.01)) + (V_cv4_B * 0.9))
y5.backward_bias(X141, V_cv4_B, 2f * lrn_rate, momentum);
// cv4_B = (V_cv4_B + cv4_B)
cv4_B.update(V_cv4_B, 1f, 1f);
// val X145 = X141 * d_Convolv(1,1)(cv4_W)/d_X45
JCudaTensor X145 = y5.backward_data(X141, cv4_W);
// V_cv4_W = ((X141 * d_Convolv(1,1)(X45)/d_cv4_W * -0.01) + (V_cv4_W * 0.9))
y5.backward_filter(X141, X45, V_cv4_W, lrn_rate, momentum);
// dealloc X141
X141.free();
// cv4_W = (V_cv4_W + (cv4_W * (1 + (5.0E-4 * -0.01))))
cv4_W.update(V_cv4_W, 1f, 1f + decay * lrn_rate);
// val X166 = X145 * d_ReLU()(X45)/d_X44
JCudaTensor X166 = y6.backward(X145, X45);
// dealloc X45
X45.free();
// V_cv3_B = ((X166 * d_Convolv(1,1)()/d_cv3_B * (2 * -0.01)) + (V_cv3_B * 0.9))
y7.backward_bias(X166, V_cv3_B, 2f * lrn_rate, momentum);
// cv3_B = (V_cv3_B + cv3_B)
cv3_B.update(V_cv3_B, 1f, 1f);
// val X162 = X166 * d_Convolv(1,1)(cv3_W)/d_X43
JCudaTensor X162 = y7.backward_data(X166, cv3_W);
// V_cv3_W = ((X166 * d_Convolv(1,1)(X43)/d_cv3_W * -0.01) + (V_cv3_W * 0.9))
y7.backward_filter(X166, X43, V_cv3_W, lrn_rate, momentum);
// dealloc X166
X166.free();
// cv3_W = (V_cv3_W + (cv3_W * (1 + (5.0E-4 * -0.01))))
cv3_W.update(V_cv3_W, 1f, 1f + decay * lrn_rate);
// val X174 = X162 * d_Pooling(2,2,0,true)(X43,X42)/d_X42
JCudaTensor X174 = y8.backward(X162, X43, X42);
// dealloc X162
X162.free();
// dealloc X43
X43.free();
// val X138 = X174 * d_ReLU()(X42)/d_X41
JCudaTensor X138 = y9.backward(X174, X42);
// dealloc X42
X42.free();
// V_cv2_B = ((X138 * d_Convolv(1,2)()/d_cv2_B * (2 * -0.01)) + (V_cv2_B * 0.9))
y10.backward_bias(X138, V_cv2_B, 2f * lrn_rate, momentum);
// cv2_B = (V_cv2_B + cv2_B)
cv2_B.update(V_cv2_B, 1f, 1f);
// val X143 = X138 * d_Convolv(1,2)(cv2_W)/d_X40
JCudaTensor X143 = y10.backward_data(X138, cv2_W);
// V_cv2_W = ((X138 * d_Convolv(1,2)(X40)/d_cv2_W * -0.01) + (V_cv2_W * 0.9))
y10.backward_filter(X138, X40, V_cv2_W, lrn_rate, momentum);
// dealloc X138
X138.free();
// cv2_W = (V_cv2_W + (cv2_W * (1 + (5.0E-4 * -0.01))))
cv2_W.update(V_cv2_W, 1f, 1f + decay * lrn_rate);
// val X136 = X143 * d_Pooling(2,2,0,true)(X40,X39)/d_X39
JCudaTensor X136 = y11.backward(X143, X40, X39);
// dealloc X143
X143.free();
// dealloc X40
X40.free();
// val X172 = X136 * d_ReLU()(X39)/d_X38
JCudaTensor X172 = y12.backward(X136, X39);
// dealloc X39
X39.free();
// V_cv1_B = ((X172 * d_Convolv(4,0)()/d_cv1_B * (2 * -0.01)) + (V_cv1_B * 0.9))
y13.backward_bias(X172, V_cv1_B, 2f * lrn_rate, momentum);
// cv1_B = (V_cv1_B + cv1_B)
cv1_B.update(V_cv1_B, 1f, 1f);
// V_cv1_W = ((X172 * d_Convolv(4,0)(X123)/d_cv1_W * -0.01) + (V_cv1_W * 0.9))
y13.backward_filter(X172, X123, V_cv1_W, lrn_rate, momentum);
// dealloc X172
X172.free();
// dealloc X123
X123.free();
// cv1_W = (V_cv1_W + (cv1_W * (1 + (5.0E-4 * -0.01))))
cv1_W.update(V_cv1_W, 1f, 1f + decay * lrn_rate);

return _loss; 
}

public JCudaTensor testFunction(JTensorFloat X) {
 // val X14 = Cuda(X)
JCudaTensor X14 = X.asJCudaTensor();
// val X15 = Convolv(4,0)(X14,cv1_W,cv1_B)
JCudaTensor X15 = y13.forward(X14, cv1_W, cv1_B);
// dealloc X14
X14.free();
// val X16 = ReLU()(X15)
JCudaTensor X16 = y12.forward(X15);
// val X17 = Pooling(2,2,0,true)(X16)
JCudaTensor X17 = y11.forward(X16);
// dealloc X16
X16.free();
// val X18 = Convolv(1,2)(X17,cv2_W,cv2_B)
JCudaTensor X18 = y10.forward(X17, cv2_W, cv2_B);
// dealloc X17
X17.free();
// val X19 = ReLU()(X18)
JCudaTensor X19 = y9.forward(X18);
// val X20 = Pooling(2,2,0,true)(X19)
JCudaTensor X20 = y8.forward(X19);
// dealloc X19
X19.free();
// val X21 = Convolv(1,1)(X20,cv3_W,cv3_B)
JCudaTensor X21 = y7.forward(X20, cv3_W, cv3_B);
// dealloc X20
X20.free();
// val X22 = ReLU()(X21)
JCudaTensor X22 = y6.forward(X21);
// val X23 = Convolv(1,1)(X22,cv4_W,cv4_B)
JCudaTensor X23 = y5.forward(X22, cv4_W, cv4_B);
// dealloc X22
X22.free();
// val X24 = ReLU()(X23)
JCudaTensor X24 = y3.forward(X23);
// val X25 = Convolv(1,1)(X24,cv5_W,cv5_B)
JCudaTensor X25 = y4.forward(X24, cv5_W, cv5_B);
// dealloc X24
X24.free();
// val X26 = ReLU()(X25)
JCudaTensor X26 = y3.forward(X25);
// val X27 = Pooling(2,2,0,true)(X26)
JCudaTensor X27 = y2.forward(X26);
// dealloc X26
X26.free();
// val X32 = (X27[1><3])(i1 | @) * (fc6_W)(i2 | @)
JCudaTensor X32 = X27.flatten(1, new int[]{1024, 6, 6}).asMatrix(1, true).times(fc6_W.asMatrix(1, true));
// dealloc X27
X27.free();
// val X29 = (X32 + (i1) => fc6_B)
JCudaTensor X29 = fc6_B.copy(256, X32);
// val X34 = (X29)(i4 | @) * (fc7_W)(i5 | @)
JCudaTensor X34 = X29.asMatrix(1, true).times(fc7_W.asMatrix(1, true));
// dealloc X29
X29.free();
// val X30 = (X34 + (i4) => fc7_B)
JCudaTensor X30 = fc7_B.copy(256, X34);
// val X36 = (X30)(i7 | @) * (fc8_W)(i8 | @)
JCudaTensor X36 = X30.asMatrix(1, true).times(fc8_W.asMatrix(1, true));
// dealloc X30
X30.free();
// val X31 = (X36 + (i7) => fc8_B)
JCudaTensor X31 = fc8_B.copy(256, X36);

return X31; 
}

}