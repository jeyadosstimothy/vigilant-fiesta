package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.util.CudaRun;


public class Alexnet256 extends CudaRun {

public static void main(String[] args){
Alexnet256 run = new Alexnet256();
run.train(10);
run.test(1);
run.save();
run.free();
}

public Alexnet256() {
super("src/main/java/deepdsl/gen/alexnet256");
setTrainData(LmdbFactory.getFactory("dataset/imagenet/ilsvrc12_train_lmdb", 1000000, new int[]{256, 3, 224, 224}, 1000, false));
setTestData(LmdbFactory.getFactory("dataset/imagenet/ilsvrc12_val_lmdb", 10000, new int[]{256, 3, 224, 224}, 1000, true));
}

float lrn_rate = -0.01f;
float momentum = 0.1f;
float decay = 5.0E-4f;

JCudnnConvolution y10 = addConvolution(new int[]{256,256,13,13},new int[]{384,256,3,3},new int[]{384}, 1, 1);
JCudnnConvolution y7 = addConvolution(new int[]{256,384,13,13},new int[]{256,384,3,3},new int[]{256}, 1, 1);
JCudnnConvolution y9 = addConvolution(new int[]{256,384,13,13},new int[]{384,384,3,3},new int[]{384}, 1, 1);
JCudnnConvolution y14 = addConvolution(new int[]{256,96,27,27},new int[]{256,96,5,5},new int[]{256}, 1, 2);
JCudnnConvolution y18 = addConvolution(new int[]{256,3,224,224},new int[]{96,3,11,11},new int[]{96}, 4, 2);
JCudnnLRN y12 = addLRN(new int[]{256,256,27,27}, 5, 1.0E-4, 0.75);
JCudnnLRN y16 = addLRN(new int[]{256,96,55,55}, 5, 1.0E-4, 0.75);
JCudnnSoftmax y1 = addSoftmax(new int[]{256,1000}, SoftmaxAlgorithm.LOG);
JCudnnPooling y5 = addPooling(new int[]{256,256,13,13}, 3, 2, 0, PoolingType.MAX);
JCudnnPooling y11 = addPooling(new int[]{256,256,27,27}, 3, 2, 0, PoolingType.MAX);
JCudnnPooling y15 = addPooling(new int[]{256,96,55,55}, 3, 2, 0, PoolingType.MAX);
JCudnnActivation y6 = addActivation(new int[]{256,256,13,13}, ActivationMode.RELU);
JCudnnActivation y13 = addActivation(new int[]{256,256,27,27}, ActivationMode.RELU);
JCudnnActivation y8 = addActivation(new int[]{256,384,13,13}, ActivationMode.RELU);
JCudnnActivation y3 = addActivation(new int[]{256,4096}, ActivationMode.RELU);
JCudnnActivation y17 = addActivation(new int[]{256,96,55,55}, ActivationMode.RELU);
JCudnnDropout y4 = addDropout("y4", new int[]{256,4096}, 0.5f);
JCudnnDropout y2 = addDropout("y2", new int[]{256,4096}, 0.5f);
JCudaTensor V_cv1_B = addParam("V_cv1_B", "Constant", 0f, 96);
JCudaTensor V_cv1_W = addParam("V_cv1_W", "Constant", 0f, 96, 3, 11, 11);
JCudaTensor V_cv2_B = addParam("V_cv2_B", "Constant", 0f, 256);
JCudaTensor V_cv2_W = addParam("V_cv2_W", "Constant", 0f, 256, 96, 5, 5);
JCudaTensor V_cv3_B = addParam("V_cv3_B", "Constant", 0f, 384);
JCudaTensor V_cv3_W = addParam("V_cv3_W", "Constant", 0f, 384, 256, 3, 3);
JCudaTensor V_cv4_B = addParam("V_cv4_B", "Constant", 0f, 384);
JCudaTensor V_cv4_W = addParam("V_cv4_W", "Constant", 0f, 384, 384, 3, 3);
JCudaTensor V_cv5_B = addParam("V_cv5_B", "Constant", 0f, 256);
JCudaTensor V_cv5_W = addParam("V_cv5_W", "Constant", 0f, 256, 384, 3, 3);
JCudaTensor V_fc6_B = addParam("V_fc6_B", "Constant", 0f, 4096);
JCudaTensor V_fc6_W = addParam("V_fc6_W", "Constant", 0f, 4096, 9216);
JCudaTensor V_fc7_B = addParam("V_fc7_B", "Constant", 0f, 4096);
JCudaTensor V_fc7_W = addParam("V_fc7_W", "Constant", 0f, 4096, 4096);
JCudaTensor V_fc8_B = addParam("V_fc8_B", "Constant", 0f, 1000);
JCudaTensor V_fc8_W = addParam("V_fc8_W", "Constant", 0f, 1000, 4096);
JCudaTensor cv1_B = addParam("cv1_B", "Constant", 0.0f, 96);
JCudaTensor cv1_W = addParam("cv1_W", "Gaussian", 0.01f, 96, 3, 11, 11);
JCudaTensor cv2_B = addParam("cv2_B", "Constant", 0.1f, 256);
JCudaTensor cv2_W = addParam("cv2_W", "Gaussian", 0.01f, 256, 96, 5, 5);
JCudaTensor cv3_B = addParam("cv3_B", "Constant", 0.0f, 384);
JCudaTensor cv3_W = addParam("cv3_W", "Gaussian", 0.01f, 384, 256, 3, 3);
JCudaTensor cv4_B = addParam("cv4_B", "Constant", 0.1f, 384);
JCudaTensor cv4_W = addParam("cv4_W", "Gaussian", 0.01f, 384, 384, 3, 3);
JCudaTensor cv5_B = addParam("cv5_B", "Constant", 0.1f, 256);
JCudaTensor cv5_W = addParam("cv5_W", "Gaussian", 0.01f, 256, 384, 3, 3);
JCudaTensor fc6_B = addParam("fc6_B", "Constant", 0.1f, 4096);
JCudaTensor fc6_W = addParam("fc6_W", "Gaussian", 0.005f, 4096, 9216);
JCudaTensor fc7_B = addParam("fc7_B", "Constant", 0.1f, 4096);
JCudaTensor fc7_W = addParam("fc7_W", "Gaussian", 0.005f, 4096, 4096);
JCudaTensor fc8_B = addParam("fc8_B", "Constant", 0.0f, 1000);
JCudaTensor fc8_W = addParam("fc8_W", "Gaussian", 0.01f, 1000, 4096);

public float trainFunction(JTensorFloat X, JTensorFloat Y) {
 // val X152 = Cuda(X)
JCudaTensor X152 = X.asJCudaTensor();
// val X155 = Cuda(Indicator(Y, 1000))
JCudaTensor X155 = Y.asIndicator(1000).asJCudaTensor();
// val X45 = Convolv(4,2)(X152,cv1_W,cv1_B)
JCudaTensor X45 = y18.forward(X152, cv1_W, cv1_B);
// val X46 = ReLU()(X45)
JCudaTensor X46 = y17.forward(X45);
// val X47 = LRN(5,1.0E-4,0.75)(X46)
JCudaTensor X47 = y16.forward(X46);
// val X48 = Pooling(3,2,0,true)(X47)
JCudaTensor X48 = y15.forward(X47);
// val X49 = Convolv(1,2)(X48,cv2_W,cv2_B)
JCudaTensor X49 = y14.forward(X48, cv2_W, cv2_B);
// val X50 = ReLU()(X49)
JCudaTensor X50 = y13.forward(X49);
// val X51 = LRN(5,1.0E-4,0.75)(X50)
JCudaTensor X51 = y12.forward(X50);
// val X52 = Pooling(3,2,0,true)(X51)
JCudaTensor X52 = y11.forward(X51);
// val X53 = Convolv(1,1)(X52,cv3_W,cv3_B)
JCudaTensor X53 = y10.forward(X52, cv3_W, cv3_B);
// val X54 = ReLU()(X53)
JCudaTensor X54 = y8.forward(X53);
// val X55 = Convolv(1,1)(X54,cv4_W,cv4_B)
JCudaTensor X55 = y9.forward(X54, cv4_W, cv4_B);
// val X56 = ReLU()(X55)
JCudaTensor X56 = y8.forward(X55);
// val X57 = Convolv(1,1)(X56,cv5_W,cv5_B)
JCudaTensor X57 = y7.forward(X56, cv5_W, cv5_B);
// val X58 = ReLU()(X57)
JCudaTensor X58 = y6.forward(X57);
// val X59 = Pooling(3,2,0,true)(X58)
JCudaTensor X59 = y5.forward(X58);
// val X148 = (X59[1><3])(i10 | @) * (fc6_W)(i11 | @)
JCudaTensor X148 = X59.flatten(1, new int[]{256, 6, 6}).asMatrix(1, true).times(fc6_W.asMatrix(1, true));
// val X61 = (X148 + (i10) => fc6_B)
JCudaTensor X61 = fc6_B.copy(256, X148);
// val X62 = ReLU()(X61)
JCudaTensor X62 = y3.forward(X61);
// val X63 = Dropout(0.5)(X62)
JCudaTensor X63 = y4.forward(X62);
// val X153 = (X63)(i13 | @) * (fc7_W)(i14 | @)
JCudaTensor X153 = X63.asMatrix(1, true).times(fc7_W.asMatrix(1, true));
// val X64 = (X153 + (i13) => fc7_B)
JCudaTensor X64 = fc7_B.copy(256, X153);
// val X65 = ReLU()(X64)
JCudaTensor X65 = y3.forward(X64);
// val X66 = Dropout(0.5)(X65)
JCudaTensor X66 = y2.forward(X65);
// val X150 = (X66)(i16 | @) * (fc8_W)(i17 | @)
JCudaTensor X150 = X66.asMatrix(1, true).times(fc8_W.asMatrix(1, true));
// val X67 = (X150 + (i16) => fc8_B)
JCudaTensor X67 = fc8_B.copy(256, X150);
// val X68 = LogSoftmax()(X67)
JCudaTensor X68 = y1.forward(X67);
// dealloc X67
X67.free();
// val _loss = ((0 - (X155 . X68)) / |256|)
float _loss = - X155.dot(X68) / 256f;
// val X183 = - X155.copy
JCudaTensor X183 = X155.clone().times_i(-1f);;
// dealloc X155
X155.free();
// val X69 = (X183 / |256|)
JCudaTensor X69 = X183.times_i(1 / 256f);;
// val X167 = X69 * d_LogSoftmax()(X68)/d_X67
JCudaTensor X167 = y1.backward(X69, X68);
// dealloc X69
X69.free();
// dealloc X68
X68.free();
// val m2 = (i21) => X167[@, i21]
JCudaMatrix m2 = X167.asMatrix(1, false);
// V_fc8_B = ((Sum(m2) * (2 * -0.01)) + (V_fc8_B * 0.1))
m2.sum(V_fc8_B, 2f * lrn_rate, momentum);
// fc8_B = (V_fc8_B + fc8_B)
fc8_B.update(V_fc8_B, 1f, 1f);
// val m4 = (i54) => fc8_W[@, i54]
JCudaMatrix m4 = fc8_W.asMatrix(1, false);
// val m3 = (i22) => X66[@, i22]
JCudaMatrix m3 = X66.asMatrix(1, false);
// V_fc8_W = ((m2 * m3 * -0.01) + (V_fc8_W * 0.1))
m2.times(m3, V_fc8_W, lrn_rate, momentum);
// dealloc X66
X66.free();
// fc8_W = (V_fc8_W + (fc8_W * (1 + (5.0E-4 * -0.01))))
fc8_W.update(V_fc8_W, 1f, 1f + decay * lrn_rate);
// val X189 = (X167)(i53 | @) * m4
JCudaTensor X189 = X167.asMatrix(1, true).times(m4);
// dealloc X167
X167.free();
// val X191 = X189 * d_Dropout(0.5)()/d_X65
JCudaTensor X191 = y2.backward(X189);
// dealloc X189
X189.free();
// val X180 = X191 * d_ReLU()(X65)/d_X64
JCudaTensor X180 = y3.backward(X191, X65);
// dealloc X65
X65.free();
// val m5 = (i25) => X180[@, i25]
JCudaMatrix m5 = X180.asMatrix(1, false);
// V_fc7_B = ((Sum(m5) * (2 * -0.01)) + (V_fc7_B * 0.1))
m5.sum(V_fc7_B, 2f * lrn_rate, momentum);
// fc7_B = (V_fc7_B + fc7_B)
fc7_B.update(V_fc7_B, 1f, 1f);
// val m7 = (i47) => fc7_W[@, i47]
JCudaMatrix m7 = fc7_W.asMatrix(1, false);
// val m6 = (i26) => X63[@, i26]
JCudaMatrix m6 = X63.asMatrix(1, false);
// V_fc7_W = ((m5 * m6 * -0.01) + (V_fc7_W * 0.1))
m5.times(m6, V_fc7_W, lrn_rate, momentum);
// dealloc X63
X63.free();
// fc7_W = (V_fc7_W + (fc7_W * (1 + (5.0E-4 * -0.01))))
fc7_W.update(V_fc7_W, 1f, 1f + decay * lrn_rate);
// val X195 = (X180)(i46 | @) * m7
JCudaTensor X195 = X180.asMatrix(1, true).times(m7);
// dealloc X180
X180.free();
// val X181 = X195 * d_Dropout(0.5)()/d_X62
JCudaTensor X181 = y4.backward(X195);
// dealloc X195
X195.free();
// val X159 = X181 * d_ReLU()(X62)/d_X61
JCudaTensor X159 = y3.backward(X181, X62);
// dealloc X62
X62.free();
// val m1 = (i43) => X159[@, i43]
JCudaMatrix m1 = X159.asMatrix(1, false);
// V_fc6_B = ((Sum(m1) * (2 * -0.01)) + (V_fc6_B * 0.1))
m1.sum(V_fc6_B, 2f * lrn_rate, momentum);
// fc6_B = (V_fc6_B + fc6_B)
fc6_B.update(V_fc6_B, 1f, 1f);
// val m12 = (i40) => fc6_W[@, i40]
JCudaMatrix m12 = fc6_W.asMatrix(1, false);
// val m9 = (i30) => X59[1><3][@, i30]
JCudaMatrix m9 = X59.flatten(1, new int[]{256, 6, 6}).asMatrix(1, false);
// V_fc6_W = ((m1 * m9 * -0.01) + (V_fc6_W * 0.1))
m1.times(m9, V_fc6_W, lrn_rate, momentum);
// fc6_W = (V_fc6_W + (fc6_W * (1 + (5.0E-4 * -0.01))))
fc6_W.update(V_fc6_W, 1f, 1f + decay * lrn_rate);
// val X204 = (X159)(i39 | @) * m12
JCudaTensor X204 = X159.asMatrix(1, true).times(m12);
// dealloc X159
X159.free();
// val X208 = X204[1<>3] * d_Pooling(3,2,0,true)(X59,X58)/d_X58
JCudaTensor X208 = y5.backward(X204.unflatten(1, new int[]{256, 6, 6}), X59, X58);
// dealloc X59
X59.free();
// dealloc X204
X204.free();
// val X178 = X208 * d_ReLU()(X58)/d_X57
JCudaTensor X178 = y6.backward(X208, X58);
// dealloc X58
X58.free();
// V_cv5_B = ((X178 * d_Convolv(1,1)()/d_cv5_B * (2 * -0.01)) + (V_cv5_B * 0.1))
y7.backward_bias(X178, V_cv5_B, 2f * lrn_rate, momentum);
// cv5_B = (V_cv5_B + cv5_B)
cv5_B.update(V_cv5_B, 1f, 1f);
// val X205 = X178 * d_Convolv(1,1)(cv5_W)/d_X56
JCudaTensor X205 = y7.backward_data(X178, cv5_W);
// V_cv5_W = ((X178 * d_Convolv(1,1)(X56)/d_cv5_W * -0.01) + (V_cv5_W * 0.1))
y7.backward_filter(X178, X56, V_cv5_W, lrn_rate, momentum);
// dealloc X178
X178.free();
// cv5_W = (V_cv5_W + (cv5_W * (1 + (5.0E-4 * -0.01))))
cv5_W.update(V_cv5_W, 1f, 1f + decay * lrn_rate);
// val X194 = X205 * d_ReLU()(X56)/d_X55
JCudaTensor X194 = y8.backward(X205, X56);
// dealloc X56
X56.free();
// V_cv4_B = ((X194 * d_Convolv(1,1)()/d_cv4_B * (2 * -0.01)) + (V_cv4_B * 0.1))
y9.backward_bias(X194, V_cv4_B, 2f * lrn_rate, momentum);
// cv4_B = (V_cv4_B + cv4_B)
cv4_B.update(V_cv4_B, 1f, 1f);
// val X199 = X194 * d_Convolv(1,1)(cv4_W)/d_X54
JCudaTensor X199 = y9.backward_data(X194, cv4_W);
// V_cv4_W = ((X194 * d_Convolv(1,1)(X54)/d_cv4_W * -0.01) + (V_cv4_W * 0.1))
y9.backward_filter(X194, X54, V_cv4_W, lrn_rate, momentum);
// dealloc X194
X194.free();
// cv4_W = (V_cv4_W + (cv4_W * (1 + (5.0E-4 * -0.01))))
cv4_W.update(V_cv4_W, 1f, 1f + decay * lrn_rate);
// val X197 = X199 * d_ReLU()(X54)/d_X53
JCudaTensor X197 = y8.backward(X199, X54);
// dealloc X54
X54.free();
// V_cv3_B = ((X197 * d_Convolv(1,1)()/d_cv3_B * (2 * -0.01)) + (V_cv3_B * 0.1))
y10.backward_bias(X197, V_cv3_B, 2f * lrn_rate, momentum);
// cv3_B = (V_cv3_B + cv3_B)
cv3_B.update(V_cv3_B, 1f, 1f);
// val X203 = X197 * d_Convolv(1,1)(cv3_W)/d_X52
JCudaTensor X203 = y10.backward_data(X197, cv3_W);
// V_cv3_W = ((X197 * d_Convolv(1,1)(X52)/d_cv3_W * -0.01) + (V_cv3_W * 0.1))
y10.backward_filter(X197, X52, V_cv3_W, lrn_rate, momentum);
// dealloc X197
X197.free();
// cv3_W = (V_cv3_W + (cv3_W * (1 + (5.0E-4 * -0.01))))
cv3_W.update(V_cv3_W, 1f, 1f + decay * lrn_rate);
// val X163 = X203 * d_Pooling(3,2,0,true)(X52,X51)/d_X51
JCudaTensor X163 = y11.backward(X203, X52, X51);
// dealloc X203
X203.free();
// dealloc X52
X52.free();
// val X157 = X163 * d_LRN(5,1.0E-4,0.75)(X51,X50)/d_X50
JCudaTensor X157 = y12.backward(X163, X51, X50);
// dealloc X51
X51.free();
// val X174 = X157 * d_ReLU()(X50)/d_X49
JCudaTensor X174 = y13.backward(X157, X50);
// dealloc X50
X50.free();
// V_cv2_B = ((X174 * d_Convolv(1,2)()/d_cv2_B * (2 * -0.01)) + (V_cv2_B * 0.1))
y14.backward_bias(X174, V_cv2_B, 2f * lrn_rate, momentum);
// cv2_B = (V_cv2_B + cv2_B)
cv2_B.update(V_cv2_B, 1f, 1f);
// val X184 = X174 * d_Convolv(1,2)(cv2_W)/d_X48
JCudaTensor X184 = y14.backward_data(X174, cv2_W);
// V_cv2_W = ((X174 * d_Convolv(1,2)(X48)/d_cv2_W * -0.01) + (V_cv2_W * 0.1))
y14.backward_filter(X174, X48, V_cv2_W, lrn_rate, momentum);
// dealloc X174
X174.free();
// cv2_W = (V_cv2_W + (cv2_W * (1 + (5.0E-4 * -0.01))))
cv2_W.update(V_cv2_W, 1f, 1f + decay * lrn_rate);
// val X171 = X184 * d_Pooling(3,2,0,true)(X48,X47)/d_X47
JCudaTensor X171 = y15.backward(X184, X48, X47);
// dealloc X48
X48.free();
// dealloc X184
X184.free();
// val X169 = X171 * d_LRN(5,1.0E-4,0.75)(X47,X46)/d_X46
JCudaTensor X169 = y16.backward(X171, X47, X46);
// dealloc X47
X47.free();
// val X176 = X169 * d_ReLU()(X46)/d_X45
JCudaTensor X176 = y17.backward(X169, X46);
// dealloc X46
X46.free();
// V_cv1_B = ((X176 * d_Convolv(4,2)()/d_cv1_B * (2 * -0.01)) + (V_cv1_B * 0.1))
y18.backward_bias(X176, V_cv1_B, 2f * lrn_rate, momentum);
// cv1_B = (V_cv1_B + cv1_B)
cv1_B.update(V_cv1_B, 1f, 1f);
// V_cv1_W = ((X176 * d_Convolv(4,2)(X152)/d_cv1_W * -0.01) + (V_cv1_W * 0.1))
y18.backward_filter(X176, X152, V_cv1_W, lrn_rate, momentum);
// dealloc X176
X176.free();
// dealloc X152
X152.free();
// cv1_W = (V_cv1_W + (cv1_W * (1 + (5.0E-4 * -0.01))))
cv1_W.update(V_cv1_W, 1f, 1f + decay * lrn_rate);

return _loss; 
}

public JCudaTensor testFunction(JTensorFloat X) {
 // val X17 = Cuda(X)
JCudaTensor X17 = X.asJCudaTensor();
// val X18 = Convolv(4,2)(X17,cv1_W,cv1_B)
JCudaTensor X18 = y18.forward(X17, cv1_W, cv1_B);
// dealloc X17
X17.free();
// val X19 = ReLU()(X18)
JCudaTensor X19 = y17.forward(X18);
// val X20 = LRN(5,1.0E-4,0.75)(X19)
JCudaTensor X20 = y16.forward(X19);
// dealloc X19
X19.free();
// val X21 = Pooling(3,2,0,true)(X20)
JCudaTensor X21 = y15.forward(X20);
// dealloc X20
X20.free();
// val X22 = Convolv(1,2)(X21,cv2_W,cv2_B)
JCudaTensor X22 = y14.forward(X21, cv2_W, cv2_B);
// dealloc X21
X21.free();
// val X23 = ReLU()(X22)
JCudaTensor X23 = y13.forward(X22);
// val X24 = LRN(5,1.0E-4,0.75)(X23)
JCudaTensor X24 = y12.forward(X23);
// dealloc X23
X23.free();
// val X25 = Pooling(3,2,0,true)(X24)
JCudaTensor X25 = y11.forward(X24);
// dealloc X24
X24.free();
// val X26 = Convolv(1,1)(X25,cv3_W,cv3_B)
JCudaTensor X26 = y10.forward(X25, cv3_W, cv3_B);
// dealloc X25
X25.free();
// val X27 = ReLU()(X26)
JCudaTensor X27 = y8.forward(X26);
// val X28 = Convolv(1,1)(X27,cv4_W,cv4_B)
JCudaTensor X28 = y9.forward(X27, cv4_W, cv4_B);
// dealloc X27
X27.free();
// val X29 = ReLU()(X28)
JCudaTensor X29 = y8.forward(X28);
// val X30 = Convolv(1,1)(X29,cv5_W,cv5_B)
JCudaTensor X30 = y7.forward(X29, cv5_W, cv5_B);
// dealloc X29
X29.free();
// val X31 = ReLU()(X30)
JCudaTensor X31 = y6.forward(X30);
// val X32 = Pooling(3,2,0,true)(X31)
JCudaTensor X32 = y5.forward(X31);
// dealloc X31
X31.free();
// val X39 = (X32[1><3])(i10 | @) * (fc6_W)(i11 | @)
JCudaTensor X39 = X32.flatten(1, new int[]{256, 6, 6}).asMatrix(1, true).times(fc6_W.asMatrix(1, true));
// dealloc X32
X32.free();
// val X34 = (X39 + (i10) => fc6_B)
JCudaTensor X34 = fc6_B.copy(256, X39);
// val X35 = ReLU()(X34)
JCudaTensor X35 = y3.forward(X34);
// val X41 = (X35)(i13 | @) * (fc7_W)(i14 | @)
JCudaTensor X41 = X35.asMatrix(1, true).times(fc7_W.asMatrix(1, true));
// dealloc X35
X35.free();
// val X36 = (X41 + (i13) => fc7_B)
JCudaTensor X36 = fc7_B.copy(256, X41);
// val X37 = ReLU()(X36)
JCudaTensor X37 = y3.forward(X36);
// val X43 = (X37)(i16 | @) * (fc8_W)(i17 | @)
JCudaTensor X43 = X37.asMatrix(1, true).times(fc8_W.asMatrix(1, true));
// dealloc X37
X37.free();
// val X38 = (X43 + (i16) => fc8_B)
JCudaTensor X38 = fc8_B.copy(256, X43);

return X38; 
}

}