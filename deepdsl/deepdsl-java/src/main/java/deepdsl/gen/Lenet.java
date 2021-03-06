package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.util.CudaRun;


public class Lenet extends CudaRun {

public static void main(String[] args){
Lenet run = new Lenet();
run.train(10);
run.test(1);
run.infer(1);
run.save();
run.free();
}

public Lenet() {
super("src/main/java/deepdsl/gen/lenet");
setTrainData(MnistFactory.getFactory(true, new int[]{512, 1, 28, 28}));
setTestData(MnistFactory.getFactory(false, new int[]{512, 1, 28, 28}));
}

float lrn_rate = -0.01f;
float momentum = 0.1f;
float decay = 5.0E-4f;

JCudnnConvolution y6 = addConvolution(new int[]{512,1,28,28},new int[]{20,1,5,5},new int[]{20}, 1, 0);
JCudnnConvolution y4 = addConvolution(new int[]{512,20,12,12},new int[]{50,20,5,5},new int[]{50}, 1, 0);
JCudnnPooling y5 = addPooling(new int[]{512,20,24,24}, 2, 2, 0, PoolingType.MAX);
JCudnnPooling y3 = addPooling(new int[]{512,50,8,8}, 2, 2, 0, PoolingType.MAX);
JCudnnActivation y2 = addActivation(new int[]{512,500}, ActivationMode.RELU);
JCudnnSoftmax y1 = addSoftmax(new int[]{512,10}, SoftmaxAlgorithm.ACCURATE);
JCudaTensor V_cv1_B = addParam("V_cv1_B", "Constant", 0f, 20);
JCudaTensor V_cv1_W = addParam("V_cv1_W", "Constant", 0f, 20, 1, 5, 5);
JCudaTensor V_cv2_B = addParam("V_cv2_B", "Constant", 0f, 50);
JCudaTensor V_cv2_W = addParam("V_cv2_W", "Constant", 0f, 50, 20, 5, 5);
JCudaTensor V_fc1_B = addParam("V_fc1_B", "Constant", 0f, 500);
JCudaTensor V_fc1_W = addParam("V_fc1_W", "Constant", 0f, 500, 800);
JCudaTensor V_fc2_B = addParam("V_fc2_B", "Constant", 0f, 10);
JCudaTensor V_fc2_W = addParam("V_fc2_W", "Constant", 0f, 10, 500);
JCudaTensor cv1_B = addParam("cv1_B", "Constant", 0.0f, 20);
JCudaTensor cv1_W = addParam("cv1_W", "Random", 0.28284273f, 20, 1, 5, 5);
JCudaTensor cv2_B = addParam("cv2_B", "Constant", 0.0f, 50);
JCudaTensor cv2_W = addParam("cv2_W", "Random", 0.06324555f, 50, 20, 5, 5);
JCudaTensor fc1_B = addParam("fc1_B", "Constant", 0.0f, 500);
JCudaTensor fc1_W = addParam("fc1_W", "Random", 0.05f, 500, 800);
JCudaTensor fc2_B = addParam("fc2_B", "Constant", 0.0f, 10);
JCudaTensor fc2_W = addParam("fc2_W", "Random", 0.06324555f, 10, 500);

public float trainFunction(JTensorFloat X, JTensorFloat Y) {
 // val X70 = Cuda(Indicator(Y, 10))
JCudaTensor X70 = Y.asIndicator(10).asJCudaTensor();
// val X67 = Cuda(X)
JCudaTensor X67 = X.asJCudaTensor();
// val X23 = Convolv(1,0)(X67,cv1_W,cv1_B)
JCudaTensor X23 = y6.forward(X67, cv1_W, cv1_B);
// val X24 = Pooling(2,2,0,true)(X23)
JCudaTensor X24 = y5.forward(X23);
// val X25 = Convolv(1,0)(X24,cv2_W,cv2_B)
JCudaTensor X25 = y4.forward(X24, cv2_W, cv2_B);
// val X26 = Pooling(2,2,0,true)(X25)
JCudaTensor X26 = y3.forward(X25);
// val X65 = (X26[1><3])(i10 | @) * (fc1_W)(i11 | @)
JCudaTensor X65 = X26.flatten(1, new int[]{50, 4, 4}).asMatrix(1, true).times(fc1_W.asMatrix(1, true));
// val X28 = (X65 + (i10) => fc1_B)
JCudaTensor X28 = fc1_B.copy(512, X65);
// val X29 = ReLU()(X28)
JCudaTensor X29 = y2.forward(X28);
// val X68 = (X29)(i13 | @) * (fc2_W)(i14 | @)
JCudaTensor X68 = X29.asMatrix(1, true).times(fc2_W.asMatrix(1, true));
// val X30 = (X68 + (i13) => fc2_B)
JCudaTensor X30 = fc2_B.copy(512, X68);
// val X31 = Softmax()(X30)
JCudaTensor X31 = y1.forward(X30);
// dealloc X30
X30.free();
// val X71 = Log X31.copy
JCudaTensor X71 = X31.clone().log();
// val _loss = ((0 - (X70 . X71)) / |512|)
float _loss = - X70.dot(X71) / 512f;
// dealloc X71
X71.free();
// val m4 = (i40) => fc2_W[@, i40]
JCudaMatrix m4 = fc2_W.asMatrix(1, false);
// val m7 = (i19) => X29[@, i19]
JCudaMatrix m7 = X29.asMatrix(1, false);
// val X93 = 1/(X31.copy)
JCudaTensor X93 = X31.clone().pow(-1f);
// val X94 = X70.copy .* X93
JCudaTensor X94 = X70.clone().times_i(X93);;
// dealloc X70
X70.free();
// dealloc X93
X93.free();
// val X95 = - X94
JCudaTensor X95 = X94.times_i(-1f);;
// val X32 = (X95 / |512|)
JCudaTensor X32 = X95.times_i(1 / 512f);;
// val X88 = X32 * d_Softmax()(X31)/d_X30
JCudaTensor X88 = y1.backward(X32, X31);
// dealloc X31
X31.free();
// dealloc X32
X32.free();
// val m3 = (i43) => X88[@, i43]
JCudaMatrix m3 = X88.asMatrix(1, false);
// V_fc2_W = ((m3 * m7 * -0.01) + (V_fc2_W * 0.1))
m3.times(m7, V_fc2_W, lrn_rate, momentum);
// fc2_W = (V_fc2_W + (fc2_W * (1 + (5.0E-4 * -0.01))))
fc2_W.update(V_fc2_W, 1f, 1f + decay * lrn_rate);
// V_fc2_B = ((Sum(m3) * -0.01) + (V_fc2_B * 0.1))
m3.sum(V_fc2_B, lrn_rate, momentum);
// fc2_B = (V_fc2_B + (fc2_B * (1 + (5.0E-4 * -0.01))))
fc2_B.update(V_fc2_B, 1f, 1f + decay * lrn_rate);
// val X74 = (X88)(i39 | @) * m4
JCudaTensor X74 = X88.asMatrix(1, true).times(m4);
// dealloc X88
X88.free();
// val X86 = X74 * d_ReLU()(X29)/d_X28
JCudaTensor X86 = y2.backward(X74, X29);
// dealloc X29
X29.free();
// val m1 = (i22) => X86[@, i22]
JCudaMatrix m1 = X86.asMatrix(1, false);
// V_fc1_B = ((Sum(m1) * -0.01) + (V_fc1_B * 0.1))
m1.sum(V_fc1_B, lrn_rate, momentum);
// fc1_B = (V_fc1_B + (fc1_B * (1 + (5.0E-4 * -0.01))))
fc1_B.update(V_fc1_B, 1f, 1f + decay * lrn_rate);
// val m8 = (i33) => fc1_W[@, i33]
JCudaMatrix m8 = fc1_W.asMatrix(1, false);
// val m2 = (i23) => X26[1><3][@, i23]
JCudaMatrix m2 = X26.flatten(1, new int[]{50, 4, 4}).asMatrix(1, false);
// V_fc1_W = ((m1 * m2 * -0.01) + (V_fc1_W * 0.1))
m1.times(m2, V_fc1_W, lrn_rate, momentum);
// fc1_W = (V_fc1_W + (fc1_W * (1 + (5.0E-4 * -0.01))))
fc1_W.update(V_fc1_W, 1f, 1f + decay * lrn_rate);
// val X83 = (X86)(i32 | @) * m8
JCudaTensor X83 = X86.asMatrix(1, true).times(m8);
// dealloc X86
X86.free();
// val X76 = X83[1<>3] * d_Pooling(2,2,0,true)(X26,X25)/d_X25
JCudaTensor X76 = y3.backward(X83.unflatten(1, new int[]{50, 4, 4}), X26, X25);
// dealloc X26
X26.free();
// dealloc X25
X25.free();
// dealloc X83
X83.free();
// V_cv2_W = ((X76 * d_Convolv(1,0)(X24)/d_cv2_W * -0.01) + (V_cv2_W * 0.1))
y4.backward_filter(X76, X24, V_cv2_W, lrn_rate, momentum);
// val X82 = X76 * d_Convolv(1,0)(cv2_W)/d_X24
JCudaTensor X82 = y4.backward_data(X76, cv2_W);
// cv2_W = (V_cv2_W + (cv2_W * (1 + (5.0E-4 * -0.01))))
cv2_W.update(V_cv2_W, 1f, 1f + decay * lrn_rate);
// V_cv2_B = ((X76 * d_Convolv(1,0)()/d_cv2_B * -0.01) + (V_cv2_B * 0.1))
y4.backward_bias(X76, V_cv2_B, lrn_rate, momentum);
// dealloc X76
X76.free();
// cv2_B = (V_cv2_B + (cv2_B * (1 + (5.0E-4 * -0.01))))
cv2_B.update(V_cv2_B, 1f, 1f + decay * lrn_rate);
// val X97 = X82 * d_Pooling(2,2,0,true)(X24,X23)/d_X23
JCudaTensor X97 = y5.backward(X82, X24, X23);
// dealloc X23
X23.free();
// dealloc X82
X82.free();
// dealloc X24
X24.free();
// V_cv1_W = ((X97 * d_Convolv(1,0)(X67)/d_cv1_W * -0.01) + (V_cv1_W * 0.1))
y6.backward_filter(X97, X67, V_cv1_W, lrn_rate, momentum);
// dealloc X67
X67.free();
// cv1_W = (V_cv1_W + (cv1_W * (1 + (5.0E-4 * -0.01))))
cv1_W.update(V_cv1_W, 1f, 1f + decay * lrn_rate);
// V_cv1_B = ((X97 * d_Convolv(1,0)()/d_cv1_B * -0.01) + (V_cv1_B * 0.1))
y6.backward_bias(X97, V_cv1_B, lrn_rate, momentum);
// dealloc X97
X97.free();
// cv1_B = (V_cv1_B + (cv1_B * (1 + (5.0E-4 * -0.01))))
cv1_B.update(V_cv1_B, 1f, 1f + decay * lrn_rate);

return _loss; 
}

public JCudaTensor testFunction(JTensorFloat X) {
 // val X10 = Cuda(X)
JCudaTensor X10 = X.asJCudaTensor();
// val X11 = Convolv(1,0)(X10,cv1_W,cv1_B)
JCudaTensor X11 = y6.forward(X10, cv1_W, cv1_B);
// dealloc X10
X10.free();
// val X12 = Pooling(2,2,0,true)(X11)
JCudaTensor X12 = y5.forward(X11);
// dealloc X11
X11.free();
// val X13 = Convolv(1,0)(X12,cv2_W,cv2_B)
JCudaTensor X13 = y4.forward(X12, cv2_W, cv2_B);
// dealloc X12
X12.free();
// val X14 = Pooling(2,2,0,true)(X13)
JCudaTensor X14 = y3.forward(X13);
// dealloc X13
X13.free();
// val X19 = (X14[1><3])(i10 | @) * (fc1_W)(i11 | @)
JCudaTensor X19 = X14.flatten(1, new int[]{50, 4, 4}).asMatrix(1, true).times(fc1_W.asMatrix(1, true));
// dealloc X14
X14.free();
// val X16 = (X19 + (i10) => fc1_B)
JCudaTensor X16 = fc1_B.copy(512, X19);
// val X17 = ReLU()(X16)
JCudaTensor X17 = y2.forward(X16);
// val X21 = (X17)(i13 | @) * (fc2_W)(i14 | @)
JCudaTensor X21 = X17.asMatrix(1, true).times(fc2_W.asMatrix(1, true));
// dealloc X17
X17.free();
// val X18 = (X21 + (i13) => fc2_B)
JCudaTensor X18 = fc2_B.copy(512, X21);

return X18; 
}

}