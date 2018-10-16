package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.util.CudaRun;


public class MyNet extends CudaRun {

public static void main(String[] args){
MyNet run = new MyNet();
run.train(50);
run.test(1);
run.save();
run.free();
}

public MyNet() {
super("src/main/java/deepdsl/gen/MyNet");
setTrainData(MnistFactory.getFactory(true, new int[]{32, 1, 28, 28}));
setTestData(MnistFactory.getFactory(false, new int[]{32, 1, 28, 28}));
}

float lrn_rate = -0.01f;
float momentum = 0.1f;
float decay = 5.0E-4f;

JCudnnConvolution y6 = addConvolution(new int[]{32,1,28,28},new int[]{20,1,5,5},new int[]{20}, 1, 0);
JCudnnConvolution y4 = addConvolution(new int[]{32,20,12,12},new int[]{20,20,5,5},new int[]{20}, 1, 0);
JCudnnPooling y5 = addPooling(new int[]{32,20,24,24}, 2, 2, 0, PoolingType.MAX);
JCudnnPooling y3 = addPooling(new int[]{32,20,8,8}, 2, 2, 0, PoolingType.MAX);
JCudnnActivation y2 = addActivation(new int[]{32,500}, ActivationMode.RELU);
JCudnnSoftmax y1 = addSoftmax(new int[]{32,10}, SoftmaxAlgorithm.ACCURATE);
JCudaTensor V_cv0_B = addParam("V_cv0_B", "Constant", 0f, 20);
JCudaTensor V_cv0_W = addParam("V_cv0_W", "Constant", 0f, 20, 1, 5, 5);
JCudaTensor V_cv1_B = addParam("V_cv1_B", "Constant", 0f, 20);
JCudaTensor V_cv1_W = addParam("V_cv1_W", "Constant", 0f, 20, 20, 5, 5);
JCudaTensor V_fc0_B = addParam("V_fc0_B", "Constant", 0f, 500);
JCudaTensor V_fc0_W = addParam("V_fc0_W", "Constant", 0f, 500, 320);
JCudaTensor V_fc1_B = addParam("V_fc1_B", "Constant", 0f, 10);
JCudaTensor V_fc1_W = addParam("V_fc1_W", "Constant", 0f, 10, 500);
JCudaTensor cv0_B = addParam("cv0_B", "Constant", 0.0f, 20);
JCudaTensor cv0_W = addParam("cv0_W", "Random", 0.28284273f, 20, 1, 5, 5);
JCudaTensor cv1_B = addParam("cv1_B", "Constant", 0.0f, 20);
JCudaTensor cv1_W = addParam("cv1_W", "Random", 0.06324555f, 20, 20, 5, 5);
JCudaTensor fc0_B = addParam("fc0_B", "Constant", 0.0f, 500);
JCudaTensor fc0_W = addParam("fc0_W", "Random", 0.07905694f, 500, 320);
JCudaTensor fc1_B = addParam("fc1_B", "Constant", 0.0f, 10);
JCudaTensor fc1_W = addParam("fc1_W", "Random", 0.06324555f, 10, 500);

public float trainFunction(JTensorFloat X, JTensorFloat Y) {
 // val X72 = Cuda(Indicator(Y, 10))
JCudaTensor X72 = Y.asIndicator(10).asJCudaTensor();
// val X67 = Cuda(X)
JCudaTensor X67 = X.asJCudaTensor();
// val X25 = Convolv(1,0)(X67,cv0_W,cv0_B)
JCudaTensor X25 = y6.forward(X67, cv0_W, cv0_B);
// val X26 = Pooling(2,2,0,true)(X25)
JCudaTensor X26 = y5.forward(X25);
// val X27 = Convolv(1,0)(X26,cv1_W,cv1_B)
JCudaTensor X27 = y4.forward(X26, cv1_W, cv1_B);
// val X28 = Pooling(2,2,0,true)(X27)
JCudaTensor X28 = y3.forward(X27);
// val X68 = (X28[1><3])(i10 | @) * (fc0_W)(i11 | @)
JCudaTensor X68 = X28.flatten(1, new int[]{20, 4, 4}).asMatrix(1, true).times(fc0_W.asMatrix(1, true));
// val X30 = (X68 + (i10) => fc0_B)
JCudaTensor X30 = fc0_B.copy(32, X68);
// val X31 = ReLU()(X30)
JCudaTensor X31 = y2.forward(X30);
// val X70 = (X31)(i13 | @) * (fc1_W)(i14 | @)
JCudaTensor X70 = X31.asMatrix(1, true).times(fc1_W.asMatrix(1, true));
// val X32 = (X70 + (i13) => fc1_B)
JCudaTensor X32 = fc1_B.copy(32, X70);
// val X33 = Softmax()(X32)
JCudaTensor X33 = y1.forward(X32);
// dealloc X32
X32.free();
// val X73 = Log X33.copy
JCudaTensor X73 = X33.clone().log();
// val _loss = ((0 - (X72 . X73)) / |32|)
float _loss = - X72.dot(X73) / 32f;
// dealloc X73
X73.free();
// val X86 = 1/(X33.copy)
JCudaTensor X86 = X33.clone().pow(-1f);
// val X87 = X72.copy .* X86
JCudaTensor X87 = X72.clone().times_i(X86);;
// dealloc X72
X72.free();
// dealloc X86
X86.free();
// val X88 = - X87
JCudaTensor X88 = X87.times_i(-1f);;
// val X34 = (X88 / |32|)
JCudaTensor X34 = X88.times_i(1 / 32f);;
// val X81 = X34 * d_Softmax()(X33)/d_X32
JCudaTensor X81 = y1.backward(X34, X33);
// dealloc X34
X34.free();
// dealloc X33
X33.free();
// val m4 = (i18) => X81[@, i18]
JCudaMatrix m4 = X81.asMatrix(1, false);
// V_fc1_B = ((Sum(m4) * -0.01) + (V_fc1_B * 0.1))
m4.sum(V_fc1_B, lrn_rate, momentum);
// fc1_B = (V_fc1_B + (fc1_B * (1 + (5.0E-4 * -0.01))))
fc1_B.update(V_fc1_B, 1f, 1f + decay * lrn_rate);
// val m3 = (i40) => fc1_W[@, i40]
JCudaMatrix m3 = fc1_W.asMatrix(1, false);
// val m5 = (i19) => X31[@, i19]
JCudaMatrix m5 = X31.asMatrix(1, false);
// V_fc1_W = ((m4 * m5 * -0.01) + (V_fc1_W * 0.1))
m4.times(m5, V_fc1_W, lrn_rate, momentum);
// fc1_W = (V_fc1_W + (fc1_W * (1 + (5.0E-4 * -0.01))))
fc1_W.update(V_fc1_W, 1f, 1f + decay * lrn_rate);
// val m1 = (i33) => fc0_W[@, i33]
JCudaMatrix m1 = fc0_W.asMatrix(1, false);
// val m8 = (i23) => X28[1><3][@, i23]
JCudaMatrix m8 = X28.flatten(1, new int[]{20, 4, 4}).asMatrix(1, false);
// val X82 = (X81)(i39 | @) * m3
JCudaTensor X82 = X81.asMatrix(1, true).times(m3);
// dealloc X81
X81.free();
// val X77 = X82 * d_ReLU()(X31)/d_X30
JCudaTensor X77 = y2.backward(X82, X31);
// dealloc X31
X31.free();
// val m2 = (i36) => X77[@, i36]
JCudaMatrix m2 = X77.asMatrix(1, false);
// V_fc0_W = ((m2 * m8 * -0.01) + (V_fc0_W * 0.1))
m2.times(m8, V_fc0_W, lrn_rate, momentum);
// fc0_W = (V_fc0_W + (fc0_W * (1 + (5.0E-4 * -0.01))))
fc0_W.update(V_fc0_W, 1f, 1f + decay * lrn_rate);
// V_fc0_B = ((Sum(m2) * -0.01) + (V_fc0_B * 0.1))
m2.sum(V_fc0_B, lrn_rate, momentum);
// fc0_B = (V_fc0_B + (fc0_B * (1 + (5.0E-4 * -0.01))))
fc0_B.update(V_fc0_B, 1f, 1f + decay * lrn_rate);
// val X75 = (X77)(i32 | @) * m1
JCudaTensor X75 = X77.asMatrix(1, true).times(m1);
// dealloc X77
X77.free();
// val X84 = X75[1<>3] * d_Pooling(2,2,0,true)(X28,X27)/d_X27
JCudaTensor X84 = y3.backward(X75.unflatten(1, new int[]{20, 4, 4}), X28, X27);
// dealloc X28
X28.free();
// dealloc X27
X27.free();
// dealloc X75
X75.free();
// V_cv1_W = ((X84 * d_Convolv(1,0)(X26)/d_cv1_W * -0.01) + (V_cv1_W * 0.1))
y4.backward_filter(X84, X26, V_cv1_W, lrn_rate, momentum);
// val X74 = X84 * d_Convolv(1,0)(cv1_W)/d_X26
JCudaTensor X74 = y4.backward_data(X84, cv1_W);
// cv1_W = (V_cv1_W + (cv1_W * (1 + (5.0E-4 * -0.01))))
cv1_W.update(V_cv1_W, 1f, 1f + decay * lrn_rate);
// V_cv1_B = ((X84 * d_Convolv(1,0)()/d_cv1_B * -0.01) + (V_cv1_B * 0.1))
y4.backward_bias(X84, V_cv1_B, lrn_rate, momentum);
// dealloc X84
X84.free();
// cv1_B = (V_cv1_B + (cv1_B * (1 + (5.0E-4 * -0.01))))
cv1_B.update(V_cv1_B, 1f, 1f + decay * lrn_rate);
// val X90 = X74 * d_Pooling(2,2,0,true)(X26,X25)/d_X25
JCudaTensor X90 = y5.backward(X74, X26, X25);
// dealloc X74
X74.free();
// dealloc X25
X25.free();
// dealloc X26
X26.free();
// V_cv0_W = ((X90 * d_Convolv(1,0)(X67)/d_cv0_W * -0.01) + (V_cv0_W * 0.1))
y6.backward_filter(X90, X67, V_cv0_W, lrn_rate, momentum);
// dealloc X67
X67.free();
// cv0_W = (V_cv0_W + (cv0_W * (1 + (5.0E-4 * -0.01))))
cv0_W.update(V_cv0_W, 1f, 1f + decay * lrn_rate);
// V_cv0_B = ((X90 * d_Convolv(1,0)()/d_cv0_B * -0.01) + (V_cv0_B * 0.1))
y6.backward_bias(X90, V_cv0_B, lrn_rate, momentum);
// dealloc X90
X90.free();
// cv0_B = (V_cv0_B + (cv0_B * (1 + (5.0E-4 * -0.01))))
cv0_B.update(V_cv0_B, 1f, 1f + decay * lrn_rate);

return _loss; 
}

public JCudaTensor testFunction(JTensorFloat X) {
 // val X11 = Cuda(X)
JCudaTensor X11 = X.asJCudaTensor();
// val X12 = Convolv(1,0)(X11,cv0_W,cv0_B)
JCudaTensor X12 = y6.forward(X11, cv0_W, cv0_B);
// dealloc X11
X11.free();
// val X13 = Pooling(2,2,0,true)(X12)
JCudaTensor X13 = y5.forward(X12);
// dealloc X12
X12.free();
// val X14 = Convolv(1,0)(X13,cv1_W,cv1_B)
JCudaTensor X14 = y4.forward(X13, cv1_W, cv1_B);
// dealloc X13
X13.free();
// val X15 = Pooling(2,2,0,true)(X14)
JCudaTensor X15 = y3.forward(X14);
// dealloc X14
X14.free();
// val X21 = (X15[1><3])(i10 | @) * (fc0_W)(i11 | @)
JCudaTensor X21 = X15.flatten(1, new int[]{20, 4, 4}).asMatrix(1, true).times(fc0_W.asMatrix(1, true));
// dealloc X15
X15.free();
// val X17 = (X21 + (i10) => fc0_B)
JCudaTensor X17 = fc0_B.copy(32, X21);
// val X18 = ReLU()(X17)
JCudaTensor X18 = y2.forward(X17);
// val X23 = (X18)(i13 | @) * (fc1_W)(i14 | @)
JCudaTensor X23 = X18.asMatrix(1, true).times(fc1_W.asMatrix(1, true));
// dealloc X18
X18.free();
// val X19 = (X23 + (i13) => fc1_B)
JCudaTensor X19 = fc1_B.copy(32, X23);
// val X20 = Softmax()(X19)
JCudaTensor X20 = y1.forward(X19);
// dealloc X19
X19.free();

return X20; 
}

}