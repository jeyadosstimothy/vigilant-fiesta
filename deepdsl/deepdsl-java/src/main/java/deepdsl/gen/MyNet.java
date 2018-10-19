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
run.infer(1);
run.save();
run.free();
}

public MyNet() {
super("src/main/java/deepdsl/gen/MyNet");
setTrainData(MnistFactory.getFactory(true, new int[]{512, 1, 28, 28}));
setTestData(MnistFactory.getFactory(false, new int[]{512, 1, 28, 28}));
}

float lrn_rate = -0.01f;
float momentum = 0.1f;
float decay = 5.0E-4f;

JCudnnConvolution y6 = addConvolution(new int[]{512,1,28,28},new int[]{20,1,5,5},new int[]{20}, 1, 0);
JCudnnConvolution y4 = addConvolution(new int[]{512,20,12,12},new int[]{20,20,5,5},new int[]{20}, 1, 0);
JCudnnPooling y5 = addPooling(new int[]{512,20,24,24}, 2, 2, 0, PoolingType.MAX);
JCudnnPooling y3 = addPooling(new int[]{512,20,8,8}, 2, 2, 0, PoolingType.MAX);
JCudnnActivation y2 = addActivation(new int[]{512,500}, ActivationMode.RELU);
JCudnnSoftmax y1 = addSoftmax(new int[]{512,10}, SoftmaxAlgorithm.ACCURATE);
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
 // val X69 = Cuda(X)
JCudaTensor X69 = X.asJCudaTensor();
// val X72 = Cuda(Indicator(Y, 10))
JCudaTensor X72 = Y.asIndicator(10).asJCudaTensor();
// val X25 = Convolv(1,0)(X69,cv0_W,cv0_B)
JCudaTensor X25 = y6.forward(X69, cv0_W, cv0_B);
// val X26 = Pooling(2,2,0,true)(X25)
JCudaTensor X26 = y5.forward(X25);
// val X27 = Convolv(1,0)(X26,cv1_W,cv1_B)
JCudaTensor X27 = y4.forward(X26, cv1_W, cv1_B);
// val X28 = Pooling(2,2,0,true)(X27)
JCudaTensor X28 = y3.forward(X27);
// val X67 = (X28[1><3])(i10 | @) * (fc0_W)(i11 | @)
JCudaTensor X67 = X28.flatten(1, new int[]{20, 4, 4}).asMatrix(1, true).times(fc0_W.asMatrix(1, true));
// val X30 = (X67 + (i10) => fc0_B)
JCudaTensor X30 = fc0_B.copy(512, X67);
// val X31 = ReLU()(X30)
JCudaTensor X31 = y2.forward(X30);
// val X70 = (X31)(i13 | @) * (fc1_W)(i14 | @)
JCudaTensor X70 = X31.asMatrix(1, true).times(fc1_W.asMatrix(1, true));
// val X32 = (X70 + (i13) => fc1_B)
JCudaTensor X32 = fc1_B.copy(512, X70);
// val X33 = Softmax()(X32)
JCudaTensor X33 = y1.forward(X32);
// dealloc X32
X32.free();
// val X73 = Log X33.copy
JCudaTensor X73 = X33.clone().log();
// val _loss = ((0 - (X72 . X73)) / |512|)
float _loss = - X72.dot(X73) / 512f;
// dealloc X73
X73.free();
// val X84 = 1/(X33.copy)
JCudaTensor X84 = X33.clone().pow(-1f);
// val X85 = X72.copy .* X84
JCudaTensor X85 = X72.clone().times_i(X84);;
// dealloc X72
X72.free();
// dealloc X84
X84.free();
// val X86 = - X85
JCudaTensor X86 = X85.times_i(-1f);;
// val X34 = (X86 / |512|)
JCudaTensor X34 = X86.times_i(1 / 512f);;
// val X80 = X34 * d_Softmax()(X33)/d_X32
JCudaTensor X80 = y1.backward(X34, X33);
// dealloc X34
X34.free();
// dealloc X33
X33.free();
// val m3 = (i18) => X80[@, i18]
JCudaMatrix m3 = X80.asMatrix(1, false);
// V_fc1_B = ((Sum(m3) * -0.01) + (V_fc1_B * 0.1))
m3.sum(V_fc1_B, lrn_rate, momentum);
// fc1_B = (V_fc1_B + (fc1_B * (1 + (5.0E-4 * -0.01))))
fc1_B.update(V_fc1_B, 1f, 1f + decay * lrn_rate);
// val m5 = (i40) => fc1_W[@, i40]
JCudaMatrix m5 = fc1_W.asMatrix(1, false);
// val m4 = (i19) => X31[@, i19]
JCudaMatrix m4 = X31.asMatrix(1, false);
// V_fc1_W = ((m3 * m4 * -0.01) + (V_fc1_W * 0.1))
m3.times(m4, V_fc1_W, lrn_rate, momentum);
// fc1_W = (V_fc1_W + (fc1_W * (1 + (5.0E-4 * -0.01))))
fc1_W.update(V_fc1_W, 1f, 1f + decay * lrn_rate);
// val m1 = (i33) => fc0_W[@, i33]
JCudaMatrix m1 = fc0_W.asMatrix(1, false);
// val m8 = (i23) => X28[1><3][@, i23]
JCudaMatrix m8 = X28.flatten(1, new int[]{20, 4, 4}).asMatrix(1, false);
// val X92 = (X80)(i39 | @) * m5
JCudaTensor X92 = X80.asMatrix(1, true).times(m5);
// dealloc X80
X80.free();
// val X78 = X92 * d_ReLU()(X31)/d_X30
JCudaTensor X78 = y2.backward(X92, X31);
// dealloc X31
X31.free();
// val m2 = (i36) => X78[@, i36]
JCudaMatrix m2 = X78.asMatrix(1, false);
// V_fc0_W = ((m2 * m8 * -0.01) + (V_fc0_W * 0.1))
m2.times(m8, V_fc0_W, lrn_rate, momentum);
// fc0_W = (V_fc0_W + (fc0_W * (1 + (5.0E-4 * -0.01))))
fc0_W.update(V_fc0_W, 1f, 1f + decay * lrn_rate);
// V_fc0_B = ((Sum(m2) * -0.01) + (V_fc0_B * 0.1))
m2.sum(V_fc0_B, lrn_rate, momentum);
// fc0_B = (V_fc0_B + (fc0_B * (1 + (5.0E-4 * -0.01))))
fc0_B.update(V_fc0_B, 1f, 1f + decay * lrn_rate);
// val X74 = (X78)(i32 | @) * m1
JCudaTensor X74 = X78.asMatrix(1, true).times(m1);
// dealloc X78
X78.free();
// val X95 = X74[1<>3] * d_Pooling(2,2,0,true)(X28,X27)/d_X27
JCudaTensor X95 = y3.backward(X74.unflatten(1, new int[]{20, 4, 4}), X28, X27);
// dealloc X28
X28.free();
// dealloc X74
X74.free();
// dealloc X27
X27.free();
// V_cv1_W = ((X95 * d_Convolv(1,0)(X26)/d_cv1_W * -0.01) + (V_cv1_W * 0.1))
y4.backward_filter(X95, X26, V_cv1_W, lrn_rate, momentum);
// val X88 = X95 * d_Convolv(1,0)(cv1_W)/d_X26
JCudaTensor X88 = y4.backward_data(X95, cv1_W);
// cv1_W = (V_cv1_W + (cv1_W * (1 + (5.0E-4 * -0.01))))
cv1_W.update(V_cv1_W, 1f, 1f + decay * lrn_rate);
// V_cv1_B = ((X95 * d_Convolv(1,0)()/d_cv1_B * -0.01) + (V_cv1_B * 0.1))
y4.backward_bias(X95, V_cv1_B, lrn_rate, momentum);
// dealloc X95
X95.free();
// cv1_B = (V_cv1_B + (cv1_B * (1 + (5.0E-4 * -0.01))))
cv1_B.update(V_cv1_B, 1f, 1f + decay * lrn_rate);
// val X99 = X88 * d_Pooling(2,2,0,true)(X26,X25)/d_X25
JCudaTensor X99 = y5.backward(X88, X26, X25);
// dealloc X25
X25.free();
// dealloc X26
X26.free();
// dealloc X88
X88.free();
// V_cv0_W = ((X99 * d_Convolv(1,0)(X69)/d_cv0_W * -0.01) + (V_cv0_W * 0.1))
y6.backward_filter(X99, X69, V_cv0_W, lrn_rate, momentum);
// dealloc X69
X69.free();
// cv0_W = (V_cv0_W + (cv0_W * (1 + (5.0E-4 * -0.01))))
cv0_W.update(V_cv0_W, 1f, 1f + decay * lrn_rate);
// V_cv0_B = ((X99 * d_Convolv(1,0)()/d_cv0_B * -0.01) + (V_cv0_B * 0.1))
y6.backward_bias(X99, V_cv0_B, lrn_rate, momentum);
// dealloc X99
X99.free();
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
JCudaTensor X17 = fc0_B.copy(512, X21);
// val X18 = ReLU()(X17)
JCudaTensor X18 = y2.forward(X17);
// val X23 = (X18)(i13 | @) * (fc1_W)(i14 | @)
JCudaTensor X23 = X18.asMatrix(1, true).times(fc1_W.asMatrix(1, true));
// dealloc X18
X18.free();
// val X19 = (X23 + (i13) => fc1_B)
JCudaTensor X19 = fc1_B.copy(512, X23);
// val X20 = Softmax()(X19)
JCudaTensor X20 = y1.forward(X19);
// dealloc X19
X19.free();

return X20; 
}

}