package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.util.CudaRun;


public class MyLenet extends CudaRun {

public static void main(String[] args){
MyLenet run = new MyLenet();
run.train(20);
run.test(5);
run.save();
run.free();
}

public MyLenet() {
super("src/main/java/deepdsl/gen/myLenet");
setTrainData(MnistFactory.getFactory(true, new int[]{256, 1, 28, 28}));
setTestData(MnistFactory.getFactory(false, new int[]{256, 1, 28, 28}));
}

float lrn_rate = -0.005f;
float momentum = 0.9f;
float decay = 5.0E-4f;

JCudnnActivation y2 = addActivation(new int[]{256,500}, ActivationMode.RELU);
JCudnnSoftmax y1 = addSoftmax(new int[]{256,10}, SoftmaxAlgorithm.ACCURATE);
JCudaTensor V_fc1_B = addParam("V_fc1_B", "Constant", 0f, 500);
JCudaTensor V_fc1_W = addParam("V_fc1_W", "Constant", 0f, 500, 1);
JCudaTensor V_fc2_B = addParam("V_fc2_B", "Constant", 0f, 10);
JCudaTensor V_fc2_W = addParam("V_fc2_W", "Constant", 0f, 10, 500);
JCudaTensor fc1_B = addParam("fc1_B", "Constant", 0.0f, 500);
JCudaTensor fc1_W = addParam("fc1_W", "Random", 1.4142135f, 500, 1);
JCudaTensor fc2_B = addParam("fc2_B", "Constant", 0.0f, 10);
JCudaTensor fc2_W = addParam("fc2_W", "Random", 0.06324555f, 10, 500);

public float trainFunction(JTensorFloat X, JTensorFloat Y) {
 // val X39 = Cuda(Indicator(Y, 10))
JCudaTensor X39 = Y.asIndicator(10).asJCudaTensor();
// val X34 = Cuda(X)
JCudaTensor X34 = X.asJCudaTensor();
// val X35 = (X34)(i1 | @, @, @) * (fc1_W)(i2 | @)
JCudaTensor X35 = X34.asMatrix(1, true).times(fc1_W.asMatrix(1, true));
// dealloc X34
X34.free();
// val X15 = (X35 + (i1) => fc1_B)
JCudaTensor X15 = fc1_B.copy(256, X35);
// val X16 = ReLU()(X15)
JCudaTensor X16 = y2.forward(X15);
// val X37 = (X16)(i4 | @) * (fc2_W)(i5 | @)
JCudaTensor X37 = X16.asMatrix(1, true).times(fc2_W.asMatrix(1, true));
// val X17 = (X37 + (i4) => fc2_B)
JCudaTensor X17 = fc2_B.copy(256, X37);
// val X18 = Softmax()(X17)
JCudaTensor X18 = y1.forward(X17);
// dealloc X17
X17.free();
// val X40 = Log X18.copy
JCudaTensor X40 = X18.clone().log();
// val _loss = ((0 - (X39 . X40)) / |256|)
float _loss = - X39.dot(X40) / 256f;
// dealloc X40
X40.free();
// val X51 = 1/(X18.copy)
JCudaTensor X51 = X18.clone().pow(-1f);
// val X52 = X39.copy .* X51
JCudaTensor X52 = X39.clone().times_i(X51);;
// dealloc X51
X51.free();
// dealloc X39
X39.free();
// val X53 = - X52
JCudaTensor X53 = X52.times_i(-1f);;
// val X19 = (X53 / |256|)
JCudaTensor X19 = X53.times_i(1 / 256f);;
// val X47 = X19 * d_Softmax()(X18)/d_X17
JCudaTensor X47 = y1.backward(X19, X18);
// dealloc X18
X18.free();
// dealloc X19
X19.free();
// val m5 = (i24) => X47[@, i24]
JCudaMatrix m5 = X47.asMatrix(1, false);
// V_fc2_B = ((Sum(m5) * -0.005) + (V_fc2_B * 0.9))
m5.sum(V_fc2_B, lrn_rate, momentum);
// fc2_B = (V_fc2_B + (fc2_B * (1 + (5.0E-4 * -0.005))))
fc2_B.update(V_fc2_B, 1f, 1f + decay * lrn_rate);
// val m4 = (i21) => fc2_W[@, i21]
JCudaMatrix m4 = fc2_W.asMatrix(1, false);
// val m7 = (i10) => X16[@, i10]
JCudaMatrix m7 = X16.asMatrix(1, false);
// V_fc2_W = ((m5 * m7 * -0.005) + (V_fc2_W * 0.9))
m5.times(m7, V_fc2_W, lrn_rate, momentum);
// fc2_W = (V_fc2_W + (fc2_W * (1 + (5.0E-4 * -0.005))))
fc2_W.update(V_fc2_W, 1f, 1f + decay * lrn_rate);
// val X45 = (X47)(i20 | @) * m4
JCudaTensor X45 = X47.asMatrix(1, true).times(m4);
// dealloc X47
X47.free();
// val X43 = X45 * d_ReLU()(X16)/d_X15
JCudaTensor X43 = y2.backward(X45, X16);
// dealloc X16
X16.free();
// val m1 = (i17) => X43[@, i17]
JCudaMatrix m1 = X43.asMatrix(1, false);
// V_fc1_B = ((Sum(m1) * -0.005) + (V_fc1_B * 0.9))
m1.sum(V_fc1_B, lrn_rate, momentum);
// fc1_B = (V_fc1_B + (fc1_B * (1 + (5.0E-4 * -0.005))))
fc1_B.update(V_fc1_B, 1f, 1f + decay * lrn_rate);
// val m3 = (i14) => Cuda(X)[@, i14]
JCudaMatrix m3 = X.asJCudaTensor().asMatrix(1, false);
// V_fc1_W = ((m1 * m3 * -0.005) + (V_fc1_W * 0.9))
m1.times(m3, V_fc1_W, lrn_rate, momentum);
// dealloc X43
X43.free();
// fc1_W = (V_fc1_W + (fc1_W * (1 + (5.0E-4 * -0.005))))
fc1_W.update(V_fc1_W, 1f, 1f + decay * lrn_rate);

return _loss; 
}

public JCudaTensor testFunction(JTensorFloat X) {
 // val X6 = Cuda(X)
JCudaTensor X6 = X.asJCudaTensor();
// val X11 = (X6)(i1 | @, @, @) * (fc1_W)(i2 | @)
JCudaTensor X11 = X6.asMatrix(1, true).times(fc1_W.asMatrix(1, true));
// dealloc X6
X6.free();
// val X7 = (X11 + (i1) => fc1_B)
JCudaTensor X7 = fc1_B.copy(256, X11);
// val X8 = ReLU()(X7)
JCudaTensor X8 = y2.forward(X7);
// val X13 = (X8)(i4 | @) * (fc2_W)(i5 | @)
JCudaTensor X13 = X8.asMatrix(1, true).times(fc2_W.asMatrix(1, true));
// dealloc X8
X8.free();
// val X9 = (X13 + (i4) => fc2_B)
JCudaTensor X9 = fc2_B.copy(256, X13);
// val X10 = Softmax()(X9)
JCudaTensor X10 = y1.forward(X9);
// dealloc X9
X9.free();

return X10; 
}

}