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
run.save();
run.free();
}

public Lenet() {
super("src/main/java/deepdsl/gen/lenet");
setTrainData(MnistFactory.getFactory(true, new int[]{500, 1, 28, 28}));
setTestData(MnistFactory.getFactory(false, new int[]{500, 1, 28, 28}));
}

float lrn_rate = -0.01f;
float momentum = 0.1f;
float decay = 5.0E-4f;

JCudnnActivation y2 = addActivation(new int[]{500,500}, ActivationMode.RELU);
JCudnnSoftmax y1 = addSoftmax(new int[]{500,10}, SoftmaxAlgorithm.ACCURATE);
JCudaTensor V_fc1_B = addParam("V_fc1_B", "Constant", 0f, 500);
JCudaTensor V_fc1_W = addParam("V_fc1_W", "Constant", 0f, 500, 1);
JCudaTensor V_fc2_B = addParam("V_fc2_B", "Constant", 0f, 10);
JCudaTensor V_fc2_W = addParam("V_fc2_W", "Constant", 0f, 10, 500);
JCudaTensor fc1_B = addParam("fc1_B", "Constant", 0.0f, 500);
JCudaTensor fc1_W = addParam("fc1_W", "Random", 1.4142135f, 500, 1);
JCudaTensor fc2_B = addParam("fc2_B", "Constant", 0.0f, 10);
JCudaTensor fc2_W = addParam("fc2_W", "Random", 0.06324555f, 10, 500);

public float trainFunction(JTensorFloat X, JTensorFloat Y) {
 // val X34 = Cuda(X)
JCudaTensor X34 = X.asJCudaTensor();
// val X39 = Cuda(Indicator(Y, 10))
JCudaTensor X39 = Y.asIndicator(10).asJCudaTensor();
// val X35 = (X34)(i10 | @, @, @) * (fc1_W)(i11 | @)
JCudaTensor X35 = X34.asMatrix(1, true).times(fc1_W.asMatrix(1, true));
// dealloc X34
X34.free();
// val X15 = (X35 + (i10) => fc1_B)
JCudaTensor X15 = fc1_B.copy(500, X35);
// val X16 = ReLU()(X15)
JCudaTensor X16 = y2.forward(X15);
// val X37 = (X16)(i13 | @) * (fc2_W)(i14 | @)
JCudaTensor X37 = X16.asMatrix(1, true).times(fc2_W.asMatrix(1, true));
// val X17 = (X37 + (i13) => fc2_B)
JCudaTensor X17 = fc2_B.copy(500, X37);
// val X18 = Softmax()(X17)
JCudaTensor X18 = y1.forward(X17);
// dealloc X17
X17.free();
// val X40 = Log X18.copy
JCudaTensor X40 = X18.clone().log();
// val _loss = ((0 - (X39 . X40)) / |500|)
float _loss = - X39.dot(X40) / 500f;
// dealloc X40
X40.free();
// val X42 = 1/(X18.copy)
JCudaTensor X42 = X18.clone().pow(-1f);
// val X43 = X39.copy .* X42
JCudaTensor X43 = X39.clone().times_i(X42);;
// dealloc X42
X42.free();
// dealloc X39
X39.free();
// val X44 = - X43
JCudaTensor X44 = X43.times_i(-1f);;
// val X19 = (X44 / |500|)
JCudaTensor X19 = X44.times_i(1 / 500f);;
// val X50 = X19 * d_Softmax()(X18)/d_X17
JCudaTensor X50 = y1.backward(X19, X18);
// dealloc X19
X19.free();
// dealloc X18
X18.free();
// val m4 = (i33) => X50[@, i33]
JCudaMatrix m4 = X50.asMatrix(1, false);
// V_fc2_B = ((Sum(m4) * -0.01) + (V_fc2_B * 0.1))
m4.sum(V_fc2_B, lrn_rate, momentum);
// fc2_B = (V_fc2_B + (fc2_B * (1 + (5.0E-4 * -0.01))))
fc2_B.update(V_fc2_B, 1f, 1f + decay * lrn_rate);
// val m7 = (i30) => fc2_W[@, i30]
JCudaMatrix m7 = fc2_W.asMatrix(1, false);
// val m6 = (i19) => X16[@, i19]
JCudaMatrix m6 = X16.asMatrix(1, false);
// V_fc2_W = ((m4 * m6 * -0.01) + (V_fc2_W * 0.1))
m4.times(m6, V_fc2_W, lrn_rate, momentum);
// fc2_W = (V_fc2_W + (fc2_W * (1 + (5.0E-4 * -0.01))))
fc2_W.update(V_fc2_W, 1f, 1f + decay * lrn_rate);
// val X53 = (X50)(i29 | @) * m7
JCudaTensor X53 = X50.asMatrix(1, true).times(m7);
// dealloc X50
X50.free();
// val X52 = X53 * d_ReLU()(X16)/d_X15
JCudaTensor X52 = y2.backward(X53, X16);
// dealloc X16
X16.free();
// val m1 = (i26) => X52[@, i26]
JCudaMatrix m1 = X52.asMatrix(1, false);
// V_fc1_B = ((Sum(m1) * -0.01) + (V_fc1_B * 0.1))
m1.sum(V_fc1_B, lrn_rate, momentum);
// fc1_B = (V_fc1_B + (fc1_B * (1 + (5.0E-4 * -0.01))))
fc1_B.update(V_fc1_B, 1f, 1f + decay * lrn_rate);
// val m3 = (i23) => Cuda(X)[@, i23]
JCudaMatrix m3 = X.asJCudaTensor().asMatrix(1, false);
// V_fc1_W = ((m1 * m3 * -0.01) + (V_fc1_W * 0.1))
m1.times(m3, V_fc1_W, lrn_rate, momentum);
// dealloc X52
X52.free();
// fc1_W = (V_fc1_W + (fc1_W * (1 + (5.0E-4 * -0.01))))
fc1_W.update(V_fc1_W, 1f, 1f + decay * lrn_rate);

return _loss; 
}

public JCudaTensor testFunction(JTensorFloat X) {
 // val X7 = Cuda(X)
JCudaTensor X7 = X.asJCudaTensor();
// val X11 = (X7)(i10 | @, @, @) * (fc1_W)(i11 | @)
JCudaTensor X11 = X7.asMatrix(1, true).times(fc1_W.asMatrix(1, true));
// dealloc X7
X7.free();
// val X8 = (X11 + (i10) => fc1_B)
JCudaTensor X8 = fc1_B.copy(500, X11);
// val X9 = ReLU()(X8)
JCudaTensor X9 = y2.forward(X8);
// val X13 = (X9)(i13 | @) * (fc2_W)(i14 | @)
JCudaTensor X13 = X9.asMatrix(1, true).times(fc2_W.asMatrix(1, true));
// dealloc X9
X9.free();
// val X10 = (X13 + (i13) => fc2_B)
JCudaTensor X10 = fc2_B.copy(500, X13);

return X10; 
}

}