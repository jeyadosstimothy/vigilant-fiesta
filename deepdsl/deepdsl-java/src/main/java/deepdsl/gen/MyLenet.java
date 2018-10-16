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
 // val X40 = Cuda(Indicator(Y, 10))
JCudaTensor X40 = Y.asIndicator(10).asJCudaTensor();
// val X35 = Cuda(X)
JCudaTensor X35 = X.asJCudaTensor();
// val X36 = (X35)(i10 | @, @, @) * (fc1_W)(i11 | @)
JCudaTensor X36 = X35.asMatrix(1, true).times(fc1_W.asMatrix(1, true));
// dealloc X35
X35.free();
// val X16 = (X36 + (i10) => fc1_B)
JCudaTensor X16 = fc1_B.copy(256, X36);
// val X17 = ReLU()(X16)
JCudaTensor X17 = y2.forward(X16);
// val X38 = (X17)(i13 | @) * (fc2_W)(i14 | @)
JCudaTensor X38 = X17.asMatrix(1, true).times(fc2_W.asMatrix(1, true));
// val X18 = (X38 + (i13) => fc2_B)
JCudaTensor X18 = fc2_B.copy(256, X38);
// val X19 = Softmax()(X18)
JCudaTensor X19 = y1.forward(X18);
// dealloc X18
X18.free();
// val X41 = Log X19.copy
JCudaTensor X41 = X19.clone().log();
// val _loss = ((0 - (X40 . X41)) / |256|)
float _loss = - X40.dot(X41) / 256f;
// dealloc X41
X41.free();
// val X51 = 1/(X19.copy)
JCudaTensor X51 = X19.clone().pow(-1f);
// val X52 = X40.copy .* X51
JCudaTensor X52 = X40.clone().times_i(X51);;
// dealloc X40
X40.free();
// dealloc X51
X51.free();
// val X53 = - X52
JCudaTensor X53 = X52.times_i(-1f);;
// val X20 = (X53 / |256|)
JCudaTensor X20 = X53.times_i(1 / 256f);;
// val X43 = X20 * d_Softmax()(X19)/d_X18
JCudaTensor X43 = y1.backward(X20, X19);
// dealloc X20
X20.free();
// dealloc X19
X19.free();
// val m4 = (i33) => X43[@, i33]
JCudaMatrix m4 = X43.asMatrix(1, false);
// V_fc2_B = ((Sum(m4) * -0.005) + (V_fc2_B * 0.9))
m4.sum(V_fc2_B, lrn_rate, momentum);
// fc2_B = (V_fc2_B + (fc2_B * (1 + (5.0E-4 * -0.005))))
fc2_B.update(V_fc2_B, 1f, 1f + decay * lrn_rate);
// val m3 = (i30) => fc2_W[@, i30]
JCudaMatrix m3 = fc2_W.asMatrix(1, false);
// val m6 = (i19) => X17[@, i19]
JCudaMatrix m6 = X17.asMatrix(1, false);
// V_fc2_W = ((m4 * m6 * -0.005) + (V_fc2_W * 0.9))
m4.times(m6, V_fc2_W, lrn_rate, momentum);
// fc2_W = (V_fc2_W + (fc2_W * (1 + (5.0E-4 * -0.005))))
fc2_W.update(V_fc2_W, 1f, 1f + decay * lrn_rate);
// val X45 = (X43)(i29 | @) * m3
JCudaTensor X45 = X43.asMatrix(1, true).times(m3);
// dealloc X43
X43.free();
// val X48 = X45 * d_ReLU()(X17)/d_X16
JCudaTensor X48 = y2.backward(X45, X17);
// dealloc X17
X17.free();
// val m1 = (i22) => X48[@, i22]
JCudaMatrix m1 = X48.asMatrix(1, false);
// V_fc1_B = ((Sum(m1) * -0.005) + (V_fc1_B * 0.9))
m1.sum(V_fc1_B, lrn_rate, momentum);
// fc1_B = (V_fc1_B + (fc1_B * (1 + (5.0E-4 * -0.005))))
fc1_B.update(V_fc1_B, 1f, 1f + decay * lrn_rate);
// val m2 = (i23) => Cuda(X)[@, i23]
JCudaMatrix m2 = X.asJCudaTensor().asMatrix(1, false);
// V_fc1_W = ((m1 * m2 * -0.005) + (V_fc1_W * 0.9))
m1.times(m2, V_fc1_W, lrn_rate, momentum);
// dealloc X48
X48.free();
// fc1_W = (V_fc1_W + (fc1_W * (1 + (5.0E-4 * -0.005))))
fc1_W.update(V_fc1_W, 1f, 1f + decay * lrn_rate);

return _loss; 
}

public JCudaTensor testFunction(JTensorFloat X) {
 // val X7 = Cuda(X)
JCudaTensor X7 = X.asJCudaTensor();
// val X12 = (X7)(i10 | @, @, @) * (fc1_W)(i11 | @)
JCudaTensor X12 = X7.asMatrix(1, true).times(fc1_W.asMatrix(1, true));
// dealloc X7
X7.free();
// val X8 = (X12 + (i10) => fc1_B)
JCudaTensor X8 = fc1_B.copy(256, X12);
// val X9 = ReLU()(X8)
JCudaTensor X9 = y2.forward(X8);
// val X14 = (X9)(i13 | @) * (fc2_W)(i14 | @)
JCudaTensor X14 = X9.asMatrix(1, true).times(fc2_W.asMatrix(1, true));
// dealloc X9
X9.free();
// val X10 = (X14 + (i13) => fc2_B)
JCudaTensor X10 = fc2_B.copy(256, X14);
// val X11 = Softmax()(X10)
JCudaTensor X11 = y1.forward(X10);
// dealloc X10
X10.free();

return X11; 
}

}