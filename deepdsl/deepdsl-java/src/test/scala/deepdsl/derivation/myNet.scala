package deepdsl.derivation

import deepdsl.analysis._
import deepdsl.ast.{Lmdb, Mnist, _}
import deepdsl.derivation.MemoryAnalysis._
import deepdsl.lang._
import deepdsl.layer._
import deepdsl.run._
import org.junit.Test

class myNet {
  val K = 1000 // # of classes for ImageNet
  val path = "deepdsl/gen"
  val env = new Env(Map())

  @Test
  def testLenet = lenet(500, 0.01f, 0.1f, 0.0005f, 10, 1, "lenet")

  private def lenet(batch_size: Int, learn_rate: Float, momentum: Float, decay: Float, train_iter: Int, test_iter: Int, name: String) {
    val K = 10 // # of classes
    val N = batch_size; val C = 1; val N1 = 28; val N2 = 28 // batch size, channel, and x/y size

    val dim = List(N, C, N1, N2)

    // Specifying train dataSet
    val mnist = Mnist(dim)
    val y = T._new("Y", List(N))
    val x = T._new("X", dim)

    val cv1 = CudaLayer.convolv("cv1", 5, 20)
    val cv2 = CudaLayer.convolv("cv2", 5, 50)
    val mp = CudaLayer.max_pool(2)
    val flat = Layer.flatten(4, 1)
    val f = Layer.full("fc1", 500)
    val f2 = Layer.full("fc2", K)
    val softmax = CudaLayer.softmax
    val relu = CudaLayer.relu(2)

    val network = f2 o relu o f o flat o mp o cv2 o mp o cv1

    println(typeof(network))
    val x1 = x.asCuda
    val y1 = y.asIndicator(K).asCuda
    val c = (Layer.log_loss(y1) o softmax o network) (x1)
    val p = network(x1)

    val param = c.freeParam.toList
    val solver = Train(name, train_iter, test_iter, learn_rate, momentum, decay, 0)

    val loop = Loop(c, p, mnist, (x, y), param, solver)

    runtimeMemory(loop.train.lst)
    parameterMemory(param, momentum)
    // generate training and testing file
    CudaCompile(path).print(loop)
  }

}
