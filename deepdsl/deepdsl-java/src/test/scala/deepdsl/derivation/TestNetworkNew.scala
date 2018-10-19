
        package deepdsl.derivation

        import deepdsl.analysis._
        import deepdsl.ast.{Lmdb, Mnist, _}
        import deepdsl.derivation.MemoryAnalysis._
        import deepdsl.lang._
        import deepdsl.layer._
        import deepdsl.run._
        import org.junit.Test
        
        class TestNetworkNew {
            val path = "deepdsl/gen"
        
            private def mynet(batch_size: Int, learn_rate: Float, momentum: Float, decay: Float, train_iter: Int, test_iter: Int, name: String) {
            
                val N = batch_size
                val dim = List(N, 1, 28, 28)
                val K = 10

                val dataset = Mnist(dim)
                val y = T._new("Y", List(N))
                val x = T._new("X", dim)
            
                val cv0 = CudaLayer.convolv("cv0", 5, 20)
                val mp0 = CudaLayer.max_pool(2)
                val cv1 = CudaLayer.convolv("cv1", 5, 20)
                val mp1 = CudaLayer.max_pool(2)
                val fl0 = Layer.flatten(4, 1)
                val fc0 = Layer.full("fc0", 500)
                val relu = CudaLayer.relu(2)
                val fc1 = Layer.full("fc1", 10)
                val softmax = CudaLayer.softmax

                val network = softmax o fc1 o relu o fc0 o fl0 o mp1 o cv1 o mp0 o cv0
                println(typeof(network))

                val xCuda = x.asCuda
                val yCuda = y.asIndicator(K).asCuda
                val loss = (Layer.log_loss(yCuda) o network)(xCuda)
                val accuracy = network(xCuda)
                val param = loss.freeParam.toList
                val solver = Train(name, train_iter, test_iter, learn_rate, momentum, decay, 0)
                val loop = Loop(loss, accuracy, dataset, (x, y), param, solver)

                runtimeMemory(loop.train.lst)
                parameterMemory(param, momentum)
                CudaCompile(path).print(loop)
            }
            @Test
            def testMyNet = mynet(512, 0.01f, 0.1f, 0.0005f, 50, 1, "MyNet")
            
        }
        