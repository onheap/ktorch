package core.value

import org.junit.jupiter.api.Test

// https://github.com/karpathy/micrograd/blob/master/demo.ipynb
internal class ModelTest {
    class MoonTest {
        val X =
            listOf(
                listOf(1.12211461e+00, 8.14771734e-02),
                listOf(-8.18829413e-01, 5.87900639e-02),
                listOf(1.61370966e+00, -1.24645900e-01),
                listOf(-9.23009184e-01, 3.65228899e-01),
                listOf(1.43851462e-01, 4.43800492e-02),
                listOf(1.64472466e-01, 1.17383457e-01),
                listOf(1.33877062e+00, -2.38009933e-01),
                listOf(8.71148615e-01, -4.22717587e-01),
                listOf(1.83129946e+00, -1.41043828e-01),
                listOf(4.87571202e-01, 6.39092830e-01),
                listOf(3.74623511e-02, 4.23588090e-01),
                listOf(-4.43916853e-01, 8.96739312e-01),
                listOf(-8.12229494e-01, 9.12090924e-01),
                listOf(1.63552312e+00, -3.49996760e-01),
                listOf(4.73539037e-01, 9.57342599e-01),
                listOf(7.53549316e-01, 6.23727143e-01),
                listOf(2.64212818e-01, -2.42419828e-01),
                listOf(1.42755726e+00, -3.72510358e-01),
                listOf(-3.72356057e-01, 9.56691710e-01),
                listOf(-9.61301967e-01, 3.26090112e-01),
                listOf(7.80858468e-01, 7.97489402e-01),
                listOf(9.16609029e-01, -4.27638438e-01),
                listOf(1.04703809e+00, -5.44492470e-01),
                listOf(-6.03630542e-02, 1.19609088e-01),
                listOf(2.91895380e-02, 3.06838997e-01),
                listOf(-3.95732255e-01, 8.96543895e-01),
                listOf(-1.04645910e-01, 1.11788313e+00),
                listOf(1.88110004e+00, 2.99202568e-01),
                listOf(8.27408779e-01, 3.44977171e-01),
                listOf(1.29777112e+00, -3.66543151e-01),
                listOf(-6.76892847e-01, 8.55599574e-01),
                listOf(5.29529953e-01, 9.47355941e-01),
                listOf(-8.43802291e-01, 6.04739822e-01),
                listOf(2.65984708e-01, 8.87321986e-01),
                listOf(1.37403862e-01, 3.97856894e-01),
                listOf(-9.10439360e-01, -9.70966409e-02),
                listOf(1.33740031e+00, -3.67411974e-01),
                listOf(1.02257719e+00, -3.97526493e-01),
                listOf(1.02490132e+00, -5.48639298e-01),
                listOf(-7.50895897e-01, 2.53287722e-01),
                listOf(1.20281632e+00, 8.11538186e-02),
                listOf(-4.69102122e-01, 7.80796219e-01),
                listOf(7.40836677e-01, 4.59232537e-01),
                listOf(7.86905117e-01, 7.66247461e-01),
                listOf(-1.30051914e-01, 1.11938940e+00),
                listOf(8.04023061e-01, -4.23154474e-01),
                listOf(2.83303670e-01, -2.19440711e-01),
                listOf(-7.11055396e-01, 7.11638805e-01),
                listOf(3.02624521e-01, -9.33890861e-02),
                listOf(8.07914632e-01, 3.36538326e-01),
                listOf(-9.41626914e-01, 1.68018576e-01),
                listOf(1.14081485e+00, -4.62897185e-01),
                listOf(-1.57137522e-01, 9.32106642e-01),
                listOf(1.71504370e+00, -1.83620088e-01),
                listOf(3.72465754e-01, -1.26078412e-01),
                listOf(-7.43095776e-01, 6.98951667e-01),
                listOf(6.81424891e-01, 6.85634239e-01),
                listOf(8.68612480e-01, -3.72786175e-01),
                listOf(1.00229575e+00, 2.16874785e-01),
                listOf(-1.02219391e+00, 3.97508213e-01),
                listOf(-6.10762581e-01, 8.31171369e-01),
                listOf(-7.76683047e-01, 6.43609323e-01),
                listOf(1.10530001e+00, 2.19162811e-01),
                listOf(-1.78904597e-01, 1.06959774e+00),
                listOf(4.04983057e-01, 8.26478337e-01),
                listOf(1.81457300e+00, 3.45253762e-02),
                listOf(-7.91285627e-01, 2.00068809e-01),
                listOf(1.98173149e+00, 4.60760535e-01),
                listOf(7.32427956e-01, -3.99657543e-01),
                listOf(2.11141647e+00, 1.80640708e-01),
                listOf(2.16205598e+00, 4.94233819e-01),
                listOf(8.96245420e-01, 4.61536063e-01),
                listOf(3.73493759e-01, 1.04498301e+00),
                listOf(5.84964554e-01, -3.21259841e-01),
                listOf(1.76329788e-01, 1.97455719e-01),
                listOf(1.09115646e-01, 4.81557011e-01),
                listOf(2.97532748e-01, 9.95836387e-01),
                listOf(9.60465720e-02, -4.62698630e-02),
                listOf(4.74643637e-01, -1.05126474e-01),
                listOf(1.11634711e+00, -4.15534373e-01),
                listOf(5.53627588e-01, -4.23125821e-01),
                listOf(1.89528099e-01, 1.01835655e+00),
                listOf(1.94566914e+00, -9.53034040e-02),
                listOf(-8.46266211e-02, 1.07262342e+00),
                listOf(1.16858008e+00, -2.84810701e-02),
                listOf(1.24468467e-01, 1.05725031e+00),
                listOf(2.03389924e+00, 2.84729799e-01),
                listOf(-2.80005380e-02, 1.70767639e-01),
                listOf(7.40934813e-01, 4.14114383e-01),
                listOf(7.93947034e-01, 5.59725800e-01),
                listOf(8.68742625e-01, -5.30147199e-01),
                listOf(1.61874435e+00, -3.25845287e-01),
                listOf(1.42986564e+00, -4.73342103e-01),
                listOf(1.97480435e+00, -1.77931622e-01),
                listOf(1.85356347e+00, 3.42263958e-01),
                listOf(1.74912164e+00, 2.83390247e-02),
                listOf(-6.85668888e-01, 4.65356936e-01),
                listOf(1.75237435e+00, 1.64520514e-01),
                listOf(1.80789551e-01, -2.95417619e-04),
                listOf(1.21082973e-01, 1.06555225e+00))

        val y =
            listOf(
                -1,
                -1,
                1,
                -1,
                1,
                1,
                1,
                1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                -1,
                1,
                1,
                1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                1,
                1,
                1,
                -1,
                1,
                1,
                -1)

        private fun loss(model: Model): Pair<Value, Double> {
            // Xb, yb = X, y
            val Xb = X
            val yb = y

            // inputs = [list(map(Value, xrow)) for xrow in Xb]
            val inputs = Xb.map { listOf(Value(it.first()), Value(it.last())) }
            //            val inputs = Xb.map { xRow -> xRow.map { Value(it) } }

            // # forward the model to get scores,
            // scores = list(map(model, inputs))
            val scores = inputs.map { input -> model.prediction(input) }

            // # svm "max-margin" loss
            // losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
            // data_loss = sum(losses) * (1.0 / len(losses))
            val losses = (yb zip scores).map { (yi, scorei) -> (1 + -yi * scorei).relu() }
            val dataLoss = losses.sum() * (1.0 / losses.size)

            // # L2 regularization
            // alpha = 1e-4
            // reg_loss = alpha * sum((p*p for p in model.parameters()))
            // total_loss = data_loss + reg_loss

            val alpha = 1e-4
            val regLoss = alpha * model.parameters().map { p -> p * p }.sum()
            val totalLoss = dataLoss + regLoss

            // # also get accuracy
            // accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
            // return total_loss, sum(accuracy) / len(accuracy)
            val accuracy =
                (yb zip scores).map { (yi, scorei) -> if ((yi > 0) == (scorei.data > 0)) 1 else 0 }

            return totalLoss to accuracy.sum() / accuracy.size.toDouble()
        }

        private fun optimization(model: Model, generation: Int) {
            for (k in 0 until generation) {
                // forward
                val (totalLoss, acc) = loss(model)

                // backward
                model.zeroGrad()
                totalLoss.backward()

                // update (sgd)
                val learningRate = 1.0 - 0.9 * k / generation
                model.parameters().forEach { p -> p.data -= learningRate * p.grad }

                println("step $k loss ${totalLoss.data}, accuracy ${acc * 100}%")
            }
        }

        private val mlpModel =
            MLP(2, listOf(16, 16, 1)).let { mlp ->
                object : Model {
                    override fun parameters(): List<Value> = mlp.parameters()

                    override fun prediction(initInput: List<Value>): Value = mlp(initInput).single()
                }
            }

        private val fixedModel = RawModel(2, listOf(16, 16, 1))

        private val testModel = mlpModel

        @Test
        fun testModel() {
            println("parameters size: ${testModel.parameters().size}")
            println("$testModel")
        }

        @Test
        fun testLoss() {
            val (totalLoss, acc) = loss(testModel)
            println("$totalLoss, $acc")
        }

        @Test
        fun testOptimization() {
            optimization(testModel, 100)
        }

        @Test
        fun testRand() {
            fun rand(i: Int): Double {
                val l = (1103515245L * (i + 1) + 12345) % (4294967296)
                val rd = l / 4294967295.0
                return (1 - -1) * rd + -1
            }

            for (i in 0 until 337) {
                println(rand(i))
            }
        }
    }
}
