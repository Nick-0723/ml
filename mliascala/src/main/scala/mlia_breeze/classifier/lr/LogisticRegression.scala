package mlia_breeze.classifier.lr
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Rand
import breeze.stats.distributions.RandBasis
import breeze.stats.distributions.{Rand, RandBasis, Uniform}

object LogisticRegression {
 
  def gradAscent(dataMatIn: List[Array[Double]], classLabels: Array[Int]): Matrix[Double] = {
    val alpha = 0.001
    val maxCycle = 500
    val dataMatrix = DenseMatrix(dataMatIn: _*)
    val labelMat = DenseMatrix(classLabels.map(_.toDouble)).t
    (0 until maxCycle).foldLeft(DenseMatrix.ones[Double](dataMatrix.cols, 1))((curWeight, cycle) => {
       val h = sigmoid(dataMatrix * curWeight)
      val error = labelMat :- h
      val ttt = dataMatrix.t
      val test = (dataMatrix.t :* alpha )
      curWeight :+ (dataMatrix.t :* alpha) * error
    })
  }

  def stocGradAscent0(dataMatIn: List[Array[Double]], classLabels: Array[Int]): Vector[Double]={
        val alpha = 0.01
        dataMatIn.zipWithIndex.foldLeft(DenseVector.ones[Double](dataMatIn.head.length))( (curWeight, row) => {
          val h = sigmoid(sum(DenseVector(row._1) * curWeight))
          val error = classLabels(row._2) .toDouble - h
          
          curWeight + (alpha * error *:* DenseVector(row._1))
        })
  }
  
 def stocGradAscent1(dataMatIn: List[Array[Double]], classLabels: Array[Int], numIter: Int = 150)(implicit rand: RandBasis = Rand): Vector[Double]={
        (0 until numIter).foldLeft(DenseVector.ones[Double](dataMatIn.head.size))((outerState, i) =>{
          dataMatIn.zipWithIndex.foldLeft((outerState, (0 until dataMatIn.size).toArray))((curr, row) => {
            val alpha = (4 / (1.0 + i + row._2)) + 0.01
            val randIndex = Uniform(0, curr._2.size).sample().toInt
            val vec = DenseVector(dataMatIn(randIndex))
            val h = sigmoid(sum(vec :* curr._1))
            val error = classLabels(randIndex) - h
            (curr._1 :+ (vec :* (alpha * error)), curr._2.tail)
          })._1
        })
  }  
   def stocGradAscent2(dataMatIn: List[Array[Double]], classLabels: Array[Int], numIter: Int = 150)(implicit rand: RandBasis = Rand): DenseVector[Double] = {

    (0 until numIter).foldLeft(DenseVector.ones[Double](dataMatIn.head.size)) { (outerState, i) =>
      (0 until dataMatIn.size).foldLeft((outerState, (0 until dataMatIn.size).toArray)) { case ((curWeights, indices), j) =>
        val alpha = (4 / (1.0 + i + j)) + 0.01
        val randIndex = Uniform(0, indices.size).sample().toInt
        val vec = DenseVector(dataMatIn(randIndex))

        val h = sigmoid((vec :* curWeights: DenseVector[Double]).sum)
        val error = classLabels(randIndex) - h

        val newWeights = (curWeights :+ (vec :* (alpha * error): DenseVector[Double]), indices.tail)
        newWeights
      }._1
    }
  }

  def classifyVector(inX: Vector[Double], weights: Vector[Double]) = {
    val prob = sigmoid((inX :* weights).sum)
    if (prob > 0.5) 1 else 0
  }
}