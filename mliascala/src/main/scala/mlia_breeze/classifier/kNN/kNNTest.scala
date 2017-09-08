package mlia_breeze.classifier.kNN

import breeze.linalg.Matrix
import breeze.linalg._
import breeze.stats.distributions._
import breeze.numerics._

object kNNTest extends App{
  def createDataSet(): Tuple2[Matrix[Double], Vector[String]] = {
    val dataSet = DenseMatrix.zeros[Double](4, 2)
    dataSet(0, ::).:=(DenseVector[Double](1.0, 1.1).t)
    dataSet(1, ::).:=(DenseVector[Double](1.0, 1.0).t)
    dataSet(2, ::).:=(DenseVector[Double](0, 0).t)
    dataSet(3, ::).:=(DenseVector[Double](0, 1.1).t)
    val labels = DenseVector[String]("A", "A", "B", "B")
    return Tuple2(dataSet, labels)
  }
  
  override def main(args: Array[String]): Unit = {
    val (group, labels) = kNNTest.createDataSet()
    val inX = DenseVector[Double](0.0, 1.1)

    printf("classify: %s\n", kNN.classify0(inX, group, labels, 1))
  }
}