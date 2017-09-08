package mlia_breeze.cluster

import breeze.linalg._
import mlia_breeze.cluster.Common._
import mlia_breeze.cluster.KMeans.State

object kMeansTest extends App{
  override def main(args: Array[String]): Unit = {
    val data = Prep.loadDataSet("kmeans/testSet.txt")
    val dataSet = DenseMatrix(data: _*)
    printf("%f, %f, %f, %f\n", min(dataSet(::, 0)), max(dataSet(::, 0)), min(dataSet(::, 1)), max(dataSet(::, 1)))
    println(randCent(dataSet, 2))
    val res:  State = KMeans(dataSet, 4)(Common.distEuclid, randCent)
    println(res.centroids)
  }
}
