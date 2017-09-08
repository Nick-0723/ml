package mlia_breeze.cluster

import breeze.linalg.{DenseMatrix, max, min}
import mlia_breeze.cluster.Common.randCent

object BiKMeansTest extends App{
  override def main(args: Array[String]): Unit = {
    val data = Prep.loadDataSet("kmeans/testSet2.txt")
    val dataSet = DenseMatrix(data: _*)
     val res  = BisectingKMeans(dataSet, 3)(Common.distEuclid, randCent)
    res.centroids.foreach(println)
  }
}
