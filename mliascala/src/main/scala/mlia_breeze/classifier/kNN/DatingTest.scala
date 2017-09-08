package mlia_breeze.classifier.kNN

import java.io.PrintWriter

import scala.io.Source

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

object DatingTest extends App {
  def file_to_matrix(fileName: String): (DenseMatrix[Double], DenseVector[String]) = {
    val fileSize = Source.fromFile(fileName).getLines().size
    val dataSet = DenseMatrix.zeros[Double](fileSize, 3)
    val labels = DenseVector.zeros[String](fileSize)

    var index = 0
    val source = Source.fromFile(fileName).getLines().toArray.map(x => {
      val tokens = x.trim().split("\\t")
      dataSet(index, ::) := DenseVector(tokens(0).toDouble, tokens(1).toDouble, tokens(2).toDouble).t
      labels(index) = tokens(3)
      index = index + 1

    })
    (dataSet, labels)
  }
  override def main(args: Array[String]): Unit = {
    val hoRatio = 0.05
    val (datingDataMat, datingLabels) = file_to_matrix("dating_set.txt")
    val (normMat, ranges, minVals) = kNN.autoNorm(datingDataMat)
    val m = normMat.rows
    val numTestVects = (m*hoRatio).toInt
    val errorCount = (0 until numTestVects).foldLeft(1)((x, y) => {
      val denseMat = normMat.toDenseMatrix
      val datingVector = denseMat(y, ::).t
      val model = denseMat(numTestVects until m, ::)
      val labelss = datingLabels(numTestVects until m)
      val classifierResult = kNN.classify0(datingVector, model, labelss, 3)
      printf("the classifier came back widh:%s, the real answer is: %s\n", classifierResult, datingLabels(y))
      val res = if(!classifierResult.equals(datingLabels(y))) x + 1 else x
      res
    })
    printf("the total error rate is: %f\n", (errorCount/numTestVects.toDouble))
    val testData = (DenseVector(10000.0, 8, 1) - minVals) /:/ ranges
    println(testData)
    println(kNN.classify0(testData, datingDataMat, datingLabels, 7))
    println()
  }
}