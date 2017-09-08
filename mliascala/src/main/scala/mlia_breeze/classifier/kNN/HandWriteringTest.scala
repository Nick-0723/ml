package mlia_breeze.classifier.kNN
import java.io.File

import scala.io.Source

import breeze.linalg._
import breeze.linalg.DenseMatrix
import breeze.linalg._
import breeze.stats.distributions._
import breeze.numerics._

object HandWriteringTest extends App {
  def image_to_vector(fileName: String): Vector[Double] = DenseVector[Double](Source.fromFile(fileName).getLines().toArray.flatMap(x => x.map(y => (y.toInt - 48).toDouble)))
  def image_to_vector(fileName: File): Vector[Double] = DenseVector[Double](Source.fromFile(fileName).getLines().toArray.flatMap(x => x.map(y => (y.toInt - 48).toDouble)))

  def listDir(dir: String) = {
    val file = new File(dir)
    file.list()
  }

  def readDatas(dataDir: String): (Matrix[Double], Vector[String], Int) = {
    val fileCount = new File(dataDir).list().size
    val dataMat = DenseMatrix.zeros[Double](fileCount, 1024)
    val labels = DenseVector.zeros[String](fileCount)
    var index = 0
    listDir(dataDir).foreach(x => {
      val src = image_to_vector(new File(dataDir, x)).toDenseVector
      dataMat(index, ::) := src.t
      labels(index) = (x.split("\\."))(0).split("_")(0)
      index = index + 1
    })
    (dataMat, labels, fileCount)
  }
  override def main(args: Array[String]) = {
    val dataDir = "digits/trainingDigits"
    val (dataMat, dataLabels, dataCounts) = readDatas(dataDir)

    val testDir = "digits/testDigits"
    val (testMat, testLabels, testCounts) = readDatas(testDir)
    val vec = testMat.toDenseMatrix

    val errorCount = (0 until testCounts).foldLeft(0)((x, y) => {
      val label = kNN.classify0(vec(y, ::).t, dataMat, dataLabels, 3)
      if (!label.equals(testLabels(y))) {
        printf("the classifier came back with: %s, the real answer is %s\n", label, testLabels(y))

        x + 1
      } else x
    })
    printf("the total number of samples is: %d\n", testCounts)
    printf("the total number of errors is: %d\n", errorCount)
    printf("the total error rate is: %f\n", errorCount / testCounts.toDouble)
  }
}