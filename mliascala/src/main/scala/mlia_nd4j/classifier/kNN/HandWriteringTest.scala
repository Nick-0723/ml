package mlia_nd4j.classifier.kNN

import java.io.File

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import scala.io.Source

object HandWriteringTest extends App {
  def image_to_vector(fileName: String): INDArray =
    Nd4j.create(Source.fromFile(fileName).getLines().toArray.flatMap(x => x.map(y => (y.toInt - 48).toDouble)))
  def image_to_vector(fileName: File): INDArray = Nd4j.create(Source.fromFile(fileName).getLines().toArray.flatMap(x => x.map(y => (y.toInt - 48).toDouble)))

  def listDir(dir: String): Array[String] = {
    val file = new File(dir)
    file.list()
  }

  def readDatas(dataDir: String): (INDArray, Array[String], Int) = {
    val fileCount = new File(dataDir).list().size
     listDir(dataDir).foldLeft((Nd4j.create(fileCount, 1024), Array.empty[String], 0)){
      case ((dataMat, labels, idx), x) =>
      val src: INDArray = image_to_vector(new File(dataDir, x))
      dataMat(idx, ->) = src
      val l = labels :+ (x.split("\\."))(0).split("_")(0)
      (dataMat, l, idx + 1)
    }

  }

  override def main(args: Array[String]) = {
    val dataDir = "digits/trainingDigits"
    val (dataMat: INDArray, dataLabels: Array[String], dataCounts: Int) = readDatas(dataDir)

    val testDir = "digits/testDigits"
    val (testMat: INDArray, testLabels: Array[String], testCounts: Int) = readDatas(testDir)
    val vec: INDArray = testMat

    val errorCount = (0 until testCounts).foldLeft(0)((x, y) => {
      val label = kNN.classify0(vec(y, ->), dataMat, dataLabels, 3)
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