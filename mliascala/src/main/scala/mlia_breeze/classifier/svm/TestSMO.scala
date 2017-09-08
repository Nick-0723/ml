package mlia_breeze.classifier.svm

import scala.io.Source
import breeze.linalg.DenseVector

object TestSMO extends App {
  def loadDataSet(fileName: String): (Array[Array[Double]], Array[Double]) = {
    Source.fromFile(fileName).getLines().toArray.foldLeft((Array.empty[Array[Double]], Array.emptyDoubleArray))((res, line) => {
      val tokens = line.split("\\t")
      val array = Array(tokens(0).toDouble, tokens(1).toDouble)
      val t1 = res._1 :+ array
      val t2 = res._2 ++ Array(tokens(2).toDouble)
      (t1.toArray, t2)
    })
  }
  override def main(args: Array[String]) = {
    val (dataSet, labels) = loadDataSet("svm/testSet.txt")
    val (alphas, b) = SimplifiedSMO.smoSimple2(dataSet, labels, 0.6, 0.001, 40)
  //  val (alphas, b) = FullSMO.smoP(dataSet, labels, 0.6, 0.001, 40)
    println(b)

    alphas.toArray.zipWithIndex.filter(x => x._1 > 0.000001).foreach(i => printf("%f, %f, %f, %f\n", i._1, dataSet(i._2)(0), dataSet(i._2)(1), labels(i._2)))

    val ws = SimplifiedSMO.calcWs(alphas.toDenseVector, dataSet, labels)
    val error =(0 until dataSet.length).foldLeft(0)((c, i) => {
      val s = if ((DenseVector(dataSet(i)) dot ws) + b > 0) 1 else -1
      if(s != labels(i)) c + 1 else c
    })
    println(error/dataSet.length.toDouble)
  }
}