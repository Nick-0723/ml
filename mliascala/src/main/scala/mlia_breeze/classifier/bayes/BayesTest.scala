package mlia_breeze.classifier.bayes

import breeze.linalg.DenseMatrix
import scala.collection.mutable.ArrayBuffer
import breeze.linalg.DenseVector

object BayesTest extends App {
  def loadDataSet(): (DenseVector[String], DenseVector[Int]) = {
    (DenseVector("my dog has flea problems help please",
      "maybe not take him to dog park stupid",
      "my dalmation is so cute i love him",
      "stop posting stupid worthless garbage",
      "mr licks ate my steak how to stop him",
      "quit buying worthless dog food stupid"),
      DenseVector(0, 1, 0, 1, 0, 1))
  }

  def createVocabList(dataSet: DenseVector[String]): Array[String] =  
    (0 until dataSet.length).foldLeft(Array.empty[String])((set, i) => dataSet(i).split(" ") ++ set).distinct
 

  def setOfWordsToVec(vocabList: Array[String], inputStr: String): DenseVector[Int] = {
    val numWords = vocabList.size
    val returnVec = DenseVector.zeros[Int](numWords)
    inputStr.split(" ").foreach(x =>  returnVec(vocabList.indexOf(x)) = 1)
    returnVec
  }
  
  override def main(args: Array[String]): Unit = {
    val wordList = createVocabList(loadDataSet()._1)
    wordList.foreach(x => print(x + ","))
    println(wordList.size)
    val matrix = (0 until loadDataSet()._1.length).foldLeft(DenseMatrix.zeros[Int](loadDataSet()._1.length, wordList.length))((ma, index) => {
      ma(index, ::) := setOfWordsToVec(wordList, loadDataSet()._1(index)).t
      ma
    })
    val trainMatrix = NativeBayes.trainNB0(matrix, loadDataSet()._2)
    println(trainMatrix._1)
    println(trainMatrix._2)
    println(trainMatrix._3)
    val index = wordList.indexOf("stupid")
    println(index)
    val inputStr = setOfWordsToVec(wordList, "love my dalmation")
    printf("%s, classified as:%d\n", inputStr, NativeBayes.classifyNB(inputStr, trainMatrix._1.probability, trainMatrix._2.probability, trainMatrix._3))
    val inputStr1 = setOfWordsToVec(wordList, "stupid garbage")
    printf("%s, classified as:%d\n", inputStr1, NativeBayes.classifyNB(inputStr1, trainMatrix._1.probability, trainMatrix._2.probability, trainMatrix._3))
    println(trainMatrix._1.probability(wordList.indexOf("stupid")))
    println(trainMatrix._2.probability(wordList.indexOf("stupid")))

  }
}