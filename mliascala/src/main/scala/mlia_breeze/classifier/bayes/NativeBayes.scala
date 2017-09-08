package mlia_breeze.classifier.bayes

import breeze.linalg._
import breeze.numerics._

object NativeBayes {
  case class Prob(num: Vector[Int], denom: Double){
    def probability: Vector[Double] = num.mapValues(_.toDouble) :/ denom
    def logProbability: Vector[Double] = log(probability)
  }
  
  object Prob{
    def apply(size: Int): Prob = Prob(DenseVector.zeros(size), 2.0d)
  }
  
  def trainNB0(trainMatrix: DenseMatrix[Int], trainCategory: Vector[Int]) = {
    val numTrainDocs = trainMatrix.rows
    val numWords = trainMatrix.cols
    
    val probs = (0 until numTrainDocs).foldLeft((Prob(numWords), Prob(numWords)))((state, i) =>{
      val v = trainMatrix(i, ::).t
      if(trainCategory(i) == 1) (Prob(state._1.num + v, state._1.denom + sum(v)), state._2)
      else (state._1, Prob(state._2.num + v, state._2.denom + sum(v)))
     
    } )
    (probs._2, probs._1, sum(trainCategory)/numTrainDocs.toDouble ) //probability, class=0, 
  }
  
  def classifyNB(vec2Classify: Vector[Int], p0Vec: Vector[Double], p1Vec: Vector[Double], pClass1: Double) = {
    val p1 = sum((vec2Classify.mapValues(_.toDouble) :* p1Vec)) + log(pClass1)
    val p0 = sum((vec2Classify.mapValues(_.toDouble) :* p0Vec)) + log(1.0 - pClass1)
    if (p1 > p0) 1 else 0
  }
}