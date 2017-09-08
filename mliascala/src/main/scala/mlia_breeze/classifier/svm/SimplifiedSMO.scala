package mlia_breeze.classifier.svm
import scala.annotation.tailrec
import breeze.stats.distributions.Uniform
import breeze.linalg._

object SimplifiedSMO {
  def selectJrand(i: Int, m: Int): Int = {
    val rand = Uniform(0, m)
    @tailrec
    def step(i: Int, j: Int): Int = if (i != j) j else step(i, rand.sample().toInt)

    step(i, rand.sample().toInt)
  }

  def clipAlpha(aj: Double, h: Double, l: Double): Double = if (aj > h) h else if (aj < l) l else aj

  def smoSimple(dataMatIn: Array[Array[Double]], classLabels: Array[Double], c: Double, toler: Double, maxIter: Int) = {
    val dataMat = DenseMatrix(dataMatIn: _*)
    val labelMat = DenseMatrix(classLabels).t
    @tailrec
    def outerLoop(curNum: Int = 0, curAlphas: DenseMatrix[Double] = DenseMatrix.zeros(dataMat.rows, 1), curB: Double = 0): (DenseMatrix[Double], Double) = {
      if (curNum == maxIter) (curAlphas, curB)
      else {
        val (iteAlphas, iteB, changeCount) = (0 until dataMat.rows).foldLeft(curAlphas, curB, 0) {
          case ((alphas, b, change), i) => {
            val fXi = ((alphas :* labelMat).t * (dataMat * dataMat(i, ::).t)) :+ b
            val ei = fXi(0) - labelMat(i, 0)
            if ((labelMat(i, 0) * ei < -toler && alphas(i, 0) < c) || (labelMat(i, 0) * ei > toler && alphas(i, 0) > 0)) {
              val j = selectJrand(i, dataMat.rows)
              val fXj = ((alphas :* labelMat).t * (dataMat * dataMat(j, ::).t)) :+ b
              val ej = fXj(0) - labelMat(j, 0)
              val alphaIold = alphas(i, 0)
              val alphaJold = alphas(j, 0)
              val (low, high) = if (labelMat(i, 0) != labelMat(j, 0)) {
                val dev = alphas(j, 0) - alphas(i, 0)
                (max(0.0, dev), min(c, c + dev))
              } else {
                val total = alphas(j, 0) + alphas(i, 0)
                (max(0.0, total - c), min(c, total))
              }

              if (low == high) {
                println(s"L == H[$low]")
                (alphas, b, change)
              } else {
                val eta1 = (dataMat(i, ::) * dataMat(j, ::).t) * 2.0
                val eta2 = eta1 - dataMat(i, ::) * dataMat(j, ::).t
                val eta3 = eta2 - dataMat(j, ::) * dataMat(j, ::).t
                if (eta3 >= 0) {
                  println(s"eta3[$eta3] >= 0")
                  (alphas, b, change)
                } else {
                  alphas(j, 0) -= labelMat(j, 0) * (ei - ej) / eta3
                  alphas(j, 0) = clipAlpha(alphas(j, 0), high, low)
                  if ((alphas(j, 0) - alphaJold).abs < 0.00001) {
                    println(s"j not moving enough[${(alphas(j, 0) - alphaJold).abs}]")
                    (alphas, b, change)

                  } else {
                    alphas(i, 0) += (labelMat(i, 0) * labelMat(j, 0) * (alphaJold - alphas(j, 0)))
                    val b1 = b - ei - (labelMat(i, 0) * (alphas(i, 0) - alphaIold)) * (dataMat(i, ::) * dataMat(j, ::).t) - (labelMat(j, 0) * (alphas(j, 0) - alphaJold)) * (dataMat(i, ::) * dataMat(j, ::).t)
                    val b2 = b - ej - (labelMat(i, 0) * (alphas(i, 0) - alphaIold)) * (dataMat(i, ::) * dataMat(j, ::).t) - (labelMat(j, 0) * (alphas(j, 0) - alphaJold)) * (dataMat(i, ::) * dataMat(j, ::).t)
                    val newB = {
                      if (alphas(i, 0) > 0 && alphas(i, 0) < c) b1
                      else if (alphas(j, 0) > 0 && alphas(j, 0) < c) b2
                      else (b1 + b2) / 2.0
                    }
                    println(s"iter: $curNum i:$i, pairs changed ${change + 1}")
                    (alphas, newB, change + 1)
                  }
                }
              }
            } else {
              (alphas, b, change)
            }
          }
        }
        val nextNum = if (changeCount == 0) curNum + 1 else 0
        println(s"iteration number: $nextNum")
        outerLoop(nextNum, iteAlphas, iteB)

      }
    }
    outerLoop(0)
  }

  def smoSimple1(dataMatIn: Array[Array[Double]], classLabels: Array[Double], c: Double, toler: Double, maxIter: Int): (DenseMatrix[Double], Double) = {

    val dataMat = DenseMatrix(dataMatIn: _*)
    val labelMat = DenseMatrix(classLabels).t

    @tailrec
    def outerLoop(curNum: Int = 0,
      curAlphas: DenseMatrix[Double] = DenseMatrix.zeros(dataMat.rows, 1),
      curB: Double = 0): (DenseMatrix[Double], Double) = {
      if (curNum == maxIter) (curAlphas, curB)
      else {
        val (iteAlphas, iteB, changeCount) = (0 until dataMat.rows).foldLeft(curAlphas, curB, 0) {
          case ((alphas, b, change), i) =>

            // this is our prediction of the class
            val s = (alphas :* labelMat).t
            val s2 = DenseMatrix.zeros[Double](dataMat.rows, 1)
            s2(::, 0) := (dataMat * dataMat(i, ::).t)
            //  val fXi: DenseMatrix[Double] = ((alphas :* labelMat).t * (dataMat * dataMat(i, ::).t): DenseMatrix[Double]) :+ b
            val fXi = (alphas :* labelMat).t * s2 :+ b
            // error = fXi - real class
            val ei = fXi(0, 0) - labelMat(i, 0)
            if ((labelMat(i, 0) * ei < -toler && alphas(i, 0) < c) ||
              (labelMat(i, 0) * ei > toler && alphas(i, 0) > 0)) {
              // enter optimization
              val j = selectJrand(i, dataMat.rows)
              //  val fXj: DenseMatrix[Double] = ((alphas :* labelMat).t * (dataMat * dataMat(j, ::).t): DenseMatrix[Double]) :+ b
              val s3 = DenseMatrix.zeros[Double](dataMat.rows, 1)
              s3(::, 0) := (dataMat * dataMat(i, ::).t)
              val fXj: DenseMatrix[Double] = (alphas :* labelMat).t * s3 :+ b

              val ej = fXj(0, 0) - labelMat(j, 0)
              val alphaIold = alphas(i, 0)
              val alphaJold = alphas(j, 0)
              // guarantee alphas stay between 0 and c
              val (low, high) = if (labelMat(i, 0) != labelMat(j, 0)) {
                val dev: Double = alphas(j, 0) - alphas(i, 0)
                (scala.math.max(0.0, dev), scala.math.min(c, c + dev))
              } else {
                val total: Double = alphas(j, 0) + alphas(i, 0)
                (scala.math.max(0.0, total - c), scala.math.min(c, total))
              }

              if (low == high) {
                println(s"L == H[$low]")
                (alphas, b, change)
              } else {
                // calculate optimal amount to change alpha[j]
                val eta1: DenseMatrix[Double] = DenseMatrix((dataMat(i, ::) * dataMat(j, ::).t)) :* 2.0
                val eta2: DenseMatrix[Double] = eta1 :- dataMat(i, ::) * dataMat(i, ::).t
                val eta3: DenseMatrix[Double] = eta2 :- dataMat(j, ::) * dataMat(j, ::).t
                val eta = eta3(0, 0)
                if (eta >= 0) {
                  println(s"eta[$eta] >= 0")
                  (alphas, b, change)
                } else {
                  alphas(j, 0) -= labelMat(j, 0) * (ei - ej) / eta
                  alphas(j, 0) = clipAlpha(alphas(j, 0), high, low)

                  if ((alphas(j, 0) - alphaJold).abs < 0.00001) {
                    println(s"j not moving enough[${(alphas(j, 0) - alphaJold).abs}]")
                    (alphas, b, change)
                  } else {
                    // update i by same amount as j in opposite direction
                    alphas(i, 0) += (labelMat(j, 0) * labelMat(i, 0) * (alphaJold - alphas(j, 0)))
                    val s1 = DenseMatrix(dataMat(i, ::) * dataMat(i, ::).t)
                    val s2 = DenseMatrix(dataMat(i, ::) * dataMat(j, ::).t)
                    val s3 = DenseMatrix(dataMat(i, ::) * dataMat(j, ::).t)
                    val s4 = DenseMatrix(dataMat(j, ::) * dataMat(j, ::).t)
                    val b1 = b - ei - (labelMat(i, 0) * (alphas(i, 0) - alphaIold)) * s1(0, 0) - (labelMat(j, 0) * (alphas(j, 0) - alphaJold)) * s2(0, 0)
                    val b2 = b - ej - (labelMat(i, 0) * (alphas(i, 0) - alphaIold)) * s3(0, 0) - (labelMat(j, 0) * (alphas(j, 0) - alphaJold)) * s4(0, 0)
                    val newB = {
                      if (alphas(i, 0) > 0 && alphas(i, 0) < c) b1
                      else if (alphas(j, 0) > 0 && alphas(j, 0) < c) b2
                      else (b1 + b2) / 2.0
                    }
                    println(s"iter: $curNum i:$i, pairs changed ${change + 1}")
                    (alphas, newB, change + 1)
                  }
                }
              }
            } else {
              (alphas, b, change)
            }
        }
        val nextNum = if (changeCount == 0) curNum + 1 else 0
        println(s"iteration number: $nextNum")
        outerLoop(nextNum, iteAlphas, iteB)
      }
    }
    outerLoop(0)
  }

  def smoSimple2(dataMatIn: Array[Array[Double]], classLabels: Array[Double], c: Double, toler: Double, maxIter: Int): (DenseVector[Double], Double) = {
    val dataMat = DenseMatrix(dataMatIn: _*)
    val labelVec = DenseVector(classLabels)
    @tailrec
    def outerLoop(curNum: Int = 0, curAlphas: DenseVector[Double] = DenseVector.zeros(dataMat.rows), curB: Double = 0): (DenseVector[Double], Double) = {
      if (curNum == maxIter) (curAlphas, curB)
      else {
        val (iteAlphas, iteB, changeCount) = (0 until dataMat.rows).foldLeft(curAlphas, curB, 0) {
          case ((alphasVec, b, change), i) => {
            // this is our prediction of the class     *:* elementwise multiplication
            val fXi = ((alphasVec *:* labelVec).t * (dataMat * dataMat(i, ::).t)) + b
            //  error between prediction and real class
            val ei = fXi - labelVec(i)
            if ((labelVec(i) * ei < -toler && alphasVec(i) < c) || (labelVec(i) * ei > toler && alphasVec(i) > 0)) {
              // enter optimization
              val j = selectJrand(i, dataMat.rows)
              // this is our prediction of the class     *:* elementwise multiplication              
              val fXj = ((alphasVec *:* labelVec).t * (dataMat * dataMat(j, ::).t)) + b
              val ej = fXj - labelVec(j)
              val alphaIold = alphasVec(i)
              val alphaJold = alphasVec(j)
              // guarantee alphas stay between 0 and c
              val (low, high) = if (labelVec(i) != labelVec(j)) {
                val dev = alphasVec(j) - alphasVec(i)
                (max(0.0, dev), min(c, c + dev))
              } else {
                val total = alphasVec(j) + alphasVec(i)
                (max(0.0, total - c), min(c, total))
              }

              if (low == high) {
                println(s"L == H[$low]")
                (alphasVec, b, change)
              } else {
                // calculate optimal amount to change alpha[j]                
               // val eta1 = (dataMat(i, ::) dot dataMat(j, ::)) * 2.0
               // val eta2 = eta1 - (dataMat(i, ::) dot dataMat(j, ::))
                val eta = (dataMat(i, ::) dot dataMat(j, ::)) * 2.0 - (dataMat(i, ::) dot dataMat(j, ::)) - (dataMat(j, ::) dot dataMat(j, ::))
                if (eta >= 0) {
                  println(s"eta[$eta] >= 0")
                  (alphasVec, b, change)
                } else {
                  alphasVec(j) -= labelVec(j) * (ei - ej) / eta
                  alphasVec(j) = clipAlpha(alphasVec(j), high, low)
                  if ((alphasVec(j) - alphaJold).abs < 0.00001) {
                    println(s"j not moving enough[${(alphasVec(j) - alphaJold).abs}]")
                    (alphasVec, b, change)
                  } else {
                    // update i by same amount as j in opposite direction
                    alphasVec(i) += (labelVec(i) * labelVec(j) * (alphaJold - alphasVec(j)))
                    val b1 = b - ei - (labelVec(i) * (alphasVec(i) - alphaIold)) * (dataMat(i, ::) * dataMat(j, ::).t) - (labelVec(j) * (alphasVec(j) - alphaJold)) * (dataMat(i, ::) * dataMat(j, ::).t)
                    val b2 = b - ej - (labelVec(i) * (alphasVec(i) - alphaIold)) * (dataMat(i, ::) * dataMat(j, ::).t) - (labelVec(j) * (alphasVec(j) - alphaJold)) * (dataMat(i, ::) * dataMat(j, ::).t)
                    val newB = {
                      if (alphasVec(i) > 0 && alphasVec(i) < c) b1
                      else if (alphasVec(j) > 0 && alphasVec(j) < c) b2
                      else (b1 + b2) / 2.0
                    }
                    println(s"iter: $curNum i:$i, pairs changed ${change + 1}")
                    (alphasVec, newB, change + 1)
                  }
                }
              }
            } else {
              (alphasVec, b, change)
            }
          }
        }
        val nextNum = if (changeCount == 0) curNum + 1 else 0
        println(s"iteration number: $nextNum")
        outerLoop(nextNum, iteAlphas, iteB)

      }
    }
    outerLoop(0)
  }
 def calcWs(alphas: DenseVector[Double], dataArr: Seq[Array[Double]], classLabels: Array[Double]) = {
    val x = DenseMatrix(dataArr: _*)
    val labelMat = DenseVector(classLabels).t
    (0 until x.rows).foldLeft(DenseVector.zeros[Double](x.cols)) { (state, i) =>
      state :+  (x(i, ::).t :*= (alphas(i) * labelMat(i)))
    }
     
  }
}