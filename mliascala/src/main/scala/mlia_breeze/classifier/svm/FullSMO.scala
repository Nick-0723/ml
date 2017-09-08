package mlia_breeze.classifier.svm

import scala.annotation.tailrec

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.Vector
import breeze.stats.distributions.Uniform
import breeze.linalg.Matrix

object FullSMO {
  case class OptStruct(dataMat: DenseMatrix[Double], labelMat: Vector[Double], alphas: Vector[Double], b: Double = 0.0, constant: Double, tolerance: Double) {
    val eCache: DenseMatrix[Double] = DenseMatrix.zeros[Double](dataMat.rows, 2)
    val rows = dataMat.rows
    def label(i: Int): Double = labelMat(i)
    def alpha(idx: Int) = alphas(idx)
    def validEcacheArr: Array[Int] = eCache(::, 0).findAll(_ != 0).toArray

    def calcETA(i: Int, j: Int): Double = {
      val iVec = dataMat(i, ::).t
      val jVec = dataMat(j, ::).t
      (iVec.t * jVec * 2.0) - (iVec.t * iVec) - (jVec.t * jVec)
    }

    def calcEk(k: Int): Double = {
      val fXk = (alphas *:* labelMat).t * (dataMat * dataMat(k, ::).t) + b
      fXk - label(k)
    }

    def nonBoundIndices: Seq[Int] = alphas.findAll(x => x > 0 && x < constant)

    def cache(i: Int, ei: Double) = {
      eCache(i, ::) := DenseVector(1, ei).t
    }

    def updateEk(k: Int) = cache(k, calcEk(k))

    def newB(b: Double) = copy(b = b)

    case class JOpt(maxK: Int = dataMat.rows - 1, maxDeltaE: Double = 0.0, ej: Double = 0.0)

    def selectJ(i: Int, ei: Double) = {
      cache(i, ei)

      if (validEcacheArr.size > 1) {
        val opt = validEcacheArr.filter(_ != i).foldLeft(JOpt())((jOpt, k) => {
          val ek = calcEk(k)
          val deltaE = (ei - ek).abs
          if (deltaE > jOpt.maxDeltaE) JOpt(k, deltaE, ek) else jOpt
        })
        (opt.maxK, opt.ej)

      } else {
        val j = selectJrand(i, rows)

        val ej = calcEk(j)
        (j, ej)
      }
    }
  }
  def innerL(i: Int, oS: OptStruct) = {
    val ei = oS.calcEk(i)
    if ((oS.label(i) * ei < -oS.tolerance && oS.alphas(i) < oS.constant) || (oS.label(i) * ei > oS.tolerance && oS.alphas(i) > 0)) {
      val (j, ej) = oS.selectJ(i, ei)
      val (alphaIold, alphaJold) = (oS.alphas(i), oS.alphas(j))
      val (low, high) = calcLH(oS, i, j)
      if (low == high) {
        println(s"L == H[$low]")
        (0, oS)
      } else {
        val eta = oS.calcETA(i, j)
        if (eta >= 0) {
          println(s"eta[$eta] >= 0")
          (0, oS)
        } else {
          oS.alphas(j) -= oS.label(j) * (ei - ej) / eta
          oS.alphas(j) = clipAlpha(oS.alphas(j), high, low)
          oS.updateEk(j)
          if ((oS.alphas(j) - alphaJold).abs < 0.00001) {
            println(s"j not moving enough[${(oS.alphas(j) - alphaJold).abs}]")
            (0, oS)
          } else {
            oS.alphas(i) += (oS.label(j) * oS.label(i) * (alphaJold - oS.alphas(j)))
            oS.updateEk(i)

            val (b1, b2) = calcB(oS, i, alphaIold, ei, j, alphaJold, ej)
            val newB = {
              if (oS.alphas(i) > 0 && oS.alphas(i) < oS.constant) b1
              else if (oS.alphas(j) > 0 && oS.alphas(j) < oS.constant) b2
              else (b1 + b2) / 2.0
            }
            (1, oS.newB(newB))
          }
        }
      }
    } else (0, oS)
  }

  def smoP(dataMatIn: Array[Array[Double]], classLabels: Array[Double], c: Double, toler: Double, maxIter: Int) = {

    @tailrec
    def outerL(oS: OptStruct, iter: Int = 0, entireSet: Boolean = true, curAlphaPairsChanged: Int = 0): (Vector[Double], Double) = {
      if (iter >= maxIter || (curAlphaPairsChanged == 0 && !entireSet)) {
        (oS.alphas, oS.b)
      } else {
        val (alphaPairsChanged, updatedOS) = if (entireSet) {
          // go over all values
          (0 until oS.rows).foldLeft(0, oS) {
            case ((totalChanged, curOS), i) =>
              val (changed, newOS) = innerL(i, curOS)
              println(s"fullSet, iter: $iter i:$i, pairs changed $totalChanged")
              (totalChanged + changed, newOS)
          }
        } else {
          // go over non-bound (railed) alphas
          oS.nonBoundIndices.foldLeft(0, oS) {
            case ((totalChanged, curOS), i) =>
              val (changed, newOS) = innerL(i, curOS)
              println(s"non-bound, iter: $iter i:$i, pairs changed $totalChanged")
              (totalChanged + changed, newOS)
          }
        }
        // toggle entire set loop
        val updatedEntireSet = if (entireSet) false else if (alphaPairsChanged == 0) true else entireSet
        println(s"iteration number: $iter")
        outerL(updatedOS, iter + 1, updatedEntireSet, alphaPairsChanged)
      }
    }

    outerL(OptStruct(
      dataMat = DenseMatrix(dataMatIn: _*),
      labelMat = DenseVector(classLabels),
      alphas = DenseVector.zeros[Double](dataMatIn.size),
      b = 0.0, constant = c, tolerance = toler))
  }

  private def calcB(os: OptStruct, i: Int, alphaIold: Double, ei: Double, j: Int, alphaJold: Double, ej: Double): (Double, Double) = {
    val b1 = os.b - ei - (os.label(i) * (os.alpha(i) - alphaIold)) *
      (os.dataMat(i, ::).t dot os.dataMat(i, ::).t) -
      (os.label(j) * (os.alpha(j) - alphaJold)) * (os.dataMat(i, ::).t dot os.dataMat(j, ::).t)

    val b2 = os.b - ej - (os.label(i) * (os.alpha(i) - alphaIold)) *
      (os.dataMat(i, ::).t dot os.dataMat(j, ::).t) -
      (os.label(j) * (os.alpha(j) - alphaJold)) * (os.dataMat(j, ::).t dot os.dataMat(j, ::).t)
    (b1, b2)
  }

  private def calcLH(os: OptStruct, i: Int, j: Int): (Double, Double) = {
    if (os.label(i) != os.label(j)) {
      val dev = os.alpha(j) - os.alpha(i)
      (scala.math.max(0.0, dev), scala.math.min(os.constant, os.constant + dev))
    } else {
      val total = os.alpha(j) + os.alpha(i)
      (scala.math.max(0.0, total - os.constant), scala.math.min(os.constant, total))
    }
  }
  def selectJrand(i: Int, m: Int): Int = {
    val rand = Uniform(0, m)
    @tailrec
    def step(i: Int, j: Int): Int = if (i != j) j else step(i, rand.sample().toInt)

    step(i, rand.sample().toInt)
  }

  def clipAlpha(aj: Double, h: Double, l: Double): Double = if (aj > h) h else if (aj < l) l else aj
  
  def calcWs(alphas: DenseVector[Double], dataArr: Seq[Array[Double]], classLabels: Array[Double]) = {
    val x = DenseMatrix(dataArr: _*)
    val labelMat = DenseVector(classLabels).t
    (0 until x.rows).foldLeft(DenseVector.zeros[Double](x.cols)) { (state, i) =>
      state :+ (x(i, ::).t :*= (alphas(i) * labelMat(i)))
    }

  }
}