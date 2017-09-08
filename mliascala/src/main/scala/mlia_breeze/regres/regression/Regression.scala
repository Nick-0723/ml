package mlia_breeze.regres.regression
import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import breeze.stats.mean.reduce_Double
import breeze.stats.variance.reduceDouble
object Regression {
  def corrcoef(x1: DenseVector[Double], x2: DenseVector[Double]) = {
    val cr = DenseMatrix.zeros[Double](2, 2)
    cr(0, 0) = corrcoef1(x1, x1)
    cr(0, 1) = corrcoef1(x1, x2)
    cr(1, 0) = corrcoef1(x2, x1)
    cr(1, 1) = corrcoef1(x2, x2)
    cr
  }
  def corrcoef1(x1: DenseVector[Double], x2: DenseVector[Double]) = {
    val x3 = x1 :+= -mean(x1)
    val x4 = x2 :+= -mean(x2)

    (x3 dot x4) / sqrt(sum(x3 * x3) * sum(x4 * x4))
  }

  def rssError(yArr: Array[Double], yHatArr: Array[Double]): Double =
    yArr.zip(yHatArr).foldLeft(0.0) { case (state, (y, yHat)) => state + scala.math.pow(y - yHat, 2) }

  /**
   * line regression
   */
  def standRegres(xArr: Array[Array[Double]], yArr: Array[Double]): DenseVector[Double] = {
    val xMat = DenseMatrix(xArr: _*)
    val yVec = DenseVector(yArr)
    standRegres(xMat, yVec)
     
  }
   def standRegres(xArr: DenseMatrix[Double], yArr: DenseVector[Double]): DenseVector[Double] = {
    val xMat = xArr //DenseMatrix(xArr: _*)
    val yVec = yArr //DenseVector(yArr)
    val xTx = xMat.t * xMat
    if (det(xTx) == 0.0) {
      println("this matrix is singular, cannot do inverse")
      DenseVector.zeros[Double](0)
    } else {
      inv(xTx) * (xMat.t * yVec)
    }
  }
  /**
   * Locally weighted liner regression.
   */
  def lwlr(testPoint: Array[Double], xArr: Array[Array[Double]], yArr: Array[Double], k: Double = 1.0): Double = {
    val xMat = DenseMatrix(xArr: _*)
    val yVec = DenseVector(yArr)
    val weights = (0 until xMat.rows).foldLeft(DenseMatrix.eye[Double](xMat.rows))((weights, index) => {
      val diffMat = DenseVector(testPoint) - xMat(index, ::).t
      weights(index, index) = exp((diffMat dot diffMat) / (-2.0 * pow(k, 2.0)))
      weights
    })

    val xTx = xMat.t * (weights * xMat)
    if (det(xTx) == 0.0) {
      println("this matrix is singular, cannot do inverse")
      println(xTx)
      0.0
    } else {
      val ws = inv(xTx) * (xMat.t * (weights * yVec))
      DenseVector(testPoint) dot ws
    }
  }
  def lwlrTest(tests: Array[Array[Double]], xArr: Array[Array[Double]], yArr: Array[Double], k: Double = 1.0) = {
    (0 until tests.length).foldLeft(DenseVector.zeros[Double](tests.length))((yHat, i) => {
      val s = tests(i)
      yHat(i) = lwlr(s, xArr, yArr, k)
      yHat
    })
  }

  /**
   * Ridge regression
   */
  def ridgeRegres(xMat: DenseMatrix[Double], yVec: DenseVector[Double], lam: Double = 0.2): DenseVector[Double] = {
    val xTx = xMat.t * xMat
     val denom = xTx + DenseMatrix.eye[Double](xMat.cols) * lam
    //    val denom = xTx + diag(DenseVector.ones[Double](xMat.cols)) * lam
     if (det(denom) == 0) {
      println("this matrix is singular, cannot do inverse")
      DenseVector.zeros[Double](0)
    } else {
      inv(denom) * (xMat.t * yVec)
    }
  }
  def ridgeRegres(xArr: Array[Array[Double]], yArr: Array[Double], lam: Double): DenseVector[Double] =  
    ridgeRegres(DenseMatrix(xArr: _*), DenseVector(yArr), lam)
  

  def ridgeTest(xArr: Array[Array[Double]], yArr: Array[Double]) = {
    val xMat = DenseMatrix(xArr: _*)
    val yVec = DenseVector(yArr)
    val yMean = mean(yVec)  
    val yDev = yVec - yMean       //:-= element wise sub
    val xReg = regularize(xMat)

    (0 until 30).foldLeft(DenseMatrix.zeros[Double](30, xReg.cols))((state, i) => {
      val ws = ridgeRegres(xReg, yDev, exp(i - 10.0))
      state(i, ::) := ws.t
      state
    })
  }
  /**
   * utility: regularize by columns.
   */
  def regularize(xMat: DenseMatrix[Double]): DenseMatrix[Double] = {
    // calc mean then subtract it off
     val xMeans  = mean(xMat, Axis._0).t
   
    // calc variance of Xi then divide by it
    val xVar = variance(xMat, Axis._0).t
    (0 until xMat.rows).foldLeft(DenseMatrix.zeros[Double](xMat.rows, xMat.cols)) { (state, i) =>
       state(i, ::) := (xMat(i, ::).t - xMeans /:/ xVar).t
      state
    }
  }

  case class StageWiseState(ws: DenseVector[Double], wsMax: DenseVector[Double], lowestError: Double = Inf) {
    def initError(): StageWiseState = copy(lowestError = Inf)
  }
  
  object StageWiseState{
    def apply(ws: DenseVector[Double]): StageWiseState = StageWiseState(ws, ws)
  }
  
  def stageWise(xArr: Array[Array[Double]], yArr: Array[Double], eps: Double = 0.01, numIt: Int =100) = {
    val xMat = DenseMatrix(xArr: _*)
    val yVec = DenseVector(yArr)
    val yMean = mean(yVec)
    val yDev = yVec :- yMean
    val xReg = regularize(xMat)
    (0 until numIt).foldLeft(StageWiseState(DenseVector.zeros(xMat.cols)))((outerState, i) => {
      println("" + outerState.ws + ":" + outerState.lowestError)
      val curState = (0 until xMat.cols).foldLeft(outerState.initError())((innerState, j) => {
        Array(-1, 1).foldLeft(innerState)((state, sign) => {
          val wsTest = state.ws.copy
          wsTest(j) += eps*sign
          val yTest = xReg * wsTest
          val rssE = rssError(yTest.toArray, yDev.toArray)
          if(rssE < state.lowestError)state.copy(lowestError = rssE, wsMax = wsTest) else state
        })
      })
      curState.copy(ws = curState.wsMax)
    })
  }.ws
  
  
 
}