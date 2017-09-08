package mlia_breeze.cluster

import breeze.linalg.{*, DenseMatrix, DenseVector, max, min, sum}
import breeze.numerics.{Inf, sqrt}
import breeze.stats.distributions.{Rand, RandBasis, Uniform}
import breeze.linalg.{DenseMatrix, DenseVector, _}
import breeze.numerics._

import scala.collection.mutable.ArrayBuffer

object Common {
  implicit  def distEuclid(vecA: DenseVector[Double], vecB: DenseVector[Double]): Double = sqrt(sum((vecA - vecB) :^ 2.0))

  implicit  def randCent(dataSet: DenseMatrix[Double], k: Int)(implicit rand: RandBasis = Rand): DenseMatrix[Double] = {
    (0 until dataSet.cols).foldLeft(DenseMatrix.zeros[Double](k, dataSet.cols))(op = (centroids, idx) => {
      val minJ: Double = min(dataSet(::, idx))
      val rangeJ: Double = max(dataSet(::, idx)) - minJ
      centroids(::, idx) := DenseVector(new Uniform(0, 1).sample(k).map(_ * rangeJ + minJ): _*)
      centroids
    })
  }

  final class Assessment(val clusterIndex: Int, val error: Double, val changed: Boolean = false) {
    override def toString = f"[clusterIndex: $clusterIndex, error: $error]"

    override def equals(other: scala.Any): Boolean = other match {
      case that: Assessment => this.clusterIndex == that.clusterIndex && this.error == that.error && that.changed == this.changed
      case _ => false
    }
    def copy(c: Boolean)  = if (c) new Assessment(clusterIndex, error, true) else this
    override def hashCode(): Int = abs(127 * clusterIndex.hashCode() + error.hashCode() + (if(changed) 1 else 0))
  }

  object Assessment {

    def apply(index: Int, dist: Double) = new Assessment(index, scala.math.pow(dist, 2))
    def apply(index: Int, dist: Double, changed: Boolean) = new Assessment(index, scala.math.pow(dist, 2), changed)

    def zero = new Assessment(0, 0)
  }


  def kMeans(dataSet: DenseMatrix[Double], k: Int)(implicit distMeas: (DenseVector[Double], DenseVector[Double]) => Double, createCent: (DenseMatrix[Double], Int) => DenseMatrix[Double]) = {

    def iterate(dataSet: DenseMatrix[Double], k: Int, centroids: DenseMatrix[Double], clusterAssment: DenseMatrix[Double], clusterChanged: Boolean):(DenseMatrix[Double], DenseMatrix[Double], Boolean) = {
      if (!clusterChanged) (centroids, clusterAssment, clusterChanged)
      else{
        (0 until dataSet.rows).foldLeft(clusterAssment){(r, i) =>
          val (distance, index) = (0 until k).foldLeft((Inf, -1)){(dist, j) =>
            val distance = distMeas(centroids(j, ::).t, dataSet(i, ::).t)
            if(distance < dist._1)(distance, j)
            else dist
          }
          if(r(i, 0) != index.toDouble){
            r(i, ::) := DenseVector(index.toDouble, distance*distance).t
            (r, true)
          }else{
            (r, false)
          }
          r
        }
        (centroids, clusterAssment, clusterChanged)

      }

    }
    val res = iterate(dataSet, k, DenseMatrix.zeros[Double](4, dataSet.cols), DenseMatrix.zeros[Double](dataSet.rows, 2), true)
    (res._1, res._2)
  }



}
