package mlia_breeze.cluster

import breeze.linalg.{Axis, DenseMatrix, DenseVector}
import breeze.numerics.Inf
import mlia_breeze.cluster.Common.Assessment
import breeze.stats._

import scala.collection.mutable.ArrayBuffer

object KMeans {
  case class RowState(minDist: Double = Inf, closestIndex: Int = -1)

  case class State(centroids: DenseMatrix[Double], clusterAssment: Array[Assessment], clusterChanged: Boolean){
    def update(i: Int, updateAssment: Assessment): State ={
     // val changed = clusterAssment(i).clusterIndex != updateAssment.clusterIndex
      val buf = ArrayBuffer(clusterAssment:_* )
      buf.update(i, updateAssment)
      val changed = clusterAssment.foldLeft(false){(r, a) =>
        r || a.changed
      }
      copy(clusterAssment = buf.toArray, clusterChanged = changed)
    }

    def getIndices(cent: Int): Array[Int] = clusterAssment.zipWithIndex.filter(x => x._1.clusterIndex == cent).map(_._2)
    override def toString: String = s"centroid:\n $centroids\ndataPoints:${clusterAssment.mkString(", ")}"

  }


  def apply(dataSet: DenseMatrix[Double], k: Int)(implicit distMeas: (DenseVector[Double], DenseVector[Double]) => Double, createCent: (DenseMatrix[Double], Int) => DenseMatrix[Double]) = {
    def iterate(state: State): State = {
      if(!state.clusterChanged){
       // println("centroid does not moved anymore.")
        state
      }else{
       // println("centroid moved. calculate distance between each point and centroid...")
        val outerResult = (0 until dataSet.rows).foldLeft(state){ (outer, i) =>
          val innerResult = (0 until k).foldLeft(RowState()){ (inner, clusterIdx) =>
            val distJI = distMeas(outer.centroids(clusterIdx, ::).t, dataSet(i, ::).t)
            if(distJI < inner.minDist)RowState(distJI, clusterIdx) else inner
          }
          val changed = outer.clusterAssment(i).clusterIndex != innerResult.closestIndex
          outer.update(i, Assessment(innerResult.closestIndex, innerResult.minDist, changed))

        }
        val centroid = (0 until k).foldLeft(DenseMatrix.zeros[Double](k, dataSet.cols)){(curCentroid, cent) =>
          val ptsInClust = DenseMatrix(outerResult.getIndices(cent).map(i => dataSet(i, ::).t.valuesIterator.toArray): _*)
          curCentroid(cent, ::) := mean(ptsInClust, Axis._0)
          curCentroid
        }
         iterate(State(centroid, outerResult.clusterAssment, outerResult.clusterChanged))
      }
    }
    iterate(State(createCent(dataSet, k), Array.fill(dataSet.rows)(Assessment.zero), clusterChanged = true))
  }
}

