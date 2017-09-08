package mlia_breeze.cluster

import breeze.linalg.{Axis, DenseMatrix, DenseVector, Transpose}
import breeze.numerics.Inf
import mlia_breeze.cluster.Common._
import breeze.stats._

import scala.collection.mutable.ArrayBuffer

object BisectingKMeans {
  case class State(centroids: Array[DenseMatrix[Double]], clusterAssment: Array[Assessment]) {

    // mutate function
    def update(i: Int, updateAssment: Assessment): State = {
      import scala.collection.mutable.ArrayBuffer
      val buf = ArrayBuffer(clusterAssment: _*)
      buf.update(i, updateAssment)
      copy(clusterAssment = buf.toArray)
    }

    def getIndices(cent: Int): Array[Int] =
      clusterAssment.zipWithIndex.filter(x => x match {case (ass, i) => ass.clusterIndex == cent}).map(_._2)

    override def toString = s"centroid:\n $centMat\ndataPoints:${clusterAssment.mkString(", ")}"

    def centMat = DenseMatrix(centroids.map(_.valuesIterator.toArray): _*)
  }

  case class SplitState(bestCentToSplit: Int = -1, bestNewCents: DenseMatrix[Double] = DenseMatrix.zeros[Double](0,0),
                        bestClustAss: Array[Assessment] = Array.empty, lowestSSE:Double = Inf){
    def reassignIndex(newIdx: Int) = bestClustAss.map(ass =>{
      if(ass.clusterIndex == 1)new Assessment(newIdx, ass.error) else new Assessment(bestCentToSplit, ass.error)
    })
  }

  def apply(dataSet: DenseMatrix[Double], k: Int)(implicit distMeas: (DenseVector[Double], DenseVector[Double]) => Double,
                                                  createCent: (DenseMatrix[Double], Int) => DenseMatrix[Double])  = {
    val centroid0: DenseVector[Double]  = mean(dataSet, Axis._0).t

    //calculate initial error and update initial state
    val initialState = (0 until dataSet.rows).foldLeft(State(Array(centroid0.toDenseMatrix), Array.fill(dataSet.rows)(Assessment.zero))){
      (state, j) =>
        val error = distMeas(centroid0, dataSet(j, ::).t)
        state.update(j, Assessment(0, error))
     }

    def iterate(state: State, curK: Int): State = {
      if(curK == k){
        state
      }else{
        val bestSplit = (0 until state.centroids.size).foldLeft(SplitState()){(inner, centIdx) =>
          val ptsInClust = DenseMatrix(state.getIndices(centIdx).map(i => dataSet(i, ::).t.valuesIterator.toArray): _*)
          val clustered = KMeans(ptsInClust, 2)
          val sseSplit = clustered.clusterAssment.map(_.error).sum
          val sseNotSplit = state.clusterAssment.filter(_.clusterIndex != centIdx).map(_.error).sum
          println(f"sseSplit: $sseSplit%.5f, sseNotSplit: $sseNotSplit%.5f")
          if (sseSplit + sseNotSplit < inner.lowestSSE) {
            SplitState(centIdx, clustered.centroids, clustered.clusterAssment, sseSplit + sseNotSplit)
          } else inner
        }
        val splitAss = bestSplit.reassignIndex(state.centroids.size)
        println(s"the bestCentToSplit is: ${bestSplit.bestCentToSplit}")
        println(s"the len of bestClustAss is: ${bestSplit.bestClustAss.length}")

        //update the old centroid
        val buf = ArrayBuffer[DenseMatrix[Double]](state.centroids: _* )
        buf.update(bestSplit.bestCentToSplit, bestSplit.bestNewCents(0, ::).t.toDenseMatrix)

        //append the new centroid
        buf += bestSplit.bestNewCents(1, ::).t.toDenseMatrix

        //replace new cluster assessment(error) into current centroid
        val sss = state.getIndices(bestSplit.bestCentToSplit)
        val zippedAss: Array[(Assessment, Int)] = splitAss.zip(state.getIndices(bestSplit.bestCentToSplit))
        val newAss: Array[Assessment] = state.clusterAssment.zipWithIndex.map{
          case (curAss, i) =>
            if(curAss.clusterIndex == bestSplit.bestCentToSplit)
              zippedAss.find(_._2 == i).map(_._1).head else curAss
        }
        iterate(State(buf.toArray, newAss), curK + 1)
      }
    }
    iterate(initialState, 1)
  }
}
