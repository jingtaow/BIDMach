package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

object TestLDALearner {
  
  def runLDALearner(mat0: Mat, blockSize0: Int, ndims0: Int, useGPU0: Boolean): Unit = {
    val ds = new MatDataSource(Array(mat0))
    ds.opts.blockSize = blockSize0
    
    val model = new LDAModel()
    model.opts.dim = ndims0
    model.opts.useGPU = useGPU0
    
    
    val updater = new IncNormUpdater()
    
    val lopts = new Learner.Options()
    
    val learner = new Learner(ds, model, null, updater, lopts)
 
    learner.init
    
    learner.run
  }
  
  def main(args: Array[String]): Unit = {
    val dirname = args(0)
    val blockSize0 = args(1).toInt
    val ndims0 = args(2).toInt
    //val nthreads = args(3).toInt
    val useGPU0 = args(3).toBoolean
    Mat.checkMKL
    Mat.checkCUDA
    val mat0: SMat = loadSMat(dirname)
    runLDALearner(mat0, blockSize0, ndims0, useGPU0)
  } 

}