package BIDMach.models

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._

abstract class RegressionModel(override val opts:RegressionModel.Opts) extends Model {
  var targmap:Mat = null
  var targets:Mat = null
  var mask:Mat = null
  
  override def init(datasource:DataSource) = {
    super.init(datasource)
    useGPU = opts.useGPU && Mat.hasCUDA > 0
    val data0 = mats(0)
    val m = size(data0, 1)
    val targetData = mats.length > 1
    val d = if (targetData) mats(1).nrows else opts.targmap.nrows
    val sdat = (sum(data0,2).t + 0.5f).asInstanceOf[FMat]
    val sp = sdat / sum(sdat)
    println("initial perplexity=%f" format (sp ddot ln(sp)) )
    
    val rmat = rand(d,m) 
    rmat ~ rmat *@ sdat
    val msum = sum(rmat, 2)
    rmat ~ rmat / msum
    val mm = rmat
    modelmats = Array[Mat](1)
    modelmats(0) = if (useGPU) GMat(mm) else mm 
    updatemats = new Array[Mat](1)
    updatemats(0) = modelmats(0).zeros(mm.nrows, mm.ncols)
    if (! targetData) {
      targets = if (useGPU) GMat(opts.targets) else opts.targets
      targmap = if (useGPU) GMat(opts.targmap) else opts.targmap
      mask = if (useGPU) GMat(opts.rmask) else opts.rmask
    }
  } 
  
  def mupdate(data:Mat):FMat
  
  def mupdate2(data:Mat, targ:Mat):FMat
  
  def doblock(gmats:Array[Mat], ipass:Int, i:Long) = {
    if (gmats.length == 1) {
      mupdate(gmats(0))
    } else {
      mupdate2(gmats(0), gmats(1))
    }
  }
  
  def evalblock(mats:Array[Mat], ipass:Int):FMat = {
    if (gmats.length == 1) {
      mupdate(gmats(0))
    } else {
      mupdate2(gmats(0), gmats(1))
    }
  }
}

object RegressionModel {
  trait Opts extends Model.Opts {
    var targets:FMat = null
    var targmap:FMat = null
    var rmask:FMat = null
  }
  
  class Options extends Opts {}
}
