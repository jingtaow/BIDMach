package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.CUMAT._

class ALSModel(override val opts:ALSModel.Options = new ALSModel.Options) extends FactorModel(opts) { 
  var mm:Mat = null
  var alpha:Mat = null
  
  var traceMem = false
  
  override def init(datasource:DataSource) = {
    super.init(datasource)
    mm = modelmats(0)
    modelmats = new Array[Mat](2)
    modelmats(0) = mm
    mm = 0.5 * rand(mm.nrows, mm.ncols)
    modelmats(1) = mm.ones(mm.nrows, 1)
    updatemats = new Array[Mat](2)
    updatemats(0) = mm.zeros(mm.nrows, mm.ncols)
    updatemats(1) = mm.zeros(mm.nrows, 1)
    

  }
  
  def uupdate(sdata:Mat, user:Mat):Unit =  {

    if (opts.putBack < 0) user.set(1f)
    val bu = mm * sdata
    val lambdau = sum((sdata>0f), 1) * opts.lambda
    
    //conjugate gradient setup
    val Ax = lambdau *@ user + mm * DDS(mm, user, sdata)
    val r = bu - Ax
    val p = r
        
    val rsold = (r dot r)
    val rsnew = rsold * 0
    //println("uupdate checkpoint: " + mm + user + bu + lambdau + Ax + r + p + rsold + rsnew)    
    //conjugate gradient loop
    for (i <- 0 until opts.uiter) {
       val Ap = (lambdau *@ p) + mm * DDS(mm, user, sdata)
       val pAp = (p dot Ap)
       max(opts.weps, pAp, pAp)
	   val alpha = rsold / pAp
	   
	   user ~ user + p *@ alpha
	   r ~ r - (Ap *@ alpha)
	   rsnew ~ (r dot r)
	   max(opts.weps, rsold, rsold)
	   p ~ r + (p *@ (rsnew / rsold))
	  
	   rsold <-- rsnew
    }
  }
  
  def mupdate(sdata:Mat, user:Mat):Unit = {
	val bm = user xT sdata
	val lambdam = sum((sdata>0f), 2).t * opts.lambda	
	
	// conjugate gradient setup
	val Ax = (lambdam *@ mm) + (user xT DDS(mm, user, sdata))
	val r = bm - Ax
    val p = r
        
    val rsold = (r dot r)
    val rsnew = rsold * 0
    
    //println("uupdate checkpoint: " + mm + user + bm + lambdam + Ax + r + p + rsold + rsnew)  
    //conjugate gradient loop
    for (i <- 0 until opts.miter) {
       val Ap = (lambdam *@ p) + (user xT DDS(mm, user, sdata))
       val pAp = (p dot Ap)
       max(opts.weps, pAp, pAp)
	   val alpha = rsold / pAp

	   mm ~ mm + p*@alpha
	   r ~ r - (Ap *@ alpha)
	   rsnew ~ (r dot r)
	   max(opts.weps, rsold, rsold)
	   p ~ r + (p *@ (rsnew/rsold))
	
	   rsold <-- rsnew
    }
  	
  }
  def evalfun(sdata:Mat, user:Mat):FMat = {  
  	val preds = DDS(mm, user, sdata)
  	val dc = sdata.contents
  	val pc = preds.contents
  	val vv = (dc - pc) ddot (dc - pc)
  	println("pc: " + pc)
  	row(vv)
  }
}

object ALSModel  {
  class Options extends FactorModel.Options {
    var LDAeps = 1e-6f
	var lambda = 0.05f
	var miter = 8
  }
  
}
