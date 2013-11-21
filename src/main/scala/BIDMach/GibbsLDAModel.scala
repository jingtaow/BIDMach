package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.CUMAT._
import java.io._

class GibbsLDAModel(sdata: GSMat, k: Int, nsamps: Float) {
	
  var A: GMat = null;
  var B: GMat = null;
  var AN: GMat = null;
  var BN: GMat = null;
  var (nfeats, nusers) = size(sdata)
  
  def init = {
    
    A = grand(k, nfeats)
    B = grand(k, nusers)
    AN = gzeros(k, nfeats)
    BN = gzeros(k, nusers)
  }
  
  def update = {
    for (i <- 0 until 10) {
        val preds = DDS(A, B, sdata)	
        val dc = sdata.contents
	  	val pc = preds.contents
	  	max(1e-6f, pc, pc)
	  	pc ~ dc / pc
    	LDAgibbs(k, sdata.nnz, A.data, B.data, AN.data, BN.data, sdata.ir, sdata.ic, pc.data, nsamps)
        A = A + AN
        B = B + BN
        AN.zeros(k, nfeats)
        BN.zeros(k, nusers)
        
        println("iteration: %d, perplexity: %f".format(i, mean(perplexity).dv))
    }
  }
  
  def perplexity:FMat = {  
  	val preds = DDS(A, B, sdata)
  	val dc = sdata.contents
  	val pc = preds.contents
  	max(1e-6f, pc, pc)
  	ln(pc, pc)
  	val sdat = sum(sdata,1)
  	val mms = sum(A,2)
  	val suu = ln(mms ^* B) 
  	val vv = ((pc ddot dc) - (sdat ddot suu))/sum(sdat,2).dv
  	row(vv, math.exp(-vv))
  }
  

}

object GibbsLDAModel{
  
  def loadFile(fname: String, oname: String)={
    val f = new BufferedReader(new FileReader(fname))
    val D = f.readLine().toInt
    val W = f.readLine().toInt
    val nnz = f.readLine().toInt
    
    var r = izeros(nnz, 1)
    var c = izeros(nnz, 1)
    var v = zeros(nnz, 1)
    
    var i = 0
    
    var l = f.readLine()
    while(l != null){
      var t = l.split(" ")
      r(i) = t(0).toInt-1
      c(i) = t(1).toInt-1
      v(i) = t(2).toFloat
      i = i+1
      l = f.readLine()
    }
    
    //val smat = sparse(r(0 to i-1), c(0 to i-1), v(0 to i-1))
    val smat = sparse(c(0 to i-1), r(0 to i-1), v(0 to i-1))
    
    saveSMat(oname, smat)
  }
  
  def main(args: Array[String]) = {
	val fname = args(0)
	
	//val oname = args(1)
	//loadFile(fname, oname)
	
	
	val k = args(1).toInt
	val nsamps = args(2).toFloat
	// sdata dimension nfeats (words) * nusers (documents)
	val data:SMat = loadSMat(fname)
	println("size of smat: " + size(data))
	println("nnz of smat: " + data.nnz)
	
	Mat.checkMKL
	Mat.checkCUDA

	println("gpu: " + Mat.hasCUDA)
    val sdata = GSMat(data)
    
    val model = new GibbsLDAModel(sdata, k, nsamps)
	model.init
	model.update
	
	
    
  }
  
}