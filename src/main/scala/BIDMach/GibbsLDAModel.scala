package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.CUMAT._
import java.io._

class GibbsLDAModel(sdata: SMat, k: Int, nsamps: Float, w: Float, alpha: Float, beta: Float, sbatch: Int) {
	
  var A: GMat = null;
  var B: GMat = null;
  var AN: GMat = null;
  var BN: GMat = null;
  var bdata:GSMat = null;
  val (nfeats, nusers) = size(sdata)
  val nbatch = nusers/sbatch
  
  def init = {
    
    A = grand(k, nfeats)
    B = grand(k, nbatch)
    AN = gzeros(k, nfeats)
    BN = gzeros(k, nbatch)
  }
  
  def update = {
    
    A ~ A + alpha
    B ~ B + beta
    
    // iteration
    for (i <- 0 until 10) {
      
      //mini-batch
      for(j <- 0 until nbatch){  
    	bdata = GSMat(sdata(?, j*nbatch until (j+1)*nbatch))
        val preds = DDS(A, B, bdata)	
        val dc = bdata.contents
	  	val pc = preds.contents
	  	//max(1e-6f, pc, pc)
	  	//pc ~ dc / pc
        pc ~ pc / dc
    	LDAgibbs(k, bdata.nnz, A.data, B.data, AN.data, BN.data, bdata.ir, bdata.ic, pc.data, nsamps)
        A = w*A + (1-w)*AN + alpha
        B = BN + beta
        AN.clear
        BN.clear
      }
      println("iteration: %d, perplexity: %f".format(i, perplexity))
    }
  }
  
  def perplexity:Double = {  
    A = A / sum(A)
    B = B / sum(B)
  	val preds = DDS(A, B, bdata)
  	val dc = bdata.contents
  	val pc = preds.contents
  	max(1e-6f, pc, pc)
  	ln(pc, pc)
  	val sdat = sum(bdata,1)
  	val mms = sum(A,2)
  	val suu = ln(mms ^* B) 
  	val vv = ((pc ddot dc) - (sdat ddot suu))/sum(sdat,2).dv
  	//row(vv, math.exp(-vv))
  	math.exp(-vv)
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
	val w = args(3).toFloat
	val alpha = args(4).toFloat
	val beta = args(5).toFloat
	val sbatch = args(6).toInt
	// sdata dimension nfeats (words) * nusers (documents)
	val data:SMat = loadSMat(fname)
	println("size of smat: " + size(data))
	println("nnz of smat: " + data.nnz)
	
	Mat.checkMKL
	Mat.checkCUDA

	println("gpu: " + Mat.hasCUDA)
    //val sdata = GSMat(data)
    
    val model = new GibbsLDAModel(data, k, nsamps, w, alpha, beta, sbatch)
	model.init
	model.update
	
	
    
  }
  
}