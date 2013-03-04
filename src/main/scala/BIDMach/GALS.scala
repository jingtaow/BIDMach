import BIDMat.{Mat, FMat, DMat, IMat, CMat, BMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat, HMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import java.io.BufferedReader
import java.io.FileReader
import java.io.IOException

class GALS(data0: SMat, dim: Int, nIteration: Int, nConjIteration: Int, bSize: Int){
	
	// in main memory
	val data:SMat = data0;
	val datatrans:SMat = data.t;
	
	val d = dim;
	val nIter = nIteration;
	val nConjIter = nConjIteration;
	val lambda = 0.065f;
	val block_size = bSize;
	var block_pointer = 0;
	
	val (nu, nm) = size(data);
	var U:FMat = null;
	
	
	var lambdau:FMat = null;
	var lambdam:GMat = null;
	
	
	// GPU memory
	var Ub:GMat = null;
	var M:GMat = null;
	
	var R:GSMat = null;
	var Rtrans:GSMat = null;
	
	var bu:GMat = null;
	var bm:GMat = null;
	
	var lambdaub:GMat = null;

	// utility variables
	var t1 = 0l;
	var t2 = 0l;
		
	// temp variables
	var temp:GSMat = null;
	var Ax:GMat = null;
	var Ap:GMat = null;
	var tmp2:GMat = null;
	var tmp3:GMat = null;
	var r:GMat = null;
	var p:GMat = null;
	var alpha:GMat = null;
	var rsold:GMat = null;
	var rsnew:GMat = null;
	
	var tempM:GSMat = null;
	var AxM:GMat = null;
	var ApM:GMat = null;
	var tmp2M:GMat = null;
	var tmp3M:GMat = null;
	var rM:GMat = null;
	var pM:GMat = null;
	var alphaM:GMat = null;
	var rsoldM:GMat = null;
	var rsnewM:GMat = null;

	
	def init(){
		U = zeros(d, nu);
		M = GMat(0.5*rand(d, nm));
		lambdau = sum((datatrans>0f), 1)*lambda;
		lambdam = GMat(sum((data>0f), 1)*lambda);
		
		R = GSMat(data( 0 to (block_size - 1), ?));
		Rtrans = GSMat(datatrans(?, 0 to (block_size - 1)));
		Ub = GMat(U(?, 0 to (block_size - 1)));
		lambdaub = GMat(lambdau(0 to (block_size - 1)));
		
	}
	
	def train(){
		
		init();
		var k = nu/block_size - 1;
		println("k: " + k);
		var i = 0;
		for( i <- 0 to nIter-1){
			println("processing iteration: " + i);
			var j = 0;
			for( j <- 0 to k){
				println("processing block: " + j);
				block_pointer = j*block_size;
				initBlock();
				trainBlock();
			}
		}
	}
	
	def initBlock(){
	
		R <-- data( block_pointer to (block_pointer + block_size - 1), ?);
		Rtrans <-- datatrans(?, block_pointer to (block_pointer + block_size - 1));
		Ub <-- U(?, block_pointer to (block_pointer + block_size - 1));
		lambdaub <-- lambdau(block_pointer to (block_pointer + block_size - 1));
		//GSMat.fromSMat(data( block_pointer to (block_pointer + block_size - 1), ?), R);
		//GSMat.fromSMat(datatrans(?, block_pointer to (block_pointer + block_size - 1)), Rtrans);
	
		bu = M * Rtrans;
		bm = Ub * R;
		
		temp = DDS(M,Ub,Rtrans,temp);
		tmp2 = M*temp;
		tmp3 = lambdaub *@ Ub;
		Ax = tmp2 + tmp3;

		r = bu - Ax;
		p = bu - Ax;

		
		tempM = DDS(Ub,M,R,tempM);
		tmp2M = Ub*tempM;
		tmp3M = lambdam *@ M;
		AxM = tmp2M + tmp3M;

		rM = bm - AxM;
		pM = bm - AxM;
	}
	
	def trainBlock(){
		
		//build matrix by user
		//bu ~ M * Rtrans;
		conjGradU(1);
			
		//build matrix by movie
		//bm ~ U*R;
		conjGradM(1);


	}
	// modification: pass A, S, lambda as a fucntion
	def conjGradU( nConjIter:Int ) {

		//var rsquare = r*@r;
		rsold = sum(GMat(r*@r), 1);
		rsnew = sum(GMat(r*@r), 1);
		
		var i: Int = 1;
		for( i <- 1 to nConjIter){

			temp = DDS(M,p,Rtrans,temp);
			tmp2 = M*temp;
			tmp3 = lambdaub *@ p;
			Ax = tmp2+tmp3;
			
			alpha = rsold / max((p dot Ax), GMat(1e-10f));

			Ub = Ub + GMat(p*@alpha);

			r = r - (Ax*@alpha);
			rsnew = sum(GMat(r*@r), 1);
			p = r + (p*@(rsnew / max(rsold, GMat(1e-10f))));
			rsold = rsnew;
		}
		
	}
	
	// modification: pass A, S, lambda as a fucntion
	def conjGradM( nConjIter:Int ) = {
	
		//var rsquare = r*@r;
		rsoldM = (rM dot rM);
		rsnewM = (rM dot rM);
		
		var i: Int = 1;
		for( i <- 1 to nConjIter){
			tempM = DDS(Ub,pM,R,tempM);
			tmp2M = Ub*tempM;
			tmp3M = lambdam *@ pM;
			AxM = tmp2M+tmp3M;
			
			alphaM = rsoldM / max((pM dot AxM), GMat(1e-10f));

			M = M + GMat(pM*@alphaM);

			rM = rM - (AxM*@alphaM);
			rsnewM = (rM dot rM);
			pM = rM + (pM*@(rsnewM / max(rsoldM, GMat(1e-10f))));
			rsoldM = rsnewM;
		}
	}
		
}

object GALS{
	
	def main(args: Array[String]){
		
		Mat.useCache = true;

		var trFileName = "";
		var cvFileName = "";
		var lambda = 0.065f;
		var d = 10;
		var bSize = 1000;
		var nIter = 1;
		var nConjIter = 1;
		var fast = true;
		var test = false;
		var dataset = "smallnetflix";
		
		var Rmat:SMat = null;
		var Vmat:SMat = null;
		
		val arglen = args.length;
		
		for( i <- 0 to arglen-1){
			if(args(i) == "-trPath"){
				trFileName = args(i+1);
				println("training path: " + trFileName);
			}
			else if(args(i) == "-cvPath"){
				cvFileName = args(i+1);
				println("cross validation path: " + cvFileName);
			}
			else if(args(i) == "-d"){
				d = args(i+1).toInt;
			}
			else if(args(i) == "-nIter"){
				nIter = args(i+1).toInt;
			}
			else if(args(i) == "-lambda"){
				lambda = args(i+1).toFloat;
			}
			else if(args(i) == "-test"){
				test = true;
			}
			else if(args(i) == "-noDDS"){
				fast = false;
			}
			else if(args(i) == "-dataset"){
				dataset = args(i+1);
			}
			else if(args(i) == "-blocksize"){
				bSize = args(i+1).toInt;
			}
		}
		
		
		//val trFileName = args(0);
		//val cvFileName = args(1);
		//val Rmat = MatReader(trFileName, 16384);
		//val Vmat = MatReader(cvFileName, 16384);
		
		//saveAs("/home/hzhao/code/ALS/netflix/smallnetflix.mat", Rmat, "train", Vmat,"cv");
		
		if( dataset == "smallnetflix" ){
			Rmat = load("/home/hzhao/code/ALS/netflix/smallnetflix.mat", "train");
			Vmat = load("/home/hzhao/code/ALS/netflix/smallnetflix.mat", "cv");
		} else {
			Rmat = load("/home/hzhao/code/ALS/netflix/netflix.mat", "train");
			Vmat = load("/home/hzhao/code/ALS/netflix/smallnetflix.mat", "cv");
		}
		
		if(fast){
			var gALS1 = new GALS(Rmat, d, nIter, nConjIter, bSize);
			flip;
			gALS1.train();
			val ff=gflop;
			println("(Gflops, time): " + ff);
		}	
		
	}
}
