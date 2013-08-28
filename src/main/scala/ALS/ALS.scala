import BIDMat.{Mat, FMat, DMat, IMat, CMat, BMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat, HMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import java.io.BufferedReader
import java.io.FileReader
import java.io.IOException

object ALS{

	def MatReader(fileName: String, bufferSize: Int): SMat = {

		var r: IMat = null;
		var c: IMat = null;
		var y: FMat = null;
		
		// init buffer
		var row = izeros(bufferSize, 1);
		var col = izeros(bufferSize, 1);
		var yy = zeros(bufferSize, 1);
		
		try{
			val br = new BufferedReader(new FileReader(fileName));
			var line = br.readLine()
			var i: Int = 0;
			while(line != null){
				var tokens = line.split("\\s+");
				line = br.readLine();
				row(i) = tokens(0).toInt - 1;
				col(i) = tokens(1).toInt - 1;
				yy(i) = tokens(2).toFloat;

				i = i+1;
				
				if(i>=bufferSize || line == null){
				
					if(r == null){
						r = row(0 to i-1,?);
						c = col(0 to i-1,?);
						y = yy(0 to i-1,?);
					}
					else{
						r = r on row(0 to i-1,?);
						c = c on col(0 to i-1,?);
						y = y on yy(0 to i-1,?);
					}
					i = 0;

				}

			}
			br.close();
		}catch{
			case e: IOException => e.printStackTrace;
			case _ => println("Some sort of exception happened");
		}

		val mat = sparse(r, c, y);
		mat
		
	}
	
	def ALS(R: SMat, V: SMat, d: Int, lambda: Float, nIter: Int, test: Boolean): (FMat, FMat, FMat) = {
	
		val (nu, nm) = size(R);
		var U = zeros(d, nu);
		var M = 0.5*rand(d, nm);
		M(1, ?) = FMat(mean(R));
		var loss = zeros(nIter, 1);		
		val Rtrans = R.t;
		
		var i = 0;

		for( i <- 0 to nIter-1){
			val E = FMat(ones(d, 1).mkdiag);
			var lamI = lambda * E;
			
			var bu = zeros(d, nu);
			
			for( u <- 0 to nu-1 ){
				//if(u%10000 == 0)
					//println("processing the %dth user.".format(u));
				//var movies = find(Rtrans( 0 to nm-1, IMat(u)));
				var movies = find(Rtrans( ?, IMat(u)));
				if(length(movies) > 0){
					var Mu = M(?, movies);
					bu(?, IMat(u)) = Mu * Rtrans(movies, IMat(u));
					var vector = Mu * Rtrans(movies, IMat(u));
					//println("size movies: %d, %d; size Mu: %d, %d; size vector: %d, %d".format(size(movies)._1, size(movies)._2, size(Mu)._1, size(Mu)._2, size(vector)._1, size(vector)._2));
					var matrix = (Mu xT Mu) + length(movies)*lamI;					
					var Xu = matrix \\ vector;
					U(?, u) = Xu;
					
					
				}
			}
			//var XX = conjGrad(M, Rtrans, bu, zeros(d, nu), 5, lambda);
			//println(U-XX);
			
			for(m <- 0 to nm-1){
				//if(m%1000 == 0)
					//println("processing the %dth movie.".format(m));
				//var users = find(R(0 to nu-1, IMat(m)));
				var users = find(R(?, IMat(m)));
				if(length(users) > 0){
					var Um = U(?, users);
					var vector = Um * R(users, IMat(m));
					//println("size movies: %d, %d; size Mu: %d, %d; size vector: %d, %d".format(size(movies)._1, size(movies)._2, size(Mu)._1, size(Mu)._2, size(vector)._1, size(vector)._2));
					var matrix = (Um xT Um) + length(users)*lamI;
					var Xm = matrix \\ vector;
					M(?, m) = Xm;
					
				}
			}

			if(test){
				val (r, c, v) = find3(V);
				for(k <- 0 to (length(r)-1)){
				//	//println("loss loss pair: " + (U(?, r(k)).t * M(?, c(k))) +  v(k));
					var v_pred = U(?, r(k)).t * M(?, c(k));
					loss(i) = loss(i) + ( v_pred - v(k))*( v_pred - v(k));
				}
				//var pred = blank;
				//pred = DDS(U(?, 0 to 95312), M, V);
				//loss(i) = sum(sum(SMat(V-pred)));
				//loss(i) = sqrt(loss(i)/length(find(V)));
				loss(i) = sqrt(loss(i)/length(r));
				println("loss at iteration %d: %f".format(i, loss(i)));
			}
		}
		
		(U, M, loss)
		
	}
	
	// modification: pass A, S, lambda as a fucntion
	def conjGrad( A:FMat, S:SMat, b:FMat, X:FMat, nConjIter:Int, lambdaX: FMat): FMat = {
	
		var Ax = (lambdaX *@ X) + A*DDS(A, X, S);
		//var Ax = lambda*X + A*DDS(A, X, S);
		var r = b - Ax;
		var p = r;
		var XX = X;
			
		var rsold = sum(FMat(r *@ r));
		var rsnew = zeros(1, length(rsold));
			
		for(i <- 1 to nConjIter){
			var Ap = (lambdaX *@ p) + A*DDS(A, p, S);
			//var Ap = (lambda * p) + A*DDS(A, p, S);
			
			var alpha = rsold /@ max(sum(FMat(p *@ Ap)), 1e-10f);

			XX = XX + FMat(p*@alpha);
			//println(XX);
			r = r - (Ap*@alpha);
			rsnew = sum(FMat(r *@ r));
			p = r + (p*@(rsnew /@ max(rsold, 1e-10f)));
			rsold = rsnew;

			//println(rsold);
		}
		
		//var Axx = (lambdaX *@ XX) + A*DDS(A, XX, S);
		//var rx = b - Axx;
		//println(rx);
		
		XX
	}
	
	def fastALS(R: SMat, V: SMat, d: Int, lambda: Float, nIter: Int, nConjIter: Int, test: Boolean): (FMat, FMat, FMat) = {
		val (nu, nm) = size(R);
		var U = zeros(d, nu);
		var M = 0.5*rand(d, nm);
		M(1, ?) = FMat(mean(R));
		var loss = zeros(nIter, 1);	
		val Rtrans = R.t;
		
		for( i <- 0 to nIter-1){
			//build matrix buser

			var bu = M * Rtrans;
			//println("dense sparse multiplication.");
			var lambdau = sum(Rtrans, 1)*lambda;
			//var lambdau = sum((Rtrans>0f), 1)*lambda;
			for( u <- 0 to nu-1 ){
				//if(u%10000 == 0)
				//	println("processing the %dth user.".format(u));
				var movies = find(Rtrans( ?, IMat(u)));
				lambdau(u) = length(movies)*lambda;
				//if(length(movies) > 0){
				//	var Mu = M(?, movies);
				//	bu(?, IMat(u)) = Mu * Rtrans(movies, IMat(u));
				//}
			}

			//println("assigning lambda.");

			if( i == 0){
				U = conjGrad(M, Rtrans, bu, U, 5, lambdau);
			}else{
				U = conjGrad(M, Rtrans, bu, U, nConjIter, lambdau);
			}

			//println("conjugate gradient.");
			var bm = U*R;
			var lambdam = sum(R, 1)*lambda;
			//var lambdam = sum((R>0f), 1)*lambda;
			for(m <- 0 to nm-1){
				//if(m%1000 == 0)
				//	println("processing the %dth movie.".format(m));
				//var users = find(R(0 to nu-1, IMat(m)));
				var users = find(R(?, IMat(m)));
				lambdam(m) = length(users)*lambda;
				//if(length(users) > 0){
				//	var Um = U(?, users);
				//	bm(?, IMat(m)) = Um * R(users, IMat(m));
				//}
			}
			
			if( i ==0 ){
				M = conjGrad(U, R, bm, M, 5, lambdam);
			}else{
				M = conjGrad(U, R, bm, M, 1, lambdam);
			}

		//var M1 = M + U*DDS(U, M, R);
		
			if(test){
				val (r, c, v) = find3(V);

				for(k <- 0 to (length(r)-1)){
					var v_pred = U(?, r(k)).t * M(?, c(k));
					loss(i) = loss(i) + (v_pred - v(k))*(v_pred - v(k));
				}
				loss(i) = sqrt(loss(i)/length(r));
				println("loss at iteration %d: %f".format(i, loss(i)));
			}
		}
		(U, M, loss)
	}
	def main(args: Array[String]){
		
		var trFileName = "";
		var cvFileName = "";
		var lambda = 0.065f;
		var d = 20;
		var nIter = 10;
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
			else if(args(i) == "-nConjIter"){
				nConjIter = args(i+1).toInt;
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
		}
		
		
		//val trFileName = args(0);
		//val cvFileName = args(1);
		//Rmat = MatReader("/home/hzhao/code/ALS/smallnetflix/smallnetflix_mm.train", 16384);
		//Vmat = MatReader("/home/hzhao/code/ALS/smallnetflix/smallnetflix_mm.validate", 16384);
		
		//saveAs("/home/hzhao/code/ALS/netflix/smallnetflix.mat", Rmat, "train", Vmat,"cv");
		
		if( dataset == "smallnetflix" ){
			Rmat = load("/home/hzhao/code/ALS/netflix/smallnetflix.mat", "train");
			Vmat = load("/home/hzhao/code/ALS/netflix/smallnetflix.mat", "cv");
		} else {
			Rmat = load("/home/hzhao/code/ALS/netflix/netflix.mat", "train");
			Vmat = load("/home/hzhao/code/ALS/netflix/smallnetflix.mat", "cv");
		}
		
		//val lambda = 0.065f;
		//var d = 500;
		//val nIter = 10;
		//val nConjIter = 1;
		//val test = false;
		//ALS(Rmat, Vmat, d, lambda, nIter, test);
		flip;
		if(fast)
			fastALS(Rmat, Vmat, d, lambda, nIter, nConjIter, test);
		else
			ALS(Rmat, Vmat, d, lambda, nIter, test);
		val ff=gflop;
		println("(Gflops, time): " + ff);
	}
}
