package core

import BIDMat._
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import java.util.Properties
import java.io.IOException
import java.io.FileInputStream


class SVMSolver(args: Array[String]) extends AbstractSolver(args) {
  
  	// state variables	
	var lambda: Double = 1.0;
	var stepSize: Double = 1.0;
	
	// algorithm parameters

	
	// data in memory
	var y: DMat = null;
	var X: SDMat = null;
	var D: DMat = null;
	var ty: DMat = null;
	var tX: SDMat = null;

	override def update(yBlock: DMat, XBlock: SDMat, i: Int){
	  
		beta = (1.0 - lambda*stepSize/java.lang.Math.sqrt(i.toDouble)) * beta + XBlock * yBlock/blockSize*stepSize/java.lang.Math.sqrt(i.toDouble);
		
		val betaNorm = sqrt( sum(beta *@ beta) );
		
		beta = min(1.0, 1.0/java.lang.Math.sqrt(lambda)/betaNorm) * beta;
	}
	
	override def loadConfig(){
		
		val properties = new Properties();
		
		try{
			properties.load(new FileInputStream(CONFIG_PATH));
			
			iterations = properties.getProperty("iterations").toInt;
			dimension = properties.getProperty("dimension").toInt;
			blockSize = properties.getProperty("block_size").toInt;
			lambda = properties.getProperty("lambda").toDouble;
			stepSize = properties.getProperty("step_size").toDouble;
			
			dataPath = properties.getProperty("data_path");
			comm = properties.getProperty("comm");
			dataSet = properties.getProperty("data_set");
			dataMaxId = properties.getProperty("data_max_id").toInt;
			testOn = properties.getProperty("testOn").toBoolean;
			wallClockBenchmark = properties.getProperty("wallClockBenchmark").toBoolean;
			
		}catch{
		  	case ioex: IOException => ioex.printStackTrace();
		  	case ex: Exception => ex.printStackTrace();
		}
	}
	
	override def loadData(dataSet: String){
		
		if(dataSet.equals("RCV1")){
			// incomplete: should convert DMat to DMat
			X = load(dataPath + myRank + ".mat", "X");
			y = load(dataPath + myRank + ".mat", "y");
			D = load(dataPath + "test.mat", "D");
			tX = load(dataPath + "test.mat", "tX");
			ty = load(dataPath + "test.mat", "ty");
			
		}else if(dataSet.equals("twitter")){

			X = load(dataPath + "%05d".format(myRank) + ".mat", "X");
			y = load(dataPath + "%05d".format(myRank) + ".mat", "y");
			y = (y - 0.5) * 2.0;
			D = ones(dimension, 1);
			tX = load(dataPath + "00155.mat", "X");
			tX = (tX t);
			ty = load(dataPath + "00155.mat", "y");
			ty = (ty - 0.5) * 2.0;
		  
		}
	}
		
	def reLoadData(dataSet: String){
		if(dataSet.equals("RCV1")){
			dataSeqId += 1;
			var id = (dataSeqId * worldSize + myRank) % dataMaxId;
			X = load(dataPath + "%d".format(id) + ".mat", "X");
			y = load(dataPath + "%d".format(id) + ".mat", "y");
		}else if(dataSet.equals("twitter")){
			dataSeqId += 1;
			var id = (dataSeqId * worldSize + myRank) % dataMaxId;
			X = load(dataPath + "%05d".format(id) + ".mat", "X");
			y = load(dataPath + "%05d".format(id) + ".mat", "y");
			y = (y - 0.5) * 2.0;
		  
		}
	}
	
	override def getXBlock(pos: Int): SDMat = {
		
		val end = pos + blockSize - 1;
		val dim = size(X)._2;
		
		if( end < dim ){
			return X(?, pos to end);
		}else{			
			var tempX: SDMat = X(?, pos to (dim-1)) \ X(?, 0 to (end-dim));
			return tempX;
		}
	}
	
	override def getyBlock(pos: Int): DMat = {
		
		val end = pos + blockSize - 1;
		val dim = size(X)._2;
		
		if( end < dim ){
			return y(pos to end, ?);
		}else{
			var tempy: DMat = y(pos to (dim-1), ?) on y( 0 to (end-dim), ?);
			reLoadData(dataSet); 
			println(myRank + ": " +"Reloading data: " + dataSeqId);
			return tempy;
		} 
	}
	
	override def getNextPos(pos: Int): Int = if((pos+blockSize) < size(X)._2) (pos+blockSize) else 0
	
	override def loss(beta: DMat): DMat = {

		val s = (ty) *@ (tX*beta);  
		sum(max(0.0, 1.0 - s));
	}

}

object SVMSolver{
  	
	def main(args: Array[String]){
		
		val sVMSolver: SVMSolver = new SVMSolver(args);
		sVMSolver.init(args);
		sVMSolver.run();
		sVMSolver.terminate();
		
	}
}