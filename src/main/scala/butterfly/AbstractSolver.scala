package core

import mpi._
import BIDMat._
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import java.lang.Math.pow
import java.lang.Math.log
import java.io._
import java.util.logging._


abstract class AbstractSolver(args: Array[String]) {

	// configuration path
	var CONFIG_PATH = "config/twitter.config";
	
	// data path
	var dataPath: String = null;
	
	// data set
	var dataSet: String = "RCV1";
	var dataSeqId: Int = 0;
	var dataMaxId: Int = 1;
	
	// mpi state variables
	var myRank: Int = 0;
	var worldSize: Int = 0;
	var dimension: Int = 47236;
	var iterations: Int = 100;
	var blockSize: Int = 1;
		
	// communication method
	var comm = "Butterfly";
	
	// communication buffer
	var buffer:Array[Double] = Array.fill[Double](dimension)(0.0);
	
	// algorithm parameter
	var beta:DMat = zeros(dimension,1);
	var betaBenchmark:DMat = zeros(dimension, 1);
	
	// test loss on/off
	var testOn = true;
	
	// wall clock time benchmark on/off
	var wallClockBenchmark = true;
	
	// logger
	//val logger: Logger = Logger.getLogger("Solver");
	var timeLogWriter: FileWriter = null;
	var lossLogWriter: FileWriter = null;
	
	def init(args: Array[String]){
		
		// mpi initialization
		val input = MPI.Init( args );
		myRank = MPI.COMM_WORLD.Rank();
		worldSize = MPI.COMM_WORLD.Size();
		
		// load config file path
		CONFIG_PATH = input(0);
		
		// algorithm configuration
		loadConfig();
		
		if(input.size > 1){
			comm = input(1);
		}
		// data initialization
		loadData(dataSet);
		
		// initialize logger
		val fileName = "%s-%s-k%d-b%d-%s.%d.txt".format("benchmark/time", dataSet, worldSize, blockSize, comm.stripLineEnd, myRank);
		//val h: FileHandler = new FileHandler(fileName);
		//h.setFormatter(new SimpleFormatter());
		//logger.addHandler(h);
		timeLogWriter = new FileWriter(fileName);
		timeLogWriter.write("This is node %d\n".format(myRank));
		
		if(myRank == 0){
			val fileName2 = "%s-%s-k%d-b%d-%s.%d.txt".format("benchmark/loss", dataSet, worldSize, blockSize, comm.stripLineEnd, myRank);
			lossLogWriter = new FileWriter(fileName2);
			lossLogWriter.write("This is node %d\n".format(myRank));
		}
		
		//init algorithm parameter
		buffer = Array.fill[Double](dimension)(0.0);
		beta = zeros(dimension, 1);
		betaBenchmark = zeros(dimension, 1);
		
		System.out.println("MPI initialized. World Size: " + worldSize);
	}
	
	
	def loadConfig();
	
	def loadData(dataSet: String);
	
	def getXBlock(pos: Int): SDMat;
	
	def getyBlock(pos: Int): DMat;
	
	def getNextPos(pos: Int): Int;
	
	def update(yBlock: DMat, XBlock: SDMat, i: Int); 
	
	def reduce(comm: String, stage: Int){
		
		if(comm.equals("AllReduce")){
		
			MPI.COMM_WORLD.Allreduce(beta.data, 0, buffer, 0, dimension, MPI.DOUBLE, MPI.SUM);	
			
			beta = DMat( dimension, 1, buffer.clone()) / worldSize;
			
		}else if(comm.equals("Butterfly")){
			
			val pos = myRank % pow(2, stage).toInt;
			val base = myRank - pos;
			val dst = base + ((pos + pow(2, stage-1)).toInt % pow(2, stage).toInt);
			
			MPI.COMM_WORLD.Sendrecv(beta.data, 0, dimension, MPI.DOUBLE, dst, 0, 
               buffer, 0, dimension, MPI.DOUBLE, dst, 0); 

			beta = (DMat( dimension, 1, buffer.clone()) + beta) / 2;	
		  
		}else if(comm.equals("Periodic")){
			if(stage == (log(worldSize)/log(2)).toInt - 1){
			  	MPI.COMM_WORLD.Allreduce(beta.data, 0, buffer, 0, dimension, MPI.DOUBLE, MPI.SUM);			
			  	beta = DMat( dimension, 1, buffer.clone()) / worldSize;
			}
		}
		
		
		if(testOn){
			MPI.COMM_WORLD.Allreduce(beta.data, 0, buffer, 0, dimension, MPI.DOUBLE, MPI.SUM);	
			betaBenchmark = DMat( dimension, 1, buffer.clone()) / worldSize;
			if(myRank == 0){
				println(loss(betaBenchmark) + "\n");
				lossLogWriter.write(loss(betaBenchmark) + "\n");
			}
		}

	}
	
	def loss(beta: DMat): DMat;
		
	def run(){
		
		var pos = 0;
		
		for(i <- 1 to iterations){
			
			MPI.COMM_WORLD.Barrier();
			
			// Getting X before y, because getyBlock reloads new data at boundary
			
			val newpos = getNextPos(pos);
			
			var XBlock = getXBlock(pos);
			var yBlock = getyBlock(pos);
			
			pos = newpos;
			
			val t0 = System.nanoTime();
			update(yBlock, XBlock, i);
			
			val t1 = System.nanoTime();
			if(worldSize > 1){
				reduce(comm, i % (log(worldSize)/log(2)).toInt);
			}
			
			val t2 = System.nanoTime();
			
			//logger.info((t1-t0)/1000.0 + "\n");
			if(wallClockBenchmark){
				timeLogWriter.write("%f, %f, %f, %f\n".format((t1-t0)/1000000.0, (t2-t1)/1000000.0, t0/1000000.0, t2/1000000.0));
			}
			
		}
				
	}
	
	def terminate(){
		MPI.Finalize();
		timeLogWriter.close();
		
		if(myRank == 0){
		  lossLogWriter.close();
		}
	}
}