package sparsecomm;

import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedList;

import mpi.*;

public class SparseComm {

	public int rank;
	public int size;
	
	
	//private LinkedList<Long> [] gatherDest;
	private LinkedList<Integer> [] gatherDest;
	//private LinkedList<Long> [] scatterDest;
	private LinkedList<Integer> [] scatterOrigin;

	private int byteCount = 0;
	private long nanoTime = 0;
	private long sTime = 0;
	private long eTime = 0;
	
	public SparseComm( String[] args ) throws MPIException{
		
		MPI.Init(args);
		rank = MPI.COMM_WORLD.Rank();
		size = MPI.COMM_WORLD.Size();
		
		//gatherDest = (LinkedList<Long>[]) new LinkedList[size];
		gatherDest = (LinkedList<Integer>[]) new LinkedList[size];
		//scatterDest = (LinkedList<Long>[]) new LinkedList[size];
		scatterOrigin = (LinkedList<Integer>[]) new LinkedList[size];

	}
	
	public void DenseAllReduce( float[] model, int[] partition_offsets, int[] partition_sizes ) throws MPIException{
		
		int right = 0;
		int left = 0;
		float[] buffer = new float[model.length];
		for( int i = 0; i<model.length; i++){
			buffer[i] = 0;
		}
		
		// reduce scatter
		for( int i = 1; i < size; i++ ){
			right = (rank + i) % size;
			left = rank - i;
			if( left < 0 ){
				left = left + size;
			}
			
			MPI.COMM_WORLD.Sendrecv(model, partition_offsets[right], partition_sizes[right], MPI.FLOAT, right, 0, buffer, partition_offsets[rank], partition_sizes[rank], MPI.FLOAT, left, 0);
			
			for( int j = 0; j < partition_sizes[rank]; j++ ){
				int k = partition_offsets[rank] + j;
				model[k] = model[k] + buffer[k];
			}
			
		}
		
		for( int j = 0; j < partition_sizes[rank]; j++ ){
			int k = partition_offsets[rank] + j;
			model[k] /= size;
		}
		
		// allgather
		for( int i = 1; i < size; i++ ){
			right = (rank + i) % size;
			left = rank - i;
			if( left < 0 ){
				left = left + size;
			}
			
			MPI.COMM_WORLD.Sendrecv(model, partition_offsets[rank], partition_sizes[rank], MPI.FLOAT, right, 0, buffer, partition_offsets[left], partition_sizes[left], MPI.FLOAT, left, 0);
			
			for( int j = 0; j < partition_sizes[left]; j++ ){
				int k = partition_offsets[left] + j;
				model[k] = buffer[k];
			}
		}
	
	}
	
	// sendbuf inbound vertex indices sorted by destination
	public void gatherConfig(int [] sendbuf, int [] sendcounts, int [] displs) throws MPIException{
		
		int right = 0;
		int left = 0;
		int [] buffer;
		int [] bufcounts = new int[size];
		
		for(int i = 0; i < size; i++){
			
			right = (rank + i) % size;
			left = rank - i;
			if( left < 0 ){
				left += size;
			}
			MPI.COMM_WORLD.Sendrecv(sendcounts, right, 1, MPI.INT, right, 0, bufcounts, left, 1, MPI.INT, left, 0);
		}
		
		//System.out.println(String.format("%d: sendcounts %s, bufcounts %s\n", rank, Arrays.toString(sendcounts), Arrays.toString(bufcounts)));
		
		for(int i = 0; i < size; i++){
						
			right = (rank + i) % size;
			left = rank - i;
			if( left < 0 ){
				left += size;
			}
			
			buffer = new int[bufcounts[left]];
			
			MPI.COMM_WORLD.Sendrecv(sendbuf, displs[right], sendcounts[right], MPI.INT, right, 0, buffer, 0, bufcounts[left], MPI.INT, left, 0);
			
			//gatherDest[right] = new LinkedList<Long>(Arrays.asList(sendbuf));
			gatherDest[left] = new LinkedList<Integer>();
			
			for(int j = 0; j<bufcounts[left]; j++){
				gatherDest[left].add(buffer[j]);
			}
		}
		
		
	}
	
	// for both gather and scatter try not to sendrecv to myself; handle i = 0 separately.
	public void gather(float [] sendbuf, float [] recvbuf, int [] recvcounts, int dim_per_proc) throws MPIException{
		
		int right = 0;
		int left = 0;
		float [] sendbuffer;
		float [] recvbuffer;
		int recvpointer = 0;
		
		//left = rank - 1;
		//if( left < 0 ){
		//	left += size;
		//}
		for(int i = 0; i<rank; i++){
			recvpointer += recvcounts[i];
		}
		
		for(int i = 0; i < size; i++){
			
			right = (rank + i) % size;
			left = rank - i;
			if( left < 0 ){
				left += size;
			}			
			
			int sendcount = gatherDest[left].size();
			sendbuffer = new float[sendcount];
			
			Iterator<Integer> itr = gatherDest[left].iterator();
			int j = 0;
			while(itr.hasNext()){
				int next = itr.next();
				int k = (next % dim_per_proc);
				sendbuffer[j] = sendbuf[k];
				j++;
			}
			
			recvbuffer = new float[recvcounts[right]];
			
			sTime = System.nanoTime();

			MPI.COMM_WORLD.Sendrecv(sendbuffer, 0, sendcount, MPI.FLOAT, left, 0, recvbuffer, 0, recvcounts[right], MPI.FLOAT, right, 0);
	
			eTime = System.nanoTime();

			byteCount += sendcount*4;
			byteCount += recvcounts[right]*4;

			nanoTime += (eTime - sTime);
			
			for( j = 0; j < recvcounts[right]; j++){
				recvbuf[recvpointer % recvbuf.length] = recvbuffer[j];
				recvpointer++;
			}
		}
	}
	
	public void scatterConfig(int [] sendbuf, int [] sendcounts, int [] displs) throws MPIException{
		
		int right = 0;
		int left = 0;
		int [] buffer;
		int [] bufcounts = new int[size];
		
		for(int i = 0; i < size; i++){
			
			right = (rank + i) % size;
			left = rank - i;
			if( left < 0 ){
				left += size;
			}
			MPI.COMM_WORLD.Sendrecv(sendcounts, right, 1, MPI.INT, right, 0, bufcounts, left, 1, MPI.INT, left, 0);
		}
		
		for(int i = 0; i < size; i++){
						
			right = (rank + i) % size;
			left = rank - i;
			if( left < 0 ){
				left += size;
			}
			
			buffer = new int[bufcounts[left]];

			MPI.COMM_WORLD.Sendrecv(sendbuf, displs[right], sendcounts[right], MPI.INT, right, 0, buffer, 0, bufcounts[left], MPI.INT, left, 0);
			
			//gatherDest[right] = new LinkedList<Long>(Arrays.asList(sendbuf));
			scatterOrigin[left] = new LinkedList<Integer>();
			for(int j = 0; j<bufcounts[left]; j++){
				scatterOrigin[left].add(buffer[j]);
			}
		}
		
	
	}
	

	public void scatter(float [] sendbuf, int [] sendcounts, int [] displs, float [] recvbuf,  int dim_per_proc) throws MPIException{
		int right = 0;
		int left = 0;
		float [] buffer;
		
		for(int i = 0; i < size; i++){
			
			right = (rank + i) % size;
			left = rank - i;
			if( left < 0 ){
				left += size;
			}
			
			int bufcount = scatterOrigin[left].size();
			buffer = new float[bufcount];

			sTime = System.nanoTime();
			MPI.COMM_WORLD.Sendrecv(sendbuf, displs[right], sendcounts[right], MPI.FLOAT, right, 0, buffer, 0, bufcount, MPI.FLOAT, left, 0);
			
			eTime = System.nanoTime();

			byteCount += sendcounts[right]*4;
			byteCount += bufcount*4;
			
			nanoTime += (eTime - sTime);

			Iterator<Integer> itr = scatterOrigin[left].iterator();
			int j = 0;
			while(itr.hasNext()){
				int next = itr.next();
				int k = (next % dim_per_proc);
				recvbuf[k] += buffer[j];
				j++;
			}
		}
	}
	
	public void printConfig(){
		
		System.out.println(String.format("Processor %d gatherDest:\n", rank));
			for(int i = 0; i<size; i++){
				if(i!=rank){
					System.out.println( String.format("%d: %s", i, gatherDest[i].toString()) );
			}
		}
		
		System.out.println(String.format("Processor %d scatterOrigin:\n", rank));
		for(int i = 0; i<size; i++){
			if(i!=rank){
				System.out.println( String.format("%d: %s\n", i, scatterOrigin[i].toString()));
			}
		}
		
	}

	public int getByteCount(){
	
		return byteCount;

	}
	
	public float getThroughput(){
		return (float)byteCount/nanoTime;
	}
	
	public void terminate() throws MPIException{
		MPI.Finalize();
	}
	
}
