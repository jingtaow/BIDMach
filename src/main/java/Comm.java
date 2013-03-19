import mpi.*;
 
class Comm {

	public int rank;
	public int size;

	public Comm( String[] args ) throws MPIException{
		
		MPI.Init(args);
		rank = MPI.COMM_WORLD.Rank();
		size = MPI.COMM_WORLD.Size();

	}
	
	public void DenseAllReduce( float[] model, int[] partition_offsets, int[] partition_sizes ) throws MPIException{
		
		int right = 0;
		int left = 0;
		float[] buffer = new float[model.length];
		for( int i = 0; i<model.length; i++){
			buffer[i] = 0;
		}
		
		
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
	
	public void terminate() throws MPIException{
		MPI.Finalize();
	}
	
}
