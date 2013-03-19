import mpi.*;
 
class Model {

	public float[] model;
	// dimension of model
	public int dim;

	public Comm comm;
	// number of processors
	public int size;
	
	public int[] partition_offsets;
	public int[] partition_sizes;
	
	// command line arguments
	public String[] argv;
	
	public Model(String[] args){
		argv = args;
		dim = Integer.parseInt(argv[0]);
	}
	
	public void initModel() throws MPIException{
		
		model = new float[dim];
		
		comm = new Comm(argv);
		size = comm.size;
		
		// dummy model for now
		for( int i = 0; i<dim; i++ ){
			model[i] = i*comm.rank;
		}
		
		// partition
		partition();

	}
	
	public void partition(){
			
		int dim_per_proc = (dim + size -1)/size;
		partition_offsets = new int[size+1];
		partition_sizes = new int[size];

		for( int i = 0; i < size + 1; i++ ){
			partition_offsets[i] = Math.min(i*dim_per_proc, dim);
		}
		
		for( int i = 0; i < size; i++ ){
			partition_sizes[i] = partition_offsets[i+1] - partition_offsets[i];
		}
	
	}
	
	public void reduce() throws MPIException{
		comm.DenseAllReduce(model, partition_offsets, partition_sizes);
	}
	
	// update the model
	public void update(){
	
	}

	public void run() throws MPIException{
	
		update();
		System.out.println(String.format("Rank: %d, Model: %f, %f, %f, %f\n", comm.rank, model[0], model[1], model[2], model[3]));
		reduce();
		System.out.println(String.format("Rank: %d, Model: %f, %f, %f, %f\n", comm.rank, model[0], model[1], model[2], model[3]));
	
	}
	public void terminate() throws MPIException{
		comm.terminate();
	}
	
	
    static public void main(String[] args) throws MPIException{
		
		Model m = new Model(args);
		
		m.initModel();
		m.run();
		m.terminate();
      
    }
}
