# MyEncoder

Main program file is program.py. It takes in one argument which is the input file to read in from.

Input fileformat is as follows

typeofoperation,outputfolder,encoder_layers,dnn_layers,split(for validation),encoder_epochs,encoder_batchsize,enc_learning_rate,dnn_epochs,dnn_batchsize,dnn_learningrate,memsize,easy(for using easy nsldataset),attack(train encoder on attack samples),numberofclasses,max_num(for sampling),shrink_threshold(for shrinkage operations),number_of_iterations


constants.py has the functions to retrive dictionaries of the filepaths to each dataset file. 

future_attack.py is to make an autoencoder excluding a specific class

mem_unit.py contains the definition of the memory unit. 

memory_mapping.py takes in command line arguments of type: dataset, list of folders of results. Will go through each folder, map memory, plot it, map contents of memory and create spread file.

process_data.py just reads in results, averages them and writes them out to folder.

transplant.py uses already made autoencoders and stitches them together to make final autoencoder and train model doing so. 

