Steps to retrain the model:

Setup Environment
	- install miniconda and create a new conda environment
	- after activating this environment, run "pip install -r requirements.txt"
	- go through NotesFromKelton.txt and set up your system as instructed
		- Opt: Download and install Cuda if you have a GPU
	
Run Training
	- run "set_env_vars.bat"
		- modify .bat file depending on how you have your folders setup
	- run "run_train.bat"
		--pipeline_config_path --> pipeline.config file
		--model_dir --> output folder for model checkpoints
		--checkpoint_every_n --> will create checkpoint files after n steps
		--num_workers --> number of threads/cores training will run on
		--alsologtostderr --> output status to terminal log
	- run "run_build.bat"
		--input_type --> MUST be 'input_tensor'
		--pipeline_config_path --> pipeline.config file
		--trained_checkpoint_dir --> folder containing checkpoint files (should equal --model_dir)
		--output_directory --> output folder for new model