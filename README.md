Necessary Installations and versions
Python 3.7
Tensorflow 2.1.0
Opencv-Python
Version matters for TensorFlow and python but not OpenCV

To Run:
Download file
Open mainfile.py
Run program and a window should open using your default webcam for image recognition

I got this working on two machines running spyder through anaconda so I would recommend using the same setup to avoid issues.


HDH - Additional installation help

1. Head to the miniconda website to get a small and lightweight version of anaconda matching your current operating system
'''https://docs.conda.io/en/latest/miniconda.html'''

2. Create a new environment using this script, where myenv is whatever you would like to call it
'''$ conda create -n [myenv] python==3.7'''

3. Activate the environment by running this script, '''$ conda activate [myenv]''' with the name matching the one you entered in step 2, you should see the name of your environment in the command line '''[myenv]'''

4. Either navigate to or open a new command line window at the root level of this repository (if you open a new window you may have repeat step 3)

5. Install the required packages using '''pip install -r "requirements.txt"

6. Finally run '''python mainfile.py''' to start the object recognition using your webcam (it takes a little while to load, sometimes, leave the command line alone, make sure webcam is detected before running mainfile.py)

7. When you are done, enter ctrl+c in the command line you used to start the program, it will close the window and stop the program
