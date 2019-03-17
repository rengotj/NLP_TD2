The whole program contains several steps :
	* Data processing -> prepare_data.py
	* PCFG processing -> extract_pcfg.py
	* Out Of Vocabulary (OOV) words handling -> OOV.py
	* Evaluation of the result -> evaluate.py
The main file (main.py) gathers this steps and use the CYK parser class of CYK_parser_class.py and the results of spelling_error_proba.py.

The programs can easily be run with run.sh that has 9 arguments :
	* The first one is a boolean. If it is True, prepare_data.py is used. From "sequoia-corpus+fct.mrg_strict", it creates several files : "sequoia_dev.tb", "sequoia_dev.txt", "sequoia_test.tb", "sequoia_test.txt", "sequoia_test_tree.txt", "sequoia_train.tb", "sequoia_test.txt". If these files already exist (otherwise True is needed), we can skip this step by giving the boolean the value False.
	* The second one is a boolean. If it is True, extract_pcfg.py is used. It creates several files : "NT-set.pkl", "PCFB_binary_dict.pkl", "PCFB_binary_freq.pkl", "PCFB_postags_dict.pkl", "PCFB_postags_freq.pkl", "PCFB_unary_dict.pkl", "PCFB_unary_freq.pkl", "postags_set.pkl", "T_set.pkl", "words_set.pkl". If these files already exist (otherwise True is needed), we can skip this step by giving the boolean the value False.
	* The third one is a boolean. It it is True, OOV.py is used. It creates one file : "sequoia_test_corrected.txt". If this file already exists (otherwise True is needed), we can skip this step by giving the boolean the value False.
	* The fourth one is a boolean. If it is True, the Damereau extension (swap between adjacent characters) is used in Levenshtein distance computation
	* The fifth one is a boolean. If it is True, the unigram model is used to compute language model in Levenshtein distance.
	* The sixth one is a boolean. If it is True, the mle estimation is used to compute a language model with context information in Levenshtein distance.
	* The seventh is a booean. If it is True, evaluate.py is used and returns the precision, recall, F-score and tag accuracy of the obtained tree stored in "output.txt". It creates the file "evaluation_data.parser_output"
	* The eighth is a string. It receives the current file path
	* The last one is a string. It receives the python file path


All the created files are available in "Created files during the running" folder. The results (corrected test text, generated tree and evaluation output) are available in the folder "result" for different experiments.
