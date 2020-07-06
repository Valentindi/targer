for filename in *.hdf5; do
	echo "$filename"

	    python run_tagger.py "$filename" data/targer/targer_test.txt --gpu -2 --evaluator f1-macro --output "$filename.json"
	done