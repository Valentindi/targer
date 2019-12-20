export PYTHONIOENCODING=utf8

train="../wstud-thesis-dittmar/data/targer/targer_train.txt"
dev="../wstud-thesis-dittmar/data/targer/targer_dev.txt"
test="../wstud-thesis-dittmar/data/targer/targer_test.txt"
targetdir="../wstud-thesis-dittmar/targer-results"

e=250




cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="run-f1-alpha-match-10"
eval="f1-alpha-match-10"
python3 main.py --train $train --dev $dev --test $test --report-fn $name.txt --save $name.hdf5 --gpu -1 --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e
python3 run_tagger.py $name.hdf5 $test -g -1 -o $name.json
python3 eval.py $test $name.json $name.eval.txt
python3 generate_detailed_report.py $test $name.json $name.report.html

mv $name* targetdir
$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer ($name)")
$(cd $targetdir && git push)



cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="run-f1-alpha-match-05"
eval="f1-alpha-match-95"
python3 main.py --train $train --dev $dev --test $test --report-fn $name.txt --save $name.hdf5 --gpu -1 --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e
python3 run_tagger.py $name.hdf5 $test -g -1 -o $name.json
python3 eval.py $test $name.json $name.eval.txt
python3 generate_detailed_report.py $test $name.json $name.report.html

mv $name* targetdir

$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer ($name)")
$(cd $targetdir && git push)



cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="run-f1-macro"
eval="f1-macro"
python3 main.py --train $train --dev $dev --test $test --report-fn $name.txt --save $name.hdf5 --gpu -1 --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e
python3 run_tagger.py $name.hdf5 $test -g -1 -o $name.json
python3 eval.py $test $name.json $name.eval.txt
python3 generate_detailed_report.py $test $name.json $name.report.html

mv $name* targetdir
$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer ($name)")
$(cd $targetdir && git push)



cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="run-token-acc"
eval="token-acc"
python3 main.py --train $train --dev $dev --test $test --report-fn $name.txt --save $name.hdf5 --gpu -1 --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e
python3 run_tagger.py $name.hdf5 $test -g -1 -o $name.json
python3 eval.py $test $name.json $name.eval.txt
python3 generate_detailed_report.py $test $name.json $name.report.html

mv $name* targetdir
$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer ($name)")
$(cd $targetdir && git push)

