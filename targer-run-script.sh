export PYTHONIOENCODING=utf8

train="../wstud-thesis-dittmar/data/targer/targer_train.txt"
dev="../wstud-thesis-dittmar/data/targer/targer_dev.txt"
test="../wstud-thesis-dittmar/data/targer/targer_test.txt"
targetdir="../wstud-thesis-dittmar/targer-results"

e=1000

git checkout embeddings/glove.6B.100d.txt

# First run: Test for evaluator f1-connl,f1-alpha-match-10,f1-alpha-match-05,f1-macro,token-acc

python3 main.py --train $train --dev $dev --test $test --save ../targer-save.sav --gpu -1 --save-best yes --cross-folds-num 5 -v f1-connl -p 25 -e $e

mv 2019* $targetdir
$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer (f1-conn)")
$(cd $targetdir && git push)
git checkout embeddings/glove.6B.100d.txt

python3 main.py --train $train --dev $dev --test $test --save ../targer-save.sav --gpu -1 --save-best yes --cross-folds-num 5 -v f1-alpha-match-10 -p 25 -e $e

mv 2019* $targetdir
$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer (f1-alpha-match-10)")
$(cd $targetdir && git push)

git checkout embeddings/glove.6B.100d.txt

python3 main.py --train $train --dev $dev --test $test --save ../targer-save.sav --gpu -1 --save-best yes --cross-folds-num 5 -v f1-alpha-match-05 -p 25 -e $e

mv 2019* $targetdir
$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer (f1-alpha-match-05)")
$(cd $targetdir && git push)

git checkout embeddings/glove.6B.100d.txt

python3 main.py --train $train --dev $dev --test $test --save ../targer-save.sav --gpu -1 --save-best yes --cross-folds-num 5 -v f1-macro -p 25 -e $e

mv 2019* $targetdir
$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer (f1-macro)")
$(cd $targetdir && git push)


git checkout embeddings/glove.6B.100d.txt


python3 main.py --train $train --dev $dev --test $test --save ../targer-save.sav --gpu -1 --save-best yes --cross-folds-num 5 -v token-acc -p 25 -e $e

mv 2019* $targetdir
$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer (token-acc)")
$(cd $targetdir && git push)

