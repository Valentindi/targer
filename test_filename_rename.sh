FILE=$(find . -iname "2019*report.txt")
FILEID=$(echo $FILE | tr "report.txt" "\n")
echo $FILE 
echo $FILEID
mkdir $FILEID
mv embeddings/glove.6B.100d.txt $FILEID 
